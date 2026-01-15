import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Use absolute imports based on your root directory (/app)
from agents.state import AgentState
from agents.prompts import PLANNER_SYSTEM, WRITER_SYSTEM, GRADER_SYSTEM
from agents.retriever import get_pinecone_retriever
from services.pinecone_llamaindex import query_pinecone_llamaindex

# 1. Initialize Global Models
# Senior Reasoning: GPT-4o for complex planning and writing
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. Local Hugging Face Setup (The "Lockheed" Expert Way)
model_id = "microsoft/Phi-3-mini-4k-instruct"

# Quantization Config: Reduces 16-bit to 4-bit (Saves 60-70% VRAM/RAM)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load Model with local cache persistence
tokenizer = AutoTokenizer.from_pretrained(model_id)
hf_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=quant_config,
    device_map="auto" # Automatically handles CPU/GPU split
)

# Wrap in LangChain Pipeline
hf_pipe = pipeline(
    "text-generation",
    model=hf_model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.1,
    return_full_text=False
)
eval_model = HuggingFacePipeline(pipeline=hf_pipe)

# --- Node Definitions ---

def planner_node(state: AgentState):
    new_count = state.get("loop_count", 0) + 1
    response = model.invoke([
        {"role": "system", "content": PLANNER_SYSTEM}, 
        {"role": "user", "content": state["task"]}
    ])
    return {"plan": [response.content], "loop_count": new_count}

def researcher_node(state: AgentState):
    # Cross-service call to your LlamaIndex service
    internal_context = query_pinecone_llamaindex(state["task"])
    return {"research_notes": [f"Retrieved Context: {internal_context}"]}

def writer_node(state: AgentState):
    full_context = "\n".join(state["research_notes"])
    response = model.invoke([
        {"role": "system", "content": WRITER_SYSTEM}, 
        {"role": "user", "content": f"Use this context: {full_context} to complete: {state['task']}"}
    ])
    return {"report": response.content}

def grader_node(state: AgentState):
    # Utilizing the local Hugging Face model for the grading task
    prompt = f"{GRADER_SYSTEM}\n\nReview this report and provide a score out of 100:\n{state['report']}"
    response = eval_model.invoke(prompt)
    
    # Robust numeric extraction
    nums = re.findall(r'\d+', str(response))
    if nums:
        # Take the last number to avoid metadata/dates
        score = int(nums[-1])
        # Auto-normalize if model gives 1-10 instead of 1-100
        if score <= 10: score *= 10
    else:
        score = 0
        
    return {"score": min(score, 100)}

# --- Graph Assembly ---

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)    
workflow.add_node("grader", grader_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "grader")

def decide_to_end(state: AgentState):
    if state.get("loop_count", 0) >= 3:
        return "end"
    if state.get("score", 0) < 85:
        return "researcher"
    return "end"

workflow.add_conditional_edges(
    "grader", 
    decide_to_end,
    {
        "researcher": "researcher",
        "end": END
    }
)

app = workflow.compile()