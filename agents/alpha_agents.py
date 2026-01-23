"""
Alpha Generation Agents
========================

Production-grade agents for finding investment edge, modeled after
how real hedge fund analysts and platforms like AlphaSense work.

Key Insight: Consensus analysis is already priced in. Alpha comes from:
1. Variant Perception - Where is the market wrong?
2. Alternative Data - What signals are others missing?
3. Catalyst Identification - What will prove the thesis?
4. Sentiment Analysis - What is the market psychology?

Reference: AlphaSense uses AI to search 500M+ documents and provides
sentence-level citations with no hallucinations. We aim for similar rigor.

Sources:
- https://www.alpha-sense.com/
- https://intuitionlabs.ai/articles/alphasense-platform-review
"""

import os
from typing import Dict, List, Optional, Literal
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# =============================================================================
# PYDANTIC MODELS FOR STRICT TYPING
# =============================================================================

class VariantScenario(BaseModel):
    """A variant scenario that differs from consensus."""
    scenario_type: Literal["bull", "bear"]
    description: str = Field(description="What could happen differently than expected")
    probability: float = Field(ge=0, le=1, description="Probability 0-1")
    price_target: float = Field(gt=0, description="Target price if scenario plays out")
    key_drivers: List[str] = Field(description="What would cause this scenario")
    evidence: List[str] = Field(description="Current evidence supporting this scenario")
    invalidation: str = Field(description="What would prove this scenario wrong")


class VariantPerceptionOutput(BaseModel):
    """Output from the Variant Perception Agent."""
    ticker: str
    current_price: float

    # What's priced in
    market_expectations: str = Field(description="What the current price implies")
    consensus_narrative: str = Field(description="The story everyone believes")
    priced_in_growth: float = Field(description="Implied growth rate from valuation")

    # Variant scenarios
    bull_variant: VariantScenario
    bear_variant: VariantScenario

    # The edge
    variant_thesis: str = Field(description="Our differentiated view vs consensus")
    information_gaps: List[str] = Field(description="What we don't know that matters")

    # Confidence
    conviction_level: Literal["HIGH", "MEDIUM", "LOW", "PASS"]
    reasoning: str


class CatalystEvent(BaseModel):
    """An upcoming event that could move the stock."""
    event_type: Literal["earnings", "guidance", "product_launch", "fda_decision",
                        "analyst_day", "macro_event", "index_rebalancing", "other"]
    date: str = Field(description="Expected date or date range")
    description: str
    expected_impact: Literal["HIGH", "MEDIUM", "LOW"]
    probability_positive: float = Field(ge=0, le=1)
    upside_if_positive: float = Field(description="Expected % move if positive")
    downside_if_negative: float = Field(description="Expected % move if negative")
    variant_opportunity: str = Field(description="How this relates to our variant thesis")


class AlternativeDataSignal(BaseModel):
    """A signal from alternative data sources."""
    data_source: str = Field(description="Where this signal comes from")
    signal_type: Literal["bullish", "bearish", "neutral"]
    signal_strength: Literal["strong", "moderate", "weak"]
    description: str
    lead_time: str = Field(description="How far ahead this typically leads fundamentals")
    evidence: str = Field(description="The actual data point observed")
    citation: str = Field(description="Source URL or reference")


class SentimentAnalysis(BaseModel):
    """Sentiment analysis like AlphaSense's proprietary model."""
    overall_sentiment: Literal["very_positive", "positive", "neutral", "negative", "very_negative"]
    management_confidence: float = Field(ge=0, le=100, description="Management tone score 0-100")
    uncertainty_score: float = Field(ge=0, le=100, description="Level of hedging language 0-100")
    forward_guidance_tone: Literal["optimistic", "cautious", "pessimistic", "no_guidance"]
    key_phrases: List[Dict[str, str]] = Field(description="Important phrases with sentiment")
    sentiment_change: str = Field(description="How sentiment changed vs prior period")


class RiskRewardAnalysis(BaseModel):
    """Final risk/reward scoring with position sizing."""
    ticker: str
    current_price: float

    # Scenario analysis
    bull_case_price: float
    bull_case_probability: float
    base_case_price: float
    base_case_probability: float
    bear_case_price: float
    bear_case_probability: float

    # Calculated metrics
    expected_value: float = Field(description="Probability-weighted expected return %")
    risk_reward_ratio: float = Field(description="Upside/downside ratio")
    kelly_position_size: float = Field(ge=0, le=0.25, description="Optimal position 0-25%")

    # Recommendation
    action: Literal["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL", "PASS"]
    position_size_recommendation: str = Field(description="Recommended % of portfolio")
    entry_price: float = Field(description="Suggested entry point")
    stop_loss: float = Field(description="Stop loss price")
    take_profit_1: float = Field(description="First profit target")
    take_profit_2: Optional[float] = Field(description="Second profit target")

    # Reasoning
    investment_thesis: str
    key_risks: List[str]
    thesis_invalidation: str = Field(description="What would make us exit")


# =============================================================================
# VARIANT PERCEPTION AGENT
# =============================================================================

VARIANT_PERCEPTION_PROMPT = """You are a senior equity research analyst at a top hedge fund.
Your job is NOT to summarize consensus, but to find where the market might be WRONG.

Think like Stanley Druckenmiller: "If you don't have a variant perception, you don't have an edge."

## CONTEXT
Ticker: {ticker}
Current Price: ${current_price}
Consensus Report: {consensus_report}
Financial Data: {financial_data}

## FINANCIAL REASONING FRAMEWORK

### 1. VALUATION REVERSE-ENGINEERING (What's Priced In?)
Using the provided financial data, calculate:

a) **Implied Growth Rate** from current valuation:
   - If P/E is given: Implied growth = (P/E - market_average_PE) / PEG_assumption
   - Use Gordon Growth: P = D1 / (r - g), solve for g
   - Example: P/E of 35 vs market 20 implies ~15% excess growth expectations

b) **DCF Implied Assumptions**:
   - Back-solve for terminal growth rate given current price
   - What WACC is the market using? (hint: higher P/E = lower implied WACC)
   - What FCF growth is embedded in EV/EBITDA?

c) **Margin Expectations**:
   - Current gross margin vs implied steady-state
   - Operating leverage assumptions (fixed vs variable cost structure)
   - SG&A as % of revenue trajectory

### 2. BULL VARIANT - Financial Analysis (Consensus Too Pessimistic)
Identify specific financial drivers that could exceed expectations:

a) **Revenue Upside**:
   - TAM expansion not modeled (new products, geographies)
   - Market share gains (calculate: 1% share gain = $X revenue)
   - Pricing power (elasticity analysis)

b) **Margin Expansion**:
   - Operating leverage math: Fixed costs / (Revenue growth rate)
   - Gross margin drivers (mix shift, input costs, scale)
   - SG&A efficiency (revenue per employee trends)

c) **Capital Efficiency**:
   - ROIC improvement potential (NOPAT / Invested Capital)
   - Working capital optimization (days inventory, receivables, payables)
   - CapEx intensity declining (maintenance vs growth capex)

d) **Calculate Bull Case Price Target**:
   - Use: Bull EPS × Justified P/E = Bull Price
   - Or: Bull FCF × (1+g)/(WACC-g) = Bull Enterprise Value → Equity Value

### 3. BEAR VARIANT - Financial Analysis (Consensus Too Optimistic)
Identify specific financial risks:

a) **Revenue Downside**:
   - Customer concentration risk (top 10 customers = X% revenue)
   - Competitive threats (market share loss = $X revenue impact)
   - Cyclicality (peak-to-trough revenue decline historically)

b) **Margin Compression**:
   - Input cost inflation (COGS as % of revenue sensitivity)
   - Competitive pricing pressure
   - Deleverage math: If revenue falls X%, EBIT falls Y%

c) **Balance Sheet Risks**:
   - Debt/EBITDA and interest coverage trends
   - Covenant headroom analysis
   - Refinancing risk (debt maturity schedule)
   - Goodwill impairment risk (goodwill as % of equity)

d) **Calculate Bear Case Price Target**:
   - Use: Trough EPS × Trough P/E = Bear Price
   - Or: Stressed FCF with higher discount rate

### 4. FINANCIAL STATEMENT RED FLAGS
Look for accounting quality issues:
- Revenue recognition changes
- Receivables growing faster than revenue (DSO trend)
- Inventory buildup (DOI trend)
- Capitalized vs expensed costs
- Related party transactions
- Non-GAAP adjustments magnitude

### 5. VARIANT THESIS
State your differentiated financial view:
- Which specific line item will surprise (revenue, margins, capex)?
- Quantify the impact: "$X EPS surprise = $Y price target"
- What financial catalyst will prove you right?

IMPORTANT: Use actual numbers from the financial data provided.
Calculate specific impacts. This is professional financial analysis.
If you don't have a variant, say PASS - don't force an edge that isn't there.
"""

class VariantPerceptionAgent:
    """
    Identifies where market expectations may be wrong.

    This is the core alpha-generation engine. It takes consensus analysis
    and finds exploitable mispricings.
    """

    def __init__(self, model: str = "gpt-5-nano"):
        self.llm = ChatOpenAI(model=model, temperature=0.1)
        self.structured_llm = self.llm.with_structured_output(VariantPerceptionOutput)

    def analyze(
        self,
        ticker: str,
        current_price: float,
        consensus_report: str,
        financial_data: Dict
    ) -> VariantPerceptionOutput:
        """
        Generate variant perception analysis.

        Args:
            ticker: Stock symbol
            current_price: Current stock price
            consensus_report: Output from your existing consensus agent
            financial_data: Dict with PE, growth rates, etc.

        Returns:
            VariantPerceptionOutput with variant thesis and scenarios
        """
        prompt = ChatPromptTemplate.from_template(VARIANT_PERCEPTION_PROMPT)

        chain = prompt | self.structured_llm

        result = chain.invoke({
            "ticker": ticker,
            "current_price": current_price,
            "consensus_report": consensus_report,
            "financial_data": str(financial_data)
        })

        return result


# =============================================================================
# CATALYST AGENT
# =============================================================================

CATALYST_PROMPT = """You are a catalyst-focused equity research analyst.

Your job is to identify UPCOMING EVENTS that could move this stock and validate
(or invalidate) the investment thesis through FINANCIAL IMPACT.

Think: "Being right isn't enough. You need a catalyst that proves it - in the numbers."

## CONTEXT
Ticker: {ticker}
Current Date: {current_date}
Investment Thesis: {investment_thesis}
Financial Calendar: {financial_calendar}
Industry Events: {industry_events}

## FINANCIAL CATALYST FRAMEWORK

For each catalyst, provide QUANTIFIED financial impact:

### 1. EARNINGS CATALYSTS
- **Consensus Estimates**: What's the Street expecting? (EPS, Revenue)
- **Whisper Number**: What do buy-side analysts really think?
- **Beat/Miss Math**:
  - $0.05 EPS beat × 15 P/E = $0.75 price impact
  - But multiple expansion on beat could add another 10-20%
- **Guidance Impact**: Forward estimates matter more than backward results
- **Estimate Revision Cycle**: When do analysts update models?

### 2. MARGIN CATALYSTS
- **Gross Margin Inflection**: Input cost changes, pricing power realization
- **Operating Leverage Events**: Revenue crossing fixed cost threshold
- **Restructuring Completion**: When do cost savings hit P&L?
- **Calculate**: 100bps margin improvement × Revenue = $X EBIT impact

### 3. REVENUE CATALYSTS
- **Product Launch Dates**: Revenue contribution timeline
- **Contract Announcements**: Deal size × probability × margin = EPS impact
- **Backlog Conversion**: When does backlog convert to revenue?
- **Seasonality**: Which quarters have easiest/hardest comps?

### 4. BALANCE SHEET CATALYSTS
- **Debt Refinancing**: Interest expense savings = EPS impact
- **Buyback Acceleration**: Share count reduction × EPS = accretion
- **Dividend Initiation/Increase**: Yield support for stock price
- **M&A Announcements**: Accretion/dilution math

### 5. VALUATION CATALYSTS
- **Index Inclusion/Exclusion**: Forced buying (calculate shares needed)
- **Analyst Initiation**: Coverage expansion = multiple expansion
- **Peer Re-rating**: Sector rotation, comp multiple changes
- **Short Interest Changes**: Days to cover, squeeze potential

### 6. BINARY EVENTS
- **FDA Decisions**: Approval probability × peak sales × margin = NPV
- **Legal Outcomes**: Liability range, settlement probability
- **Regulatory Rulings**: Revenue at risk calculation

## OUTPUT FORMAT
For each catalyst:
1. Event and expected date
2. Positive outcome scenario + probability + price impact (with math)
3. Negative outcome scenario + probability + price impact (with math)
4. How it validates/invalidates the variant thesis

IMPORTANT: QUANTIFY EVERYTHING. "$X revenue at Y margin = $Z EPS = $W price target"
Focus on the NEXT 6 MONTHS. Prioritize by expected value (probability × impact).
"""

class CatalystAgent:
    """
    Maps upcoming events that could move the stock.

    Key insight: Timing matters. A thesis without a catalyst is just a hope.
    """

    def __init__(self, model: str = "gpt-5-nano"):
        self.llm = ChatOpenAI(model=model, temperature=0.1)

    def identify_catalysts(
        self,
        ticker: str,
        investment_thesis: str,
        financial_calendar: Dict = None,
        industry_events: List[str] = None
    ) -> List[CatalystEvent]:
        """
        Identify upcoming catalysts for the investment thesis.
        """
        structured_llm = self.llm.with_structured_output(
            # Return list of catalysts
            type("CatalystList", (), {"catalysts": (List[CatalystEvent], Field(description="List of catalysts"))})
        )

        prompt = ChatPromptTemplate.from_template(CATALYST_PROMPT)

        chain = prompt | structured_llm

        result = chain.invoke({
            "ticker": ticker,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "investment_thesis": investment_thesis,
            "financial_calendar": str(financial_calendar or {}),
            "industry_events": "\n".join(industry_events or [])
        })

        return result.catalysts


# =============================================================================
# SENTIMENT AGENT (AlphaSense-style)
# =============================================================================

SENTIMENT_PROMPT = """You are a senior earnings call analyst, similar to AlphaSense's
proprietary sentiment model trained on 10+ years of earnings calls.

Analyze this earnings call transcript for FINANCIAL SENTIMENT SIGNALS that predict
future financial performance and estimate revisions.

## TRANSCRIPT
{transcript}

## FINANCIAL SENTIMENT FRAMEWORK

### 1. REVENUE CONFIDENCE INDICATORS (Score 0-100)
Analyze language around top-line:

**Strong Revenue Signals (High Score)**:
- "Demand exceeds our capacity" → Pricing power
- "Record backlog" → Revenue visibility
- "Pipeline at all-time high" → Future growth
- "Taking market share" → Competitive strength
- Specific quantitative guidance with narrow ranges

**Weak Revenue Signals (Low Score)**:
- "Macro uncertainty impacting demand" → Cyclical risk
- "Customers delaying decisions" → Pipeline risk
- "Competitive pricing pressure" → Margin compression
- Wide guidance ranges, qualitative hedging
- "Depends on" / "Subject to" / "If market conditions"

### 2. MARGIN CONFIDENCE INDICATORS (Score 0-100)
Analyze language around profitability:

**Positive Margin Signals**:
- "Operating leverage kicking in" → Margin expansion
- "Price increases sticking" → Pricing power realized
- "Cost actions ahead of schedule" → Restructuring working
- "Mix shift toward higher margin" → Quality of revenue
- Specific gross margin / operating margin guidance

**Negative Margin Signals**:
- "Investing for growth" → Near-term margin pressure
- "Input cost inflation" → COGS pressure
- "Promotional environment" → Pricing weakness
- "Ramping new facilities" → Startup costs
- "One-time charges" (how many quarters is "one-time"?)

### 3. BALANCE SHEET / CASH FLOW LANGUAGE
- "Strong cash generation" → Financial flexibility
- "Returning capital to shareholders" → Confidence + low reinvestment needs
- "Strengthening the balance sheet" → Deleveraging priority
- "Strategic optionality" → M&A interest
- "Working capital normalization" → Cash flow improvement

### 4. GUIDANCE QUALITY ANALYSIS
Compare guidance language to actual financial impact:

**High Quality Guidance**:
- Specific ranges: "Revenue of $4.2-4.3B" (2.4% range)
- EPS guidance with clear bridge from prior quarter
- Margin assumptions explicitly stated
- CapEx and FCF guidance provided

**Low Quality Guidance**:
- Wide ranges: "Revenue of $4.0-4.5B" (12.5% range = uncertainty)
- "In line with Street" (no conviction)
- Guidance withdrawn or not provided
- Excessive non-GAAP adjustments in outlook

### 5. MANAGEMENT TONE VS REALITY
Compare actual results to language:
- Did they beat/miss? By how much?
- Is tone consistent with results? (Upbeat on a miss = concerning)
- Are they managing expectations down for sandbagging?
- What's NOT being discussed? (Often more important)

### 6. KEY FINANCIAL PHRASE EXTRACTION
Extract 5-10 phrases with FINANCIAL IMPACT assessment:

Format: "[Quote]" → Financial Impact → Sentiment
Examples:
- "Gross margin expanded 150bps" → $X EBIT impact → POSITIVE
- "Macro headwinds in Q4" → Revenue risk $X-Y → NEGATIVE
- "Record contracted backlog" → Revenue visibility 12+ months → POSITIVE
- "Customer inventory destocking" → Near-term revenue pressure → NEGATIVE

### 7. ESTIMATE REVISION PREDICTOR
Based on this call, predict analyst reaction:
- Will consensus estimates go UP, DOWN, or UNCHANGED?
- Which line items are most likely to be revised?
- What's the magnitude? ($X EPS revision = $Y price impact)

IMPORTANT: Every observation must tie to a FINANCIAL OUTCOME.
Provide exact quotes with sentence-level citations. This is professional analysis.
"""

class SentimentAgent:
    """
    Analyzes management sentiment from earnings calls and filings.

    Modeled after AlphaSense's proprietary sentiment analysis.
    Key: This catches subtle changes in management tone that precede results.
    """

    def __init__(self, model: str = "gpt-5-nano"):
        self.llm = ChatOpenAI(model=model, temperature=0.0)  # Zero temp for consistency
        self.structured_llm = self.llm.with_structured_output(SentimentAnalysis)

    def analyze_transcript(self, transcript: str) -> SentimentAnalysis:
        """
        Analyze earnings call transcript for sentiment signals.

        Args:
            transcript: Full earnings call transcript text

        Returns:
            SentimentAnalysis with management confidence, tone, key phrases
        """
        prompt = ChatPromptTemplate.from_template(SENTIMENT_PROMPT)
        chain = prompt | self.structured_llm

        return chain.invoke({"transcript": transcript})


# =============================================================================
# FINANCIAL ANALYSIS HELPERS
# =============================================================================

def calculate_implied_growth_rate(pe_ratio: float, market_pe: float = 20, peg_assumption: float = 1.5) -> float:
    """
    Reverse-engineer implied growth rate from P/E ratio.

    Formula: Implied Growth = (P/E - Market P/E) / PEG
    Example: P/E of 35 vs market 20 with PEG 1.5 = (35-20)/1.5 = 10% implied growth premium
    """
    if pe_ratio <= 0:
        return 0.0
    excess_pe = pe_ratio - market_pe
    implied_growth = excess_pe / peg_assumption
    return round(implied_growth / 100, 4)  # Return as decimal


def calculate_dcf_implied_value(
    fcf: float,
    growth_rate: float,
    terminal_growth: float = 0.03,
    wacc: float = 0.10,
    years: int = 5,
    shares_outstanding: float = 1.0
) -> float:
    """
    Calculate intrinsic value per share using DCF.

    Two-stage model:
    1. High growth period (years 1-5)
    2. Terminal value with perpetual growth
    """
    if wacc <= terminal_growth:
        return 0.0

    # Stage 1: Project FCF for growth period
    projected_fcf = []
    current_fcf = fcf
    for year in range(1, years + 1):
        current_fcf *= (1 + growth_rate)
        discount_factor = 1 / ((1 + wacc) ** year)
        projected_fcf.append(current_fcf * discount_factor)

    # Stage 2: Terminal value
    terminal_fcf = current_fcf * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    terminal_pv = terminal_value / ((1 + wacc) ** years)

    # Enterprise value
    enterprise_value = sum(projected_fcf) + terminal_pv

    # Per share (simplified - should subtract debt and add cash)
    return round(enterprise_value / shares_outstanding, 2)


def calculate_operating_leverage(
    fixed_costs: float,
    revenue: float,
    variable_margin: float
) -> dict:
    """
    Calculate operating leverage and breakeven.

    Operating Leverage = Contribution Margin / Operating Income
    Higher leverage = bigger swings in profitability from revenue changes
    """
    contribution_margin = revenue * variable_margin
    operating_income = contribution_margin - fixed_costs

    if operating_income <= 0:
        return {"leverage": float('inf'), "breakeven_revenue": fixed_costs / variable_margin}

    leverage = contribution_margin / operating_income
    breakeven = fixed_costs / variable_margin

    return {
        "leverage": round(leverage, 2),
        "breakeven_revenue": round(breakeven, 2),
        "margin_of_safety": round((revenue - breakeven) / revenue * 100, 1)
    }


def calculate_roic(
    nopat: float,
    invested_capital: float
) -> float:
    """
    Return on Invested Capital = NOPAT / Invested Capital

    NOPAT = Operating Income × (1 - Tax Rate)
    Invested Capital = Total Equity + Total Debt - Cash

    ROIC > WACC creates value. ROIC < WACC destroys value.
    """
    if invested_capital <= 0:
        return 0.0
    return round(nopat / invested_capital * 100, 2)


def calculate_eps_sensitivity(
    revenue_change_pct: float,
    current_revenue: float,
    gross_margin: float,
    operating_margin: float,
    tax_rate: float = 0.21,
    shares_outstanding: float = 1.0
) -> float:
    """
    Calculate EPS impact from revenue change.

    Uses gross margin to estimate variable vs fixed cost structure,
    then applies operating leverage to project EPS impact.

    Useful for variant scenario analysis.
    """
    revenue_delta = current_revenue * revenue_change_pct

    # Estimate operating leverage from margin structure
    # Higher gross margin relative to operating margin = more fixed costs = higher leverage
    if operating_margin > 0:
        implied_leverage = gross_margin / operating_margin
        operating_leverage = min(max(implied_leverage, 1.5), 4.0)  # Bound between 1.5x-4x
    else:
        operating_leverage = 2.0  # Default assumption

    ebit_delta = revenue_delta * operating_margin * operating_leverage
    net_income_delta = ebit_delta * (1 - tax_rate)
    eps_delta = net_income_delta / shares_outstanding

    return round(eps_delta, 2)


# =============================================================================
# RISK/REWARD SCORER
# =============================================================================

class RiskRewardScorer:
    """
    Calculates expected value and position sizing using financial analysis.

    Uses professional hedge fund frameworks:
    1. Probability-weighted scenario analysis
    2. Kelly Criterion for position sizing
    3. Risk-adjusted return metrics (Sortino-style)
    4. Valuation-based entry/exit points
    """

    def calculate(
        self,
        ticker: str,
        current_price: float,
        variant_analysis: VariantPerceptionOutput,
        catalysts: List[CatalystEvent] = None,
        financial_data: Dict = None
    ) -> RiskRewardAnalysis:
        """
        Calculate risk/reward and recommended position size using financial metrics.
        """
        bull = variant_analysis.bull_variant
        bear = variant_analysis.bear_variant

        # Use financial data for valuation context if available
        pe_ratio = financial_data.get("pe_ratio", 20) if financial_data else 20
        implied_growth = calculate_implied_growth_rate(pe_ratio) if pe_ratio > 0 else 0

        # Base case is current price (consensus already priced in)
        base_case_price = current_price
        base_case_prob = 1 - bull.probability - bear.probability
        base_case_prob = max(0.1, min(0.5, base_case_prob))  # Sanity check

        # Calculate returns
        bull_return = (bull.price_target - current_price) / current_price
        base_return = 0  # Consensus = no alpha
        bear_return = (bear.price_target - current_price) / current_price

        # Valuation adjustment: reduce position size for expensive stocks
        # High implied growth = high expectations = more downside risk if missed
        valuation_haircut = 1.0
        if implied_growth > 0.15:  # >15% implied growth is aggressive
            valuation_haircut = 0.7  # 30% smaller position for expensive stocks
        elif implied_growth > 0.10:
            valuation_haircut = 0.85

        # Expected value (probability-weighted return)
        ev = (bull_return * bull.probability +
              base_return * base_case_prob +
              bear_return * bear.probability)

        # Risk/reward ratio (asymmetric)
        upside = bull_return * bull.probability
        downside = abs(bear_return * bear.probability)
        risk_reward = upside / downside if downside > 0 else 10

        # Kelly Criterion for position sizing
        # f* = (p * b - q) / b
        # where p = win prob, q = lose prob, b = win/loss ratio
        win_prob = bull.probability + (base_case_prob * 0.3)  # Partial credit for base
        lose_prob = bear.probability + (base_case_prob * 0.3)
        win_loss_ratio = abs(bull_return / bear_return) if bear_return != 0 else 2

        kelly_full = (win_prob * win_loss_ratio - lose_prob) / win_loss_ratio
        kelly_full = max(0, kelly_full)

        # Use fractional Kelly (half Kelly is industry standard for reduced volatility)
        kelly_half = kelly_full * 0.5
        kelly = min(kelly_half, 0.25)  # Cap at 25% of portfolio

        # Apply valuation haircut (reduce position for expensive stocks)
        kelly = kelly * valuation_haircut

        # Adjust for catalyst proximity (higher position if catalyst imminent)
        catalyst_multiplier = 1.0
        if catalysts:
            # Find nearest high-impact catalyst
            high_impact_catalysts = [c for c in catalysts if c.expected_impact == "HIGH"]
            if high_impact_catalysts:
                catalyst_multiplier = 1.2  # 20% boost for imminent catalyst

        kelly = min(kelly * catalyst_multiplier, 0.25)

        # Determine action based on financial thresholds
        if ev > 0.15 and risk_reward > 3 and variant_analysis.conviction_level == "HIGH":
            action = "STRONG_BUY"
            position = f"{kelly*100:.0f}% of portfolio (full conviction)"
        elif ev > 0.08 and risk_reward > 2:
            action = "BUY"
            position = f"{kelly*100*0.7:.0f}% of portfolio"
        elif ev > 0.03 and risk_reward > 1.5:
            action = "HOLD"
            position = "Maintain current position"
        elif ev > -0.05:
            action = "SELL"
            position = "Reduce by 50%"
        else:
            action = "STRONG_SELL"
            position = "Exit position"

        # Handle PASS case
        if variant_analysis.conviction_level == "PASS":
            action = "PASS"
            position = "No position - no edge identified"
            kelly = 0

        # Calculate entry/exit points based on valuation
        # Entry: Look for 2-3% margin of safety
        entry_price = round(current_price * 0.975, 2)

        # Stop loss: Just below bear case with 5% buffer for volatility
        stop_loss = round(bear.price_target * 0.95, 2)

        # Take profits: Scale out at 50% and 90% of bull target
        take_profit_1 = round(current_price + (bull.price_target - current_price) * 0.5, 2)
        take_profit_2 = round(current_price + (bull.price_target - current_price) * 0.9, 2)

        # Build investment thesis with financial reasoning
        thesis_parts = [
            variant_analysis.variant_thesis,
            f"EV: {ev*100:+.1f}% | R:R {risk_reward:.1f}:1",
            f"Entry ${entry_price} | Stop ${stop_loss} | TP ${take_profit_1}/{take_profit_2}"
        ]
        investment_thesis = " | ".join(thesis_parts)

        return RiskRewardAnalysis(
            ticker=ticker,
            current_price=current_price,
            bull_case_price=bull.price_target,
            bull_case_probability=bull.probability,
            base_case_price=base_case_price,
            base_case_probability=base_case_prob,
            bear_case_price=bear.price_target,
            bear_case_probability=bear.probability,
            expected_value=round(ev * 100, 2),  # As percentage
            risk_reward_ratio=round(risk_reward, 2),
            kelly_position_size=round(kelly, 4),
            action=action,
            position_size_recommendation=position,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            investment_thesis=investment_thesis,
            key_risks=bear.key_drivers,
            thesis_invalidation=bear.invalidation
        )


# =============================================================================
# MAIN ALPHA WORKFLOW
# =============================================================================

def run_alpha_analysis(
    ticker: str,
    current_price: float,
    consensus_report: str,
    financial_data: Dict,
    earnings_transcript: str = None
) -> Dict:
    """
    Run the complete alpha generation workflow.

    This is the main entry point that coordinates all alpha agents.

    Args:
        ticker: Stock symbol
        current_price: Current stock price
        consensus_report: Output from existing consensus analysis
        financial_data: Dict with financials (PE, growth, etc)
        earnings_transcript: Optional latest earnings call

    Returns:
        Complete alpha analysis with actionable recommendation
    """
    print(f"\n{'='*70}")
    print(f"ALPHA ANALYSIS: {ticker}")
    print(f"{'='*70}")

    # Step 1: Variant Perception
    print("\n[1/4] Running Variant Perception Analysis...")
    variant_agent = VariantPerceptionAgent()
    variant = variant_agent.analyze(
        ticker=ticker,
        current_price=current_price,
        consensus_report=consensus_report,
        financial_data=financial_data
    )
    print(f"  Conviction: {variant.conviction_level}")
    print(f"  Variant Thesis: {variant.variant_thesis[:100]}...")

    # Step 2: Catalyst Identification
    print("\n[2/4] Identifying Catalysts...")
    catalyst_agent = CatalystAgent()
    catalysts = catalyst_agent.identify_catalysts(
        ticker=ticker,
        investment_thesis=variant.variant_thesis
    )
    print(f"  Found {len(catalysts)} catalysts")
    for cat in catalysts[:3]:
        print(f"    - {cat.event_type}: {cat.date} ({cat.expected_impact} impact)")

    # Step 3: Sentiment Analysis (if transcript provided)
    sentiment = None
    if earnings_transcript:
        print("\n[3/4] Analyzing Management Sentiment...")
        sentiment_agent = SentimentAgent()
        sentiment = sentiment_agent.analyze_transcript(earnings_transcript)
        print(f"  Overall: {sentiment.overall_sentiment}")
        print(f"  Management Confidence: {sentiment.management_confidence}/100")
    else:
        print("\n[3/4] Sentiment Analysis skipped (no transcript)")

    # Step 4: Risk/Reward Scoring
    print("\n[4/4] Calculating Risk/Reward...")
    scorer = RiskRewardScorer()
    risk_reward = scorer.calculate(
        ticker=ticker,
        current_price=current_price,
        variant_analysis=variant,
        catalysts=catalysts
    )
    print(f"  Expected Value: {risk_reward.expected_value:+.1f}%")
    print(f"  Risk/Reward: {risk_reward.risk_reward_ratio:.1f}:1")
    print(f"  Action: {risk_reward.action}")
    print(f"  Position Size: {risk_reward.position_size_recommendation}")

    # Compile final output
    result = {
        "ticker": ticker,
        "current_price": current_price,
        "analysis_date": datetime.now().isoformat(),

        # Variant analysis
        "variant_perception": variant.model_dump(),

        # Catalysts
        "catalysts": [c.model_dump() for c in catalysts],

        # Sentiment
        "sentiment": sentiment.model_dump() if sentiment else None,

        # Final recommendation
        "risk_reward": risk_reward.model_dump(),

        # Summary
        "summary": {
            "action": risk_reward.action,
            "conviction": variant.conviction_level,
            "expected_value": f"{risk_reward.expected_value:+.1f}%",
            "risk_reward_ratio": f"{risk_reward.risk_reward_ratio:.1f}:1",
            "position_size": risk_reward.position_size_recommendation,
            "thesis": variant.variant_thesis,
            "next_catalyst": catalysts[0].description if catalysts else "None identified",
            "key_risk": risk_reward.key_risks[0] if risk_reward.key_risks else "None identified"
        }
    }

    print(f"\n{'='*70}")
    print("ALPHA ANALYSIS COMPLETE")
    print(f"{'='*70}")

    return result


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Example usage
    result = run_alpha_analysis(
        ticker="NVDA",
        current_price=145.00,
        consensus_report="""
        NVIDIA is a leading AI chip company. Analysts expect 25% revenue growth.
        P/E ratio is 35x, above historical average. Strong data center demand.
        Consensus price target: $160.
        """,
        financial_data={
            "pe_ratio": 35,
            "revenue_growth": 0.25,
            "gross_margin": 0.65,
            "debt_to_equity": 0.4
        }
    )

    print("\n\nFINAL RECOMMENDATION:")
    print(f"Action: {result['summary']['action']}")
    print(f"Expected Value: {result['summary']['expected_value']}")
    print(f"Position Size: {result['summary']['position_size']}")
    print(f"Thesis: {result['summary']['thesis']}")
