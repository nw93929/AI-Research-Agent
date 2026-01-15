import os, psycopg2, redis, requests

def test_connections():
    try:
        # Test Postgres
        conn = psycopg2.connect(os.getenv("POSTGRES_URI"))
        print("Postgres: Connected")
        
        # Test Redis
        r = redis.from_url(os.getenv("REDIS_URL"))
        print(f"Redis: Ping {r.ping()}")
        
        # Test Ollama
        response = requests.get(os.getenv("OLLAMA_HOST"))
        if response.status_code == 200:
            print("Ollama: Accessible")
    except Exception as e:
        print(f"Connection Failed: {e}")

test_connections()