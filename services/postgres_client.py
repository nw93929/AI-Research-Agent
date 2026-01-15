import psycopg2
import os

'''
query internal Postgres database
'''

def query_internal_db(sql_query: str):
    conn = psycopg2.connect(os.getenv("POSTGRES_URI"))
    cur = conn.cursor()
    try:
        cur.execute(sql_query)
        results = cur.fetchall()
        return f"Database Results: {str(results)}"
    finally:
        cur.close()
        conn.close()