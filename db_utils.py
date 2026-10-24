import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "data/assistant.db")
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")
THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.72"))

LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "api")
EMBED_MODEL = os.getenv("EMBED_MODEL")

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

# 向量函数分支：只有local模式才导入
if EMBED_PROVIDER == "api":
    def get_embedding(text: str):
        resp = client.embeddings.create(model=EMBED_MODEL, input=text)
        return resp.data[0].embedding
else:
    # 【重点】import写在if里面！只有本地local模式才会执行这行
    from sentence_transformers import SentenceTransformer
    _embedding_model = SentenceTransformer(EMBED_MODEL)
    def get_embedding(text: str):
        return _embedding_model.encode(text).tolist()


chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name="knowledge_base")

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS qa
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  question TEXT, answer TEXT, source TEXT, created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS unanswered
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  question TEXT, status TEXT, created_at TEXT)''')
    conn.commit()
    conn.close()

def add_qa(question, answer, source="manual"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO qa (question, answer, source, created_at) VALUES (?,?,?,?)",
              (question, answer, source, now))
    conn.commit()
    qid = c.lastrowid
    conn.close()
    emb = get_embedding(question)
    collection.add(
        ids=[str(qid)],
        embeddings=[emb],
        documents=[question],
        metadatas=[{"answer": answer}]
    )
    return qid

def get_all_qa():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id,question,answer,source,created_at FROM qa ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    res = []
    for r in rows:
        res.append({
            "id": r[0],
            "question": r[1],
            "answer": r[2],
            "source": r[3],
            "created_at": r[4]
        })
    return res

def delete_qa(qid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    conn.commit()
    conn.close()
    collection.delete(ids=[str(qid)])

def search_similar(query, top_k=3):
    query_emb = get_embedding(query)
    res = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )
    out = []
    if res and res["ids"] and len(res["ids"][0])>0:
        for idx, qid in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][idx]
            score = 1 - res["distances"][0][idx]
            if score >= THRESHOLD:
                out.append({
                    "id": int(qid),
                    "question": res["documents"][0][idx],
                    "answer": meta["answer"],
                    "score": score
                })
    return out

def generate_answer(question, ctx_list, allow_free=False):
    if ctx_list:
        context_text = "\n".join([f"Q:{x['question']}\nA:{x['answer']}" for x in ctx_list])
        prompt = f"""参考下面知识库内容回答用户问题：
{context_text}
用户问题：{question}
"""
    else:
        if not allow_free:
            return "知识库中没有相关内容"
        prompt = f"回答用户问题：{question}"
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

def record_unanswered(q):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO unanswered (question, answer, source, created_at) VALUES (?,?,?,?)",
              (q, "pending", now))
    conn.commit()
    conn.close()

