import os
import sqlite3
from datetime import datetime
import chromadb
import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ============ 基础配置 ============
# 默认值已直接指向火山方舟：即使云端漏配环境变量，应用也能正常启动
DB_PATH = os.getenv("DB_PATH", "data/assistant.db")
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")
THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.50"))

LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "ep-20261023195449-bfmjn")

# ============ 向量化配置（固定走方舟多模态向量接口，纯文本输入）============
# 不再使用本地 sentence-transformers 模型，因此云端无需安装它和 torch，
# 任何 EMBED_PROVIDER 残留值都不会再触发 ModuleNotFoundError
EMBED_MODEL = os.getenv("EMBED_MODEL", "doubao-embedding-vision-251215")
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "1024"))

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
# 多模态向量接口路径与 OpenAI SDK 内置的 /embeddings 不同，用 httpx 直连
_http = httpx.Client(timeout=30.0)


def get_embedding(text: str):
    """调用方舟 /embeddings/multimodal 接口，返回定长浮点向量。"""
    payload = {
        "model": EMBED_MODEL,
        "input": [{"type": "text", "text": text}],
        "encoding_format": "float",
        "dimensions": EMBED_DIMENSIONS,
    }
    resp = _http.post(
        LLM_BASE_URL.rstrip("/") + "/embeddings/multimodal",
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    resp.raise_for_status()
    return resp.json()["data"]["embedding"]


# ============ SQLite 初始化 ============
def init_db():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 问答表
    c.execute('''CREATE TABLE IF NOT EXISTS qa
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  question TEXT, answer TEXT, source TEXT, created_at TEXT)''')
    # 未回答记录表
    c.execute('''CREATE TABLE IF NOT EXISTS unanswered
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  question TEXT, status TEXT, created_at TEXT)''')
    conn.commit()
    conn.close()


# ============ Chroma 向量库（余弦距离，1 - distance 即余弦相似度）============
def _init_collection():
    chroma_dir = os.path.dirname(CHROMA_DIR)
    if chroma_dir:
        os.makedirs(chroma_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    cosine_meta = {"hnsw:space": "cosine"}
    try:
        col = chroma_client.get_collection(name="knowledge_base")
    except Exception:
        col = None
    if col is None:
        return chroma_client.create_collection(name="knowledge_base", metadata=cosine_meta)
    # 旧集合若不是 cosine 空间则重建（此前向量化从未成功写入，重建无数据损失），
    # 保证相似度分数与 THRESHOLD 阈值口径一致
    space = (col.metadata or {}).get("hnsw:space", "l2")
    if space != "cosine":
        chroma_client.delete_collection(name="knowledge_base")
        return chroma_client.create_collection(name="knowledge_base", metadata=cosine_meta)
    return col


collection = _init_collection()


# ============ 知识库 CRUD ============
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
    c.execute("DELETE FROM qa WHERE id=?", (qid,))
    conn.commit()
    conn.close()
    collection.delete(ids=[str(qid)])


def search_similar(query, top_k=3):
    """语义检索，只返回相似度 >= THRESHOLD 的结果。"""
    query_emb = get_embedding(query)
    res = collection.query(query_embeddings=[query_emb], n_results=top_k)
    out = []
    if res and res["ids"] and len(res["ids"][0]) > 0:
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
        prompt = f"""参考下面知识库内容回答用户问题，只使用参考内容回答：
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


# ============ 未回答问题管理 ============
def record_unanswered(q):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO unanswered (question, status, created_at) VALUES (?,?,?)",
              (q, "pending", now))
    conn.commit()
    conn.close()


def get_unanswered(status="pending"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id,question,created_at FROM unanswered WHERE status=? ORDER BY id DESC",
              (status,))
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "question": r[1], "created_at": r[2]} for r in rows]


def resolve_unanswered(uid, ans):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 必须先取出问题：状态改成 resolved 后就无法再按 pending 查到
    row = c.execute("SELECT question FROM unanswered WHERE id=?", (uid,)).fetchone()
    c.execute("UPDATE unanswered SET status='resolved' WHERE id=?", (uid,))
    conn.commit()
    conn.close()
    # 处理后自动入库
    if row:
        add_qa(row[0], ans, source="unresolved_add")


def ignore_unanswered(uid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE unanswered SET status='ignored' WHERE id=?", (uid,))
    conn.commit()
    conn.close()


def import_records(json_list):
    cnt = 0
    for item in json_list:
        q = item.get("question")
        a = item.get("answer")
        if q and a:
            add_qa(q, a, source="json_import")
            cnt += 1
    return cnt
