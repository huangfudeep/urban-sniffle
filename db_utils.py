import os
import sqlite3
import chromadb
import httpx
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

# 读取配置
DB_PATH = os.getenv("DB_PATH", "data/assistant.db")
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")
THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))

LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
# 方舟多模态向量模型的可选参数
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS")) if os.getenv("EMBED_DIMENSIONS") else None
EMBED_INSTRUCTIONS = os.getenv("EMBED_INSTRUCTIONS", "")

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
# 向量化接口走普通 HTTP（多模态向量接口与 OpenAI SDK 的 embeddings 路径不同）
_http = httpx.Client(timeout=30.0)

# 本地embedding模型，正确初始化
_embedding_model = None
if EMBED_PROVIDER == "local":
    _embedding_model = SentenceTransformer(EMBED_MODEL)


def get_embedding(text: str):
    if EMBED_PROVIDER == "local":
        emb = _embedding_model.encode(text)
        return emb.tolist()

    if EMBED_PROVIDER == "ark":
        # 方舟多模态向量化：/api/v3/embeddings/multimodal，纯文本输入
        payload = {
            "model": EMBED_MODEL,
            "input": [{"type": "text", "text": text}],
            "encoding_format": "float",
        }
        if EMBED_DIMENSIONS:
            payload["dimensions"] = EMBED_DIMENSIONS
        if EMBED_INSTRUCTIONS:
            payload["instructions"] = EMBED_INSTRUCTIONS
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

    # 标准 OpenAI 兼容 /embeddings 接口
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


# 初始化sqlite库
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


# chroma向量库（余弦距离：1 - distance 即余弦相似度，与 0.70 阈值语义一致）
def _init_collection():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    cosine_meta = {"hnsw:space": "cosine"}
    try:
        col = chroma_client.get_collection(name="qa_collection")
    except Exception:
        col = None
    if col is None:
        return chroma_client.create_collection(name="qa_collection", metadata=cosine_meta)
    # 旧版本代码建出来的集合默认是 l2 空间；此前向量化一直 404，库里没有向量，
    # 直接删除重建为 cosine，避免相似度分数与阈值口径不一致
    space = (col.metadata or {}).get("hnsw:space", "l2")
    if space != "cosine":
        chroma_client.delete_collection(name="qa_collection")
        return chroma_client.create_collection(name="qa_collection", metadata=cosine_meta)
    return col


collection = _init_collection()


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
    emb = get_embedding(query)
    res = collection.query(query_embeddings=[emb], n_results=top_k)
    out = []
    if not res["ids"] or not res["ids"][0]:
        return out
    for idx, qid in enumerate(res["ids"][0]):
        meta = res["metadatas"][0][idx]
        doc = res["documents"][0][idx]
        score = 1 - res["distances"][0][idx]
        out.append({
            "id": qid,
            "question": doc,
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
    c.execute("SELECT id,question,created_at FROM unanswered WHERE status=? ORDER BY id DESC", (status,))
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "question": r[1], "created_at": r[2]} for r in rows]


def resolve_unanswered(uid, ans):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 必须先取出问题：状态改成 resolved 后再按 pending 查就查不到了（原逻辑的 bug）
    row = c.execute("SELECT question FROM unanswered WHERE id=?", (uid,)).fetchone()
    c.execute("UPDATE unanswered SET status='resolved' WHERE id=?", (uid,))
    conn.commit()
    conn.close()
    # 自动入库
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
import json
def load_json_to_kb(json_path: str):
    """从json文件批量导入问答到知识库，自动去重"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    count = 0
    for item in data:
        q = item["question"].strip()
        a = item["answer"].strip()
        if not q or not a:
            continue
        # 简单查重：看问题是否已经存在
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id FROM qa WHERE question = ?", (q,))
        row = cur.fetchone()
        if row is None:
            add_qa(q, a)
            count +=1
        conn.close()
    return count
