"""RAG Document QA — 本地知识库问答（手写管道版）

管道：文档加载 → 切片 → 向量化 → Chroma 存储 → 检索 Top-K → 注入提示词 → LLM 作答
模型：本机 API 代理池（OpenAI 兼容）

用法：
    python app.py                    # 重建索引 + 跑内置问题
    python app.py "订单超时怎么办"     # 单问
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# 国内网络下加速 HuggingFace 模型下载（须在导入 sentence_transformers 之前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import chromadb
from openai import APIError, OpenAI
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("rag")

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "knowledge"

EMBED_MODEL = os.getenv("EMBED_MODEL", str(BASE_DIR / "models" / "minilm"))
POOL_BASE_URL = os.getenv("POOL_BASE_URL", "http://127.0.0.1:8787/v1")
POOL_MODELS = [m.strip() for m in os.getenv("POOL_MODELS", "flash,pro,auto").split(",")]
POOL_CONFIG = Path(
    os.getenv(
        "POOL_CONFIG",
        r"C:\Users\32492\WorkBuddy\2026-09-01-11-19-50\api-key-pool\config.json",
    )
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
GROUNDING_RULE = "若文档中没有相关信息，请如实回答「文档中未找到相关信息」，不要编造。"


def load_pool_key() -> str:
    """取代理池分发 Key：优先环境变量，否则用配置里日额度最高的用户。"""
    env_key = os.getenv("POOL_API_KEY")
    if env_key:
        return env_key
    config = json.loads(POOL_CONFIG.read_text(encoding="utf-8"))
    users: dict[str, dict[str, Any]] = config["accessKeys"]
    key, _ = max(users.items(), key=lambda kv: kv[1].get("daily", 0))
    return key


def load_documents(docs_dir: Path) -> list[dict[str, str]]:
    """加载 docs 目录下的 .md / .txt 文档。"""
    documents: list[dict[str, str]] = []
    for pattern in ("*.md", "*.txt"):
        for path in sorted(docs_dir.glob(pattern)):
            documents.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    return documents


def split_document(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """按段落聚合切片：先按空行分段，再合并到接近 size，并保留 overlap 重叠。"""
    chunks: list[str] = []
    buffer = ""
    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        if buffer and len(buffer) + len(paragraph) + 2 > size:
            chunks.append(buffer.strip())
            buffer = buffer[-overlap:] if overlap else ""
        buffer += paragraph + "\n\n"
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


def build_index(embedder: SentenceTransformer) -> Any:
    """切分 → 向量化 → 写入 Chroma，返回 collection。"""
    documents = load_documents(DOCS_DIR)
    if not documents:
        raise SystemExit(f"docs 目录为空：{DOCS_DIR}")

    chunks: list[dict[str, str]] = []
    for doc in documents:
        for piece in split_document(doc["text"]):
            chunks.append({"source": doc["source"], "text": piece})

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if COLLECTION_NAME in {c.name for c in client.list_collections()}:
        client.delete_collection(COLLECTION_NAME)  # 重建索引时先清掉旧数据
    collection = client.get_or_create_collection(COLLECTION_NAME)

    vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
    collection.add(
        ids=[f"{c['source']}#{i}" for i, c in enumerate(chunks)],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
        embeddings=[v.tolist() for v in vectors],
    )
    log.info("索引完成：%d 篇文档 → %d 个切片", len(documents), len(chunks))
    return collection


def ask(
    embedder: SentenceTransformer,
    collection: Any,
    llm: OpenAI,
    question: str,
    k: int = TOP_K,
) -> dict[str, Any]:
    """检索 Top-K → 拼提示词 → 生成答案，返回答案与来源。"""
    vector = embedder.encode([question], normalize_embeddings=True)[0].tolist()
    result = collection.query(query_embeddings=[vector], n_results=k)
    hits = list(zip(result["documents"][0], result["metadatas"][0]))

    context = "\n\n---\n\n".join(text for text, _ in hits)
    prompt = f"基于以下文档回答问题。{GROUNDING_RULE}\n\n文档内容：\n{context}\n\n问题：{question}"

    completion = None
    last_error: Exception | None = None
    for model in POOL_MODELS:
        try:
            completion = llm.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
            break
        except APIError as exc:
            last_error = exc
            log.warning("模型别名 %s 不可用，换下一个：%s", model, str(exc)[:80])
    if completion is None:
        raise RuntimeError(f"所有模型别名都不可用，最后错误：{last_error}")

    return {
        "question": question,
        "answer": completion.choices[0].message.content or "",
        "sources": sorted({meta["source"] for _, meta in hits}),
        "chunks": len(hits),
    }


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    log.info("加载向量模型：%s", EMBED_MODEL)
    embedder = SentenceTransformer(EMBED_MODEL)
    llm = OpenAI(base_url=POOL_BASE_URL, api_key=load_pool_key())

    collection = build_index(embedder)
    questions = (
        [question]
        if question
        else [
            "订单超过多久没签收需要主动联系买家？",
            "买家收到破损商品应该怎么处理？",
            "TikTok Shop 的结算周期是多久？",
        ]
    )
    for q in questions:
        result = ask(embedder, collection, llm, q)
        log.info(
            "\n❓ %s\n💡 %s\n📎 来源：%s（检索 %d 段）",
            q,
            result["answer"],
            result["sources"],
            result["chunks"],
        )


if __name__ == "__main__":
    main()
