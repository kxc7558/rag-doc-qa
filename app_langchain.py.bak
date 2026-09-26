"""RAG Document QA Demo — 本地文档智能问答

技术栈: LangChain + Chroma 向量数据库 + MiMo 免费模型
使用方式: python app.py
"""

import os
import sys
from pathlib import Path

# ============================================================
# 0. 依赖检查 （首次运行请先 pip install -r requirements.txt）
# ============================================================

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import TextLoader, DirectoryLoader
    from langchain.chains import RetrievalQA
    from langchain_community.llms import OpenAI
except ImportError:
    print("请先安装依赖: pip install -r requirements.txt")
    sys.exit(1)

# ============================================================
# 1. 配置
# ============================================================

# MiMo 免费 API（OpenAI 兼容）
LLM_BASE_URL = "https://api.xiaomimimo.com/api/free-ai/openai/v1"
LLM_API_KEY = "anonymous"
LLM_MODEL = "mimo-auto"

# 使用本地 embedding 模型（无需 API，免费）
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 向量数据库存储目录
CHROMA_DIR = "./chroma_db"

# 文档目录（放 .txt / .md 文件进去）
DOCS_DIR = "./docs"

# ============================================================
# 2. 初始化组件
# ============================================================

print("=" * 60)
print("RAG Document QA Demo")
print("=" * 60)

print(f"\n[1/4] 加载 Embedding 模型: {EMBEDDING_MODEL}")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print(f"[2/4] 初始化 LLM: {LLM_MODEL}")
llm = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    model=LLM_MODEL,
    temperature=0.3,
    max_tokens=1024,
)

# ============================================================
# 3. 文档加载与索引
# ============================================================

def build_index(docs_dir: str) -> Chroma:
    """从目录加载文档，切分、向量化并存入 Chroma。"""
    docs_path = Path(docs_dir)

    # 创建示例文档（如果没有）
    if not docs_path.exists() or not list(docs_path.glob("*.txt")):
        docs_path.mkdir(parents=True, exist_ok=True)
        sample = docs_path / "sample.txt"
        sample.write_text("""
RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术架构。
它的核心思想是：在 LLM 生成回答之前，先从外部知识库中检索相关文档片段，
然后将这些片段作为上下文注入到提示词中，从而让模型生成更准确、更有依据的回答。

RAG 的主要优势包括：
1. 减少幻觉（Hallucination）：模型基于真实文档回答，而不是凭空编造
2. 知识可更新：只需更新文档库，不需要重新训练模型
3. 可追溯性：每个回答都可以追溯到具体的文档来源
4. 领域适配：可以接入企业内部文档、技术手册等私有知识

常见的 RAG 技术栈：
- 文档加载：LangChain DirectoryLoader
- 文本切分：RecursiveCharacterTextSplitter
- 向量化：OpenAI Embeddings / HuggingFace Sentence Transformers
- 向量数据库：Chroma / Pinecone / Milvus / Weaviate
- 检索策略：相似度搜索、MMR（最大边际相关性）
- 生成模型：GPT-4 / Claude / 本地 Llama

Claude Code 是一个 AI 编程助手，支持多 Agent 协作开发。
MCP（Model Context Protocol）是 Anthropic 发布的开放协议，
允许 AI 助手安全地访问外部工具和数据源。
""".strip(), encoding="utf-8")
        print(f"  [*] 已创建示例文档: {sample}")

    print(f"\n[3/4] 加载并索引文档: {docs_dir}")
    loader = DirectoryLoader(
        docs_dir,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"  加载了 {len(documents)} 个文档")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  切分为 {len(chunks)} 个文本块")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"  向量索引已存入: {CHROMA_DIR}")
    return vectorstore


def query(vectorstore: Chroma, question: str) -> dict:
    """查询 RAG 管道并返回答案 + 来源。"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(question)

    context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)
    sources = [doc.metadata.get("source", "unknown") for doc in relevant_docs]

    prompt = f"""基于以下文档内容回答问题。如果文档中没有相关信息，请如实说"文档中未找到相关信息"。

文档内容：
{context}

问题：{question}

回答："""

    answer = llm.invoke(prompt)

    return {
        "question": question,
        "answer": answer,
        "sources": list(set(sources)),
        "chunks_retrieved": len(relevant_docs),
    }


# ============================================================
# 4. 主流程
# ============================================================

if __name__ == "__main__":
    vectorstore = build_index(DOCS_DIR)

    print("\n[4/4] RAG 管道就绪！")
    print("-" * 60)

    # 示例查询
    questions = [
        "什么是 RAG？",
        "RAG 有哪些优势？",
        "MCP 协议是什么？",
    ]

    for q in questions:
        print(f"\n❓ {q}")
        result = query(vectorstore, q)
        print(f"💡 {result['answer'][:200]}...")
        print(f"📎 来源: {result['sources']}")
        print(f"📊 检索到 {result['chunks_retrieved']} 个相关片段")

    print("\n" + "=" * 60)
    print("Demo 完成！你可以：")
    print("  1. 把 .txt 文档放到 ./docs/ 目录下")
    print("  2. 删除 ./chroma_db/ 重新索引")
    print("  3. 自己问问题：修改上面的 questions 列表")
    print("=" * 60)
