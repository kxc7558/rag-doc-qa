"""批量跑问题集，输出自建 RAG 侧的对照结果（Markdown）。

用法：python run_eval.py            # 跑 eval_questions.txt 里全部问题
     python run_eval.py 3          # 只跑前 3 题（快速验证）
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app import (
    BASE_DIR,
    EMBED_MODEL,
    POOL_BASE_URL,
    ask,
    build_index,
    load_pool_key,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
QUESTIONS_FILE = BASE_DIR / "eval_questions.txt"
OUTPUT_FILE = BASE_DIR / "eval_result_self.md"


def load_questions(path: Path) -> list[str]:
    """读取问题集，跳过注释与空行。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def strip_tag(question: str) -> str:
    """去掉行首的 [A] 分类标记，只保留问题本身。"""
    return question.split("]", 1)[1].strip() if question.startswith("[") else question


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    questions = load_questions(QUESTIONS_FILE)
    if limit:
        questions = questions[:limit]

    embedder = SentenceTransformer(EMBED_MODEL)
    llm = OpenAI(base_url=POOL_BASE_URL, api_key=load_pool_key())
    collection = build_index(embedder)

    blocks: list[str] = []
    for index, raw in enumerate(questions, start=1):
        question = strip_tag(raw)
        tag = raw.split("]", 1)[0].lstrip("[") if raw.startswith("[") else "?"
        result = ask(embedder, collection, llm, question)
        block = (
            f"### {index}. [{tag}] {question}\n\n"
            f"**答案**：{result['answer'].strip()}\n\n"
            f"**来源**：{'、'.join(result['sources'])}（检索 {result['chunks']} 段）\n"
        )
        blocks.append(block)
        print(block, flush=True)

    OUTPUT_FILE.write_text("# 自建 rag-doc-qa 侧结果\n\n" + "\n".join(blocks), encoding="utf-8")
    print(f"\n已写入：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
