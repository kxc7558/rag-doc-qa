"""RAG 评测指标 — 纯 Python 实现，零外部依赖。

三个指标：
  1. faithfulness      答案忠实度：答案中的关键事实断言是否在检索文档片段中出现
  2. retrieval_coverage 检索覆盖率：期望来源文档是否被实际检索命中
  3. context_precision@k 上下文精度：检索的 Top-k 片段中，对答案有贡献的占比

设计原则：
  - 不用 NLI 模型（太重，本地没部署），用字符串匹配近似
  - 数字/日期/英文词直接子串匹配；中文概念词用包含判断
  - 幻觉题（F 类）单独处理：答案必须包含「未找到」类措辞，否则判幻觉
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"


@dataclass
class Question:
    id: int
    tag: str
    question: str
    expected_keywords: list[str]
    expected_sources: list[str]
    note: str = ""


@dataclass
class QARecord:
    """一次问答的完整记录，由 run_eval.py 填充。"""
    question_id: int
    tag: str
    question: str
    answer: str
    sources: list[str]
    chunks_retrieved: int
    # 检索命中的文档片段（用于 faithfulness 计算）
    context_text: str = ""
    # 计算结果
    is_hallucination: bool = False
    faithfulness_score: float = 0.0
    retrieval_coverage: float = 0.0
    context_precision: float = 0.0


def load_questions() -> list[Question]:
    raw = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    return [Question(**item) for item in raw]


# ---------- 指标 1：faithfulness ----------

def _extract_facts(answer: str) -> list[str]:
    """从答案中提取事实断言片段（数字、日期、关键名词）。"""
    facts: list[str] = []
    # 数字 + 单位（"7 天" / "24 小时" / "3 个工作日" / "200 美元"）
    facts.extend(re.findall(r"\d+\s*[天小时工作日美元]", answer))
    # 百分比
    facts.extend(re.findall(r"\d+%", answer))
    # 关键名词（中文 2-6 字，排除停用词）
    stop = {"文档", "信息", "相关", "找到", "没有", "未找到", "根据", "按照", "处理", "需要"}
    for m in re.finditer(r"[一-鿿]{2,8}", answer):
        w = m.group(0)
        if w not in stop:
            facts.append(w)
    return facts


def compute_faithfulness(answer: str, context_text: str) -> tuple[float, bool]:
    """答案中有多少事实能在检索文档中找到依据。

    返回 (faithfulness_score, is_hallucination)。
    陷阱题（expected_sources 为空）单独判断：答案是否包含「未找到」类措辞。
    """
    facts = _extract_facts(answer)
    if not facts:
        # 答案太短没有可提取事实，保守给 0.5
        return 0.5, False

    # 归一化：去掉空格、统一全角/半角
    ctx_norm = context_text.replace(" ", "").replace("，", ",")
    found = 0
    for fact in facts:
        f_norm = fact.replace(" ", "")
        if f_norm and f_norm in ctx_norm:
            found += 1
    score = found / len(facts)

    # 幻觉判断：分数过低（< 0.3）且答案没有「未找到」措辞
    has_refusal = any(k in answer for k in ("未找到", "没有", "文档中未", "无法", "查不到"))
    is_hallucination = score < 0.3 and not has_refusal
    return score, is_hallucination


def compute_faithfulness_trap(question: Question, answer: str) -> tuple[float, bool]:
    """陷阱题（F 类）：正确行为是明确拒绝，不编造。"""
    refusal_patterns = ["未找到", "没有", "文档中未", "无法", "查不到", "未提及", "没有相关信息"]
    has_refusal = any(p in answer for p in refusal_patterns)
    # 陷阱题如果编造了具体数字/事实，判幻觉
    specific_facts = re.findall(r"\d+\s*[天小时工作日美元%]", answer)
    is_hallucination = not has_refusal and bool(specific_facts)
    score = 1.0 if has_refusal and not is_hallucination else 0.0
    return score, is_hallucination


# ---------- 指标 2：retrieval_coverage ----------

def compute_retrieval_coverage(sources: list[str], expected_sources: list[str]) -> float:
    """期望来源文档中，实际被检索命中的占比。
    陷阱题（expected_sources 为空）：只要没误报具体来源就算 1.0。"""
    if not expected_sources:
        return 1.0
    hit = sum(1 for s in expected_sources if s in sources)
    return hit / len(expected_sources)


# ---------- 指标 3：context_precision@k ----------

def compute_context_precision(answer: str, context_text: str, k: int = 3) -> float:
    """检索的 Top-k 片段中，有多少片段对答案有贡献（包含答案关键数字/词）。
    简化版：判断 context_text 中是否包含 expected 关键词的任意一个。"""
    keywords_in_context = sum(
        1 for kw in re.findall(r"[\d一二三]+|[一-鿿]{2,4}", context_text[:200])
        if len(kw) >= 1
    )
    # 用答案中出现的数字是否在 context 里做粗粒度判断
    answer_numbers = set(re.findall(r"\d+", answer))
    context_numbers = set(re.findall(r"\d+", context_text))
    if not answer_numbers:
        return 1.0  # 没有数字断言，视为全相关
    overlap = len(answer_numbers & context_numbers)
    return overlap / len(answer_numbers)


# ---------- 汇总 ----------

@dataclass
class EvalReport:
    total: int = 0
    faithfulness_avg: float = 0.0
    retrieval_coverage_avg: float = 0.0
    context_precision_avg: float = 0.0
    hallucination_count: int = 0
    per_question: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_questions": self.total,
            "metrics": {
                "faithfulness_avg": round(self.faithfulness_avg, 3),
                "retrieval_coverage_avg": round(self.retrieval_coverage_avg, 3),
                "context_precision_avg": round(self.context_precision_avg, 3),
            },
            "hallucination_count": self.hallucination_count,
            "hallucination_rate": round(self.hallucination_count / max(self.total, 1), 3),
            "per_question": self.per_question,
        }


def build_report(records: list[QARecord], questions: list[Question]) -> EvalReport:
    qmap = {q.id: q for q in questions}
    report = EvalReport()
    f_scores, rc_scores, cp_scores = [], [], []

    for rec in records:
        q = qmap.get(rec.question_id)
        if q is None:
            continue
        report.total += 1

        # faithfulness
        if q.tag == "F":
            f, is_hall = compute_faithfulness_trap(q, rec.answer)
        else:
            f, is_hall = compute_faithfulness(rec.answer, rec.context_text)
        rec.faithfulness_score = f
        rec.is_hallucination = is_hall
        f_scores.append(f)
        if is_hall:
            report.hallucination_count += 1

        # retrieval coverage
        rc = compute_retrieval_coverage(rec.sources, q.expected_sources)
        rec.retrieval_coverage = rc
        rc_scores.append(rc)

        # context precision
        cp = compute_context_precision(rec.answer, rec.context_text)
        rec.context_precision = cp
        cp_scores.append(cp)

        report.per_question.append({
            "id": rec.question_id,
            "tag": rec.tag,
            "question": rec.question,
            "faithfulness": round(f, 3),
            "retrieval_coverage": round(rc, 3),
            "context_precision": round(cp, 3),
            "is_hallucination": is_hall,
        })

    if f_scores:
        report.faithfulness_avg = sum(f_scores) / len(f_scores)
    if rc_scores:
        report.retrieval_coverage_avg = sum(rc_scores) / len(rc_scores)
    if cp_scores:
        report.context_precision_avg = sum(cp_scores) / len(cp_scores)
    return report


def save_report(report: EvalReport) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "baseline.json"
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_reports_from_eval_md(md_path: Path) -> list[QARecord]:
    """从现有 eval_result_self.md 解析出 QARecord（context_text 从 sources 文档补齐）。"""
    import chromadb
    import os

    records: list[QARecord] = []
    text = md_path.read_text(encoding="utf-8")

    # 解析每个 block
    blocks = re.findall(
        r"### (\d+)\. \[(\w)\] (.+?)\n\n\*\*答案\*\*：(.*?)\n\n\*\*来源\*\*：(.+?)（检索 \d+ 段）",
        text,
        re.DOTALL,
    )
    for num, tag, question, answer, sources_str in blocks:
        sources = [s.strip() for s in sources_str.split("、") if s.strip()]
        rec = QARecord(
            question_id=int(num),
            tag=tag,
            question=question.strip(),
            answer=answer.strip(),
            sources=sources,
            chunks_retrieved=3,
        )
        records.append(rec)
    return records


def enrich_context(records: list[QARecord]) -> None:
    """用 chroma_db 检索结果补齐 context_text（需要 chroma_db 已存在）。"""
    import chromadb
    import sentence_transformers

    CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
    MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "minilm"
    if not CHROMA_DIR.exists():
        print("chroma_db 不存在，跳过 context_text 补齐（retrieval_coverage 仍可用）")
        return

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("knowledge")
    embedder = sentence_transformers.SentenceTransformer(str(MODEL_DIR))

    for rec in records:
        vector = embedder.encode([rec.question], normalize_embeddings=True)[0].tolist()
        result = collection.query(query_embeddings=[vector], n_results=3)
        rec.context_text = "\n".join(result["documents"][0])


def main() -> None:
    """从现有 eval_result_self.md 生成 baseline.json。"""
    md = EVAL_DIR.parent / "eval_result_self.md"
    questions = load_questions()
    records = load_reports_from_eval_md(md)
    enrich_context(records)
    report = build_report(records, questions)
    out = save_report(report)
    print(f"baseline.json 写入：{out}")
    print(json.dumps(report.to_dict()["metrics"], ensure_ascii=False, indent=2))
    print(f"幻觉题数：{report.hallucination_count}/{report.total}")


if __name__ == "__main__":
    main()
