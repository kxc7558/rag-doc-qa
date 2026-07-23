# RAG Document QA Demo

本地文档智能问答系统 — 基于 LangChain + Chroma + MiMo 免费模型。

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 文档加载 | LangChain DirectoryLoader | 自动加载 txt/md 文件 |
| 文本切分 | RecursiveCharacterTextSplitter | 500字符块 + 100字符重叠 |
| 向量化 | sentence-transformers (all-MiniLM-L6-v2) | 本地运行，免费 |
| 向量数据库 | Chroma | 持久化到本地磁盘 |
| 生成模型 | MiMo-V2.5 (mimo-auto) | 小米免费模型 |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行 Demo（自动创建示例文档）
python app.py

# 3. 放自己的文档
# 把 .txt 文件放到 ./docs/ 目录，删除 ./chroma_db/，重新运行
```

## 架构

```
用户提问 → Embedding 向量化 → Chroma 相似度检索 → 
检索到的文档片段 + 原始问题 → LLM 生成回答 → 返回答案 + 来源
```

## 为什么做这个

学习 RAG 管道的完整流程：
1. 文档加载与预处理
2. 文本切分策略（chunk_size / overlap 对检索质量的影响）
3. 向量嵌入与相似度搜索
4. 检索结果作为上下文注入 LLM 提示词
5. 可追溯的答案来源

## License

MIT
