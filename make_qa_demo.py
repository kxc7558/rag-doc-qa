"""生成 rag-doc-qa 问答效果示例卡片，用于 GitHub README 视觉展示。"""
from PIL import Image, ImageDraw, ImageFont
import os

W, H = 620, 480
img = Image.new("RGB", (W, H), "#f7f9fc")
d = ImageDraw.Draw(img)

def font(size, bold=False):
    name = "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"
    try:
        return ImageFont.truetype(name, size)
    except Exception:
        return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)

f_title = font(20, True)
f_tag   = font(13)
f_q     = font(15, True)
f_ctx   = font(13)
f_ans_h = font(15, True)
f_ans   = font(14)
f_src_h = font(13, True)
f_src   = font(13)
f_note  = font(12)

# 主卡片
d.rounded_rectangle([20, 20, W-20, H-20], radius=14, fill="white", outline="#dde4ee", width=2)

# 标题 + tag
d.text((44, 36), "rag-doc-qa  ·  跨境电商售后知识库问答", font=f_title, fill="#1a2332")
d.rounded_rectangle([44, 68, 134, 88], radius=4, fill="#e8f5e9")
d.text((52, 72), "RAG 管道", font=f_tag, fill="#2e7d32")

# 问题区
d.rounded_rectangle([44, 104, W-44, 164], radius=8, fill="#eef4ff", outline="#c5d5f2")
d.text((56, 112), "❓  买家收到破损商品，客服该怎么处理？", font=f_q, fill="#1a3a6b")
d.text((56, 136), "德国 · 订单 250 美元 · 外包装明显破损", font=f_ctx, fill="#5a7aab")

# 分隔
d.text((56, 172), "▼", font=f_q, fill="#8899aa")

# 答案区
d.rounded_rectangle([44, 192, W-44, 348], radius=8, fill="#f2f8f4", outline="#b8dcc8")
d.text((56, 200), "💡  系统回答", font=f_ans_h, fill="#1e5c3a")

ans_lines = [
    "① 客服须在 24 小时内响应，先要求买家提供破损照片留证",
    "② 判定责任方：运输破损 → 公司/物流承担",
    "③ 财务 3 个工作日内完成退款（先留证再退款）",
    "④ 订单 250 美元 > 200 美元阈值 → 需主管审批",
    "⑤ 登记退货台账，原因分类「运输破损」",
]
y = 228
for line in ans_lines:
    d.text((60, y), line, font=f_ans, fill="#2c4a38")
    y += 27

# 来源区
d.rounded_rectangle([44, 360, W-44, H-40], radius=6, fill="#f0f4f8", outline="#d0dce8")
d.text((56, 368), "📎  检索来源（Top-3）", font=f_src_h, fill="#5a6a7a")
d.text((56, 392), "03_退货SOP.md · 01_平台规则.md · 02_物流时效.md", font=f_src, fill="#7a8a9a")
d.text((56, 416), "9 道评测题实测：幻觉测试全通过 —— 文档未覆盖的问题如实回答「未找到」", font=f_note, fill="#99aabb")

out = "D:/rag-doc-qa/assets/qa_demo.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
img.save(out)
print(f"Saved: {out}")
