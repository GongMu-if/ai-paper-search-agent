# ==========================================
# 模块 1: 依赖导入与页面基础配置
# ==========================================
import base64
import datetime
import hashlib
import io
import re
import time
from html import escape
from typing import Dict, List, Tuple

import markdown
import requests
import streamlit as st
from bs4 import BeautifulSoup, NavigableString, Tag
from openai import OpenAI
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image as RLImage
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

st.set_page_config(page_title="AI 论文检索 Agent", page_icon="📚", layout="wide")

# ==========================================
# 模块 2: 全局变量与 API 密钥配置
# ==========================================
# 说明：
# 这些密钥都从 Streamlit secrets 读取。
# 如果某个 key 未配置，对应功能会报错，因此部署前需要在 secrets.toml 中填好。

DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

QWEN_API_KEY = st.secrets["QWEN_API_KEY"]
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MODAL_API_URL = st.secrets["MODAL_API_URL"]

seen_paper_ids = set()

# ==========================================
# 模块 3: 导出与报告生成配置
# ==========================================
# 说明：
# 这里集中管理 PDF 导出和报告生成的关键参数，便于后续微调。

PDF_EXPORT_CONFIG = {
    # A4 版芯宽度；适当收窄版芯可以减少右侧裁切和长行换行不稳定问题
    "content_width_mm": 172,
    # 页边距；不要太小，否则 html2canvas/jsPDF 在边缘更容易出问题
    "margin_top_mm": 10,
    "margin_right_mm": 13,
    "margin_bottom_mm": 10,
    "margin_left_mm": 13,
    # 正文字号与行距
    "body_font_size_px": 13.5,
    "body_line_height": 1.80,
    # 普通图片尺寸上限
    "figure_max_width_pct": 64,
    "figure_max_height_mm": 92,
    # 宽图尺寸上限
    "wide_visual_max_width_pct": 82,
    "wide_visual_max_height_mm": 100,
    # 表格型图片尺寸上限；通常要更接近整页宽度
    "table_visual_max_width_pct": 96,
    "table_visual_max_height_mm": 235,
    # 竖长图尺寸上限
    "tall_visual_max_width_pct": 60,
    "tall_visual_max_height_mm": 150,
    # 真正 HTML 表格的字号与 padding
    "table_font_size_px": 10,
    "table_cell_padding_px": 4,
    # 导出清晰度；影响清晰度，不影响元素实际大小
    "html2canvas_scale": 2,
}

REPORT_SECTION_SPECS = [
    ("## 1. 研究问题与核心贡献", "请只输出“## 1. 研究问题与核心贡献”这一节。"),
    ("## 2. 背景、研究缺口与前人路线", "请只输出“## 2. 背景、研究缺口与前人路线”这一节。"),
    ("## 3. 方法总览与整体数据流", "请只输出“## 3. 方法总览与整体数据流”这一节。"),
    ("## 4. 关键模块逐层机制剖析", "请只输出“## 4. 关键模块逐层机制剖析”这一节。"),
    ("## 5. 实验设计、关键证据与论点验证", "请只输出“## 5. 实验设计、关键证据与论点验证”这一节。"),
    ("## 6. 复现要点与方法适用边界", "请只输出“## 6. 复现要点与方法适用边界”这一节。"),
    ("## 7. 局限性与未解决问题", "请只输出“## 7. 局限性与未解决问题”这一节。"),
]

# ==========================================
# 模块 4: 核心 Agent 提示词
# ==========================================
# 说明：
# 1. Text Agent 只做事实抽取与解释，不做 speculative 创新推演。
# 2. Vision Agent 输出 FIGURE_CARD，方便后续总报告引用。
# 3. Main Agent 只负责第 1-7 节。
# 4. Research Agent 单独负责第 8 节，避免“研究设想”和“原文事实”污染彼此。
# 5. Auditor Agent 只做审校，不负责大段重写。

TEXT_AGENT_PROMPT = """
你是论文事实抽取专家，不是评论员。你的任务是把论文文本拆成“可验证事实”和“可阅读解释”两层内容，供后续主报告模型调用。

【总原则】
1. 绝对忠于原文。只能基于输入文本抽取、重述、解释，不得补充外部知识。
2. 任何原文未明确给出的内容，一律写“原文未明确交代”，禁止猜测。
3. 重点保留任务定义、问题缺口、方法链路、模块名称、训练目标、数据集、评价指标、实验数值、作者自述局限。
4. 你的职责是“抽取和澄清”，不是“提出未来研究蓝图”。不要输出主编视角创新点，不要做超出原文证据的研究推演。
5. 输出要兼顾机器可读和人类可读。

【输出格式】
请严格输出以下两部分。

<FACT_BANK>
论文任务：
研究对象：
作者显式声称的核心贡献：
前人方法的主要不足：
方法总体流程：
关键模块与作用：
训练/优化目标：
数据集与划分：
评价指标：
主实验关键结果：
消融实验关键结果：
作者自述局限：
原文未回答的问题：
</FACT_BANK>

# 论文深度文本综述

## 1. 研究问题与动机
用多个自然段说明本文要解决什么问题、为什么这个问题重要、前人为什么还没解决好。

## 2. 方法总览与数据流
用多个自然段从输入、表示、模块、输出的顺序解释整套方法，明确每一步“为了解决什么问题而设计”。

## 3. 关键模块机制拆解
逐一解释关键模块的内部机制、与上下游模块的关系、它为什么是必要的。

## 4. 实验设计与证据提取
说明数据集、设置、对照组、核心数值结果，以及这些结果分别支撑了哪条方法论断。

## 5. 局限性与未解问题
只写原文明确承认的局限，以及从原文实验设计中可以直接看出的尚未回答问题；不得引入外部观点。
"""

VISION_AGENT_PROMPT = """
你是论文图表证据抽取专家。你会收到一张图或表，以及必要的论文上下文。你的目标不是泛泛“点评”，而是把它转成后续主报告可调用的证据卡。

【总原则】
1. 只根据图像可见信息和提供的上下文作答。
2. 看不清、读不出的数值或标签，必须明确写“无法可靠辨认”。
3. 不要猜作者意图，不要补外部知识。
4. 对表格和曲线图，尽可能提取“最佳项、对照项、差值、趋势”。
5. 对架构图和流程图，尽可能提取“输入、模块、连接关系、信息流、输出”。

【输出格式】
<FIGURE_CARD>
图像ID：
原文编号（如 Figure 1 / Table 2，若不可见则写“未显示”）：
图表类型：
可读性（高/中/低）：
最相关的论文部分：
这张图/表在回答什么问题：
可见的关键结构或字段：
可提取的关键数值或对比：
它能支持的结论：
它不能支持或无法确认的内容：
推荐插入位置（写成“建议插入在报告第X节的哪一段之后”）：
推荐图注：
</FIGURE_CARD>

随后再用一段不超过200字的自然语言，解释这张图/表为什么对理解论文重要。
"""

MAIN_AGENT_PROMPT = """
你是学术论文深度解读主编。你的任务不是把材料写得更华丽，而是生成一份“研究者可直接使用”的论文全维度报告：读者应当仅凭这份报告就能理解论文的问题、方法、证据、边界，并据此形成后续研究方案。

【你会收到的输入】
1. 论文原始结构化 markdown
2. Text Agent 产出的 <FACT_BANK> 与文本综述
3. Vision Agent 产出的若干 <FIGURE_CARD>
4. 当前所有可用图片ID列表

【硬性规则】
1. 所有关键判断都必须优先依据原始 markdown、FACT_BANK、FIGURE_CARD。
2. 当三者冲突时，以原始 markdown 为准；若仍无法确定，明确写“原文未明确交代”。
3. 不得捏造论文没有给出的公式、模块、参数、结果、作者动机或实验结论。
4. 每个章节必须由多个自然段组成，不能把整章写成一个大段。
5. 每次解释方法或实验时，都要回答四个问题：它要解决什么问题、具体怎么做、证据是什么、边界在哪里。
6. 插图必须使用 Markdown 图片语法，且图片占位符必须从提供的图片ID列表中原样复制，格式为：
   ![图X：学术化图注](图片ID)
7. 表格前后各保留一个空行；表标题必须单独成行。
8. 语言风格保持学术、克制、清晰，不做口语化渲染。

【报告结构】
# 论文全维度深度透视报告

## 1. 研究问题与核心贡献
定义本文试图解决的核心问题，准确概括本文相对前人工作的主要创新，并说明这些创新分别改变了哪一个技术瓶颈。

## 2. 背景、研究缺口与前人路线
还原该方向的研究背景、主流技术路线及其局限，说明本文的问题为什么值得解决，以及它切入的位置在哪里。

## 3. 方法总览与整体数据流
结合原始文本和图表证据，说明系统从输入到输出的完整链路。若有总架构图，应在这里插入。

## 4. 关键模块逐层机制剖析
按照模型真实工作顺序拆解每个关键模块。每个模块都要说明：输入是什么、变换是什么、它为何必要、它与其他模块如何耦合、它预期改善了什么问题。若有模块结构图，应在对应段落处插入。

## 5. 实验设计、关键证据与论点验证
交代数据集、评价指标、对照组、主实验、消融实验。每写一个结论，都要明确指出它由哪一组结果支持，并解释这项结果验证了哪条方法主张。关键图表在对应段落后插入。

## 6. 复现要点与方法适用边界
用多个自然段归纳读者若要复现本文，最不能忽视的输入条件、训练设置、模块依赖和评测前提。同时说明该方法适用于什么情形、不适用于什么情形。

## 7. 局限性与未解决问题
区分“作者明确承认的局限”和“从实验设计中可以直接看出的未解决问题”，但后者也必须基于论文证据，而不是外部常识。
"""

RESEARCH_AGENT_PROMPT = """
你是后续研究路线设计专家。
你只能基于以下材料提出研究路线：
1. 原始论文 markdown
2. FACT_BANK 与文本综述
3. FIGURE_CARD
4. 已生成的第1-7节主报告

硬性规则：
1. 禁止把研究设想写成原文事实。
2. 每条路线都要明确对应本文的某个模块、某条实验结论或某个未解问题。
3. 必须写清：缺口、改造方案、预期收益、验证方式、技术风险。
4. 语言要学术、克制、可执行。

请只输出：
## 8. 面向后续研究的可执行创新路线
并先用一段简短声明说明：以下内容属于基于本文机制的研究设想，不是原文结论。
"""

REPORT_AUDITOR_PROMPT = """
你是学术报告审校员。请检查以下报告是否存在：
1. 不被原始 markdown / FACT_BANK / FIGURE_CARD 支持的结论
2. 不存在的图片ID
3. 漏掉的关键模块或关键实验
4. 将研究设想误写为原文结论
5. 缺失章节或章节标题错误

输出格式：
RESULT: PASS 或 FAIL
ISSUES:
- ...
"""

# ==========================================
# 模块 5: 搜索 Agent 提示词
# ==========================================

def get_system_prompt(requirements, preprint_rule):
    """根据用户筛选条件，构造论文搜索 Agent 的 system prompt。"""
    if preprint_rule == "排除预印本 (仅限正规期刊/会议)":
        preprint_prompt = "严禁选择 Venue 为 'Unknown Venue/Preprint' 的预印本论文。"
    else:
        preprint_prompt = "可以接受预印本论文。"

    return f"""
你是一个科研论文搜索专家。你的任务是根据研究方向，在近一年内的论文中筛选六篇。
# 你的工作流程：
1. 通过`search_and_detail_papers`工具来获得相关论文的标题、摘要和Introduction。
2. 将符合要求的记录在 Thought 中作为“备选池”累加。
3. 学习同义词或近义词，作为下一次的查询词。【警告】：严禁加入用户要求中的关键词！
4. 如果不符合或无摘要：摒弃该论文。
5. 最终选出最好的六篇来作为结果。

# 用户的具体筛选要求：
{requirements}
{preprint_prompt}

# 输出格式要求：
Thought: [你的思考逻辑]
Action: [执行工具或结束]

Action格式支持：
1. search_and_detail_papers(query="关键词")
2. Finish:
在此处使用 Markdown 格式直接列出选出的 6 篇论文（1.标题 2.Venue 3.DOI 4.推荐理由）。不要多余的话。
"""

# ==========================================
# 模块 6: 通用工具函数
# ==========================================
# 说明：
# 这里放搜索、文本预处理、图表上下文抽取、Markdown 清理等通用能力。


def reconstruct_abstract(inverted_index: dict) -> str:
    """将 OpenAlex 的倒排摘要恢复成自然顺序文本。"""
    if not inverted_index:
        return ""
    word_index = [(pos, word) for word, positions in inverted_index.items() for pos in positions]
    word_index.sort(key=lambda x: x[0])
    return " ".join([word for _, word in word_index])


def search_and_detail_papers(query: str) -> str:
    """调用 Semantic Scholar + OpenAlex，对用户研究方向做候选论文搜集。"""
    global seen_paper_ids
    api_key = st.secrets["S2_API_KEY"]
    email = "gaoym3@mails.neu.edu.cn"
    s2_url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}"
        f"&limit=100&year=2025-2026&fields=paperId,title,abstract,year,externalIds,venue"
    )
    headers = {"x-api-key": api_key}

    try:
        time.sleep(2)
        s2_response = requests.get(s2_url, headers=headers, timeout=20)
        s2_response.raise_for_status()
        papers = s2_response.json().get("data", [])

        if not papers:
            return f"Observation: 未找到关于'{query}'的近一年论文。"

        results = []
        for p in papers:
            paper_id = p.get("paperId", "No ID")
            if paper_id in seen_paper_ids:
                continue

            title = p.get("title", "No Title")
            doi = p.get("externalIds", {}).get("DOI", "")
            venue = p.get("venue") or "Unknown Venue/Preprint"

            s2_abstract = p.get("abstract")
            final_abstract, openalex_mark = "", ""

            if s2_abstract and s2_abstract.strip():
                final_abstract = s2_abstract.strip()
            else:
                try:
                    oa_url = (
                        f"https://api.openalex.org/works/https://doi.org/{doi}"
                        if doi
                        else f"https://api.openalex.org/works?filter=title.search:{title}"
                    )
                    oa_res = requests.get(oa_url, params={"mailto": email}, timeout=20)
                    if oa_res.status_code == 200:
                        work_data = oa_res.json().get("results", [None])[0] if "results" in oa_res.json() else oa_res.json()
                        if work_data and work_data.get("abstract_inverted_index"):
                            final_abstract = reconstruct_abstract(work_data["abstract_inverted_index"])
                            openalex_mark = " [via OpenAlex]"
                except Exception:
                    pass

            if not final_abstract or len(final_abstract.strip()) < 10:
                continue

            seen_paper_ids.add(paper_id)
            results.append(
                f"Title: {title}\n"
                f"  - Venue: {venue}\n"
                f"  - S2_ID: {paper_id} | DOI: {doi}{openalex_mark}\n"
                f"  - Abstract: {final_abstract[:800]}..."
            )
            if len(results) >= 30:
                break

        if not results:
            return f"Observation: 搜索到关于 '{query}' 的论文，但均为已读或无摘要，未能提供新信息。"
        return f"Observation: 提取到 {len(results)} 篇全新论文：\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"Observation: 搜索出错 - {str(e)}"


available_tools = {"search_and_detail_papers": search_and_detail_papers}


def stable_file_hash(file_bytes: bytes) -> str:
    """对上传 PDF 做稳定哈希，避免 Python 内置 hash 在重启后变化。"""
    return hashlib.md5(file_bytes).hexdigest()


def sort_images_by_doc_order(md_content: str, images_dict: Dict[str, str]) -> List[Tuple[str, str]]:
    """按图片 ID 在 markdown 中首次出现的位置排序，尽量贴近论文原始逻辑顺序。"""
    def sort_key(item):
        pos = md_content.find(item[0])
        return pos if pos >= 0 else 10**9

    return sorted(images_dict.items(), key=sort_key)


def build_source_pack(md_content: str, max_chars: int = 50000) -> str:
    """为主报告模型构造一个较长但可控的论文原文摘要包。"""
    if len(md_content) <= max_chars:
        return md_content

    parts = re.split(r'(?=^#{1,6}\s)', md_content, flags=re.M)
    kept, total = [], 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        chunk = part[:7000]
        if total + len(chunk) > max_chars:
            break
        kept.append(chunk)
        total += len(chunk)

    return "\n\n".join(kept) if kept else md_content[:max_chars]


def extract_local_context(md_content: str, image_id: str, window: int = 1600) -> str:
    """根据图片 ID，从论文 markdown 中抽取图片附近的局部上下文，供 Vision Agent 使用。"""
    idx = md_content.find(image_id)
    if idx == -1:
        return build_source_pack(md_content, max_chars=6000)

    start = max(0, idx - window)
    end = min(len(md_content), idx + len(image_id) + window)

    prefix = md_content[:idx]
    headings = re.findall(r'^(#{1,6}\s+.+)$', prefix, flags=re.MULTILINE)
    nearest_headings = "\n".join(headings[-3:]) if headings else "未找到章节标题"

    local_snippet = md_content[start:end]
    return f"""【最近章节】
{nearest_headings}

【图像附近原文】
{local_snippet}
"""


def normalize_report_markdown(text: str) -> str:
    """对模型生成的报告做基础清洗，减少格式噪音。"""
    text = text.replace("\r\n", "\n")
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?m)^\s+##\s', '## ', text)
    text = re.sub(r'(?m)^\s+#\s', '# ', text)
    return text.strip()


def extract_report_section(report_md: str, heading: str) -> str:
    """从完整 markdown 报告中抽取指定标题对应的 section。"""
    pattern = rf'({re.escape(heading)}.*?)(?=\n##\s+\d+\.|\Z)'
    match = re.search(pattern, report_md, flags=re.S)
    return match.group(1).strip() if match else ""


def find_missing_sections(report_md: str) -> List[Tuple[str, str]]:
    """检查第 1-7 节是否完整生成。"""
    missing = []
    for heading, task in REPORT_SECTION_SPECS:
        if heading not in report_md:
            missing.append((heading, task))
    return missing


def render_report_with_images(report_md: str, images_dict: Dict[str, str]):
    """在 Streamlit 页面里同时支持 Markdown 图片语法和旧的 [REF_IMG: ...] 占位协议。"""
    pattern = r'(\[REF_IMG:\s*.*?\]|!\[.*?\]\(.*?\))'
    sections = re.split(pattern, report_md)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        ref_match = re.fullmatch(r'\[REF_IMG:\s*(.*?)\]', section)
        md_match = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', section)

        if ref_match:
            alt_text = ref_match.group(1).strip()
            img_key = ref_match.group(1).strip()
        elif md_match:
            alt_text = md_match.group(1).strip()
            img_key = md_match.group(2).strip()
        else:
            st.markdown(section)
            continue

        matched = False
        for img_name, b64 in images_dict.items():
            if img_key in img_name or img_name in img_key:
                st.image(base64.b64decode(b64), caption=alt_text, use_container_width=True)
                matched = True
                break

        if not matched:
            st.markdown(section)


def embed_base64_images(md_text: str, images_dict: Dict[str, str]) -> str:
    """把 Markdown 中的图片占位符替换成 base64 data URI，供浏览器端导出 PDF。"""

    def build_figure_html(alt_text: str, b64: str, img_name: str) -> str:
        return (
            '\n'
            f'<div class="pdf-figure" data-img-key="{img_name}" data-alt="{alt_text}">'
            f'<img src="data:image/jpeg;base64,{b64}" alt="{alt_text}" />'
            f'<div class="img-caption">{alt_text}</div>'
            '</div>\n'
        )

    def replace_markdown_img(match):
        alt_text = match.group(1).strip()
        img_placeholder = match.group(2).strip()

        for img_name, b64 in images_dict.items():
            if img_placeholder in img_name or img_name in img_placeholder:
                return build_figure_html(alt_text, b64, img_name)

        return match.group(0)

    def replace_ref_img(match):
        img_placeholder = match.group(1).strip()
        for img_name, b64 in images_dict.items():
            if img_placeholder in img_name or img_name in img_placeholder:
                return build_figure_html(img_placeholder, b64, img_name)
        return match.group(0)

    md_text = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_markdown_img, md_text)
    md_text = re.sub(r'\[REF_IMG:\s*(.*?)\]', replace_ref_img, md_text)
    return md_text


def wrap_markdown_tables(html_content: str) -> str:
    """为真正的 HTML 表格套上一层 table-block，方便分页控制。"""
    return re.sub(r'(<table>.*?</table>)', r'<div class="table-block">\1</div>', html_content, flags=re.S)


def html_escape_for_component(text: str) -> str:
    """保留这个兼容函数，便于旧逻辑平滑迁移；当前版本的 PDF 不再依赖浏览器端脚本。"""
    return text


# ==========================================
# 模块 7: 服务器端高质量 PDF 生成
# ==========================================
# 说明：
# 1. 这里彻底改为 ReportLab 服务器端排版生成 PDF，不再依赖 html2canvas/jsPDF 截图。
# 2. 这样生成的 PDF 文字是矢量文本，清晰度高，也不会因为截图造成页首/分页处丢字。
# 3. ReportLab 会严格根据 A4 页尺寸、边距、字号、行高自动分页；放不下的块会自动顺延到下一页。
# 4. 我们还会把每个二级章节里的图表统一后置到该章节尾部，让正文先尽量填满页面，再展示图表。
# 5. 英文与数字优先使用 Times 系列，中文使用 STSong-Light（ReportLab 内置 CID 中文字体）。


def _register_pdf_fonts():
    """注册 PDF 所需字体。STSong-Light 是 ReportLab 自带的中文 CID 字体。"""
    try:
        pdfmetrics.getFont('STSong-Light')
    except Exception:
        pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))


def _is_ascii_latin(ch: str) -> bool:
    """判断字符是否属于英文/数字/常见英文标点，用于混排字体切换。"""
    return ord(ch) < 128


def _mixed_font_markup(text: str, ascii_font: str = 'Times-Roman', cjk_font: str = 'STSong-Light') -> str:
    """把一段文本拆成英文/数字片段和中文片段，分别套不同字体。"""
    if not text:
        return ''
    chunks = []
    current = []
    current_ascii = _is_ascii_latin(text[0])
    for ch in text:
        is_ascii = _is_ascii_latin(ch)
        if is_ascii == current_ascii:
            current.append(ch)
        else:
            chunks.append((current_ascii, ''.join(current)))
            current = [ch]
            current_ascii = is_ascii
    if current:
        chunks.append((current_ascii, ''.join(current)))
    rendered = []
    for is_ascii, chunk in chunks:
        safe = escape(chunk)
        font_name = ascii_font if is_ascii else cjk_font
        rendered.append(f'<font name="{font_name}">{safe}</font>')
    return ''.join(rendered)


def _build_pdf_styles() -> Dict[str, ParagraphStyle]:
    """构建 PDF 用的段落样式。"""
    _register_pdf_fonts()
    cfg = PDF_EXPORT_CONFIG
    sample = getSampleStyleSheet()
    body = ParagraphStyle(
        'BodyZH',
        parent=sample['BodyText'],
        fontName='STSong-Light',
        fontSize=cfg['body_font_size_px'],
        leading=cfg['body_font_size_px'] * cfg['body_line_height'],
        alignment=TA_JUSTIFY,
        firstLineIndent=cfg['body_font_size_px'] * 2,
        spaceAfter=8,
        textColor=colors.black,
    )
    h1 = ParagraphStyle(
        'H1ZH', parent=sample['Heading1'], fontName='STSong-Light', fontSize=24,
        leading=30, alignment=TA_CENTER, spaceAfter=14, spaceBefore=6, textColor=colors.black,
    )
    h2 = ParagraphStyle(
        'H2ZH', parent=sample['Heading2'], fontName='STSong-Light', fontSize=18,
        leading=24, spaceBefore=14, spaceAfter=10, textColor=colors.black,
    )
    h3 = ParagraphStyle(
        'H3ZH', parent=sample['Heading3'], fontName='STSong-Light', fontSize=15,
        leading=20, spaceBefore=10, spaceAfter=8, textColor=colors.black,
    )
    caption = ParagraphStyle(
        'CaptionZH', parent=sample['BodyText'], fontName='STSong-Light', fontSize=10.5,
        leading=14, alignment=TA_CENTER, spaceBefore=4, spaceAfter=10, textColor=colors.HexColor('#333333'),
    )
    table_text = ParagraphStyle(
        'TableTextZH', parent=sample['BodyText'], fontName='STSong-Light',
        fontSize=max(9, cfg['table_font_size_px']),
        leading=max(12, cfg['table_font_size_px'] * 1.35),
        alignment=TA_CENTER, textColor=colors.black,
    )
    return {'h1': h1, 'h2': h2, 'h3': h3, 'body': body, 'caption': caption, 'table': table_text}


def _extract_plain_text(tag: Tag) -> str:
    """从 HTML 标签中提取尽量干净的纯文本。"""
    return tag.get_text(' ', strip=True).replace(' ', ' ').strip()


def _classify_visual(alt_text: str, width: int, height: int) -> str:
    """根据标题和宽高比，判断是普通图、宽图、表格图还是高图。"""
    alt = (alt_text or '').lower()
    ratio = width / max(height, 1)
    if '表' in alt or 'table' in alt or ratio > 1.85:
        return 'table'
    if height / max(width, 1) > 1.35:
        return 'tall'
    if ratio > 1.35:
        return 'wide'
    return 'normal'


def _scaled_image_size(width: int, height: int, visual_type: str) -> Tuple[float, float]:
    """按视觉类型和配置，把图片缩放到合适尺寸。"""
    cfg = PDF_EXPORT_CONFIG
    content_width_pt = cfg['content_width_mm'] * mm
    if visual_type == 'table':
        max_w = content_width_pt * cfg['table_visual_max_width_pct'] / 100.0
        max_h = cfg['table_visual_max_height_mm'] * mm
    elif visual_type == 'wide':
        max_w = content_width_pt * cfg['wide_visual_max_width_pct'] / 100.0
        max_h = cfg['wide_visual_max_height_mm'] * mm
    elif visual_type == 'tall':
        max_w = content_width_pt * cfg['tall_visual_max_width_pct'] / 100.0
        max_h = cfg['tall_visual_max_height_mm'] * mm
    else:
        max_w = content_width_pt * cfg['figure_max_width_pct'] / 100.0
        max_h = cfg['figure_max_height_mm'] * mm
    scale = min(max_w / max(width, 1), max_h / max(height, 1), 1.0)
    return width * scale, height * scale


def _build_image_flowable(img_key: str, caption: str, images_dict: Dict[str, str], styles: Dict[str, ParagraphStyle]):
    """把 Markdown 图片占位符转换成 ReportLab 的图片 + 图注。"""
    matched_b64 = None
    matched_name = img_key
    for img_name, b64 in images_dict.items():
        if img_key in img_name or img_name in img_key:
            matched_b64 = b64
            matched_name = img_name
            break
    if not matched_b64:
        return None
    raw = base64.b64decode(matched_b64)
    pil = PILImage.open(io.BytesIO(raw))
    width, height = pil.size
    visual_type = _classify_visual(caption or matched_name, width, height)
    draw_w, draw_h = _scaled_image_size(width, height, visual_type)
    img = RLImage(io.BytesIO(raw), width=draw_w, height=draw_h)
    img.hAlign = 'CENTER'
    cap = Paragraph(_mixed_font_markup(caption or matched_name), styles['caption'])
    return KeepTogether([img, Spacer(1, 4), cap, Spacer(1, 4)])


def _build_table_flowable(table_tag: Tag, styles: Dict[str, ParagraphStyle]):
    """把 HTML 表格转成 ReportLab Table。"""
    rows = []
    for tr in table_tag.find_all('tr'):
        cells = tr.find_all(['th', 'td'])
        if not cells:
            continue
        row = []
        for cell in cells:
            txt = _extract_plain_text(cell)
            row.append(Paragraph(_mixed_font_markup(txt), styles['table']))
        rows.append(row)
    if not rows:
        return None
    col_count = max(len(r) for r in rows)
    normalized = []
    for row in rows:
        if len(row) < col_count:
            row = row + [''] * (col_count - len(row))
        normalized.append(row)
    cfg = PDF_EXPORT_CONFIG
    content_width_pt = cfg['content_width_mm'] * mm
    col_width = content_width_pt / max(col_count, 1)
    t = Table(normalized, colWidths=[col_width] * col_count, repeatRows=1)
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'STSong-Light'),
        ('FONTSIZE', (0, 0), (-1, -1), max(9, cfg['table_font_size_px'])),
        ('LEADING', (0, 0), (-1, -1), max(12, cfg['table_font_size_px'] * 1.35)),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3f3f3')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('GRID', (0, 0), (-1, -1), 0.6, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('LEFTPADDING', (0, 0), (-1, -1), cfg['table_cell_padding_px']),
        ('RIGHTPADDING', (0, 0), (-1, -1), cfg['table_cell_padding_px']),
        ('TOPPADDING', (0, 0), (-1, -1), max(3, cfg['table_cell_padding_px'] - 1)),
        ('BOTTOMPADDING', (0, 0), (-1, -1), max(3, cfg['table_cell_padding_px'] - 1)),
    ]))
    return t


def _flush_paragraph_buffer(buffer: List[str], bucket: List, styles: Dict[str, ParagraphStyle]):
    """把累计的正文段落一次性写入 flowable 列表。"""
    if not buffer:
        return
    text = '\n'.join([x.strip() for x in buffer if x.strip()]).strip()
    buffer.clear()
    if not text:
        return
    bucket.append(Paragraph(_mixed_font_markup(text), styles['body']))
    bucket.append(Spacer(1, 4))


def _reorder_section_visuals(md_text: str, images_dict: Dict[str, str], styles: Dict[str, ParagraphStyle]):
    """解析 Markdown，并把每个二级章节中的图表移动到该章节尾部。"""
    html = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
    soup = BeautifulSoup(html, 'html.parser')
    sections = []
    current = {'text': [], 'visuals': []}
    para_buffer: List[str] = []

    def flush_current():
        _flush_paragraph_buffer(para_buffer, current['text'], styles)
        if current['text'] or current['visuals']:
            sections.append({'text': list(current['text']), 'visuals': list(current['visuals'])})
        current['text'].clear()
        current['visuals'].clear()

    for node in soup.children:
        if isinstance(node, NavigableString):
            raw = str(node).strip()
            if raw:
                para_buffer.append(raw)
            continue
        if not isinstance(node, Tag):
            continue
        if node.name == 'h1':
            _flush_paragraph_buffer(para_buffer, current['text'], styles)
            current['text'].append(Paragraph(_mixed_font_markup(_extract_plain_text(node), ascii_font='Times-Bold'), styles['h1']))
            current['text'].append(Spacer(1, 8))
            continue
        if node.name == 'h2':
            flush_current()
            current['text'].append(Paragraph(_mixed_font_markup(_extract_plain_text(node), ascii_font='Times-Bold'), styles['h2']))
            current['text'].append(Spacer(1, 4))
            continue
        if node.name in ('h3', 'h4'):
            _flush_paragraph_buffer(para_buffer, current['text'], styles)
            current['text'].append(Paragraph(_mixed_font_markup(_extract_plain_text(node), ascii_font='Times-Bold'), styles['h3']))
            current['text'].append(Spacer(1, 3))
            continue
        if node.name == 'p':
            img = node.find('img')
            if img and img.get('src'):
                _flush_paragraph_buffer(para_buffer, current['text'], styles)
                flowable = _build_image_flowable(img.get('src', '').strip(), img.get('alt', '').strip(), images_dict, styles)
                if flowable:
                    current['visuals'].append(flowable)
                    current['visuals'].append(Spacer(1, 6))
                continue
            txt = _extract_plain_text(node)
            if txt:
                para_buffer.append(txt)
            continue
        if node.name == 'table':
            _flush_paragraph_buffer(para_buffer, current['text'], styles)
            table_flow = _build_table_flowable(node, styles)
            if table_flow:
                current['visuals'].append(KeepTogether([table_flow, Spacer(1, 8)]))
            continue
        if node.name == 'pre':
            _flush_paragraph_buffer(para_buffer, current['text'], styles)
            pre_text = _extract_plain_text(node)
            if pre_text:
                current['text'].append(Paragraph(_mixed_font_markup(pre_text, ascii_font='Courier'), styles['body']))
                current['text'].append(Spacer(1, 6))
            continue
        txt = _extract_plain_text(node)
        if txt:
            para_buffer.append(txt)

    flush_current()
    flowables = []
    for section in sections:
        flowables.extend(section['text'])
        flowables.extend(section['visuals'])
    return flowables


def build_pdf_bytes_from_markdown(md_text: str, images_dict: Dict[str, str]) -> bytes:
    """把最终 Markdown 报告排版成真正的 PDF 字节流。"""
    styles = _build_pdf_styles()
    cfg = PDF_EXPORT_CONFIG
    story = _reorder_section_visuals(md_text, images_dict, styles)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=cfg['margin_left_mm'] * mm,
        rightMargin=cfg['margin_right_mm'] * mm,
        topMargin=cfg['margin_top_mm'] * mm,
        bottomMargin=cfg['margin_bottom_mm'] * mm,
        title='论文全维度透视报告',
        author='ChatGPT',
    )
    doc.build(story)
    return buffer.getvalue()
# ==========================================
# 模块 8: LLM 客户端类
# ==========================================
# 说明：
# 对 OpenAI 兼容接口做一个轻量封装，并且内置简单重试逻辑。

class LLMClient:
    def __init__(self, sys_prompt, model="deepseek-chat", api_key="", base_url=""):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.sys_prompt = sys_prompt

    def generate(self, prompt_history: List[str], max_tokens: int = 8192) -> str:
        """纯文本生成接口。max_tokens 显式给大一点，减少长输出被截断的概率。"""
        messages = [{"role": "system", "content": self.sys_prompt}]
        for msg in prompt_history:
            messages.append({"role": "user", "content": msg})

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt < 2:
                    time.sleep(3)
                else:
                    raise e

    def generate_with_images(self, user_prompt: str, base64_images: List[str], max_tokens: int = 4096) -> str:
        """多模态生成接口，支持给 Vision Agent 喂图。"""
        messages = [{"role": "system", "content": self.sys_prompt}]
        content_list = [{"type": "text", "text": user_prompt}]
        for b64 in base64_images:
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        messages.append({"role": "user", "content": content_list})

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt < 2:
                    time.sleep(3)
                else:
                    raise e

# ==========================================
# 模块 9: PDF 结构化解析接口
# ==========================================
# 说明：
# 这里调用你的 Modal 服务，把原始 PDF 解析成 markdown + 图片字典。
# 这一步是整个深度解读流程的输入基础。


def analyze_pdf_with_modal(pdf_file_bytes: bytes):
    """把上传的 PDF 发给云端解析服务，返回 markdown 与图片字典。"""
    with st.spinner("正在唤醒云端 GPU 引擎，深度解析公式与版面..."):
        try:
            files_payload = {"file": ("paper.pdf", pdf_file_bytes, "application/pdf")}
            response = requests.post(MODAL_API_URL, files=files_payload, timeout=600)
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    return result
                st.error(f"解析内部错误: {result.get('message')}")
            else:
                st.error(f"服务器响应错误: {response.status_code}")
        except Exception as e:
            st.error(f"连接云端失败: {str(e)}")
    return None

# ==========================================
# 模块 10: 报告生成辅助逻辑
# ==========================================
# 说明：
# 这里把“文本抽取 -> 图表证据抽取 -> 核心报告生成 -> 研究路线生成 -> 审校修订”拆开，
# 尽量避免 one-shot 长输出被截断，只生成前几节就结束。


def _remove_overlap(prev: str, new: str) -> str:
    """去掉多轮续写时首尾可能重复的内容。"""
    prev_tail = prev[-1200:]
    max_overlap = min(len(prev_tail), len(new), 500)
    for n in range(max_overlap, 30, -1):
        if prev_tail[-n:] == new[:n]:
            return new[n:]
    return new


def generate_until_marker(agent: LLMClient, task_prompt: str, end_marker: str = '[END_OF_SECTION]', max_rounds: int = 4, max_tokens: int = 6500) -> str:
    """带结束标记的稳健长文本生成。若首轮输出被截断，就继续续写直到看到结束标记。"""
    accumulated = ''
    for round_idx in range(max_rounds):
        if round_idx == 0:
            prompt = f"""{task_prompt}

请严格遵守以下要求：
1. 只输出当前请求的正文内容，不要输出解释。
2. 当你完整写完本次请求的全部内容后，必须单独一行输出 {end_marker}。
3. 如果一次输出不完，不要写总结，不要写结束语，直接在下一轮继续。"""
        else:
            prompt = f"""你上一轮的输出在长度限制处被截断了。请从前文结尾继续写，不要重复已经写过的内容。

【已生成内容末尾】
{accumulated[-3500:]}

请继续，并在完整结束后单独一行输出 {end_marker}。"""
        part = (agent.generate([prompt], max_tokens=max_tokens) or '').strip()
        if not part:
            break
        if accumulated:
            part = _remove_overlap(accumulated, part)
            accumulated += '\n' + part.strip()
        else:
            accumulated = part
        if end_marker in accumulated:
            accumulated = accumulated.split(end_marker)[0].strip()
            break
        if len(part.strip()) < 80:
            break
    return normalize_report_markdown(accumulated)


def build_main_context(md_content: str, text_report: str, vision_summaries: str, image_ids: List[str]) -> str:
    """给 Main Agent 构造上下文。"""
    source_pack = build_source_pack(md_content, max_chars=50000)
    available_img_ids = "\n".join([f"- {k}" for k in image_ids]) if image_ids else "无可用图片"
    return f"""
【论文原始结构化 markdown】
{source_pack}

【Text Agent 输出】
{text_report}

【Vision Agent 输出】
{vision_summaries}

【当前所有可用的图表真实标识符列表】（你必须从中复制以插入图片）
{available_img_ids}
"""


def generate_main_report_by_sections(main_agent: LLMClient, combined_prompt: str) -> str:
    """把主报告拆成第 1-7 节分别生成，降低长输出截断概率。"""
    parts = ["# 论文全维度深度透视报告"]
    for _, task in REPORT_SECTION_SPECS:
        part = main_agent.generate([combined_prompt + "\n\n" + task])
        parts.append(normalize_report_markdown(part))
    return normalize_report_markdown("\n\n".join(parts))


def patch_missing_sections(main_agent: LLMClient, combined_prompt: str, report_md: str) -> str:
    """如果第 1-7 节中有缺节，就只补缺失章节，而不是整篇重写。"""
    missing = find_missing_sections(report_md)
    if not missing:
        return report_md

    patched = [report_md]
    for _, task in missing:
        part = main_agent.generate([combined_prompt + "\n\n" + task])
        patched.append(normalize_report_markdown(part))

    return normalize_report_markdown("\n\n".join(patched))


def generate_research_section(
    research_agent: LLMClient,
    md_content: str,
    text_report: str,
    vision_summaries: str,
    core_report: str,
) -> str:
    """单独生成第 8 节，把研究设想和前 1-7 节事实层隔离。"""
    source_pack = build_source_pack(md_content, max_chars=50000)
    prompt = f"""
【论文原始结构化 markdown】
{source_pack}

【Text Agent 输出】
{text_report}

【Vision Agent 输出】
{vision_summaries}

【已生成的第1-7节主报告】
{core_report}
"""
    return generate_until_marker(research_agent, prompt)


def audit_prompt_for(
    md_content: str,
    text_report: str,
    vision_summaries: str,
    final_report: str,
    image_ids: List[str],
) -> str:
    """构造审校输入。"""
    source_pack = build_source_pack(md_content, max_chars=50000)
    available_img_ids = "\n".join([f"- {k}" for k in image_ids]) if image_ids else "无可用图片"
    return f"""
【原始 markdown】
{source_pack}

【Text Agent 输出】
{text_report}

【Vision Agent 输出】
{vision_summaries}

【当前报告】
{final_report}

【合法图片ID】
{available_img_ids}
"""


def audit_and_revise_report(
    main_agent: LLMClient,
    auditor: LLMClient,
    combined_prompt: str,
    md_content: str,
    text_report: str,
    vision_summaries: str,
    final_report: str,
    image_ids: List[str],
) -> Tuple[str, str]:
    """执行最多两轮审校。失败时只修报告，不从头重跑 Text/Vision。"""
    MAX_AUDIT_ROUNDS = 2
    last_audit_result = ""

    for round_idx in range(MAX_AUDIT_ROUNDS):
        try:
            audit_result = auditor.generate([
                audit_prompt_for(md_content, text_report, vision_summaries, final_report, image_ids)
            ])
            last_audit_result = audit_result
        except Exception as e:
            if round_idx == 0:
                continue
            st.warning(f"审校阶段失败，已返回未完成终审的报告：{e}")
            break

        if not re.search(r"RESULT\s*:\s*FAIL", audit_result, flags=re.I):
            break

        revise_prompt = f"""
{combined_prompt}

【当前报告】
{final_report}

【审校意见】
{audit_result}

请严格根据审校意见修订整篇 Markdown 报告，只输出修订后的最终报告。
"""
        final_report = generate_until_marker(main_agent, revise_prompt, end_marker="[END_OF_REPORT]", max_rounds=3, max_tokens=7000)

    return final_report, last_audit_result

# ==========================================
# 模块 11: 论文精读与渲染主管线
# ==========================================
# 说明：
# 这是“上传 PDF -> 生成报告 -> 页面展示 -> 提供导出”的核心流程。


def render_analysis_ui(pdf_bytes: bytes):
    """对上传 PDF 执行完整的深度解读工作流。"""
    file_hash = stable_file_hash(pdf_bytes)

    if st.session_state.get("current_pdf_hash") != file_hash:
        # 只要换文件，就清空和上一个 PDF 相关的状态
        st.session_state.current_pdf_hash = file_hash
        st.session_state.final_main_report = ""
        st.session_state.final_audit_result = ""
        st.session_state.temp_images = {}

        result = analyze_pdf_with_modal(pdf_bytes)
        if result and result.get("status") == "success":
            md_content = result.get("markdown", "")
            ordered_images = sort_images_by_doc_order(md_content, result.get("images", {}))
            st.session_state.temp_images = dict(ordered_images)

            # 1) Text Agent：纯事实抽取
            with st.spinner("文本专家正在精读全篇文本..."):
                text_agent = LLMClient(
                    sys_prompt=TEXT_AGENT_PROMPT,
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL,
                )
                text_report = text_agent.generate([f"请详尽解析此论文：\n{md_content}"])

            # 2) Vision Agent：给图片 + 局部原文上下文
            vision_cards = []
            if st.session_state.temp_images:
                with st.spinner(f"视觉专家正在分析 {len(st.session_state.temp_images)} 张关键图表..."):
                    vision_agent = LLMClient(
                        sys_prompt=VISION_AGENT_PROMPT,
                        model="qwen3.6-plus",
                        api_key=QWEN_API_KEY,
                        base_url=QWEN_BASE_URL,
                    )
                    for name, b64 in ordered_images:
                        local_context = extract_local_context(md_content, name)
                        vision_prompt = f"""
请基于以下论文原文上下文与图片，严格输出 FIGURE_CARD。

【图片ID】
{name}

【论文局部上下文】
{local_context}

【论文整体上下文摘要】
{build_source_pack(md_content, max_chars=4000)}
"""
                        v_res = vision_agent.generate_with_images(vision_prompt, [b64])
                        vision_cards.append(f"\n--- 图表标识: {name} ---\n{v_res}\n")

            vision_summaries = "\n".join(vision_cards)

            # 3) Main Agent：分章节生成第 1-7 节
            with st.spinner("总报告主编正在分章节生成第 1-7 节..."):
                main_agent = LLMClient(
                    sys_prompt=MAIN_AGENT_PROMPT,
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL,
                )
                combined_prompt = build_main_context(
                    md_content,
                    text_report,
                    vision_summaries,
                    list(st.session_state.temp_images.keys()),
                )
                core_report = generate_main_report_by_sections(main_agent, combined_prompt)
                core_report = patch_missing_sections(main_agent, combined_prompt, core_report)

            # 4) Research Agent：单独生成第 8 节
            with st.spinner("研究路线设计专家正在生成第 8 节..."):
                research_agent = LLMClient(
                    sys_prompt=RESEARCH_AGENT_PROMPT,
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL,
                )
                section8 = generate_research_section(
                    research_agent,
                    md_content,
                    text_report,
                    vision_summaries,
                    core_report,
                )

            final_report = normalize_report_markdown(core_report + "\n\n" + section8)

            # 5) Auditor：最多两轮审校
            with st.spinner("学术审校员正在检查章节完整性、证据一致性与图片引用合法性..."):
                auditor = LLMClient(
                    sys_prompt=REPORT_AUDITOR_PROMPT,
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL,
                )
                final_report, audit_result = audit_and_revise_report(
                    main_agent=main_agent,
                    auditor=auditor,
                    combined_prompt=combined_prompt,
                    md_content=md_content,
                    text_report=text_report,
                    vision_summaries=vision_summaries,
                    final_report=final_report,
                    image_ids=list(st.session_state.temp_images.keys()),
                )

            st.session_state.final_main_report = final_report
            st.session_state.final_audit_result = audit_result

    # 只要 final_main_report 存在，就显示报告
    if st.session_state.get("final_main_report"):
        st.success("论文全维度透视报告已生成！")
        render_report_with_images(
            st.session_state.final_main_report,
            st.session_state.temp_images,
        )

        st.divider()
        st.markdown("### 导出与下载")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "下载报告原文 (Markdown)",
                st.session_state.final_main_report,
                file_name="论文全维度透视报告.md",
                use_container_width=True,
            )
        with col2:
            pdf_cache_key = hashlib.md5(
                (st.session_state.final_main_report + '||' + '|'.join(st.session_state.temp_images.keys())).encode('utf-8')
            ).hexdigest()
            if st.session_state.get('final_pdf_hash') != pdf_cache_key:
                with st.spinner('正在生成高质量 PDF（矢量文本排版）...'):
                    try:
                        st.session_state.final_pdf_bytes = build_pdf_bytes_from_markdown(
                            st.session_state.final_main_report,
                            st.session_state.temp_images,
                        )
                        st.session_state.final_pdf_hash = pdf_cache_key
                    except Exception as e:
                        st.session_state.final_pdf_bytes = b''
                        st.session_state.final_pdf_hash = None
                        st.error(f'PDF 生成失败：{e}')

            if st.session_state.get('final_pdf_bytes'):
                st.download_button(
                    '下载高质量 PDF 报告',
                    data=st.session_state.final_pdf_bytes,
                    file_name='论文全维度透视报告.pdf',
                    mime='application/pdf',
                    use_container_width=True,
                )
            else:
                st.info('当前未成功生成 PDF，请先查看上方报错信息。')

        # 审校结果默认折叠，方便排查“为什么某些节没有生成/被修订”这类问题
        if st.session_state.get("final_audit_result"):
            with st.expander("查看审校结果", expanded=False):
                st.code(st.session_state.final_audit_result, language="text")

# ==========================================
# 模块 12: 前端 UI
# ==========================================
# 说明：
# 这里定义页面标题、侧边栏检索配置与“上传 PDF 直接精读”的入口。

st.title("AI 智能论文检索 Agent")
st.markdown("基于大模型的多轮深度挖掘，为您精准匹配 Top 6 核心前沿文献。")

with st.sidebar:
    st.header("检索配置")

    user_topic = st.text_input(
        "研究方向",
        value="",
    )

    user_requirements = st.text_area(
        "具体筛选要求",
        value="",
        placeholder="建议分点填写",
        help="要求越具体，Agent 挖掘的文献越精准。",
    )

    allow_preprint = st.radio(
        "文献收录标准",
        ("仅限同行评审文献 (排除预印本)", "接受预印本 (如 arXiv)"),
    )

    start_button = st.button("开始智能检索", type="primary", use_container_width=True)

    st.divider()

    st.header("文献直读")
    sidebar_pdf = st.file_uploader(
        "上传本地 PDF 进行结构化解析",
        type="pdf",
        key="sb_pdf",
        help="跳过检索步骤，直接对已有文献生成精读报告",
    )
    start_analyze_button = st.button(
        "开始解读",
        type="primary",
        key="start_analyze_btn",
        use_container_width=True,
        disabled=not sidebar_pdf,
    )

# ==========================================
# 模块 13: 全局应用状态初始化
# ==========================================
# 说明：
# Streamlit 每次交互都会 rerun，所以所有状态都要通过 session_state 保存。

DEFAULT_STATES = {
    "app_state": "IDLE",
    "prompt_history": [],
    "agent": None,
    "final_result": "",
    "loop_count": 0,
    "has_provided_feedback": False,
    "feedback_start_time": None,
    "ui_logs": [],
    "current_pdf_hash": None,
    "final_main_report": "",
    "final_audit_result": "",
    "final_pdf_bytes": b"",
    "final_pdf_hash": None,
    "temp_images": {},
}

for key, value in DEFAULT_STATES.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ==========================================
# 模块 14: 业务路由与主循环
# ==========================================
# 说明：
# 这一段维持你原有的“论文搜索 -> 结果确认 -> 上传 PDF 精读”的状态机逻辑。

if start_analyze_button and sidebar_pdf:
    st.markdown("---")
    st.info("正在启动【直接解析模式】，开始解构文献...")
    render_analysis_ui(sidebar_pdf.read())
    st.stop()

if start_button:
    if not user_topic:
        st.warning("请填写研究方向！")
    else:
        seen_paper_ids.clear()
        sys_prompt = get_system_prompt(user_requirements, allow_preprint)
        st.session_state.agent = LLMClient(
            sys_prompt=sys_prompt,
            model="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        st.session_state.prompt_history = [f"用户请求: {user_topic}"]
        st.session_state.app_state = "RUNNING"
        st.session_state.loop_count = 0
        st.session_state.has_provided_feedback = False
        st.session_state.ui_logs = []
        st.rerun()

if st.session_state.app_state == "IDLE":
    st.markdown("""
### 系统使用指南

欢迎使用 AI 智能论文检索 Agent。本系统旨在通过深度信息挖掘与多轮交互，为您精准匹配最具参考价值的前沿文献。为获得最佳体验，请参考以下操作规范：

**一、 智能文献检索**

1. **精准配置检索条件**
   请在左侧边栏填写宏观的“研究方向”。为进一步提升检索精度，建议在“具体筛选要求”中分点详细说明：研究的特定子领域、目标应用场景、核心算法要求或其他限制条件。您还可以根据严谨性需求，勾选是否排除预印本（如 arXiv）文献。

2. **人机协同与动态纠偏**
   首次检索完成后，系统将输出初步筛选的 6 篇高相关性候选文献。本系统支持动态调优：若结果偏离预期，您无需重新开始，只需在反馈对话框中指出理解偏差或追加新的约束条件，Agent 将据此进行下一轮定向纠偏与深度检索。

3. **会话时效管理**
   为保障系统底层计算资源的有效流转，系统在等待用户反馈时设有 30 分钟的静默超时机制。若超过此时限未收到新指令，当前检索任务将自动归档结束。

**二、 既有文献直读**

* **本地 PDF 深度解析**
  若您已有确定的目标文献，可跳过检索环节。直接通过左侧边栏底部的“上传 PDF 立即深度解读”入口提交文件，系统将自动提取文本并生成结构化的文献精读分析报告。
""")

if st.session_state.app_state != "IDLE":
    st.markdown("### Agent 检索执行轨迹")
    for log in st.session_state.ui_logs:
        with st.expander(log["title"], expanded=False):
            st.markdown(log["content"])

if st.session_state.app_state == "RUNNING":
    st.info("Agent 正在自主检索并筛选文献，请稍候...")
    current_step_container = st.container()

    with st.spinner("Agent 正在思考和执行学术工具..."):
        while True:
            st.session_state.loop_count += 1
            i = st.session_state.loop_count

            loop_reminder = "系统提示: 正在执行检索..." if i > 1 else "系统提示: 第一次循环开始..."
            st.session_state.prompt_history.append(loop_reminder)

            output = st.session_state.agent.generate(st.session_state.prompt_history)
            st.session_state.prompt_history.append(output)

            log_entry = {
                "title": f"Agent 运行日志 (第 {i} 步)",
                "content": f"**Agent思考与决策:**\n```text\n{output}\n```",
            }
            st.session_state.ui_logs.append(log_entry)
            with current_step_container.expander(log_entry["title"], expanded=True):
                st.markdown(log_entry["content"])

            action_match = re.search(r"Action:\s*(.*)", output, re.DOTALL)
            if not action_match:
                continue

            action_str = action_match.group(1).strip()

            if action_str.startswith("Finish"):
                clean_result = re.sub(r"^Finish\s*[:：\[]?\s*", "", action_str).rstrip("]").strip()
                st.session_state.final_result = clean_result
                st.session_state.app_state = "WAITING_FEEDBACK" if not st.session_state.has_provided_feedback else "COMPLETED"
                st.rerun()
                break

            tool_match = re.search(r"(\w+)\((.*)\)", action_str)
            if tool_match:
                tool_name, args_str = tool_match.group(1), tool_match.group(2)
                raw_kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
                observation = available_tools[tool_name](**raw_kwargs) if tool_name in available_tools else "错误"
                st.session_state.prompt_history.append(f"Observation: {observation}")

elif st.session_state.app_state == "WAITING_FEEDBACK":
    st.markdown("### 阶段性检索结果展示")
    with st.container(border=True):
        st.markdown(st.session_state.final_result)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("满意，确认检索组合", use_container_width=True):
            st.session_state.app_state = "COMPLETED"
            st.rerun()
    with col2:
        with st.popover("不满意，修改筛选条件", use_container_width=True):
            new_req = st.text_area("指出不符合要求的地方/添加新约束：")
            if st.button("提交纠偏指令"):
                st.session_state.prompt_history.append(f"用户反馈: {new_req}")
                st.session_state.has_provided_feedback = True
                st.session_state.app_state = "RUNNING"
                st.rerun()

elif st.session_state.app_state == "COMPLETED":
    st.success("文献检索任务已圆满完成！")
    st.markdown("### 最终确认的 Top 6 核心论文推荐")
    with st.container(border=True):
        st.markdown(st.session_state.final_result)

    st.divider()
    st.header("开启深度解读工作流")
    st.info("从上方选定并下载任意一篇论文的 PDF，在此上传，系统将立刻调动【结构拆解 + 文本解读 + 图表洞察】全自动引擎为您庖丁解牛。")

    uploaded_pdf = st.file_uploader("上传 PDF 文件以获取精读报告", type="pdf", key="bottom_pdf")
    bottom_start_btn = st.button(
        "开始深度解读",
        type="primary",
        disabled=not uploaded_pdf,
        use_container_width=True,
    )
    if bottom_start_btn and uploaded_pdf:
        render_analysis_ui(uploaded_pdf.read())

    if st.button("开启全新检索轮次", type="primary"):
        st.session_state.clear()
        st.rerun()
