# ==========================================
# 模块 1：依赖导入与页面配置
# ==========================================
import re
import os
import time
import html
import base64
import hashlib
import tempfile
import unicodedata
import json
from io import BytesIO
from typing import Any, Dict, List, Tuple, Optional
import datetime
import requests
import streamlit as st
from openai import OpenAI
from streamlit_autorefresh import st_autorefresh
# 服务器端 PDF 排版依赖
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate,
    PageTemplate,
    Frame,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    KeepTogether,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont

st.set_page_config(page_title="AI 论文检索 Agent", page_icon="📚", layout="wide")


# ==========================================
# 模块 2：全局变量与 API 配置
# ==========================================
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

QWEN_API_KEY = st.secrets["QWEN_API_KEY"]
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MODAL_API_URL = st.secrets["MODAL_API_URL"]

ANALYSIS_CACHE_VERSION = "20260412_history_single_image_v5"

# 检索时避免重复论文
seen_paper_ids = set()

# 固定的报告章节规范：后续会逐节生成，避免一次生成过长被截断
CORE_SECTION_SPECS = [
    "## 1. 研究问题与核心贡献",
    "## 2. 背景、研究缺口与前人路线",
    "## 3. 方法总览与整体数据流",
    "## 4. 关键模块逐层机制剖析",
    "## 5. 实验设计、关键证据与论点验证",
    "## 6. 局限性与未解决问题",
]
RESEARCH_SECTION_SPEC = "## 7. 面向后续研究的可执行创新路线"

# PDF 页面与版式参数
PDF_LAYOUT = {
    "page_size": A4,
    "margin_top": 16 * mm,
    "margin_right": 16 * mm,
    "margin_bottom": 15 * mm,
    "margin_left": 16 * mm,
    "body_font_size": 12.0,
    "body_leading": 20,
    "title_font_size": 24,
    "h2_font_size": 18.5,
    "h3_font_size": 15.0,
    "caption_font_size": 10.2,
    "paragraph_space_after": 8,
    "section_space_before": 10,
    "section_space_after": 10,
    "image_max_width_ratio": 0.82,
    "wide_image_max_width_ratio": 0.92,
    "tall_image_max_width_ratio": 0.66,
    "image_max_height": 155 * mm,
    "table_image_max_height": 220 * mm,
}


# ==========================================
# 模块 3：核心 Prompt
# ==========================================
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
6. 图表必须使用 Markdown 图片语法，且图片占位符必须从提供的图片ID列表中原样复制，格式为：
   ![图X：学术化图注](图片ID) 或 ![表X：学术化表注](图片ID)
   如果 FIGURE_CARD 的图表类型或原文编号显示它是表格、对比表、结果表或消融表，正文引用和图片 caption 必须写“表X”，不得写“图X”。
7. 正文引用图表时只能写“如图X所示”“见表X”等自然表达，严禁把图片ID、文件名或长代码名称写进正文。
8. 报告中凡是正文引用的图表，必须在首次引用段落之后立即插入对应 Markdown 图片；未被正文引用的图片不得插入。同一张图片在整篇报告中最多只允许插入一次。
9. 表格前后各保留一个空行；表标题必须单独成行。
9. 语言风格保持学术、克制、清晰，不做口语化渲染。

【报告结构】
# 论文全维度深度透视报告

## 1. 研究问题与核心贡献
定义本文试图解决的核心问题，准确概括本文相对前人工作的主要创新，并说明这些创新分别改变了哪一个技术瓶颈。

## 2. 背景、研究缺口与前人路线
还原该方向的研究背景、主流技术路线及其局限，说明本文的问题为什么值得解决，以及它切入的位置在哪里。

## 3. 方法总览与整体数据流
结合原始文本和图表证据，说明系统从输入到输出的完整链路。若有总架构图，应在这里插入。

## 4. 关键模块逐层机制剖析
按照模型真实工作顺序拆解每个关键模块。每个模块都要说明：输入是什么、变换是什么、它为何必要、它与其他模块如何耦合、它预期改善了什么问题。若有模块结构图，应在对应段落处插入。不要描述实验内容，实验内容应该是在第五章才说的，这章只说模块机制就行了。

## 5. 实验设计、关键证据与论点验证
交代数据集、评价指标、对照组、主实验、消融实验。每写一个结论，都要明确指出它由哪一组结果支持，并解释这项结果验证了哪条方法主张。关键图表在对应段落后插入。

## 6. 局限性与未解决问题
区分“作者明确承认的局限”和“从实验设计中可以直接看出的未解决问题”，但后者也必须基于论文证据，而不是外部常识。
"""

RESEARCH_AGENT_PROMPT = """
你是后续研究路线设计专家。
你只能基于以下材料提出研究路线：
1. 原始论文 markdown
2. FACT_BANK 与文本综述
3. FIGURE_CARD
4. 已生成的第1-6节主报告

硬性规则：
1. 禁止把研究设想写成原文事实。
2. 每条路线都要明确对应本文的某个模块、某条实验结论或某个未解问题。
3. 必须写清：缺口、改造方案、预期收益、验证方式、技术风险。
4. 语言要学术、克制、可执行。

请只输出：
## 7. 面向后续研究的可执行创新路线
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
# 模块 4：检索 Agent Prompt
# ==========================================
def get_system_prompt(requirements, preprint_rule):
    current_year = datetime.datetime.now().year
    if preprint_rule == "排除预印本 (仅限正规期刊/会议)":
        preprint_prompt = "严禁选择 Venue 为 'Unknown Venue/Preprint' 的预印本论文。"
    else:
        preprint_prompt = "可以接受预印本论文。"

    return f"""
你是一个科研论文搜索专家。你的任务是根据研究方向，在近一年内的论文中筛选六篇。

# 你的工作流程：
1. 通过`search_and_detail_papers`工具来获得相关论文的标题、摘要和Introduction。
2. 每次搜索后，阅读标题、摘要和Introduction。如果符合用户的具体要求，将其记录在你的 Thought 中作为“备选池”累加。
3. 每次阅读完一批论文后，学习同义词或近义词，作为下一次的 `search_and_detail_papers` 查询词。【警告】：新query必须是纯粹同义词，严禁加入用户要求中的关键词！
4. 如果不符合或无法获取摘要：摒弃该论文。
5. 最终在备选池中选出最好的六篇来作为结果。
6. 必须满足用户的要求。此外如果选择了排除预定本则一定要排除预定本。

# 用户的具体筛选要求：
{requirements}
{preprint_prompt}

# 输出格式要求：
Thought: [你的思考逻辑，包括学习到的新关键词]
Action: [执行工具或结束]

Action格式支持以下两种：
1. 继续搜索时输出：
search_and_detail_papers(query="关键词")

2. 已经选够6篇好论文结束时输出：
Finish:
在此处使用 Markdown 格式直接列出选出的 6 篇论文（必须包含 1.标题 2.Venue 3.DOI 4.完整摘要 5.推荐理由）。不要写多余的话。
"""


# ==========================================
# 模块 5：通用工具函数
# ==========================================
def reconstruct_abstract(inverted_index: dict) -> str:
    """把 OpenAlex 的倒排摘要重建成普通文本。"""
    if not inverted_index:
        return ""
    word_index = [(pos, word) for word, positions in inverted_index.items() for pos in positions]
    word_index.sort(key=lambda x: x[0])
    return " ".join([word for _, word in word_index])


def search_and_detail_papers(query: str) -> str:
    """
    论文检索工具：
    1) 优先走 Semantic Scholar
    2) 若摘要缺失，则尝试使用 OpenAlex 补摘要
    """
    global seen_paper_ids

    api_key = st.secrets["S2_API_KEY"]
    email = "gaoym3@mails.neu.edu.cn"
    s2_url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={query}&limit=100&year=2025-2026"
        "&fields=paperId,title,abstract,year,externalIds,venue"
    )
    headers = {"x-api-key": api_key}

    try:
        time.sleep(2)
        s2_response = requests.get(s2_url, headers=headers, timeout=20)
        s2_response.raise_for_status()
        papers = s2_response.json().get("data", [])

        if not papers:
            return f"Observation: 未找到关于“{query}”的近一年论文。"

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
                f"  - Abstract: {final_abstract}..."
            )
            if len(results) >= 30:
                break

        if not results:
            return f"Observation: 搜索到关于“{query}”的论文，但均为已读或无摘要，未能提供新信息。"

        return f"Observation: 提取到 {len(results)} 篇全新论文：\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"Observation: 搜索出错 - {str(e)}"


available_tools = {"search_and_detail_papers": search_and_detail_papers}


def build_source_pack(md_content: str, max_chars: int = 50000) -> str:
    """
    给主报告 Agent 提供较长但不过长的原始 markdown。
    这样既保留证据，又避免 token 爆炸。
    """
    if len(md_content) <= max_chars:
        return md_content

    parts = re.split(r'(?=^#{1,6}\s)', md_content, flags=re.M)
    kept = []
    total = 0
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


def translate_fullwidth_digits(text: str) -> str:
    return (text or '').translate(str.maketrans('０１２３４５６７８９', '0123456789'))


ROMAN_NUMERAL_RE = re.compile(r'^[IVXLCDM]+$', flags=re.I)
CHINESE_NUMERAL_CHARS = '零〇一二三四五六七八九十百千万两'


def int_to_roman(num: int) -> str:
    if num <= 0 or num > 3999:
        return ''
    values = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I'),
    ]
    result = []
    rest = num
    for value, symbol in values:
        while rest >= value:
            result.append(symbol)
            rest -= value
    return ''.join(result)


def roman_to_int(value: str) -> Optional[int]:
    token = (value or '').strip().upper()
    if not token or not ROMAN_NUMERAL_RE.fullmatch(token):
        return None
    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    previous = 0
    for ch in reversed(token):
        current = roman_map.get(ch, 0)
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total if int_to_roman(total) == token else None


def chinese_to_int(value: str) -> Optional[int]:
    token = (value or '').strip().replace('两', '二').replace('〇', '零')
    if not token or any(ch not in CHINESE_NUMERAL_CHARS for ch in token):
        return None
    digit_map = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
    if token in digit_map:
        return digit_map[token]
    total = 0
    section = 0
    number = 0
    unit_map = {'十': 10, '百': 100, '千': 1000, '万': 10000}
    for ch in token:
        if ch in digit_map:
            number = digit_map[ch]
        elif ch in unit_map:
            unit = unit_map[ch]
            if unit == 10000:
                section = (section + number) or 1
                total += section * unit
                section = 0
            else:
                section += (number or 1) * unit
            number = 0
    return total + section + number


def int_to_chinese(num: int) -> str:
    if num < 0 or num > 99:
        return ''
    digits = '零一二三四五六七八九'
    if num < 10:
        return digits[num]
    tens, ones = divmod(num, 10)
    if num < 20:
        return '十' + (digits[ones] if ones else '')
    return digits[tens] + '十' + (digits[ones] if ones else '')


def normalize_report_number_token(number: str) -> str:
    return re.sub(r'\s+', '', translate_fullwidth_digits(str(number or '').strip()))


def report_number_variants(number: str) -> List[str]:
    raw = normalize_report_number_token(number)
    if not raw:
        return []
    variants = {raw, raw.upper()}
    numeric_value: Optional[int] = None
    if raw.isdigit():
        numeric_value = int(raw)
    elif ROMAN_NUMERAL_RE.fullmatch(raw):
        numeric_value = roman_to_int(raw)
    elif all(ch in CHINESE_NUMERAL_CHARS for ch in raw):
        numeric_value = chinese_to_int(raw)

    if numeric_value is not None and numeric_value > 0:
        variants.add(str(numeric_value))
        roman = int_to_roman(numeric_value)
        if roman:
            variants.add(roman)
        chinese = int_to_chinese(numeric_value)
        if chinese:
            variants.add(chinese)
    return sorted({x for x in variants if x}, key=lambda x: (len(x), x))


def find_image_line_index_and_caption(md_content: str, image_id: str) -> Tuple[Optional[int], str, List[str]]:
    lines = md_content.splitlines()
    target_key = normalize_report_image_key(image_id)
    for idx, line in enumerate(lines):
        stripped = line.strip()
        match = REPORT_IMAGE_LINE_RE.match(stripped)
        if match:
            key = match.group('key').strip()
            norm_key = normalize_report_image_key(key)
            if norm_key == target_key or (target_key and (target_key in norm_key or norm_key in target_key)):
                return idx, match.group('caption').strip(), lines
        elif target_key and target_key in normalize_report_image_key(stripped):
            return idx, '', lines
    return None, '', lines


def extract_caption_label_for_image(md_content: str, image_id: str) -> Optional[Dict[str, str]]:
    image_line_idx, inline_caption, lines = find_image_line_index_and_caption(md_content, image_id)
    if image_line_idx is None:
        return None

    candidate_indices = [image_line_idx, image_line_idx + 1, image_line_idx - 1, image_line_idx + 2, image_line_idx - 2]
    candidate_texts: List[str] = []
    if inline_caption:
        candidate_texts.append(inline_caption)
    for idx in candidate_indices:
        if idx < 0 or idx >= len(lines):
            continue
        stripped = lines[idx].strip()
        if not stripped:
            continue
        match = REPORT_IMAGE_LINE_RE.match(stripped)
        if match:
            candidate = match.group('caption').strip()
            if candidate:
                candidate_texts.append(candidate)
            continue
        candidate_texts.append(stripped)

    for candidate in candidate_texts:
        label = extract_report_label(candidate)
        if label:
            return {
                'kind': label[0],
                'number': normalize_report_number_token(label[1]),
                'caption': candidate,
                'image_line_idx': image_line_idx,
            }
    return None


def build_report_reference_patterns(kind: str, number: str) -> List[re.Pattern]:
    variants = report_number_variants(number)
    if not variants:
        return []
    patterns: List[re.Pattern] = []
    if kind == '表':
        english_names = [r'Table', r'Tab\.?']
        chinese_name = '表'
    else:
        english_names = [r'Figure', r'Fig\.?']
        chinese_name = '图'
    english_alt = '|'.join(english_names)

    for variant in variants:
        escaped = re.escape(variant)
        patterns.append(re.compile(rf'(?<![A-Za-z0-9])(?:{english_alt})\.?\s*{escaped}(?![A-Za-z0-9])', flags=re.I))
        patterns.append(re.compile(rf'{chinese_name}\s*{escaped}'))
    return patterns


def extract_nearest_heading_before_position(md_content: str, position: int) -> str:
    prefix = md_content[:max(0, position)]
    headings = re.findall(r'^(#{1,6}\s+.+)$', prefix, flags=re.MULTILINE)
    return headings[-1] if headings else '未找到最近章节标题'


def find_body_reference_position(md_content: str, image_id: str, kind: str, number: str) -> Optional[int]:
    image_line_idx, _, lines = find_image_line_index_and_caption(md_content, image_id)
    if image_line_idx is None:
        return None

    best_position: Optional[int] = None
    for pattern in build_report_reference_patterns(kind, number):
        for match in pattern.finditer(md_content):
            line_idx = md_content.count('\n', 0, match.start())
            line_text = lines[line_idx].strip() if 0 <= line_idx < len(lines) else ''
            if image_line_idx == line_idx:
                continue
            if is_standalone_figure_table_caption_line(line_text):
                continue
            if normalize_report_image_key(image_id) in normalize_report_image_key(line_text):
                continue
            if best_position is None or match.start() < best_position:
                best_position = match.start()
    return best_position


def extract_local_context(md_content: str, image_id: str, window: int = 1800) -> str:
    """按 caption 识别图表编号，再以正文引用位置为中心提取上下文。"""
    label_info = extract_caption_label_for_image(md_content, image_id)
    if not label_info:
        return ''

    citation_pos = find_body_reference_position(
        md_content,
        image_id,
        label_info.get('kind', '图'),
        label_info.get('number', ''),
    )
    if citation_pos is None:
        return ''

    start = max(0, citation_pos - window)
    end = min(len(md_content), citation_pos + window)
    nearest_heading = extract_nearest_heading_before_position(md_content, citation_pos)
    local = md_content[start:end]
    return f"【最近章节】\n{nearest_heading}\n\n【图表正文引用附近原文】\n{local}"


def collect_cited_images_by_reference(md_content: str, ordered_images: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    kept: List[Tuple[str, str]] = []
    context_map: Dict[str, str] = {}
    for name, b64 in ordered_images:
        local_context = extract_local_context(md_content, name)
        if not local_context:
            continue
        kept.append((name, b64))
        context_map[name] = local_context
    return kept, context_map


def sort_images_by_doc_order(md_content: str, images_dict: Dict[str, str]) -> List[Tuple[str, str]]:
    """按图片在 markdown 中出现的先后顺序排序。"""
    def sort_key(item):
        pos = md_content.find(item[0])
        return pos if pos >= 0 else 10**9
    return sorted(images_dict.items(), key=sort_key)


def normalize_report_markdown(md_text: str) -> str:
    """
    清洗报告 markdown：
    - 去掉孤立的 0
    - 压缩多余空行
    - 统一换行
    """
    text = md_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'^\s*0\s*$', '', text, flags=re.M)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def get_adaptive_web_image_width(image_bytes: bytes) -> Optional[int]:
    """网页报告图片按原始像素自适应显示：小图不放大，大图按类型设上限。"""
    try:
        reader = ImageReader(BytesIO(image_bytes))
        iw, ih = reader.getSize()
    except Exception:
        return None

    if iw <= 0 or ih <= 0:
        return None

    ratio = iw / max(ih, 1)
    if iw <= 520:
        return int(iw)
    if ratio > 1.8:
        return int(min(iw, 960))
    if ratio < 0.72:
        return int(min(iw, 560))
    return int(min(iw, 760))


def render_report_with_images(report_md: str, images_dict: Dict[str, str]):
    """
    Streamlit 页面预览：
    - 识别 Markdown 图片语法
    - 其他内容正常 markdown 渲染
    """
    pattern = r'(!\[.*?\]\(.*?\))'
    parts = re.split(pattern, report_md)

    for part in parts:
        if not part or not part.strip():
            continue

        img_match = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', part.strip(), flags=re.S)
        if not img_match:
            st.markdown(part)
            continue

        caption = img_match.group(1).strip()
        img_key = img_match.group(2).strip()

        matched = False
        for name, b64 in images_dict.items():
            if img_key == name or img_key in name or name in img_key:
                image_bytes = base64.b64decode(b64)
                display_width = get_adaptive_web_image_width(image_bytes)
                if display_width:
                    st.image(image_bytes, caption=caption, width=display_width)
                else:
                    st.image(image_bytes, caption=caption, use_container_width=True)
                matched = True
                break

        if not matched:
            st.markdown(part)


def find_missing_sections(report_md: str) -> List[str]:
    """检查报告是否缺少固定章节。"""
    missing = []
    for section in CORE_SECTION_SPECS + [RESEARCH_SECTION_SPEC]:
        if section not in report_md:
            missing.append(section)
    return missing


def is_report_truncated(report_md: str) -> bool:
    """粗略判断报告是否明显截断。"""
    text = report_md.strip()
    if len(text) < 800:
        return True
    if find_missing_sections(text):
        return True
    if re.search(r'[：:（(，,、\-]$', text):
        return True
    if re.search(r'##\s*\d+\..+$', text.split("\n")[-1]):
        return True
    return False


# ==========================================
# 模块 6：LLM 客户端
# ==========================================
class LLMClient:
    """
    统一的大模型调用封装：
    - 支持纯文本
    - 支持图像输入
    - 自动重试
    - 显式设置 max_tokens，降低长响应被默认截断的概率
    """
    def __init__(self, sys_prompt, model="deepseek-chat", api_key="", base_url="", max_tokens=10000):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.sys_prompt = sys_prompt
        self.max_tokens = max_tokens

    def generate(self, prompt_history: List[str]) -> str:
        messages = [{"role": "system", "content": self.sys_prompt}]
        for msg in prompt_history:
            messages.append({"role": "user", "content": msg})

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt < 2:
                    time.sleep(3)
                else:
                    raise e

    def generate_with_images(self, user_prompt: str, base64_images: List[str]) -> str:
        messages = [{"role": "system", "content": self.sys_prompt}]
        content_list = [{"type": "text", "text": user_prompt}]
        for b64 in base64_images:
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        messages.append({"role": "user", "content": content_list})

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt < 2:
                    time.sleep(3)
                else:
                    raise e


# ==========================================
# 模块 7：PDF 解析服务
# ==========================================
def analyze_pdf_with_modal(pdf_file_bytes: bytes):
    """上传 PDF 到远端解析服务，拿回结构化 markdown 和图片。"""
    with st.spinner("正在唤醒云端 GPU 引擎，深度解析公式、版面与图表……"):
        try:
            files_payload = {"file": ("paper.pdf", pdf_file_bytes, "application/pdf")}
            response = requests.post(MODAL_API_URL, files=files_payload, timeout=600)
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    return result
                st.error(f"解析内部错误：{result.get('message')}")
            else:
                st.error(f"服务器响应错误：{response.status_code}")
        except Exception as e:
            st.error(f"连接云端失败：{str(e)}")
    return None


# ==========================================
# 模块 8：主报告生成逻辑
# ==========================================
def build_analysis_context(md_content: str, text_report: str, vision_summaries: str, image_ids: List[str]) -> str:
    """构造主报告阶段的统一上下文。"""
    available_img_ids = "\n".join([f"- {x}" for x in image_ids]) if image_ids else "无可用图片"
    source_pack = build_source_pack(md_content, max_chars=50000)

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


def generate_section(main_agent: LLMClient, base_context: str, section_spec: str) -> str:
    """逐节生成报告，避免整篇 one-shot 被截断。"""
    prompt = f"""
{base_context}

请只输出这一节，并确保标题与下列要求完全一致：
{section_spec}

要求：
1. 只输出该章节，不要输出其他章节。
2. 必须包含这个章节标题。
3. 不要写“后续略”“未完待续”等字样。
4. 这是一节正式学术报告，需要多个自然段，不要只给一个长段落。
"""
    result = main_agent.generate([prompt]).strip()
    if section_spec not in result:
        result = section_spec + "\n" + result
    return result


def generate_full_report(md_content: str, text_report: str, vision_summaries: str, image_ids: List[str]) -> str:
    """
    生成完整报告：
    1) 第 1-6 节逐节生成
    2) 第 7 节单独生成
    3) 若缺节，则按缺失章节补生成
    4) 经过审校后再做修订
    """
    base_context = build_analysis_context(md_content, text_report, vision_summaries, image_ids)

    main_agent = LLMClient(
        sys_prompt=MAIN_AGENT_PROMPT,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        max_tokens=10000,
    )

    research_agent = LLMClient(
        sys_prompt=RESEARCH_AGENT_PROMPT,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        max_tokens=10000,
    )

    auditor = LLMClient(
        sys_prompt=REPORT_AUDITOR_PROMPT,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        max_tokens=7000,
    )

    sections = ["# 论文全维度深度透视报告"]
    for spec in CORE_SECTION_SPECS:
        sections.append(generate_section(main_agent, base_context, spec))

    research_prompt = f"""
{base_context}

【已生成的第1-6节主报告】
{chr(10).join(sections)}

请只输出：
{RESEARCH_SECTION_SPEC}
"""
    section7 = research_agent.generate([research_prompt]).strip()
    if RESEARCH_SECTION_SPEC not in section7:
        section7 = RESEARCH_SECTION_SPEC + "\n" + section7
    sections.append(section7)

    report = normalize_report_markdown("\n\n".join(sections))

    # 缺节补齐
    missing = find_missing_sections(report)
    if missing:
        repaired = [report]
        for spec in missing:
            if spec == RESEARCH_SECTION_SPEC:
                extra = research_agent.generate([f"{base_context}\n\n请只输出：\n{spec}"])
            else:
                extra = generate_section(main_agent, base_context, spec)
            if spec not in extra:
                extra = spec + "\n" + extra
            repaired.append(extra)
        report = normalize_report_markdown("\n\n".join(repaired))

    # 审校与最多两轮修订
    for _ in range(2):
        audit_prompt = f"""
【原始论文 markdown】
{build_source_pack(md_content, max_chars=40000)}

【Text Agent 输出】
{text_report}

【Vision Agent 输出】
{vision_summaries}

【当前报告】
{report}

【合法图片ID】
{chr(10).join(image_ids) if image_ids else "无"}
"""
        try:
            audit = auditor.generate([audit_prompt])
        except Exception:
            break

        if "RESULT: FAIL" not in audit:
            break

        revise_prompt = f"""
{base_context}

【当前报告】
{report}

【审校意见】
{audit}

请在不丢失已有有效内容的前提下修订整篇报告。
必须保留完整 7个章节标题。
只输出修订后的完整 Markdown 报告。
"""
        report = normalize_report_markdown(main_agent.generate([revise_prompt]))

    # 再兜底一次
    missing = find_missing_sections(report)
    if missing:
        repaired = [report]
        for spec in missing:
            if spec == RESEARCH_SECTION_SPEC:
                extra = research_agent.generate([f"{base_context}\n\n请只输出：\n{spec}"])
            else:
                extra = generate_section(main_agent, base_context, spec)
            if spec not in extra:
                extra = spec + "\n" + extra
            repaired.append(extra)
        report = normalize_report_markdown("\n\n".join(repaired))

    return report


# ==========================================
# 模块 9：PDF 字体与样式
# ==========================================
def register_pdf_fonts():
    """
    注册 PDF 字体：
    - STSong-Light：中文主字体
    - DejaVuSans / DejaVuSans-Bold：公式、符号与西文加粗回退
    - DejaVuSansMono：代码与等宽文本
    """
    try:
        pdfmetrics.getFont("STSong-Light")
    except KeyError:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    font_candidates = {
        "DejaVuSans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSansMono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    }
    registered = set(pdfmetrics.getRegisteredFontNames())
    for name, path in font_candidates.items():
        if name in registered or not os.path.exists(path):
            continue
        try:
            pdfmetrics.registerFont(TTFont(name, path))
        except Exception:
            continue



TABLE_TITLE_LINE_RE = re.compile(
    r'^(?:表|table)\s*(?:\d+|[IVXLCDM]+|[一二三四五六七八九十百千万]+)(?:\s*[：:.-]|(?:\s+[-–—]\s+)|\s+).+',
    flags=re.I,
)
INLINE_MARKUP_TOKEN_RE = re.compile(
    r'(\*\*[^\n]+?\*\*|__[^\n]+?__|\$[^$\n]+\$|\\\([^\n]+?\\\)|\\\[[^\n]+?\\\]|`[^`\n]+`)',
    flags=re.S,
)
ASCII_TEXT_RE = re.compile(r'([A-Za-z0-9\.\,\:\;\-\+\=\(\)\/_%#@&\[\]\{\}<>\'\"\s]+)')
MATH_UNICODE_RANGE = "𝐀-𝟿"
FORMULA_BRACE_EXPR = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
FORMULA_COMBINING_MARKS = r'[\u0300-\u036F]*'
FORMULA_SUBSUP_PATTERN = (
    rf'(?:[_^](?:{FORMULA_BRACE_EXPR}|\\[A-Za-z]+|[A-Za-z0-9Α-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]+{FORMULA_COMBINING_MARKS}))+'
)
FORMULA_TERM_PATTERN = (
    rf'(?:'
    rf'\\[A-Za-z]+(?:{FORMULA_BRACE_EXPR})*(?:{FORMULA_SUBSUP_PATTERN})*'
    rf'|'
    rf'[Α-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]+{FORMULA_COMBINING_MARKS}(?:{FORMULA_SUBSUP_PATTERN})*'
    rf'|'
    rf'[A-Za-z][A-Za-z0-9]*{FORMULA_COMBINING_MARKS}(?:{FORMULA_SUBSUP_PATTERN})'
    rf'|'
    rf'\d+(?:\.\d+)?(?:{FORMULA_SUBSUP_PATTERN})'
    rf')'
)
AUTO_FORMULA_RUN_RE = re.compile(
    rf'{FORMULA_TERM_PATTERN}'
    rf'(?:\s*(?:=|<|>|/|\+|\*|·|×|÷|\\cdot)\s*(?:{FORMULA_TERM_PATTERN}|\d+(?:\.\d+)?))*'
    rf'(?:\s+{FORMULA_TERM_PATTERN})*'
)
GREEK_LATEX_MAP = {
    'α': r'\alpha', 'β': r'\beta', 'γ': r'\gamma', 'δ': r'\delta', 'ε': r'\epsilon',
    'θ': r'\theta', 'λ': r'\lambda', 'μ': r'\mu', 'µ': r'\mu', 'σ': r'\sigma', 'ω': r'\omega',
    'π': r'\pi', 'η': r'\eta', 'τ': r'\tau', 'φ': r'\phi', 'ψ': r'\psi', 'ρ': r'\rho',
    'ν': r'\nu', 'κ': r'\kappa', 'ξ': r'\xi', 'Δ': r'\Delta', 'Γ': r'\Gamma',
    'Λ': r'\Lambda', 'Σ': r'\Sigma', 'Π': r'\Pi', 'Ω': r'\Omega', 'Φ': r'\Phi', 'Ψ': r'\Psi',
}
SPECIAL_FORMULA_CHAR_MAP = {
    'ℒ': r'\mathcal{L}', '·': r'\cdot', '×': r'\times', '÷': r'\div',
    '⊙': r'\odot', '⊕': r'\oplus', '⊗': r'\otimes', '⊘': r'\oslash',
    '⊖': r'\ominus', '⊚': r'\circledcirc', '⊛': r'\circledast',
    '≤': r'\le', '≥': r'\ge', '≠': r'\neq', '≈': r'\approx',
    '≃': r'\simeq', '≅': r'\cong', '≡': r'\equiv', '∼': r'\sim',
    '±': r'\pm', '∞': r'\infty', '∑': r'\sum', '∏': r'\prod',
    '√': r'\sqrt', '∂': r'\partial', '∇': r'\nabla',
    '∈': r'\in', '∉': r'\notin', '∋': r'\ni', '∝': r'\propto',
    '∩': r'\cap', '∪': r'\cup', '⊂': r'\subset', '⊃': r'\supset',
    '⊆': r'\subseteq', '⊇': r'\supseteq', '∅': r'\emptyset',
    '∀': r'\forall', '∃': r'\exists',
}
MATH_OPERATOR_CHARS = ''.join([
    '⊙', '⊕', '⊗', '⊘', '⊖', '⊚', '⊛',
    '≤', '≥', '≠', '≈', '≃', '≅', '≡', '∼',
    '±', '∞', '∑', '∏', '√', '∂', '∇',
    '∈', '∉', '∋', '∝', '∩', '∪', '⊂', '⊃',
    '⊆', '⊇', '∅', '∀', '∃',
])
MATH_OPERATOR_CHARS_ESC = re.escape(MATH_OPERATOR_CHARS)
MATH_VARIABLE_CHARS = 'abcdefghijklmnopqrstuvwxyz'
SUPERSCRIPT_CHAR_MAP = {
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    '⁺': '+', '⁻': '-', '⁼': '=', '⁽': '(', '⁾': ')',
    'ⁿ': 'n', 'ⁱ': 'i',
}
SUBSCRIPT_CHAR_MAP = {
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
    '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
    '₊': '+', '₋': '-', '₌': '=', '₍': '(', '₎': ')',
    'ₐ': 'a', 'ₑ': 'e', 'ₕ': 'h', 'ᵢ': 'i', 'ⱼ': 'j',
    'ₖ': 'k', 'ₗ': 'l', 'ₘ': 'm', 'ₙ': 'n', 'ₒ': 'o',
    'ₚ': 'p', 'ᵣ': 'r', 'ₛ': 's', 'ₜ': 't', 'ᵤ': 'u',
    'ᵥ': 'v', 'ₓ': 'x',
}
UNICODE_SUPERSCRIPT_CHARS = ''.join(SUPERSCRIPT_CHAR_MAP.keys())
UNICODE_SUBSCRIPT_CHARS = ''.join(SUBSCRIPT_CHAR_MAP.keys())
SPECIAL_FORMULA_RUN_RE = re.compile(
    rf'(?:'
    rf'[A-Za-zΑ-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]�?[_^][A-Za-z0-9Α-Ωα-ω]+'
    rf'|'
    rf'[A-Za-zΑ-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]+[{re.escape(UNICODE_SUPERSCRIPT_CHARS + UNICODE_SUBSCRIPT_CHARS)}]+'
    rf'|'
    rf'[{MATH_OPERATOR_CHARS_ESC}]'
    rf')'
)
CAPTION_CORE_RE = re.compile(
    r'^(?:表|图|table|figure)\s*(?:\d+|[IVXLCDM]+|[一二三四五六七八九十百千万]+)?\s*[：:.-]?\s*(.+)$',
    flags=re.I,
)
BULLET_LINE_RE = re.compile(r'^[*•\-]\s+(.*)$')
ORDERED_LIST_LINE_RE = re.compile(
    r'^((?:\(?\d+[\).、])|(?:[A-Za-z][\).])|(?:[一二三四五六七八九十]+[、.]))\s*(.*)$'
)
INLINE_BULLET_SPLIT_RE = re.compile(
    r'\s+\*\s+(?=(?:对应缺口/模块|改造方案|预期收益|验证方式|技术风险|对应缺口|技术风险)[:：])'
)
PAREN_FORMULA_CANDIDATE_RE = re.compile(r'[（(][^（）()\n]{1,140}[）)]')
SCRIPTED_FORMULA_TOKEN_RE = re.compile(
    rf'(?:\\[A-Za-z]+|[A-Za-z][A-Za-z0-9]*|[Α-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]+)'
    rf'(?:\{{[^\{{}}\n]{{1,48}}\}})?'
    rf'(?:[_^](?:{FORMULA_BRACE_EXPR}|\\[A-Za-z]+|[A-Za-z0-9Α-Ωα-ω]+))+'
)
FORMULA_COMMAND_TOKEN_RE = re.compile(
    rf'\\[A-Za-z]+(?:{FORMULA_BRACE_EXPR})*(?:{FORMULA_SUBSUP_PATTERN})*'
)
FORMULA_NAME_TOKEN_PATTERN = rf'[A-Za-zΑ-Ωα-ωℒμµ{MATH_UNICODE_RANGE}][A-Za-z0-9Α-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]*'
FORMULA_SCRIPT_PART_PATTERN = rf'(?:[_^](?:{FORMULA_BRACE_EXPR}|\\[A-Za-z]+|[A-Za-z0-9Α-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]+))*'
SIMPLE_FORMULA_ATOM_PATTERN = (
    rf'(?:'
    rf'\\[A-Za-z]+(?:{FORMULA_BRACE_EXPR})*{FORMULA_SCRIPT_PART_PATTERN}'
    rf'|{FORMULA_NAME_TOKEN_PATTERN}(?:\([^()\n]{{1,80}}\))?{FORMULA_SCRIPT_PART_PATTERN}'
    rf'|[{MATH_OPERATOR_CHARS_ESC}]'
    rf'|\d+(?:\.\d+)?{FORMULA_SCRIPT_PART_PATTERN}'
    rf')'
)
INLINE_EQUATION_RUN_RE = re.compile(
    rf'(?<![A-Za-z0-9_])'
    rf'{SIMPLE_FORMULA_ATOM_PATTERN}'
    rf'(?:'
    rf'\s*(?:=|<|>|/|\+|\-|\*|·|×|÷|\\cdot|[{MATH_OPERATOR_CHARS_ESC}])\s*{SIMPLE_FORMULA_ATOM_PATTERN}'
    rf'|\s+{SIMPLE_FORMULA_ATOM_PATTERN}'
    rf')+'
    rf'(?![A-Za-z0-9_])'
)
STANDALONE_MATH_SYMBOL_RE = re.compile(
    rf'(?<![A-Za-z0-9_])(?:[{MATH_VARIABLE_CHARS}]|[Α-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]+|[{MATH_OPERATOR_CHARS_ESC}])(?![A-Za-z0-9_])'
)

LIST_ENUMERATOR_ONLY_RE = re.compile(r'^[（(]?[0-9]+[）).、]?$')
IMPLICIT_SUBSCRIPT_BASE_RE = re.compile(
    r'(?P<base>\\mathcal\{[A-Za-z]\}|\\mathbb\{[A-Za-z]\}|\\mathrm\{[A-Za-z]\}|\\[A-Za-z]+|[A-Za-z])\{(?P<sub>[A-Za-z][A-Za-z0-9]{0,15})\}'
)


def strip_outer_markdown_markers(text: str) -> str:
    value = (text or '').strip()
    patterns = [
        r'^\*\*(.*?)\*\*$',
        r'^__(.*?)__$',
        r'^`(.*?)`$',
        r'^_(.*?)_$',
    ]

    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            match = re.fullmatch(pattern, value, flags=re.S)
            if match:
                value = match.group(1).strip()
                changed = True
    return value


def normalize_compare_text(text: str) -> str:
    cleaned = strip_outer_markdown_markers(text)
    cleaned = re.sub(r'[\s`*_~：:.,，。；;、\-—–（）()\[\]{}]+', '', cleaned)
    return cleaned.lower()


def normalize_table_title_line(text: str) -> str:
    return re.sub(r'\s+', ' ', strip_outer_markdown_markers(text)).strip()


def normalize_caption_core_text(text: str) -> str:
    cleaned = normalize_table_title_line(text)
    match = CAPTION_CORE_RE.match(cleaned)
    if match:
        cleaned = match.group(1).strip()
    return normalize_compare_text(cleaned)


def is_table_title_line(text: str) -> bool:
    return bool(TABLE_TITLE_LINE_RE.match(normalize_table_title_line(text)))


def normalize_formula_spacing(text: str) -> str:
    value = (text or '').replace('\r\n', '\n').replace('\r', '\n')
    value = re.sub(r'\s*_\s*', '_', value)
    value = re.sub(r'\s*\^\s*', '^', value)
    value = re.sub(r'(\\[A-Za-z]+)\s+\{', r'\1{', value)
    return value


def collapse_spaced_math_braces(text: str) -> str:
    def _collapse(match):
        inner = match.group(1)
        tokens = inner.split()
        if len(tokens) >= 2 and all(re.fullmatch(r'[A-Za-z0-9]', tok) for tok in tokens):
            return '{' + ''.join(tokens) + '}'
        return '{' + inner.strip() + '}'

    previous = None
    value = text
    while previous != value:
        previous = value
        value = re.sub(r'\{([^{}]+)\}', _collapse, value)
    return value


def convert_unicode_scripts_to_tex(text: str) -> str:
    value = text
    if UNICODE_SUBSCRIPT_CHARS:
        sub_re = re.compile(r'[' + re.escape(UNICODE_SUBSCRIPT_CHARS) + r']+')
        value = sub_re.sub(
            lambda m: '_{' + ''.join(SUBSCRIPT_CHAR_MAP[ch] for ch in m.group(0)) + '}',
            value,
        )
    if UNICODE_SUPERSCRIPT_CHARS:
        sup_re = re.compile(r'[' + re.escape(UNICODE_SUPERSCRIPT_CHARS) + r']+')
        value = sup_re.sub(
            lambda m: '^{' + ''.join(SUPERSCRIPT_CHAR_MAP[ch] for ch in m.group(0)) + '}',
            value,
        )
    return value


def repair_broken_formula_glyphs(text: str) -> str:
    value = text or ''
    value = value.replace('￾', '').replace('￼', '').replace('\u00ad', '')
    value = re.sub(
        r'([xX])�(?=_[dDtT])',
        lambda m: r'\tilde{' + m.group(1).lower() + '}',
        value,
    )
    value = re.sub(
        r'([A-Za-zΑ-Ωα-ω])�(?=_[A-Za-z0-9])',
        lambda m: r'\hat{' + m.group(1) + '}',
        value,
    )
    value = re.sub(
        r'([A-Za-zΑ-Ωα-ω])�',
        lambda m: r'\hat{' + m.group(1) + '}',
        value,
    )
    return value


def normalize_math_unicode_to_latex(text: str) -> str:
    value = repair_broken_formula_glyphs(text)
    value = value.replace('（', '(').replace('）', ')').replace('，', ',').replace('：', ':')
    value = convert_unicode_scripts_to_tex(value)
    for raw, latex in SPECIAL_FORMULA_CHAR_MAP.items():
        value = value.replace(raw, latex)

    value = unicodedata.normalize('NFKD', value)

    accent_map = {
        '\u0302': r'\hat',
        '\u0303': r'\tilde',
        '\u0304': r'\bar',
        '\u0307': r'\dot',
    }
    for mark, cmd in accent_map.items():
        value = re.sub(
            rf'(\\[A-Za-z]+|[A-Za-zΑ-Ωα-ω]){mark}',
            lambda m, cmd=cmd: cmd + '{' + m.group(1) + '}',
            value,
        )

    value = re.sub(r'[\u0300-\u036F]+', '', value)

    for raw, latex in GREEK_LATEX_MAP.items():
        value = value.replace(raw, latex)
    return value



def is_list_enumerator_text(text: str) -> bool:
    candidate = extract_formula_text(text)
    if not candidate:
        return False
    candidate = candidate.strip().replace('（', '(').replace('）', ')')
    return bool(LIST_ENUMERATOR_ONLY_RE.fullmatch(candidate)) or bool(re.fullmatch(r'^[A-Za-z][\).]$', candidate))


def repair_implicit_subscripts(expr: str) -> str:
    allowed_symbol_cmds = {
        r'\alpha', r'\beta', r'\gamma', r'\delta', r'\epsilon', r'\theta',
        r'\lambda', r'\mu', r'\sigma', r'\omega', r'\phi', r'\psi',
        r'\rho', r'\nu', r'\kappa', r'\xi', r'\Delta', r'\Gamma',
        r'\Lambda', r'\Sigma', r'\Pi', r'\Omega', r'\Phi', r'\Psi',
    }

    def _replace(match):
        base = match.group('base')
        sub = match.group('sub')
        if base.startswith(r'\mathcal{') or base.startswith(r'\mathbb{') or base.startswith(r'\mathrm{'):
            return f'{base}_{{{sub}}}'
        if len(base) == 1 and base.isalpha():
            return f'{base}_{{{sub}}}'
        if base in allowed_symbol_cmds:
            return f'{base}_{{{sub}}}'
        return match.group(0)

    previous = None
    value = expr
    while previous != value:
        previous = value
        value = IMPLICIT_SUBSCRIPT_BASE_RE.sub(_replace, value)
    return value


def formula_has_strong_signal(text: str) -> bool:
    value = text or ''
    if not value.strip():
        return False
    if any(ch in value for ch in MATH_OPERATOR_CHARS):
        return True
    if any(ch in value for ch in UNICODE_SUPERSCRIPT_CHARS + UNICODE_SUBSCRIPT_CHARS):
        return True
    if re.search(r'[=<>^_+*/\\]|\bcdot\b|\\[A-Za-z]+', value):
        return True
    if re.search(r'[Α-Ωα-ωℒμµ]', value):
        return True
    return False


def normalize_formula_script_groups(expr: str) -> str:
    r"""修复 L_aug_i、L_{aug}_i、L_{\mathrm{aug}_i} 这类会导致 KaTeX/MathText 报错的下标。"""
    value = expr or ''

    script_body_pattern = r'(?:[^{}]|\\[A-Za-z]+\{[^{}]*\})+'

    def _strip_outer_script_braces(script: str) -> str:
        script = (script or '').strip()
        if script.startswith('{') and script.endswith('}'):
            return script[1:-1].strip()
        return script

    def _merge_script_parts(op: str, left: str, right: str) -> str:
        left = _strip_outer_script_braces(left)
        right = _strip_outer_script_braces(right)
        return f'{op}{{{left},{right}}}'

    # 合并连续的下标/上标，覆盖 z_{drug}_{specific}、L_{aug}_i、L_i_{aug}。
    previous = None
    while previous != value:
        previous = value
        value = re.sub(
            rf'([_^])\{{({script_body_pattern})\}}\s*\1\{{({script_body_pattern})\}}',
            lambda m: _merge_script_parts(m.group(1), m.group(2), m.group(3)),
            value,
        )
        value = re.sub(
            rf'([_^])\{{({script_body_pattern})\}}\s*\1([A-Za-z0-9Α-Ωα-ω]+|\\[A-Za-z]+)',
            lambda m: _merge_script_parts(m.group(1), m.group(2), m.group(3)),
            value,
        )
        value = re.sub(
            rf'([_^])([A-Za-z0-9Α-Ωα-ω]+|\\[A-Za-z]+)\s*\1\{{({script_body_pattern})\}}',
            lambda m: _merge_script_parts(m.group(1), m.group(2), m.group(3)),
            value,
        )

    def _flatten_mathrm_scripts(text: str) -> str:
        previous_inner = None
        current = text
        while previous_inner != current:
            previous_inner = current
            current = re.sub(
                r'\\mathrm\{([^{}]*)\}\s*_\s*\{([^{}]+)\}',
                lambda m: r'\mathrm{' + m.group(1).replace(r'\_', ',') + ',' + m.group(2).strip() + '}',
                current,
            )
            current = re.sub(
                r'\\mathrm\{([^{}]*)\}\s*_\s*([A-Za-z0-9Α-Ωα-ω]+)',
                lambda m: r'\mathrm{' + m.group(1).replace(r'\_', ',') + ',' + m.group(2).strip() + '}',
                current,
            )
            current = re.sub(
                r'\\mathrm\{([^{}]*?)_([A-Za-z0-9Α-Ωα-ω]+)\}',
                lambda m: r'\mathrm{' + m.group(1).replace(r'\_', ',') + ',' + m.group(2).strip() + '}',
                current,
            )
        return current

    value = _flatten_mathrm_scripts(value)

    def _normalize_script(match):
        op = match.group(1)
        body = _flatten_mathrm_scripts((match.group(2) or '').strip())
        if not body:
            return match.group(0)
        if re.fullmatch(r'\\mathrm\{[^{}]*\}', body):
            return f'{op}{{{body}}}'
        if body.startswith('\\') and not re.fullmatch(r'\\[A-Za-z]+', body):
            return f'{op}{{{body}}}'
        if re.search(r'[+\-*/=<>]', body):
            return f'{op}{{{body}}}'
        if re.search(r'[A-Za-z]{2,}|_', body):
            safe = body.replace('\\', r'\backslash ')
            safe = safe.replace('_', ',')
            safe = re.sub(r'\s+', r'\\ ', safe)
            return f'{op}{{\\mathrm{{{safe}}}}}'
        return f'{op}{{{body}}}'

    value = re.sub(rf'([_^])\{{({script_body_pattern})\}}', _normalize_script, value)
    value = _flatten_mathrm_scripts(value)

    # 最后兜底：把 L_{\mathrm{aug}}_i 合并成 L_{\mathrm{aug,i}}，避免 Double subscript。
    previous = None
    while previous != value:
        previous = value
        value = re.sub(
            r'([A-Za-zΑ-Ωα-ω])_\{\\mathrm\{([^{}]+)\}\}\s*_\s*\{?([A-Za-z0-9Α-Ωα-ω]+)\}?',
            lambda m: f'{m.group(1)}_{{\\mathrm{{{m.group(2)},{m.group(3)}}}}}',
            value,
        )
    return value

def explode_inline_numbered_segments(text: str) -> List[str]:
    value = (text or '').strip()
    if not value:
        return []

    colon_match = re.search(r'[：:]', value)
    if not colon_match:
        return [value]

    prefix = value[:colon_match.end()].strip()
    suffix = value[colon_match.end():].strip()
    numbered_parts = [
        part.strip(' ；;')
        for part in re.split(r'(?=(?:\(?\d+[\).、]))', suffix)
        if part and part.strip(' ；;')
    ]
    if len(numbered_parts) < 2:
        return [value]

    return [prefix] + numbered_parts

def append_list_item_blocks(blocks: List[Tuple[str, object]], item_text: str, prefix: str = '- '):
    pieces = explode_inline_numbered_segments(item_text)
    if not pieces:
        return
    for idx, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        if idx == 0 and prefix:
            blocks.append(("list_item", f"{prefix}{piece}"))
        elif re.match(r'^\(?\d+[\).、]', piece):
            blocks.append(("list_item", piece))
        else:
            blocks.append(("list_item", f"- {piece}"))

def looks_like_formula_text(text: str) -> bool:
    candidate = extract_formula_text(text)
    if not candidate:
        return False
    if is_list_enumerator_text(candidate):
        return False

    normalized = collapse_spaced_math_braces(normalize_formula_spacing(candidate))
    normalized = normalize_math_unicode_to_latex(normalized)
    normalized = repair_implicit_subscripts(normalized)

    if (
        AUTO_FORMULA_RUN_RE.fullmatch(normalized)
        or SPECIAL_FORMULA_RUN_RE.fullmatch(normalized)
        or SCRIPTED_FORMULA_TOKEN_RE.fullmatch(normalized)
    ):
        return True

    if FORMULA_COMMAND_TOKEN_RE.fullmatch(normalized):
        return True

    repaired = repair_broken_formula_glyphs(normalized)
    if repaired != normalized and (AUTO_FORMULA_RUN_RE.search(repaired) or '_' in repaired or '^' in repaired):
        return True

    if SCRIPTED_FORMULA_TOKEN_RE.search(normalized) or FORMULA_COMMAND_TOKEN_RE.search(normalized):
        return True

    explicit_tokens = [
        r'\frac', r'\sum', r'\prod', r'\sqrt', r'\alpha', r'\beta', r'\gamma',
        r'\theta', r'\lambda', r'\mu', r'\sigma', r'\omega', r'\mathbf',
        r'\mathrm', r'\mathcal', r'\mathbb', r'\hat', r'\tilde', r'\left',
        r'\right', r'\begin', r'\end', r'\log', r'\exp', r'\arg', r'\max',
        r'\min',
    ]
    if any(token in normalized for token in explicit_tokens):
        return True

    if re.search(r'[Α-Ωα-ωℒμµ]', normalized):
        return True

    if any(ch in normalized for ch in UNICODE_SUPERSCRIPT_CHARS + UNICODE_SUBSCRIPT_CHARS):
        return True

    if any(ch in normalized for ch in MATH_OPERATOR_CHARS):
        return True

    has_letter = bool(re.search(r'[A-Za-zΑ-Ωα-ω]', normalized))
    operator_count = sum(ch in normalized for ch in "=<>^_+-*/\\")
    brace_count = normalized.count('{') + normalized.count('}')

    if has_letter and ('_' in normalized or '^' in normalized):
        return True
    if has_letter and brace_count >= 2 and operator_count >= 1:
        return True
    return has_letter and operator_count >= 2

def extract_formula_text(text: str) -> str:
    candidate = (text or '').strip()
    if not candidate:
        return ''

    if candidate.startswith('$$') and candidate.endswith('$$') and len(candidate) >= 4:
        candidate = candidate[2:-2]
    elif candidate.startswith(r'\(') and candidate.endswith(r'\)') and len(candidate) >= 4:
        candidate = candidate[2:-2]
    elif candidate.startswith(r'\[') and candidate.endswith(r'\]') and len(candidate) >= 4:
        candidate = candidate[2:-2]
    elif candidate.startswith('`') and candidate.endswith('`') and len(candidate) >= 2:
        candidate = candidate[1:-1]

    candidate = re.sub(r'^```(?:math|latex|tex)?\s*', '', candidate, flags=re.I)
    candidate = re.sub(r'\s*```$', '', candidate)
    candidate = candidate.replace('\r\n', '\n').replace('\r', '\n')
    candidate = '\n'.join([line.strip() for line in candidate.split('\n') if line.strip()])
    return candidate.strip()


def sanitize_formula_for_render(text: str) -> str:
    expr = extract_formula_text(text)
    if not expr:
        return ''

    expr = repair_broken_formula_glyphs(expr)
    expr = expr.replace('$', ' ')
    expr = normalize_formula_spacing(expr)
    expr = collapse_spaced_math_braces(expr)
    expr = normalize_math_unicode_to_latex(expr)
    expr = repair_implicit_subscripts(expr)

    expr = re.sub(r'\\begin\{[^{}]+\}', '', expr)
    expr = re.sub(r'\\end\{[^{}]+\}', '', expr)
    expr = expr.replace('&', ' ')
    expr = expr.replace('\\\\', ' ')
    expr = expr.replace(r'\displaystyle', '')
    expr = expr.replace('−', '-').replace('–', '-').replace('—', '-')
    expr = re.sub(r'\\label\{.*?\}', '', expr)
    expr = re.sub(r'\\tag\{.*?\}', '', expr)
    expr = re.sub(r'\\nonumber\b', '', expr)
    expr = re.sub(r'(?<=\d)\s+(?=\d)', '', expr)
    expr = re.sub(
        r'\\text\s*\{([^{}]*)\}',
        lambda m: r'\mathrm{' + m.group(1).replace(' ', r'\ ') + '}',
        expr,
    )
    expr = re.sub(
        r'\\operatorname\s*\{([^{}]*)\}',
        lambda m: r'\mathrm{' + m.group(1).replace(' ', r'\ ') + '}',
        expr,
    )
    expr = re.sub(
        r'\\mathrm\{([^{}]*)\}',
        lambda m: r'\mathrm{' + ''.join(m.group(1).split()) + '}',
        expr,
    )
    expr = re.sub(r'_(?!\{)\s*(\\[A-Za-z]+)', r'_{\1}', expr)
    expr = re.sub(r'\^(?!\{)\s*(\\[A-Za-z]+)', r'^{\1}', expr)
    expr = re.sub(r'_(?!\{)\s*([A-Za-z0-9]{2,})', r'_{\1}', expr)
    expr = re.sub(r'\^(?!\{)\s*([A-Za-z0-9]{2,})', r'^{\1}', expr)
    expr = normalize_formula_script_groups(expr)
    expr = re.sub(r'\\Sigma(?=\s*[({])', r'\\sum', expr)
    expr = expr.replace('...', r'\ldots')
    expr = re.sub(r'\s*\*\s*', lambda m: r' \cdot ', expr)
    expr = re.sub(r'\s+', ' ', expr).strip()
    return expr


def create_formula_asset_context(formula_dir: str) -> Dict[str, Any]:
    return {
        'formula_dir': formula_dir,
        'formula_cache': {},
    }


def render_formula_image(
    formula_text: str,
    asset_ctx: Dict[str, Any],
    font_size: float = 12.0,
    display: bool = False,
) -> Optional[Dict[str, Any]]:
    expr = sanitize_formula_for_render(formula_text)
    if not expr:
        return None

    cache_key = hashlib.sha256(
        f"{font_size:.2f}|{int(display)}|{expr}".encode('utf-8')
    ).hexdigest()
    cached = asset_ctx['formula_cache'].get(cache_key)
    if cached and os.path.exists(cached['path']):
        return cached

    try:
        os.makedirs(asset_ctx['formula_dir'], exist_ok=True)
        mpl_config_dir = os.path.join(asset_ctx['formula_dir'], 'mplconfig')
        os.makedirs(mpl_config_dir, exist_ok=True)
        os.environ['MPLCONFIGDIR'] = mpl_config_dir

        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import rcParams
        from matplotlib.font_manager import FontProperties
        from matplotlib.mathtext import math_to_image

        rcParams['mathtext.fontset'] = 'dejavusans'
        rcParams['font.family'] = 'DejaVu Sans'

        output_path = os.path.join(asset_ctx['formula_dir'], f'formula_{cache_key}.png')
        prop = FontProperties(family='DejaVu Sans', size=font_size + (2 if display else 0))
        math_to_image(f'${expr}$', output_path, prop=prop, dpi=220, format='png', color='black')

        reader = ImageReader(output_path)
        width_px, height_px = reader.getSize()
        asset = {
            'path': output_path,
            'width': width_px * 72.0 / 220.0,
            'height': height_px * 72.0 / 220.0,
        }
        asset_ctx['formula_cache'][cache_key] = asset
        return asset
    except Exception:
        return None

def wrap_plain_text_basic(text: str, bold: bool = False) -> str:
    """
    将普通文本转换为 ReportLab Paragraph 可识别的安全行内标记。

    设计原则：
    - 常规英数字使用 Times-Roman / Times-Bold，确保英文加粗真正生效；
    - 中文与其他非 ASCII 文本默认走 STSong-Light；
    - 对中文加粗场景，不再嵌套 <font> 到 <b> 内，避免被解析成普通字重或触发字体映射异常。
    """
    escaped = html.escape(text)
    if not escaped:
        return ''

    parts = []
    ascii_font = 'Times-Bold' if bold else 'Times-Roman'
    for chunk in ASCII_TEXT_RE.split(escaped):
        if not chunk:
            continue
        if ASCII_TEXT_RE.fullmatch(chunk):
            parts.append(f'<font name="{ascii_font}">{chunk}</font>')
        else:
            if bold:
                parts.append(f'<b>{chunk}</b>')
            else:
                parts.append(f'<font name="STSong-Light">{chunk}</font>')
    return ''.join(parts)


def should_auto_render_formula(candidate: str) -> bool:
    stripped = extract_formula_text(candidate)
    if not stripped:
        return False
    if is_list_enumerator_text(stripped):
        return False

    normalized = collapse_spaced_math_braces(normalize_formula_spacing(stripped))
    normalized_latex = normalize_math_unicode_to_latex(normalized)

    if INLINE_EQUATION_RUN_RE.fullmatch(normalized) and formula_has_strong_signal(normalized):
        return True
    if SCRIPTED_FORMULA_TOKEN_RE.search(normalized) or FORMULA_COMMAND_TOKEN_RE.search(normalized):
        return True
    if any(ch in normalized for ch in MATH_OPERATOR_CHARS):
        return True
    if re.fullmatch(rf'[Α-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]+', normalized):
        return True
    if re.fullmatch(rf'[{MATH_VARIABLE_CHARS}]', normalized):
        return True

    if '�' in stripped or any(ch in stripped for ch in UNICODE_SUPERSCRIPT_CHARS + UNICODE_SUBSCRIPT_CHARS):
        return True
    if re.fullmatch(r'[A-Z]{2,}(?:-[A-Z]{2,})*', stripped):
        return False
    if re.fullmatch(r'[A-Za-z]{2,}', stripped):
        return False
    return looks_like_formula_text(normalized_latex)


def has_inline_math_context(working: str, start: int, end: int) -> bool:
    token = working[start:end].strip()
    if not token:
        return False
    if any(ch in token for ch in MATH_OPERATOR_CHARS):
        return True
    if re.fullmatch(rf'[Α-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]+', token):
        return True

    around = working[max(0, start - 18):min(len(working), end + 18)]
    if re.search(r'[=+\-*/<>_^{}]|\\cdot|' + rf'[{MATH_OPERATOR_CHARS_ESC}]', around):
        return True
    if re.search(r'[\u4e00-\u9fff]', around):
        return True
    return False


def find_standalone_math_symbol_match(working: str, start_pos: int):
    match = STANDALONE_MATH_SYMBOL_RE.search(working, start_pos)
    while match:
        if has_inline_math_context(working, match.start(), match.end()):
            return match
        match = STANDALONE_MATH_SYMBOL_RE.search(working, match.start() + 1)
    return None


def find_parenthetical_formula_match(working: str, start_pos: int):
    match = PAREN_FORMULA_CANDIDATE_RE.search(working, start_pos)
    while match:
        candidate = match.group(0).strip()
        inner = candidate[1:-1].strip()
        if should_auto_render_formula(inner):
            return match
        match = PAREN_FORMULA_CANDIDATE_RE.search(working, match.start() + 1)
    return None


def collect_formula_candidate_matches(working: str, cursor: int):
    candidate_matches = []
    for pattern in (
        INLINE_EQUATION_RUN_RE,
        SCRIPTED_FORMULA_TOKEN_RE,
        AUTO_FORMULA_RUN_RE,
        SPECIAL_FORMULA_RUN_RE,
    ):
        match = pattern.search(working, cursor)
        if match:
            candidate_matches.append(match)

    standalone_match = find_standalone_math_symbol_match(working, cursor)
    if standalone_match:
        candidate_matches.append(standalone_match)

    paren_match = find_parenthetical_formula_match(working, cursor)
    if paren_match:
        candidate_matches.append(paren_match)
    return candidate_matches

def wrap_plain_text_for_paragraph(
    text: str,
    asset_ctx: Optional[Dict[str, Any]] = None,
    bold: bool = False,
) -> str:
    if not text:
        return ''

    if asset_ctx is None:
        return wrap_plain_text_basic(text, bold=bold)

    working = collapse_spaced_math_braces(normalize_formula_spacing(text))
    working = working.replace('$', ' ')
    parts: List[str] = []
    cursor = 0

    while cursor < len(working):
        candidate_matches = collect_formula_candidate_matches(working, cursor)
        if not candidate_matches:
            parts.append(wrap_plain_text_basic(working[cursor:], bold=bold))
            break

        match = min(candidate_matches, key=lambda m: (m.start(), -(m.end() - m.start())))
        start, end = match.span()
        candidate = match.group(0).strip()

        if start > cursor:
            parts.append(wrap_plain_text_basic(working[cursor:start], bold=bold))

        if (
            not is_list_enumerator_text(candidate)
            and (
                should_auto_render_formula(candidate)
                or '�' in candidate
                or any(ch in candidate for ch in UNICODE_SUPERSCRIPT_CHARS + UNICODE_SUBSCRIPT_CHARS)
                or candidate[:1] in {'(', '（'}
            )
        ):
            parts.append(inline_math_markup(candidate, asset_ctx))
        else:
            parts.append(wrap_plain_text_basic(candidate, bold=bold))

        cursor = end

    return ''.join(parts)


def inline_math_markup(formula_text: str, asset_ctx: Dict[str, Any], font_size: float = 12.0) -> str:
    asset = render_formula_image(formula_text, asset_ctx, font_size=font_size, display=False)
    if not asset:
        fallback = extract_formula_text(formula_text)
        return wrap_plain_text_basic(fallback)

    return (
        f'<img src="{html.escape(asset["path"], quote=True)}" '
        f'width="{asset["width"]:.2f}" height="{asset["height"]:.2f}" valign="middle"/>'
    )

def mixed_inline_markup(text: str, asset_ctx: Optional[Dict[str, Any]] = None, bold: bool = False) -> str:
    """
    渲染 ReportLab Paragraph 可识别的行内富文本：
    - 普通文本保留中英文字体切换
    - **bold** / __bold__ 会转为真正的加粗样式
    - $...$ / \\(...\\) / `公式` 会优先渲染为公式图片
    - 非公式代码使用标准 Courier / Courier-Bold，避免依赖环境字体导致崩溃
    """
    value = text or ''
    if not value:
        return ''

    if asset_ctx is None:
        asset_ctx = create_formula_asset_context(tempfile.gettempdir())

    parts = []
    start = 0
    for match in INLINE_MARKUP_TOKEN_RE.finditer(value):
        if match.start() > start:
            parts.append(wrap_plain_text_for_paragraph(value[start:match.start()], asset_ctx=asset_ctx, bold=bold))

        token = match.group(0)
        if token.startswith('**') and token.endswith('**'):
            parts.append(mixed_inline_markup(token[2:-2], asset_ctx, bold=True))
        elif token.startswith('__') and token.endswith('__'):
            parts.append(mixed_inline_markup(token[2:-2], asset_ctx, bold=True))
        elif token.startswith('$') and token.endswith('$'):
            parts.append(inline_math_markup(token[1:-1], asset_ctx))
        elif token.startswith(r'\(') and token.endswith(r'\)'):
            parts.append(inline_math_markup(token[2:-2], asset_ctx))
        elif token.startswith(r'\[') and token.endswith(r'\]'):
            parts.append(inline_math_markup(token[2:-2], asset_ctx))
        elif token.startswith('`') and token.endswith('`'):
            inner = token[1:-1]
            if should_auto_render_formula(inner):
                parts.append(inline_math_markup(inner, asset_ctx))
            else:
                code_font = 'Courier-Bold' if bold else 'Courier'
                parts.append(f'<font name="{code_font}">{html.escape(inner)}</font>')
        start = match.end()

    if start < len(value):
        parts.append(wrap_plain_text_for_paragraph(value[start:], asset_ctx=asset_ctx, bold=bold))

    return ''.join(parts)

def formula_inline_markdown(formula_text: str, display: bool = False) -> str:
    expr = sanitize_formula_for_render(formula_text)
    if not expr:
        expr = extract_formula_text(formula_text)
    expr = (expr or '').strip()
    if not expr:
        return (formula_text or '').strip()

    if display or '\n' in expr:
        return f"$$\n{expr}\n$$"
    return f"${expr}$"


def wrap_plain_text_for_markdown(text: str) -> str:
    if not text:
        return ''

    working = collapse_spaced_math_braces(normalize_formula_spacing(text))
    working = working.replace('$', ' ')
    parts: List[str] = []
    cursor = 0

    while cursor < len(working):
        candidate_matches = collect_formula_candidate_matches(working, cursor)
        if not candidate_matches:
            parts.append(working[cursor:])
            break

        match = min(candidate_matches, key=lambda m: (m.start(), -(m.end() - m.start())))
        start, end = match.span()
        candidate = match.group(0).strip()

        if start > cursor:
            parts.append(working[cursor:start])

        if (
            not is_list_enumerator_text(candidate)
            and (
                should_auto_render_formula(candidate)
                or '�' in candidate
                or any(ch in candidate for ch in UNICODE_SUPERSCRIPT_CHARS + UNICODE_SUBSCRIPT_CHARS)
                or candidate[:1] in {'(', '（'}
            )
        ):
            parts.append(formula_inline_markdown(candidate, display=False))
        else:
            parts.append(candidate)

        cursor = end

    return ''.join(parts)


def convert_inline_formula_markup_to_markdown(text: str) -> str:
    value = text or ''
    if not value:
        return ''

    parts: List[str] = []
    start = 0
    for match in INLINE_MARKUP_TOKEN_RE.finditer(value):
        if match.start() > start:
            parts.append(wrap_plain_text_for_markdown(value[start:match.start()]))

        token = match.group(0)
        if token.startswith('**') and token.endswith('**'):
            parts.append(f"**{convert_inline_formula_markup_to_markdown(token[2:-2])}**")
        elif token.startswith('__') and token.endswith('__'):
            parts.append(f"__{convert_inline_formula_markup_to_markdown(token[2:-2])}__")
        elif token.startswith('$') and token.endswith('$'):
            parts.append(formula_inline_markdown(token[1:-1], display=False))
        elif token.startswith(r'\(') and token.endswith(r'\)'):
            parts.append(formula_inline_markdown(token[2:-2], display=False))
        elif token.startswith(r'\[') and token.endswith(r'\]'):
            parts.append(formula_inline_markdown(token[2:-2], display=True))
        elif token.startswith('`') and token.endswith('`'):
            inner = token[1:-1]
            if should_auto_render_formula(inner):
                parts.append(formula_inline_markdown(inner, display=False))
            else:
                parts.append(token)
        start = match.end()

    if start < len(value):
        parts.append(wrap_plain_text_for_markdown(value[start:]))

    return ''.join(parts)


def build_pdf_styles():
    """构造 PDF 所有样式，标题字号明显大于正文，并启用 keepWithNext。"""
    register_pdf_fonts()

    title_style = ParagraphStyle(
        "DocTitle",
        fontName="STSong-Light",
        fontSize=PDF_LAYOUT["title_font_size"],
        leading=30,
        alignment=TA_CENTER,
        spaceAfter=16,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "H2",
        fontName="STSong-Light",
        fontSize=PDF_LAYOUT["h2_font_size"],
        leading=25,
        alignment=TA_LEFT,
        spaceBefore=PDF_LAYOUT["section_space_before"],
        spaceAfter=PDF_LAYOUT["section_space_after"],
        keepWithNext=True,
    )

    h3_style = ParagraphStyle(
        "H3",
        fontName="STSong-Light",
        fontSize=PDF_LAYOUT["h3_font_size"],
        leading=21,
        alignment=TA_LEFT,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body",
        fontName="STSong-Light",
        fontSize=PDF_LAYOUT["body_font_size"],
        leading=PDF_LAYOUT["body_leading"],
        alignment=TA_JUSTIFY,
        firstLineIndent=2 * PDF_LAYOUT["body_font_size"],
        spaceAfter=PDF_LAYOUT["paragraph_space_after"],
    )

    list_item_style = ParagraphStyle(
        "ListItem",
        parent=body_style,
        firstLineIndent=0,
        leftIndent=12,
        spaceAfter=4,
    )

    caption_style = ParagraphStyle(
        "Caption",
        fontName="STSong-Light",
        fontSize=PDF_LAYOUT["caption_font_size"],
        leading=14,
        alignment=TA_CENTER,
        spaceBefore=2,
        spaceAfter=8,
    )

    table_title_style = ParagraphStyle(
        "TableTitle",
        fontName="STSong-Light",
        fontSize=11.0,
        leading=16,
        alignment=TA_CENTER,
        spaceBefore=6,
        spaceAfter=6,
        keepWithNext=True,
    )

    return {
        "title": title_style,
        "h2": h2_style,
        "h3": h3_style,
        "body": body_style,
        "list_item": list_item_style,
        "caption": caption_style,
        "table_title": table_title_style,
    }


# ==========================================
# 模块 10：Markdown 自定义块解析
# ==========================================
def split_markdown_blocks(md_text: str) -> List[Tuple[str, object]]:
    """
    自定义 Markdown 块解析器。

    只解析当前系统真正会生成的几类结构：
    - 标题：## / ###
    - 图片：![caption](id)
    - 表格标题：表1：... / Table 1: ...
    - 公式块：$$...$$ / \\[...\\] / ```latex``` / 常见公式环境
    - Markdown 表格：| a | b |
    - 列表项：* xxx / 1) xxx
    - 普通段落：空行分隔
    """
    text = normalize_report_markdown(md_text)
    lines = text.split("\n")
    blocks: List[Tuple[str, object]] = []

    paragraph_buffer: List[str] = []

    def flush_paragraph():
        if paragraph_buffer:
            joined = " ".join([x.strip() for x in paragraph_buffer if x.strip()]).strip()
            if joined:
                fragments = [frag.strip() for frag in INLINE_BULLET_SPLIT_RE.split(joined) if frag and frag.strip()]
                if len(fragments) > 1:
                    head = fragments[0]
                    if head:
                        blocks.append(("paragraph", head))
                    for frag in fragments[1:]:
                        cleaned = frag.lstrip("*•- ").strip()
                        append_list_item_blocks(blocks, cleaned, prefix='- ')
                else:
                    blocks.append(("paragraph", joined))
            paragraph_buffer.clear()

    def is_structural_block(line: str) -> bool:
        if not line:
            return True
        if line.startswith("## ") or line.startswith("### "):
            return True
        if line.startswith("```"):
            return True
        if re.match(r'^\\begin\{[^{}]+\}$', line):
            return True
        if line.startswith("$$") or line.startswith(r"\["):
            return True
        if re.fullmatch(r'!\[(.*?)\]\((.*?)\)', line):
            return True
        if is_table_title_line(line):
            return True
        if line.startswith("|"):
            return True
        if BULLET_LINE_RE.match(line) or ORDERED_LIST_LINE_RE.match(line):
            return True
        return False

    def consume_list_item(start_index: int, initial_text: str) -> Tuple[str, int]:
        item_lines = [initial_text.strip()]
        j = start_index + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if not nxt or is_structural_block(nxt):
                break
            item_lines.append(nxt)
            j += 1
        return " ".join([x for x in item_lines if x]).strip(), j

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if stripped.startswith("## "):
            flush_paragraph()
            blocks.append(("h2", stripped[3:].strip()))
            i += 1
            continue

        if stripped.startswith("### "):
            flush_paragraph()
            blocks.append(("h3", stripped[4:].strip()))
            i += 1
            continue

        bullet_match = BULLET_LINE_RE.match(stripped)
        if bullet_match:
            flush_paragraph()
            item_text, next_i = consume_list_item(i, bullet_match.group(1))
            append_list_item_blocks(blocks, item_text, prefix='- ')
            i = next_i
            continue

        ordered_match = ORDERED_LIST_LINE_RE.match(stripped)
        if ordered_match:
            flush_paragraph()
            marker = ordered_match.group(1).strip()
            item_text, next_i = consume_list_item(i, ordered_match.group(2))
            pieces = explode_inline_numbered_segments(item_text)
            if pieces:
                first = f"{marker} {pieces[0]}".strip()
                blocks.append(("list_item", first))
                for piece in pieces[1:]:
                    piece = piece.strip()
                    if not piece:
                        continue
                    if re.match(r'^\(?\d+[\).、]', piece):
                        blocks.append(("list_item", piece))
                    else:
                        blocks.append(("list_item", f"- {piece}"))
            i = next_i
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            fence_lang = stripped[3:].strip().lower()
            fenced_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                fenced_lines.append(lines[i].rstrip())
                i += 1
            if i < len(lines) and lines[i].strip().startswith("```"):
                i += 1

            block_text = "\n".join([x for x in fenced_lines if x.strip()]).strip()
            if block_text:
                if fence_lang in {"math", "latex", "tex"} or looks_like_formula_text(block_text):
                    blocks.append(("math_block", block_text))
                else:
                    blocks.append(("paragraph", " ".join([x.strip() for x in fenced_lines if x.strip()])))
            continue

        if re.match(r'^\\begin\{[^{}]+\}$', stripped):
            flush_paragraph()
            env_lines = [stripped]
            i += 1
            while i < len(lines):
                env_lines.append(lines[i].strip())
                if re.match(r'^\\end\{[^{}]+\}$', lines[i].strip()):
                    i += 1
                    break
                i += 1
            blocks.append(("math_block", "\n".join(env_lines)))
            continue

        if stripped.startswith("$$") or stripped.startswith(r"\["):
            flush_paragraph()
            opener, closer = ("$$", "$$") if stripped.startswith("$$") else (r"\[", r"\]")
            formula_lines = []
            current = stripped

            if current.startswith(opener):
                current = current[len(opener):]

            if current.endswith(closer) and current.strip() != closer:
                current = current[:-len(closer)]
                if current.strip():
                    formula_lines.append(current.strip())
                i += 1
            else:
                if current.strip():
                    formula_lines.append(current.strip())
                i += 1
                while i < len(lines):
                    current = lines[i].strip()
                    if current.endswith(closer):
                        tail = current[:-len(closer)].strip()
                        if tail:
                            formula_lines.append(tail)
                        i += 1
                        break
                    formula_lines.append(current)
                    i += 1

            formula_text = "\n".join([x for x in formula_lines if x])
            if formula_text:
                blocks.append(("math_block", formula_text))
            continue

        img_match = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', stripped)
        if img_match:
            flush_paragraph()
            blocks.append(("image", (img_match.group(1).strip(), img_match.group(2).strip())))
            i += 1
            continue

        if is_table_title_line(stripped):
            flush_paragraph()
            blocks.append(("table_title", normalize_table_title_line(stripped)))
            i += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            table_lines = [stripped]
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("|"):
                table_lines.append(lines[j].strip())
                j += 1
            blocks.append(("md_table", table_lines))
            i = j
            continue

        paragraph_buffer.append(stripped)
        i += 1

    flush_paragraph()
    return blocks

def parse_md_table(table_lines: List[str]) -> List[List[str]]:
    """解析最常见的 Markdown 表格格式。"""
    rows = []
    for idx, line in enumerate(table_lines):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if idx == 1 and all(re.fullmatch(r'[:\- ]+', c) for c in cells):
            continue
        rows.append(cells)

    if not rows:
        return []

    width = max(len(r) for r in rows)
    return [r + [""] * (width - len(r)) for r in rows]


def split_title_and_body(md_text: str) -> Tuple[str, str]:
    """从完整 markdown 中拆出文档主标题和正文。"""
    text = normalize_report_markdown(md_text)
    lines = text.split("\n")
    if lines and lines[0].startswith("# "):
        return lines[0][2:].strip(), "\n".join(lines[1:]).strip()
    return "论文全维度深度透视报告", text


CURRENT_REPORT_SECTION_TITLES = [
    "研究问题与核心贡献",
    "背景、研究缺口与前人路线",
    "方法总览与整体数据流",
    "关键模块逐层机制剖析",
    "实验设计、关键证据与论点验证",
    "局限性与未解决问题",
    "面向后续研究的可执行创新路线",
]
REMOVED_REPORT_SECTION_TITLES = {"复现要点与方法适用边界"}


def canonicalize_report_section_core(title: str) -> str:
    raw = (title or "").strip()
    raw = re.sub(r'^#+\s*', '', raw)
    raw = re.sub(r'^\d+\s*[.．、:：\-]?\s*', '', raw)
    normalized = normalize_compare_text(raw)

    for candidate in CURRENT_REPORT_SECTION_TITLES + list(REMOVED_REPORT_SECTION_TITLES):
        if normalize_compare_text(candidate) == normalized:
            return candidate
    return raw


def normalize_report_sections_to_current_schema(md_text: str) -> str:
    doc_title, body = split_title_and_body(md_text)
    lines = normalize_report_markdown(body).split("\n") if body else []

    prelude_lines: List[str] = []
    section_blocks: List[Tuple[str, List[str]]] = []
    current_header: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if current_header is not None:
                section_blocks.append((current_header, current_lines))
            current_header = stripped[3:].strip()
            current_lines = []
        else:
            if current_header is None:
                prelude_lines.append(line)
            else:
                current_lines.append(line)

    if current_header is not None:
        section_blocks.append((current_header, current_lines))

    merged_sections: Dict[str, List[str]] = {}
    for header, content_lines in section_blocks:
        core_title = canonicalize_report_section_core(header)
        if core_title in REMOVED_REPORT_SECTION_TITLES:
            continue
        if core_title not in CURRENT_REPORT_SECTION_TITLES:
            continue

        cleaned_content = list(content_lines)
        if core_title not in merged_sections:
            merged_sections[core_title] = cleaned_content
        else:
            existing_content = normalize_report_markdown("\n".join(merged_sections[core_title]))
            incoming_content = normalize_report_markdown("\n".join(cleaned_content))
            if incoming_content and incoming_content != existing_content:
                if merged_sections[core_title] and merged_sections[core_title][-1].strip():
                    merged_sections[core_title].append("")
                merged_sections[core_title].extend(cleaned_content)

    rebuilt_lines: List[str] = [f"# {doc_title}"]
    prelude_text = normalize_report_markdown("\n".join(prelude_lines))
    if prelude_text:
        rebuilt_lines.extend(["", prelude_text])

    for idx, section_title in enumerate(CURRENT_REPORT_SECTION_TITLES, start=1):
        if section_title not in merged_sections:
            continue
        rebuilt_lines.extend(["", f"## {idx}. {section_title}"])
        rebuilt_lines.extend(merged_sections[section_title])

    return normalize_report_markdown("\n".join(rebuilt_lines))


def prepare_report_markdown_for_display(
    md_text: str,
    images_dict: Optional[Dict[str, str]] = None,
    vision_summaries: str = '',
) -> str:
    normalized_sections = normalize_report_sections_to_current_schema(md_text)
    image_ids = list((images_dict or {}).keys())
    return postprocess_generated_report_markdown(normalized_sections, image_ids=image_ids, vision_summaries=vision_summaries)


def convert_inline_formulas_in_table_line(line: str) -> str:
    stripped = (line or '').strip()
    if not stripped.startswith('|'):
        return line

    cells = [cell.strip() for cell in stripped.strip('|').split('|')]
    if cells and all(re.fullmatch(r'[:\- ]+', cell) for cell in cells):
        return stripped

    rendered_cells = [convert_inline_formula_markup_to_markdown(cell) for cell in cells]
    return '| ' + ' | '.join(rendered_cells) + ' |'


def serialize_report_blocks(blocks: List[Tuple[str, object]], doc_title: str) -> str:
    lines: List[str] = [f"# {doc_title}"]

    for block_type, payload in blocks:
        if block_type == 'h2':
            lines.extend(['', f"## {convert_inline_formula_markup_to_markdown(str(payload))}"])
        elif block_type == 'h3':
            lines.extend(['', f"### {convert_inline_formula_markup_to_markdown(str(payload))}"])
        elif block_type == 'table_title':
            lines.extend(['', convert_inline_formula_markup_to_markdown(normalize_table_title_line(str(payload)))])
        elif block_type == 'paragraph':
            lines.extend(['', convert_inline_formula_markup_to_markdown(str(payload))])
        elif block_type == 'list_item':
            lines.extend(['', convert_inline_formula_markup_to_markdown(str(payload))])
        elif block_type == 'math_block':
            lines.extend(['', formula_inline_markdown(str(payload), display=True)])
        elif block_type == 'image':
            caption, key = payload
            rendered_caption = convert_inline_formula_markup_to_markdown(str(caption)) if caption else ''
            lines.extend(['', f"![{rendered_caption}]({key})"])
        elif block_type == 'md_table':
            lines.append('')
            lines.extend([convert_inline_formulas_in_table_line(str(row)) for row in payload])
        else:
            lines.extend(['', str(payload)])

    return normalize_report_markdown('\n'.join(lines))


def normalize_report_image_key(image_key: str) -> str:
    value = (image_key or '').strip().replace('\\', '/')
    value = value.split('#', 1)[0].split('?', 1)[0]
    value = value.split('/')[-1]
    return re.sub(r'\s+', '', value).lower()


def deduplicate_report_image_blocks(blocks: List[Tuple[str, object]]) -> List[Tuple[str, object]]:
    deduped: List[Tuple[str, object]] = []
    seen_image_keys = set()

    for block_type, payload in blocks:
        if block_type != 'image':
            deduped.append((block_type, payload))
            continue

        caption, key = payload
        dedup_key = normalize_report_image_key(str(key))
        if dedup_key and dedup_key in seen_image_keys:
            if deduped and deduped[-1][0] == 'table_title':
                last_title = str(deduped[-1][1])
                caption_text = (caption or '').strip()
                if not caption_text or captions_equivalent(caption_text, last_title):
                    deduped.pop()
            continue

        if dedup_key:
            seen_image_keys.add(dedup_key)
        deduped.append(('image', ((caption or '').strip(), key)))

    return deduped



REPORT_IMAGE_LINE_RE = re.compile(r'^!\[(?P<caption>.*?)\]\((?P<key>.*?)\)\s*$')
REPORT_FIG_TABLE_LABEL_RE = re.compile(r'(图|表)\s*([0-9一二三四五六七八九十百千万]+)')
REPORT_EN_LABEL_RE = re.compile(r'\b(Figure|Fig\.?|Table)\s*([0-9]+)\b', flags=re.I)
REPORT_IMAGE_ID_LIKE_RE = re.compile(
    r'(?:image[_\-]?[0-9a-f]{6,}|fig(?:ure)?[_\-]?\d+|table[_\-]?\d+|[0-9a-f]{16,}|[\w\-.]+\.(?:png|jpg|jpeg|webp))',
    flags=re.I,
)


def normalize_report_label(kind: str, number: str) -> str:
    return f"{(kind or '图').strip()}{re.sub(r'\\s+', '', str(number or '').strip())}"


def extract_report_label(text: str) -> Optional[Tuple[str, str]]:
    value = text or ''
    match = REPORT_FIG_TABLE_LABEL_RE.search(value)
    if match:
        return match.group(1), re.sub(r'\s+', '', match.group(2))
    match = REPORT_EN_LABEL_RE.search(value)
    if match:
        kind = '表' if match.group(1).lower().startswith('table') else '图'
        return kind, match.group(2)
    return None


def is_table_like_figure_text(text: str) -> bool:
    value = (text or '').strip()
    lower = value.lower()
    table_keywords = [
        'table', 'tabular', '表格', '数据表', '结果表', '对比表', '消融表', '指标表', '性能表', '统计表'
    ]
    figure_keywords = ['figure', 'fig.', '架构图', '流程图', '曲线图', '示意图', '模块图', '框图', '散点图', '折线图']
    if any(keyword in lower for keyword in ['table', 'tabular']) or any(keyword in value for keyword in table_keywords[2:]):
        return True
    if any(keyword in lower for keyword in ['figure', 'fig.']) or any(keyword in value for keyword in figure_keywords[2:]):
        return False
    label = extract_report_label(value)
    return bool(label and label[0] == '表')


def parse_vision_figure_metadata(vision_summaries: str) -> Dict[str, Dict[str, str]]:
    metadata: Dict[str, Dict[str, str]] = {}
    text = vision_summaries or ''
    if not text.strip():
        return metadata

    chunks = re.split(r'\n\s*---\s*图表标识\s*:\s*(.*?)\s*---\s*\n', text)
    pairs: List[Tuple[str, str]] = []
    if len(chunks) >= 3:
        for idx in range(1, len(chunks), 2):
            pairs.append((chunks[idx].strip(), chunks[idx + 1] if idx + 1 < len(chunks) else ''))
    else:
        card_chunks = re.split(r'(?=<FIGURE_CARD>)', text)
        pairs = [('', chunk) for chunk in card_chunks if chunk.strip()]

    for fallback_key, chunk in pairs:
        key = fallback_key
        id_match = re.search(r'图像ID\s*[：:]\s*(.+)', chunk)
        if id_match:
            candidate = id_match.group(1).strip().splitlines()[0].strip()
            if candidate and candidate not in {'未显示', '无'}:
                key = candidate
        if not key:
            continue

        type_match = re.search(r'图表类型\s*[：:]\s*(.+)', chunk)
        type_text = type_match.group(1).strip().splitlines()[0].strip() if type_match else ''
        original_match = re.search(r'原文编号\s*[：:]\s*(.+)', chunk)
        original_text = original_match.group(1).strip().splitlines()[0].strip() if original_match else ''
        caption_match = re.search(r'推荐图注\s*[：:]\s*(.+)', chunk)
        caption = caption_match.group(1).strip().splitlines()[0].strip() if caption_match else ''

        label = extract_report_label(original_text) or extract_report_label(caption)
        kind = '表' if is_table_like_figure_text(type_text) or is_table_like_figure_text(original_text) or (label and label[0] == '表') else '图'
        number = label[1] if label else ''

        metadata[normalize_report_image_key(key)] = {
            'key': key,
            'kind': kind,
            'number': number,
            'caption': caption,
            'type_text': type_text,
            'original_text': original_text,
        }
    return metadata


def strip_label_prefix(text: str) -> str:
    value = (text or '').strip()
    value = re.sub(r'^(?:图|表)\s*[0-9一二三四五六七八九十百千万]+\s*[：:：.．\-—–]?\s*', '', value)
    value = re.sub(r'^(?:Figure|Fig\.?|Table)\s*[0-9]+\s*[：:：.．\-—–]?\s*', '', value, flags=re.I)
    return value.strip()


def strip_internal_asset_references(text: str, known_image_keys: Optional[List[str]] = None) -> str:
    value = text or ''
    normalized_known = [normalize_report_image_key(x) for x in (known_image_keys or []) if x]

    def _paren_repl(match: re.Match) -> str:
        label = re.sub(r'\s+', '', match.group(1))
        inner = match.group(2).strip()
        norm_inner = normalize_report_image_key(inner)
        if REPORT_IMAGE_ID_LIKE_RE.search(inner) or any(k and (k in norm_inner or norm_inner in k) for k in normalized_known):
            return label
        return match.group(0)

    value = re.sub(
        r'((?:图|表)\s*[0-9一二三四五六七八九十百千万]+)\s*[（(]\s*([^）)]{3,220})\s*[）)]',
        _paren_repl,
        value,
    )
    value = re.sub(
        r'\b(?:Figure|Fig\.?|Table)\s*([0-9]+)\s*[（(]\s*([^）)]{3,220})\s*[）)]',
        lambda m: f"{'表' if m.group(0).lower().startswith('table') else '图'}{m.group(1)}"
        if REPORT_IMAGE_ID_LIKE_RE.search(m.group(2)) else m.group(0),
        value,
        flags=re.I,
    )
    return value


def replace_report_label_aliases(text: str, aliases: Dict[str, str]) -> str:
    value = text or ''
    for old_label, new_label in sorted((aliases or {}).items(), key=lambda item: -len(item[0])):
        if not old_label or not new_label or old_label == new_label:
            continue
        kind, number = old_label[0], re.escape(old_label[1:])
        pattern = re.compile(rf'{kind}\s*{number}(?![0-9一二三四五六七八九十百千万])')
        value = pattern.sub(new_label, value)
    return value


def collect_report_labels_from_text(text: str) -> List[str]:
    labels: List[str] = []
    for kind, number in REPORT_FIG_TABLE_LABEL_RE.findall(text or ''):
        labels.append(normalize_report_label(kind, number))
    for kind_text, number in REPORT_EN_LABEL_RE.findall(text or ''):
        kind = '表' if kind_text.lower().startswith('table') else '图'
        labels.append(normalize_report_label(kind, number))
    return labels


def is_standalone_figure_table_caption_line(line: str) -> bool:
    stripped = (line or '').strip()
    if not stripped:
        return False
    if REPORT_IMAGE_LINE_RE.match(stripped):
        return True
    if re.match(r'^(?:图|表)\s*[0-9一二三四五六七八九十百千万]+\s*[：:：.．\-—–]', stripped):
        return True
    if re.match(r'^(?:Figure|Fig\.?|Table)\s*[0-9]+\s*[：:：.．\-—–]', stripped, flags=re.I):
        return True
    return False


def build_report_asset_maps(
    report_md: str,
    image_ids: Optional[List[str]] = None,
    vision_summaries: str = '',
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    available_keys = list(image_ids or [])
    vision_meta = parse_vision_figure_metadata(vision_summaries)
    key_meta: Dict[str, Dict[str, str]] = {}

    for key in available_keys:
        norm_key = normalize_report_image_key(key)
        key_meta[norm_key] = {'key': key, 'kind': '图', 'number': '', 'caption': '', 'type_text': '', 'original_text': ''}
        if norm_key in vision_meta:
            key_meta[norm_key].update(vision_meta[norm_key])
            key_meta[norm_key]['key'] = key

    for line in normalize_report_markdown(report_md).splitlines():
        match = REPORT_IMAGE_LINE_RE.match(line.strip())
        if not match:
            continue
        key = match.group('key').strip()
        norm_key = normalize_report_image_key(key)
        if norm_key not in key_meta:
            key_meta[norm_key] = {'key': key, 'kind': '图', 'number': '', 'caption': '', 'type_text': '', 'original_text': ''}
        caption = strip_internal_asset_references(match.group('caption').strip(), available_keys)
        label = extract_report_label(caption)
        if label:
            if key_meta[norm_key].get('kind') != '表':
                key_meta[norm_key]['kind'] = label[0]
            key_meta[norm_key]['number'] = key_meta[norm_key].get('number') or label[1]
        if caption and not key_meta[norm_key].get('caption'):
            key_meta[norm_key]['caption'] = caption
        if is_table_like_figure_text(caption):
            key_meta[norm_key]['kind'] = '表'

    counters = {'图': 1, '表': 1}
    used_numbers = {'图': set(), '表': set()}
    for meta in key_meta.values():
        kind = meta.get('kind') or '图'
        number = str(meta.get('number') or '').strip()
        if number:
            used_numbers.setdefault(kind, set()).add(number)
            if number.isdigit():
                counters[kind] = max(counters.get(kind, 1), int(number) + 1)

    key_to_label: Dict[str, str] = {}
    label_to_key: Dict[str, str] = {}
    label_to_caption: Dict[str, str] = {}
    aliases: Dict[str, str] = {}

    for norm_key, meta in key_meta.items():
        kind = meta.get('kind') or '图'
        number = str(meta.get('number') or '').strip()
        if not number:
            while str(counters[kind]) in used_numbers.setdefault(kind, set()):
                counters[kind] += 1
            number = str(counters[kind])
            used_numbers[kind].add(number)
            counters[kind] += 1
        label = normalize_report_label(kind, number)
        key_to_label[norm_key] = label
        label_to_key.setdefault(label, meta.get('key') or norm_key)
        caption_core = strip_label_prefix(strip_internal_asset_references(meta.get('caption') or '', available_keys))
        label_to_caption[label] = caption_core

    for line in normalize_report_markdown(report_md).splitlines():
        match = REPORT_IMAGE_LINE_RE.match(line.strip())
        if not match:
            continue
        norm_key = normalize_report_image_key(match.group('key').strip())
        current_label = extract_report_label(match.group('caption').strip())
        new_label = key_to_label.get(norm_key)
        if current_label and new_label:
            old_label = normalize_report_label(*current_label)
            if old_label != new_label:
                aliases[old_label] = new_label

    return key_to_label, label_to_key, label_to_caption, aliases


def build_report_image_markdown(label: str, key: str, label_to_caption: Dict[str, str]) -> str:
    caption_core = strip_label_prefix(label_to_caption.get(label, '')).strip()
    caption = f"{label}：{caption_core}" if caption_core else label
    return f"![{caption}]({key})"


def reconcile_report_figure_table_references(
    report_md: str,
    image_ids: Optional[List[str]] = None,
    vision_summaries: str = '',
) -> str:
    text = normalize_report_markdown(report_md)
    if not text:
        return text

    available_keys = list(image_ids or [])
    key_to_label, label_to_key, label_to_caption, aliases = build_report_asset_maps(text, available_keys, vision_summaries)
    known_keys = available_keys + list(label_to_key.values())

    cleaned_lines: List[str] = []
    cited_labels = set()
    for line in text.splitlines():
        stripped = line.strip()
        img_match = REPORT_IMAGE_LINE_RE.match(stripped)
        if img_match:
            cleaned_lines.append(line)
            continue

        cleaned = strip_internal_asset_references(line, known_keys)
        cleaned = replace_report_label_aliases(cleaned, aliases)
        cleaned_lines.append(cleaned)

        if not is_standalone_figure_table_caption_line(cleaned):
            cited_labels.update(collect_report_labels_from_text(cleaned))

    inserted_labels = set()
    used_image_keys = set()
    filtered_lines: List[str] = []
    for line in cleaned_lines:
        stripped = line.strip()
        img_match = REPORT_IMAGE_LINE_RE.match(stripped)
        if not img_match:
            filtered_lines.append(line)
            continue

        caption = strip_internal_asset_references(img_match.group('caption').strip(), known_keys)
        key = img_match.group('key').strip()
        norm_key = normalize_report_image_key(key)
        label = key_to_label.get(norm_key)
        if not label:
            caption_label = extract_report_label(caption)
            label = normalize_report_label(*caption_label) if caption_label else ''

        if label and label in cited_labels and norm_key not in used_image_keys:
            canonical_key = label_to_key.get(label, key)
            filtered_lines.append(build_report_image_markdown(label, canonical_key, label_to_caption))
            inserted_labels.add(label)
            used_image_keys.add(normalize_report_image_key(canonical_key))
        else:
            continue

    missing_labels = [label for label in sorted(cited_labels, key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 10**9, x)) if label in label_to_key and label not in inserted_labels]
    for label in missing_labels:
        image_line = build_report_image_markdown(label, label_to_key[label], label_to_caption)
        inserted = False
        for idx, line in enumerate(filtered_lines):
            if REPORT_IMAGE_LINE_RE.match(line.strip()) or is_standalone_figure_table_caption_line(line):
                continue
            if label in collect_report_labels_from_text(line):
                filtered_lines.insert(idx + 1, '')
                filtered_lines.insert(idx + 2, image_line)
                inserted = True
                break
        if not inserted:
            filtered_lines.append('')
            filtered_lines.append(image_line)

    return normalize_report_markdown('\n'.join(filtered_lines))

def postprocess_generated_report_markdown(
    md_text: str,
    image_ids: Optional[List[str]] = None,
    vision_summaries: str = '',
) -> str:
    doc_title, body = split_title_and_body(md_text)
    blocks = split_markdown_blocks(body)
    cleaned_blocks: List[Tuple[str, object]] = []

    for idx, (block_type, payload) in enumerate(blocks):
        prev = cleaned_blocks[-1] if cleaned_blocks else None
        next_block = blocks[idx + 1] if idx + 1 < len(blocks) else None

        if block_type == 'table_title':
            current_title = normalize_table_title_line(str(payload))
            if prev and prev[0] == 'table_title' and captions_equivalent(current_title, str(prev[1])):
                continue
            cleaned_blocks.append(('table_title', current_title))
            continue

        if block_type == 'paragraph':
            paragraph_text = str(payload).strip()
            if prev and prev[0] == 'table_title' and captions_equivalent(paragraph_text, str(prev[1])):
                continue
            if is_table_title_line(paragraph_text) and next_block and next_block[0] in {'image', 'md_table'}:
                normalized_title = normalize_table_title_line(paragraph_text)
                if not (prev and prev[0] == 'table_title' and captions_equivalent(normalized_title, str(prev[1]))):
                    cleaned_blocks.append(('table_title', normalized_title))
                continue
            cleaned_blocks.append((block_type, paragraph_text))
            continue

        if block_type == 'image':
            caption, key = payload
            caption_text = (caption or '').strip()
            if prev and prev[0] == 'table_title' and caption_text and captions_equivalent(caption_text, str(prev[1])):
                cleaned_blocks.append(('image', ('', key)))
                continue
            cleaned_blocks.append(('image', (caption_text, key)))
            continue

        cleaned_blocks.append((block_type, payload))

    cleaned_blocks = deduplicate_report_image_blocks(cleaned_blocks)
    serialized = serialize_report_blocks(cleaned_blocks, doc_title)
    return reconcile_report_figure_table_references(serialized, image_ids=image_ids, vision_summaries=vision_summaries)
def classify_image_size(img_reader: ImageReader) -> Tuple[float, float]:
    """根据图片宽高比选择更合适的显示尺寸。"""
    iw, ih = img_reader.getSize()
    page_width = A4[0] - PDF_LAYOUT["margin_left"] - PDF_LAYOUT["margin_right"]
    ratio = iw / max(ih, 1)

    if ratio > 1.8:
        max_w = page_width * PDF_LAYOUT["wide_image_max_width_ratio"]
        max_h = PDF_LAYOUT["table_image_max_height"]
    elif ratio < 0.72:
        max_w = page_width * PDF_LAYOUT["tall_image_max_width_ratio"]
        max_h = PDF_LAYOUT["image_max_height"]
    else:
        max_w = page_width * PDF_LAYOUT["image_max_width_ratio"]
        max_h = PDF_LAYOUT["image_max_height"]

    scale = min(max_w / iw, max_h / ih, 1.0)
    return iw * scale, ih * scale


def image_flowable(
    caption: str,
    key: str,
    images_dict: Dict[str, str],
    styles,
    asset_ctx: Dict[str, Any],
    suppress_caption: bool = False,
) -> Optional[KeepTogether]:
    """构造图片块（图片 + 图注）。"""
    matched_b64 = None
    for img_name, b64 in images_dict.items():
        if key == img_name or key in img_name or img_name in key:
            matched_b64 = b64
            break

    display_caption = (caption or key).strip()
    if not matched_b64:
        if suppress_caption:
            return KeepTogether([Spacer(1, 4)])
        return KeepTogether([
            Paragraph(mixed_inline_markup(f"[未匹配到图片] {display_caption}", asset_ctx), styles["caption"]),
            Spacer(1, 4),
        ])

    try:
        img_bytes = BytesIO(base64.b64decode(matched_b64))
        reader = ImageReader(img_bytes)
        w, h = classify_image_size(reader)
        rl_img = RLImage(img_bytes, width=w, height=h)

        items = [Spacer(1, 4), rl_img]
        if display_caption and not suppress_caption:
            items.append(Paragraph(mixed_inline_markup(display_caption, asset_ctx), styles["caption"]))
        items.append(Spacer(1, 4))
        return KeepTogether(items)
    except Exception:
        if suppress_caption:
            return KeepTogether([Spacer(1, 4)])
        return KeepTogether([
            Paragraph(mixed_inline_markup(f"[图片加载失败] {display_caption}", asset_ctx), styles["caption"]),
            Spacer(1, 4),
        ])


def markdown_table_flowable(table_lines: List[str], styles, asset_ctx: Dict[str, Any]):
    """构造 Markdown 表格的 Flowable。"""
    data = parse_md_table(table_lines)
    if not data:
        return None

    usable_width = A4[0] - PDF_LAYOUT["margin_left"] - PDF_LAYOUT["margin_right"]
    cols = max(len(r) for r in data)
    col_width = usable_width / max(cols, 1)

    wrapped = []
    for row in data:
        wrapped.append([Paragraph(mixed_inline_markup(cell, asset_ctx), styles["body"]) for cell in row])

    tbl = Table(wrapped, colWidths=[col_width] * cols, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


def captions_equivalent(left: str, right: str) -> bool:
    norm_left = normalize_caption_core_text(left) or normalize_compare_text(left)
    norm_right = normalize_caption_core_text(right) or normalize_compare_text(right)
    if not norm_left or not norm_right:
        return False
    return norm_left == norm_right or norm_left in norm_right or norm_right in norm_left


def math_block_flowable(formula_text: str, asset_ctx: Dict[str, Any]) -> Optional[KeepTogether]:
    """将块级公式渲染为独立图片，避免在 PDF 中显示为代码或乱码。"""
    asset = render_formula_image(formula_text, asset_ctx, font_size=14.5, display=True)
    if not asset:
        fallback_text = extract_formula_text(formula_text)
        if not fallback_text:
            return None
        fallback_style = ParagraphStyle(
            'MathFallback',
            fontName='STSong-Light',
            fontSize=11.5,
            leading=16,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=8,
        )
        return KeepTogether([
            Paragraph(html.escape(fallback_text), fallback_style),
            Spacer(1, 6),
        ])

    max_width = A4[0] - PDF_LAYOUT["margin_left"] - PDF_LAYOUT["margin_right"]
    width = asset["width"]
    height = asset["height"]
    if width > max_width * 0.92:
        scale = (max_width * 0.92) / width
        width *= scale
        height *= scale

    eq_img = RLImage(asset["path"], width=width, height=height)
    eq_img.hAlign = 'CENTER'
    return KeepTogether([
        Spacer(1, 4),
        eq_img,
        Spacer(1, 8),
    ])

def build_story_from_markdown(md_text: str, images_dict: Dict[str, str], asset_ctx: Dict[str, Any]) -> List:
    """
    将最终 markdown 报告转换为 ReportLab story。

    关键处理：
    - 标题使用专门样式，不与正文同字号。
    - 标题 keepWithNext，尽量避免标题掉到页末单独一行。
    - 自定义块解析器确保段落按空行正确分段。
    - 行内 bold / 公式 / 代码会在 PDF 中被正确渲染。
    - 表题与图表 caption 做去重，避免重复显示。
    """
    styles = build_pdf_styles()
    doc_title, body = split_title_and_body(md_text)
    blocks = split_markdown_blocks(body)

    story = [
        Paragraph(mixed_inline_markup(doc_title, asset_ctx), styles["title"]),
        Spacer(1, 8),
    ]

    pending_table_title: Optional[str] = None

    def emit_pending_table_title():
        nonlocal pending_table_title
        if pending_table_title:
            story.append(Paragraph(mixed_inline_markup(pending_table_title, asset_ctx), styles["table_title"]))
            pending_table_title = None

    for block_type, payload in blocks:
        if block_type == "h2":
            emit_pending_table_title()
            story.append(Paragraph(mixed_inline_markup(str(payload), asset_ctx), styles["h2"]))

        elif block_type == "h3":
            emit_pending_table_title()
            story.append(Paragraph(mixed_inline_markup(str(payload), asset_ctx), styles["h3"]))

        elif block_type == "table_title":
            pending_table_title = str(payload)

        elif block_type == "paragraph":
            paragraph_text = str(payload)
            if pending_table_title and captions_equivalent(paragraph_text, pending_table_title):
                continue
            story.append(Paragraph(mixed_inline_markup(paragraph_text, asset_ctx), styles["body"]))

        elif block_type == "list_item":
            emit_pending_table_title()
            story.append(Paragraph(mixed_inline_markup(str(payload), asset_ctx), styles["list_item"]))

        elif block_type == "math_block":
            emit_pending_table_title()
            formula_block = math_block_flowable(str(payload), asset_ctx)
            if formula_block is not None:
                story.append(formula_block)

        elif block_type == "image":
            caption, key = payload
            suppress_caption = False
            if pending_table_title:
                story.append(Paragraph(mixed_inline_markup(pending_table_title, asset_ctx), styles["table_title"]))
                suppress_caption = True
                pending_table_title = None
            img_block = image_flowable(
                caption=caption,
                key=key,
                images_dict=images_dict,
                styles=styles,
                asset_ctx=asset_ctx,
                suppress_caption=suppress_caption,
            )
            if img_block:
                story.append(img_block)

        elif block_type == "md_table":
            if pending_table_title:
                story.append(Paragraph(mixed_inline_markup(pending_table_title, asset_ctx), styles["table_title"]))
                pending_table_title = None
            tbl = markdown_table_flowable(payload, styles, asset_ctx)
            if tbl is not None:
                story.append(KeepTogether([tbl, Spacer(1, 6)]))

    emit_pending_table_title()
    return story


# ==========================================
# 模块 12：PDF 文档模板
# ==========================================
class StableDocTemplate(BaseDocTemplate):
    """
    自定义 DocTemplate：
    使用单个 Frame 实现稳定分页，并加页码。
    """
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
            showBoundary=0,
        )
        template = PageTemplate(id="main", frames=[frame], onPage=self._draw_page_number)
        self.addPageTemplates([template])

    def _draw_page_number(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 9)
        canvas.drawCentredString(A4[0] / 2.0, 8 * mm, f"{doc.page}")
        canvas.restoreState()


def build_pdf_bytes_from_markdown(md_text: str, images_dict: Dict[str, str]) -> bytes:
    """
    服务器端生成高清 PDF：
    - 文字为真实矢量文本，不是截图
    - 段落、标题、图片、表格按 story 排版
    - 分页由 ReportLab 控制，避免 html2canvas 裁字
    - 公式与加粗会在这一层被正确渲染
    """
    register_pdf_fonts()
    buffer = BytesIO()

    doc = StableDocTemplate(
        buffer,
        pagesize=PDF_LAYOUT["page_size"],
        leftMargin=PDF_LAYOUT["margin_left"],
        rightMargin=PDF_LAYOUT["margin_right"],
        topMargin=PDF_LAYOUT["margin_top"],
        bottomMargin=PDF_LAYOUT["margin_bottom"],
        title="论文全维度深度透视报告",
        author="AI 论文检索 Agent",
    )

    with tempfile.TemporaryDirectory(prefix='paper_report_formula_') as formula_dir:
        asset_ctx = create_formula_asset_context(formula_dir)
        story = build_story_from_markdown(md_text, images_dict, asset_ctx)
        doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ==========================================
# 模块 13：用户账户与历史报告持久化
# ==========================================
import uuid

try:
    import psycopg
    _PG_DRIVER = "psycopg"
except Exception:
    psycopg = None
    try:
        import psycopg2
        _PG_DRIVER = "psycopg2"
    except Exception:
        psycopg2 = None
        _PG_DRIVER = None

SUPABASE_DB_URL = (st.secrets.get("SUPABASE_DB_URL", "") or "").strip()
ASYNC_MODAL_API_URL = (st.secrets.get("ASYNC_MODAL_API_URL", "") or st.secrets.get("MODAL_JOB_API_URL", "") or "").strip()
PASSWORD_HASH_ROUNDS = 120000
JOB_STATUS_REFRESH_INTERVAL_MS = 4000
DB_READ_CACHE_TTL_SECONDS = 6


def normalize_supabase_db_url(db_url: str) -> str:
    value = (db_url or "").strip()
    if value and "sslmode=" not in value:
        value = f"{value}{'&' if '?' in value else '?'}sslmode=require"
    return value


SUPABASE_DB_URL = normalize_supabase_db_url(SUPABASE_DB_URL)


def _open_db_connection(db_url: str):
    if not db_url:
        raise RuntimeError("未在 Streamlit secrets 中配置 SUPABASE_DB_URL。")
    if _PG_DRIVER == "psycopg":
        return psycopg.connect(db_url)
    if _PG_DRIVER == "psycopg2":
        return psycopg2.connect(db_url)
    raise RuntimeError("未检测到 PostgreSQL 驱动。请在 requirements 中安装 psycopg[binary] 或 psycopg2-binary。")


def get_db_connection():
    return _open_db_connection(SUPABASE_DB_URL)


def _rows_to_dicts(cursor, rows):
    columns = [desc[0] for desc in (cursor.description or [])]
    return [dict(zip(columns, row)) for row in rows]


def _execute_bootstrap_statements(db_url: str):
    statements = [
        """
        CREATE TABLE IF NOT EXISTS public.users (
            id UUID PRIMARY KEY,
            username TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_lower ON public.users ((LOWER(username)))",
        """
        CREATE TABLE IF NOT EXISTS public.analysis_jobs (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
            task_no INTEGER NOT NULL,
            title TEXT,
            pdf_name TEXT,
            pdf_path TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            progress_text TEXT DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "ALTER TABLE public.analysis_jobs ADD COLUMN IF NOT EXISTS cache_key TEXT",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_analysis_jobs_user_task_unique ON public.analysis_jobs(user_id, task_no)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_analysis_jobs_user_cache_key ON public.analysis_jobs(user_id, cache_key) WHERE cache_key IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS idx_analysis_jobs_user_id ON public.analysis_jobs(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status ON public.analysis_jobs(status)",
        """
        CREATE TABLE IF NOT EXISTS public.analysis_reports (
            id UUID PRIMARY KEY,
            job_id UUID NOT NULL UNIQUE REFERENCES public.analysis_jobs(id) ON DELETE CASCADE,
            user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
            report_markdown TEXT,
            parsed_markdown TEXT,
            text_agent_output TEXT,
            vision_output TEXT,
            images_manifest JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_analysis_reports_user_id ON public.analysis_reports(user_id)",
    ]

    conn = _open_db_connection(db_url)
    try:
        with conn.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return True


@st.cache_resource(show_spinner=False)
def _bootstrap_database_once(db_url: str):
    return _execute_bootstrap_statements(db_url)


def ensure_app_storage():
    if not SUPABASE_DB_URL:
        raise RuntimeError("未在 Streamlit secrets 中配置 SUPABASE_DB_URL。")
    _bootstrap_database_once(SUPABASE_DB_URL)


def db_fetch_one(query: str, params: Optional[Tuple[Any, ...]] = None) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params or ())
            row = cursor.fetchone()
            if row is None:
                return None
            return _rows_to_dicts(cursor, [row])[0]
    finally:
        conn.close()


def db_fetch_all(query: str, params: Optional[Tuple[Any, ...]] = None) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params or ())
            rows = cursor.fetchall()
            return _rows_to_dicts(cursor, rows)
    finally:
        conn.close()


def format_db_timestamp(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime.datetime):
        if value.tzinfo:
            value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return ""
        try:
            parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo:
                parsed = parsed.astimezone(datetime.timezone.utc).replace(tzinfo=None)
            return parsed.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return value[:19].replace("T", " ")
    return str(value)


def normalize_json_field(value: Any, default: Any):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return default


def clear_db_read_caches():
    _get_user_record_cached.clear()
    _load_user_report_index_cached.clear()
    _get_user_job_state_cached.clear()
    _get_user_job_by_cache_key_cached.clear()
    _load_user_report_record_cached.clear()
    _get_user_cached_report_cached.clear()


def db_execute(query: str, params: Optional[Tuple[Any, ...]] = None):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params or ())
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    clear_db_read_caches()


def normalize_username(username: str) -> str:
    return (username or "").strip()


def canonical_username(username: str) -> str:
    return normalize_username(username).lower()


def make_password_hash(password: str, salt_hex: Optional[str] = None) -> Tuple[str, str]:
    salt_hex = salt_hex or os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        (password or "").encode("utf-8"),
        bytes.fromhex(salt_hex),
        PASSWORD_HASH_ROUNDS,
    ).hex()
    return salt_hex, digest


def pack_password_hash(password: str) -> str:
    salt_hex, digest = make_password_hash(password)
    return f"{salt_hex}${digest}"


def verify_password_hash(password: str, packed_hash: str) -> bool:
    packed_hash = (packed_hash or "").strip()
    if "$" not in packed_hash:
        return False
    salt_hex, expected_hash = packed_hash.split("$", 1)
    _, actual_hash = make_password_hash(password, salt_hex=salt_hex)
    return actual_hash == expected_hash


@st.cache_data(show_spinner=False, ttl=DB_READ_CACHE_TTL_SECONDS)
def _get_user_record_cached(username: str) -> Optional[Dict[str, Any]]:
    return db_fetch_one(
        """
        SELECT id, username, password_hash, created_at
        FROM public.users
        WHERE LOWER(username) = LOWER(%s)
        LIMIT 1
        """,
        (username,),
    )


def get_user_record(username: str) -> Optional[Dict[str, Any]]:
    username = normalize_username(username)
    if not username:
        return None
    ensure_app_storage()
    return _get_user_record_cached(username)


def register_user(username: str, password: str) -> Tuple[bool, str]:
    username = normalize_username(username)
    password = password or ""

    if not username:
        return False, "请输入账号。"
    if len(username) < 3:
        return False, "账号至少需要 3 个字符。"
    if len(username) > 32:
        return False, "账号长度请控制在 32 个字符以内。"
    if len(password) < 6:
        return False, "密码至少需要 6 个字符。"

    try:
        ensure_app_storage()
        if get_user_record(username):
            return False, "该账号已存在，请直接登录。"

        db_execute(
            """
            INSERT INTO public.users (id, username, password_hash, created_at)
            VALUES (%s, %s, %s, NOW())
            """,
            (str(uuid.uuid4()), username, pack_password_hash(password)),
        )
        return True, username
    except Exception as e:
        if "idx_users_username_lower" in str(e) or "duplicate key" in str(e).lower():
            return False, "该账号已存在，请直接登录。"
        return False, f"注册失败：{str(e)}"


def authenticate_user(username: str, password: str) -> Tuple[bool, str]:
    username = normalize_username(username)
    password = password or ""

    try:
        ensure_app_storage()
        record = get_user_record(username)
    except Exception as e:
        return False, f"登录失败：{str(e)}"

    if not record:
        return False, "账号或密码错误。"
    if not verify_password_hash(password, record.get("password_hash", "")):
        return False, "账号或密码错误。"
    return True, record.get("username") or username


def get_history_selector_key(username: str) -> str:
    return f"history_selector_{canonical_username(username)}"


def get_history_reset_flag_key(username: str) -> str:
    return f"history_reset_pending_{canonical_username(username)}"


def reset_user_workspace_view(username: Optional[str] = None):
    username = normalize_username(username or st.session_state.get("current_user", ""))
    if not username:
        return
    reset_flag_key = get_history_reset_flag_key(username)
    st.session_state[reset_flag_key] = True
    st.session_state.selected_history_report_id = None


def start_fresh_workspace(username: Optional[str] = None):
    username = normalize_username(username or st.session_state.get("current_user", ""))
    reset_user_workspace_view(username)
    seen_paper_ids.clear()
    st.session_state.app_state = "IDLE"
    st.session_state.prompt_history = []
    st.session_state.agent = None
    st.session_state.final_result = ""
    st.session_state.loop_count = 0
    st.session_state.has_provided_feedback = False
    st.session_state.ui_logs = []
    st.session_state.feedback_start_time = None
    st.session_state.sidebar_direct_entries = []
    st.session_state.bottom_direct_entries = []


def get_user_space_dir(username: str) -> str:
    return canonical_username(username)


def ensure_user_space(username: str) -> str:
    ensure_app_storage()
    return get_user_space_dir(username)


def get_user_report_index_path(username: str) -> str:
    return canonical_username(username)


def get_user_report_file_path(username: str, report_id: str) -> str:
    return f"{canonical_username(username)}:{report_id}"


def build_report_meta_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "report_id": row.get("report_id") or row.get("job_id") or "",
        "cache_key": row.get("cache_key", "") or "",
        "source_name": row.get("source_name") or row.get("pdf_name") or row.get("report_title") or "未命名论文",
        "report_title": row.get("report_title") or row.get("title") or row.get("source_name") or "论文全维度深度透视报告",
        "created_at": format_db_timestamp(row.get("created_at") or row.get("job_created_at") or row.get("report_created_at")),
        "updated_at": format_db_timestamp(row.get("updated_at") or row.get("report_created_at") or row.get("job_created_at") or row.get("created_at")),
        "status": (row.get("status") or "").lower(),
        "progress_text": row.get("progress_text") or "",
        "has_report": bool(row.get("has_report")),
        "task_no": row.get("task_no"),
    }


@st.cache_data(show_spinner=False, ttl=DB_READ_CACHE_TTL_SECONDS)
def _load_user_report_index_cached(username: str) -> List[Dict[str, Any]]:
    rows = db_fetch_all(
        """
        SELECT
            j.id AS report_id,
            j.task_no,
            j.cache_key,
            COALESCE(NULLIF(j.pdf_name, ''), NULLIF(j.title, ''), '未命名论文') AS source_name,
            COALESCE(NULLIF(j.title, ''), NULLIF(j.pdf_name, ''), '论文全维度深度透视报告') AS report_title,
            j.status,
            j.progress_text,
            j.created_at,
            COALESCE(j.updated_at, r.created_at, j.created_at) AS updated_at,
            CASE WHEN r.job_id IS NOT NULL THEN TRUE ELSE FALSE END AS has_report
        FROM public.analysis_jobs j
        JOIN public.users u ON u.id = j.user_id
        LEFT JOIN public.analysis_reports r ON r.job_id = j.id
        WHERE LOWER(u.username) = LOWER(%s)
        ORDER BY COALESCE(j.updated_at, r.created_at, j.created_at) DESC
        """,
        (username,),
    )
    return [build_report_meta_from_row(row) for row in rows]


def load_user_report_index(username: str) -> List[Dict[str, Any]]:
    username = normalize_username(username)
    if not username:
        return []
    ensure_app_storage()
    return _load_user_report_index_cached(username)


def get_persistable_analysis_result(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_markdown": analysis_result.get("source_markdown", ""),
        "text_report": analysis_result.get("text_report", ""),
        "vision_summaries": analysis_result.get("vision_summaries", ""),
        "images": analysis_result.get("images", {}),
        "main_report": analysis_result.get("main_report", ""),
    }


def get_next_task_no(user_id: str) -> int:
    row = db_fetch_one(
        "SELECT COALESCE(MAX(task_no), 0) + 1 AS next_task_no FROM public.analysis_jobs WHERE user_id = %s",
        (user_id,),
    )
    return int((row or {}).get("next_task_no") or 1)


def save_user_report_record(
    username: str,
    source_name: str,
    cache_key: str,
    analysis_result: Dict[str, Any],
) -> Dict[str, Any]:
    ensure_app_storage()
    username = normalize_username(username)
    user_record = get_user_record(username)
    if not user_record:
        raise RuntimeError("当前账号不存在，无法保存历史报告。")

    user_id = user_record["id"]
    source_name = (source_name or "").strip() or "未命名论文"
    report_title, _ = split_title_and_body(analysis_result.get("main_report", ""))
    report_title = (report_title or source_name or "论文全维度深度透视报告").strip()
    report_payload = get_persistable_analysis_result(analysis_result)
    images_manifest_text = json.dumps(report_payload.get("images", {}), ensure_ascii=False)

    existing_job = db_fetch_one(
        """
        SELECT id AS job_id, task_no, created_at, updated_at
        FROM public.analysis_jobs
        WHERE user_id = %s AND cache_key = %s
        LIMIT 1
        """,
        (user_id, cache_key),
    )

    if existing_job:
        job_id = existing_job["job_id"]
        task_no = int(existing_job.get("task_no") or 1)
        created_at_value = existing_job.get("created_at")
        db_execute(
            """
            UPDATE public.analysis_jobs
            SET title = %s,
                pdf_name = %s,
                status = 'finished',
                progress_text = '解析完成',
                updated_at = NOW()
            WHERE id = %s
            """,
            (report_title, source_name, job_id),
        )
    else:
        task_no = get_next_task_no(user_id)
        job_id = str(uuid.uuid4())
        created_at_value = datetime.datetime.now()
        db_execute(
            """
            INSERT INTO public.analysis_jobs (
                id, user_id, task_no, title, pdf_name, pdf_path, cache_key,
                status, progress_text, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'finished', '解析完成', NOW(), NOW())
            """,
            (job_id, user_id, task_no, report_title, source_name, "", cache_key),
        )

    existing_report = db_fetch_one(
        "SELECT id, created_at FROM public.analysis_reports WHERE job_id = %s LIMIT 1",
        (job_id,),
    )

    if existing_report:
        db_execute(
            """
            UPDATE public.analysis_reports
            SET user_id = %s,
                report_markdown = %s,
                parsed_markdown = %s,
                text_agent_output = %s,
                vision_output = %s,
                images_manifest = %s::jsonb
            WHERE job_id = %s
            """,
            (
                user_id,
                report_payload.get("main_report", ""),
                report_payload.get("source_markdown", ""),
                report_payload.get("text_report", ""),
                report_payload.get("vision_summaries", ""),
                images_manifest_text,
                job_id,
            ),
        )
    else:
        db_execute(
            """
            INSERT INTO public.analysis_reports (
                id, job_id, user_id, report_markdown, parsed_markdown,
                text_agent_output, vision_output, images_manifest, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW())
            """,
            (
                str(uuid.uuid4()),
                job_id,
                user_id,
                report_payload.get("main_report", ""),
                report_payload.get("source_markdown", ""),
                report_payload.get("text_report", ""),
                report_payload.get("vision_summaries", ""),
                images_manifest_text,
            ),
        )

    return {
        "report_id": job_id,
        "cache_key": cache_key,
        "source_name": source_name,
        "report_title": report_title,
        "created_at": format_db_timestamp(created_at_value),
        "updated_at": format_db_timestamp(datetime.datetime.now()),
        "task_no": task_no,
        "status": "finished",
        "progress_text": "解析完成",
        "has_report": True,
    }


@st.cache_data(show_spinner=False, ttl=DB_READ_CACHE_TTL_SECONDS)
def _get_user_job_state_cached(username: str, report_id: str) -> Optional[Dict[str, Any]]:
    row = db_fetch_one(
        """
        SELECT
            j.id AS report_id,
            j.task_no,
            j.cache_key,
            j.title AS report_title,
            j.pdf_name AS source_name,
            j.status,
            j.progress_text,
            j.created_at,
            j.updated_at,
            CASE WHEN r.job_id IS NOT NULL THEN TRUE ELSE FALSE END AS has_report
        FROM public.analysis_jobs j
        JOIN public.users u ON u.id = j.user_id
        LEFT JOIN public.analysis_reports r ON r.job_id = j.id
        WHERE LOWER(u.username) = LOWER(%s) AND j.id = %s
        LIMIT 1
        """,
        (username, report_id),
    )
    if not row:
        return None
    return build_report_meta_from_row(row)


def get_user_job_state(username: str, report_id: str) -> Optional[Dict[str, Any]]:
    username = normalize_username(username)
    report_id = (report_id or "").strip()
    if not username or not report_id:
        return None
    ensure_app_storage()
    return _get_user_job_state_cached(username, report_id)


@st.cache_data(show_spinner=False, ttl=DB_READ_CACHE_TTL_SECONDS)
def _get_user_job_by_cache_key_cached(username: str, cache_key: str) -> Optional[Dict[str, Any]]:
    row = db_fetch_one(
        """
        SELECT
            j.id AS report_id,
            j.task_no,
            j.cache_key,
            j.title AS report_title,
            j.pdf_name AS source_name,
            j.status,
            j.progress_text,
            j.created_at,
            j.updated_at,
            CASE WHEN r.job_id IS NOT NULL THEN TRUE ELSE FALSE END AS has_report
        FROM public.analysis_jobs j
        JOIN public.users u ON u.id = j.user_id
        LEFT JOIN public.analysis_reports r ON r.job_id = j.id
        WHERE LOWER(u.username) = LOWER(%s) AND j.cache_key = %s
        LIMIT 1
        """,
        (username, cache_key),
    )
    if not row:
        return None
    return build_report_meta_from_row(row)


def get_user_job_by_cache_key(username: str, cache_key: str) -> Optional[Dict[str, Any]]:
    username = normalize_username(username)
    cache_key = (cache_key or "").strip()
    if not username or not cache_key:
        return None
    ensure_app_storage()
    return _get_user_job_by_cache_key_cached(username, cache_key)


def focus_latest_user_job(username: str):
    username = normalize_username(username)
    selector_key = get_history_selector_key(username)
    reset_flag_key = get_history_reset_flag_key(username)
    history = load_user_report_index(username)
    if history:
        latest_id = history[0].get("report_id", "")
        if latest_id:
            st.session_state[selector_key] = latest_id
            st.session_state.selected_history_report_id = latest_id
            st.session_state[reset_flag_key] = False
            return
    reset_user_workspace_view(username)


def update_analysis_job_status(job_id: str, status: str, progress_text: str):
    job_id = (job_id or "").strip()
    if not job_id:
        return
    safe_status = (status or "").strip().lower()
    if safe_status not in {"queued", "processing", "finished", "failed"}:
        safe_status = "processing"
    db_execute(
        """
        UPDATE public.analysis_jobs
        SET status = %s,
            progress_text = %s,
            updated_at = NOW()
        WHERE id = %s
        """,
        (safe_status, (progress_text or "").strip()[:1000], job_id),
    )


def create_or_reuse_analysis_job(username: str, source_name: str, cache_key: str) -> Tuple[Dict[str, Any], bool]:
    ensure_app_storage()
    username = normalize_username(username)
    user_record = get_user_record(username)
    if not user_record:
        raise RuntimeError("当前账号不存在，无法创建后台解析任务。")

    source_name = (source_name or "").strip() or "未命名论文"
    user_id = user_record["id"]
    existing_job = get_user_job_by_cache_key(username, cache_key)
    if existing_job:
        existing_status = (existing_job.get("status") or "").lower()
        if existing_status in {"queued", "processing"}:
            return existing_job, False
        if existing_status == "finished" and existing_job.get("has_report"):
            return existing_job, False

        update_analysis_job_status(existing_job["report_id"], "queued", "任务已重新提交，等待后台启动")
        db_execute(
            """
            UPDATE public.analysis_jobs
            SET title = %s,
                pdf_name = %s,
                updated_at = NOW()
            WHERE id = %s
            """,
            (source_name, source_name, existing_job["report_id"]),
        )
        refreshed_job = get_user_job_state(username, existing_job["report_id"]) or existing_job
        return refreshed_job, True

    task_no = get_next_task_no(user_id)
    job_id = str(uuid.uuid4())
    db_execute(
        """
        INSERT INTO public.analysis_jobs (
            id, user_id, task_no, title, pdf_name, pdf_path, cache_key,
            status, progress_text, created_at, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'queued', '任务已提交，等待后台启动', NOW(), NOW())
        """,
        (job_id, user_id, task_no, source_name, source_name, "", cache_key),
    )
    created_job = get_user_job_state(username, job_id)
    if not created_job:
        raise RuntimeError("后台解析任务创建成功，但未能回读任务记录。")
    return created_job, True


def submit_analysis_job_to_modal(job_id: str, source_name: str, cache_key: str, pdf_bytes: bytes):
    submit_url = (ASYNC_MODAL_API_URL or "").strip()
    if not submit_url:
        raise RuntimeError("未在 Streamlit secrets 中配置 ASYNC_MODAL_API_URL。")

    files = {
        "file": (source_name or "paper.pdf", pdf_bytes, "application/pdf"),
    }
    data = {
        "job_id": job_id,
        "source_name": source_name or "未命名论文",
        "cache_key": cache_key,
    }

    response = requests.post(submit_url, data=data, files=files, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"后台任务提交失败：HTTP {response.status_code}")

    try:
        payload = response.json()
    except Exception as e:
        raise RuntimeError(f"后台任务返回了无法解析的响应：{str(e)}")

    if payload.get("status") not in {"accepted", "queued", "processing"}:
        raise RuntimeError(payload.get("message") or "后台任务未被接受。")


def render_pending_job_notice(job_meta: Dict[str, Any], show_title: bool = False):
    source_name = job_meta.get("source_name") or job_meta.get("report_title") or "未命名论文"
    if show_title:
        st.markdown(f"### {source_name}")

    status = (job_meta.get("status") or "").lower()
    progress_text = job_meta.get("progress_text") or ""
    if status == "failed":
        st.error(progress_text or f"《{source_name}》解析失败。")
        return

    display_text = progress_text or "后台任务正在运行中。"
    st.info(f"《{source_name}》当前状态：{display_text}")
    st.caption("该任务已提交到后台 Modal。您现在可以直接关闭页面，稍后重新登录查看状态或结果。")


@st.cache_data(show_spinner=False, ttl=DB_READ_CACHE_TTL_SECONDS)
def _load_user_report_record_cached(username: str, report_id: str) -> Optional[Dict[str, Any]]:
    row = db_fetch_one(
        """
        SELECT
            j.id AS report_id,
            j.cache_key,
            j.title AS report_title,
            j.pdf_name AS source_name,
            j.created_at AS job_created_at,
            j.updated_at,
            r.created_at AS report_created_at,
            r.report_markdown,
            r.parsed_markdown,
            r.text_agent_output,
            r.vision_output,
            r.images_manifest
        FROM public.analysis_jobs j
        JOIN public.users u ON u.id = j.user_id
        JOIN public.analysis_reports r ON r.job_id = j.id
        WHERE LOWER(u.username) = LOWER(%s) AND j.id = %s
        LIMIT 1
        """,
        (username, report_id),
    )
    if not row:
        return None

    meta = build_report_meta_from_row({
        "report_id": row.get("report_id"),
        "cache_key": row.get("cache_key"),
        "source_name": row.get("source_name"),
        "report_title": row.get("report_title"),
        "created_at": row.get("job_created_at"),
        "updated_at": row.get("updated_at") or row.get("report_created_at") or row.get("job_created_at"),
        "status": "finished",
        "progress_text": "解析完成",
        "has_report": True,
    })
    analysis_result = {
        "source_markdown": row.get("parsed_markdown", "") or "",
        "text_report": row.get("text_agent_output", "") or "",
        "vision_summaries": row.get("vision_output", "") or "",
        "images": normalize_json_field(row.get("images_manifest"), {}),
        "main_report": row.get("report_markdown", "") or "",
    }
    return {"meta": meta, "analysis_result": analysis_result}


def load_user_report_record(username: str, report_id: str) -> Optional[Dict[str, Any]]:
    username = normalize_username(username)
    report_id = (report_id or "").strip()
    if not username or not report_id:
        return None
    ensure_app_storage()
    return _load_user_report_record_cached(username, report_id)


@st.cache_data(show_spinner=False, ttl=DB_READ_CACHE_TTL_SECONDS)
def _get_user_cached_report_cached(username: str, cache_key: str) -> Optional[Dict[str, Any]]:
    row = db_fetch_one(
        """
        SELECT j.id AS report_id
        FROM public.analysis_jobs j
        JOIN public.users u ON u.id = j.user_id
        JOIN public.analysis_reports r ON r.job_id = j.id
        WHERE LOWER(u.username) = LOWER(%s) AND j.cache_key = %s
        ORDER BY COALESCE(j.updated_at, r.created_at, j.created_at) DESC
        LIMIT 1
        """,
        (username, cache_key),
    )
    if not row:
        return None
    payload = _load_user_report_record_cached(username, row.get("report_id", ""))
    if payload and isinstance(payload.get("analysis_result"), dict):
        return payload["analysis_result"]
    return None


def get_user_cached_report(username: str, cache_key: str) -> Optional[Dict[str, Any]]:
    username = normalize_username(username)
    cache_key = (cache_key or "").strip()
    if not username or not cache_key:
        return None
    ensure_app_storage()
    return _get_user_cached_report_cached(username, cache_key)


def shorten_sidebar_label(text: str, max_len: int = 20) -> str:
    value = (text or "").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "…"



def format_report_history_label(meta: Dict[str, Any]) -> str:
    if not meta:
        return "当前工作区"
    display_name = meta.get("source_name") or meta.get("report_title") or "未命名论文"
    short_name = shorten_sidebar_label(display_name, max_len=18)
    status = (meta.get("status") or "").lower()
    if status in {"queued", "processing"}:
        return f"{short_name}｜正在解析中"
    if status == "failed":
        return f"{short_name}｜解析失败"
    timestamp = (meta.get("updated_at") or meta.get("created_at") or "")[:16]
    return f"{short_name}｜{timestamp}" if timestamp else short_name



def render_auth_ui():
    st.title("AI 智能论文检索 Agent")
    st.markdown("请先登录或注册账号。登录后，系统会为每个账号自动保存历史论文解析报告，下一次登录可直接查看，无需重新解析。")

    login_tab, register_tab = st.tabs(["登录", "注册"])

    with login_tab:
        with st.form("login_form"):
            login_username = st.text_input("账号", key="login_username")
            login_password = st.text_input("密码", type="password", key="login_password")
            login_submit = st.form_submit_button("登录", use_container_width=True)
            if login_submit:
                ok, result = authenticate_user(login_username, login_password)
                if ok:
                    st.session_state.current_user = result
                    focus_latest_user_job(result)
                    st.rerun()
                else:
                    st.error(result)

    with register_tab:
        with st.form("register_form"):
            register_username = st.text_input("账号", key="register_username")
            register_password = st.text_input("密码", type="password", key="register_password")
            confirm_password = st.text_input("确认密码", type="password", key="register_password_confirm")
            register_submit = st.form_submit_button("注册并进入系统", use_container_width=True)
            if register_submit:
                if register_password != confirm_password:
                    st.error("两次输入的密码不一致。")
                else:
                    ok, result = register_user(register_username, register_password)
                    if ok:
                        st.session_state.current_user = result
                        reset_user_workspace_view(result)
                        st.rerun()
                    else:
                        st.error(result)

    st.stop()



def render_history_sidebar(username: str):
    history = load_user_report_index(username)
    selector_key = get_history_selector_key(username)
    reset_flag_key = get_history_reset_flag_key(username)
    options = ["__workspace__"] + [item.get("report_id", "") for item in history if item.get("report_id")]
    meta_map = {item.get("report_id", ""): item for item in history if item.get("report_id")}

    if (
        selector_key not in st.session_state
        or st.session_state[selector_key] not in options
        or st.session_state.get(reset_flag_key, False)
    ):
        st.session_state[selector_key] = "__workspace__"
        st.session_state[reset_flag_key] = False

    st.header("历史报告记录")
    selected_id = st.radio(
        "历史报告记录",
        options=options,
        format_func=lambda option_id: "当前工作区" if option_id == "__workspace__" else format_report_history_label(meta_map.get(option_id, {})),
        key=selector_key,
        label_visibility="collapsed",
    )
    st.session_state.selected_history_report_id = None if selected_id == "__workspace__" else selected_id

    if history:
        st.caption("重新登录后也可在这里直接打开历史报告。")
    else:
        st.caption("当前账号还没有历史报告。")

# ==========================================
# 模块 14：论文精读主流程
# ==========================================
def get_pdf_cache_key(pdf_bytes: bytes) -> str:
    """为每篇论文生成稳定缓存键，支持多文件分别缓存。"""
    return hashlib.sha256(ANALYSIS_CACHE_VERSION.encode('utf-8') + pdf_bytes).hexdigest()



def build_analysis_result(pdf_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    单篇论文的完整分析流程：
    1. 远端解析 PDF
    2. Text Agent 输出 FACT_BANK + 文本综述
    3. Vision Agent 输出 FIGURE_CARD
    4. 主报告逐节生成
    """
    result = analyze_pdf_with_modal(pdf_file_bytes=pdf_bytes)
    if not (result and result.get("status") == "success"):
        return None

    md_content = result["markdown"]
    ordered_images = sort_images_by_doc_order(md_content, result.get("images", {}))
    ordered_images, image_local_contexts = collect_cited_images_by_reference(md_content, ordered_images)
    images_dict = dict(ordered_images)

    with st.spinner("文本专家正在精读全篇文本……"):
        text_agent = LLMClient(
            sys_prompt=TEXT_AGENT_PROMPT,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            max_tokens=10000,
        )
        text_report = text_agent.generate([f"请详尽解析此论文：\n{md_content}"])

    vision_summaries = ""
    if images_dict:
        with st.spinner(f"视觉专家正在分析 {len(images_dict)} 张关键图表……"):
            vision_agent = LLMClient(
                sys_prompt=VISION_AGENT_PROMPT,
                model="qwen3.6-plus-2026-04-02",
                api_key=QWEN_API_KEY,
                base_url=QWEN_BASE_URL,
                max_tokens=6000,
            )
            cards = []
            for name, b64 in ordered_images:
                local_context = image_local_contexts.get(name) or extract_local_context(md_content, name)
                vision_prompt = f"""
请基于以下论文上下文与图片，严格输出 FIGURE_CARD。

【图片ID】
{name}

【论文局部上下文】
{local_context}

【论文整体上下文摘要】
{build_source_pack(md_content, max_chars=4000)}
"""
                v_res = vision_agent.generate_with_images(vision_prompt, [b64])
                cards.append(f"\n--- 图表标识: {name} ---\n{v_res}\n")
            vision_summaries = "\n".join(cards)

    with st.spinner("总策划正在逐节生成终极报告……"):
        report = generate_full_report(
            md_content=md_content,
            text_report=text_report,
            vision_summaries=vision_summaries,
            image_ids=list(images_dict.keys()),
        )
        final_main_report = prepare_report_markdown_for_display(report, images_dict=images_dict, vision_summaries=vision_summaries)

    if is_report_truncated(final_main_report):
        with st.spinner("检测到报告可能截断，正在补全缺失内容……"):
            report = generate_full_report(
                md_content=md_content,
                text_report=text_report,
                vision_summaries=vision_summaries,
                image_ids=list(images_dict.keys()),
            )
            final_main_report = prepare_report_markdown_for_display(report, images_dict=images_dict, vision_summaries=vision_summaries)

    return {
        "source_markdown": md_content,
        "text_report": text_report,
        "vision_summaries": vision_summaries,
        "images": images_dict,
        "main_report": final_main_report,
    }



def get_or_create_analysis_result(pdf_bytes: bytes, source_name: str) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """读取、提交或轮询单篇论文分析结果。"""
    cache_key = get_pdf_cache_key(pdf_bytes)
    cache_pool = st.session_state.analysis_results
    if cache_key in cache_pool and cache_pool[cache_key] is not None:
        return cache_key, cache_pool[cache_key], "session_cache"

    current_user = normalize_username(st.session_state.get("current_user", ""))
    if not current_user:
        return cache_key, None, "failed"

    cached_report = get_user_cached_report(current_user, cache_key)
    if cached_report is not None:
        cache_pool[cache_key] = cached_report
        return cache_key, cached_report, "history_cache"

    existing_job = get_user_job_by_cache_key(current_user, cache_key)
    if existing_job:
        existing_status = (existing_job.get("status") or "").lower()
        if existing_status in {"queued", "processing"}:
            return cache_key, None, "pending"
        if existing_status == "failed":
            return cache_key, None, "failed"
        if existing_status == "finished":
            return cache_key, None, "processing_finalize"

    try:
        job_meta, should_submit = create_or_reuse_analysis_job(current_user, source_name, cache_key)
        if should_submit:
            submit_analysis_job_to_modal(job_meta["report_id"], source_name, cache_key, pdf_bytes)
            update_analysis_job_status(job_meta["report_id"], "processing", "后台任务已启动，等待离线解析完成")
            return cache_key, None, "submitted"
        status = (job_meta.get("status") or "").lower()
        if status in {"queued", "processing"}:
            return cache_key, None, "pending"
        if status == "finished" and job_meta.get("has_report"):
            cached_report = get_user_cached_report(current_user, cache_key)
            if cached_report is not None:
                cache_pool[cache_key] = cached_report
                return cache_key, cached_report, "history_cache"
        return cache_key, None, "pending"
    except Exception as e:
        existing_job = get_user_job_by_cache_key(current_user, cache_key)
        if existing_job:
            update_analysis_job_status(existing_job["report_id"], "failed", f"后台任务提交失败：{str(e)}")
        cache_pool.pop(cache_key, None)
        return cache_key, None, "failed"



def build_export_filename(source_name: str, suffix: str) -> str:
    base_name = re.sub(r'(?i)\.pdf$', '', (source_name or '').strip())
    base_name = re.sub(r'[\/:*?"<>|]+', '_', base_name).strip() or '论文'
    return f"{base_name}{suffix}"



def collect_pdf_entries(pdf_inputs) -> List[Tuple[str, bytes]]:
    entries: List[Tuple[str, bytes]] = []

    if isinstance(pdf_inputs, bytes):
        entries.append(("论文.pdf", pdf_inputs))
        return entries

    if isinstance(pdf_inputs, tuple) and len(pdf_inputs) == 2:
        name, data = pdf_inputs
        if isinstance(data, (bytes, bytearray)):
            entries.append((str(name) or "论文.pdf", bytes(data)))
        return entries

    for idx, uploaded in enumerate(pdf_inputs or [], start=1):
        if uploaded is None:
            continue
        if isinstance(uploaded, tuple) and len(uploaded) == 2:
            name, data = uploaded
            if isinstance(data, (bytes, bytearray)):
                entries.append((str(name) or f"paper_{idx}.pdf", bytes(data)))
            continue
        if isinstance(uploaded, dict):
            name = str(uploaded.get("name") or f"paper_{idx}.pdf")
            data = uploaded.get("bytes")
            if isinstance(data, (bytes, bytearray)):
                entries.append((name, bytes(data)))
            continue
        paper_name = getattr(uploaded, 'name', f'paper_{idx}.pdf')
        if hasattr(uploaded, 'getvalue'):
            entries.append((paper_name, uploaded.getvalue()))
        elif isinstance(uploaded, (bytes, bytearray)):
            entries.append((paper_name, bytes(uploaded)))
        else:
            entries.append((paper_name, uploaded))

    return entries



def render_single_analysis_result(
    analysis_result: Dict[str, Any],
    cache_key: str,
    source_name: str,
    show_paper_title: bool = False,
    status_text: str = "论文深度透视报告已生成！",
):
    if show_paper_title:
        st.markdown(f"### {source_name}")

    display_report_md = prepare_report_markdown_for_display(
        analysis_result.get("main_report", ""),
        images_dict=analysis_result.get("images", {}),
        vision_summaries=analysis_result.get("vision_summaries", ""),
    )
    st.success(status_text)
    render_report_with_images(display_report_md, analysis_result["images"])

    st.divider()
    st.markdown("### 导出")
    st.download_button(
        label="下载报告原文（Markdown）",
        data=display_report_md,
        file_name=build_export_filename(source_name, "_论文全维度深度透视报告.md"),
        mime="text/markdown",
        use_container_width=True,
        key=f"download_md_{cache_key}",
    )



def render_saved_history_report(username: str, report_id: str):
    job_meta = get_user_job_state(username, report_id)
    if not job_meta:
        st.warning("未找到这条历史报告记录，请重新解析论文。")
        return

    status = (job_meta.get("status") or "").lower()
    source_name = job_meta.get("source_name") or job_meta.get("report_title") or "历史报告"

    if status in {"queued", "processing"}:
        render_pending_job_notice(job_meta, show_title=True)
        st_autorefresh(interval=JOB_STATUS_REFRESH_INTERVAL_MS, key=f"job_status_poll_{report_id}")
        return

    if status == "failed":
        render_pending_job_notice(job_meta, show_title=True)
        return

    payload = load_user_report_record(username, report_id)
    if not payload:
        st.info(f"《{source_name}》的后台任务已完成，正在同步最终报告，请稍候自动刷新。")
        st_autorefresh(interval=JOB_STATUS_REFRESH_INTERVAL_MS, key=f"job_finalize_poll_{report_id}")
        return

    meta = payload.get("meta", {})
    analysis_result = payload.get("analysis_result", {})
    timestamp = meta.get("updated_at") or meta.get("created_at") or "未知时间"

    st.info(f"已载入历史报告：{source_name}（保存时间：{timestamp}）。")
    render_single_analysis_result(
        analysis_result=analysis_result,
        cache_key=meta.get("cache_key", report_id),
        source_name=source_name,
        show_paper_title=True,
        status_text="历史报告已载入，无需重新解析。",
    )



def render_batch_status_overview(batch_rows: List[Dict[str, Any]]):
    st.markdown("### 批量解析进度")
    for row in batch_rows:
        idx = row.get("index")
        source_name = row.get("source_name") or "未命名论文"
        status = (row.get("status") or "").lower()
        progress_text = row.get("progress_text") or ""
        if status == "finished":
            icon = "✅"
            text = progress_text or "解析完成"
        elif status in {"queued", "processing"}:
            icon = "⏳"
            text = progress_text or "正在解析中"
        else:
            icon = "❌"
            text = progress_text or "解析失败"
        st.markdown(f"**{icon} 第 {idx} 篇《{source_name}》：** {text}")



def render_analysis_ui(pdf_inputs):
    """
    上传论文后的主工作流：
    - 支持单篇 PDF 分析
    - 支持多篇 PDF 同时上传，并分别生成各自报告
    - 任务会立即写入 analysis_jobs，并交给新的后台 Modal 继续运行
    - 多篇 PDF 场景下，全部完成后再统一展示所有报告
    """
    entries = collect_pdf_entries(pdf_inputs)
    if not entries:
        return

    current_user = normalize_username(st.session_state.get("current_user", ""))
    multi_mode = len(entries) > 1
    should_poll = False
    batch_rows: List[Dict[str, Any]] = []
    ready_reports: List[Dict[str, Any]] = []

    for idx, (paper_name, pdf_bytes) in enumerate(entries, start=1):
        cache_key, analysis_result, result_source = get_or_create_analysis_result(pdf_bytes, paper_name)
        job_meta = get_user_job_by_cache_key(current_user, cache_key) if current_user else None

        row = {
            "index": idx,
            "source_name": paper_name,
            "cache_key": cache_key,
            "status": "processing",
            "progress_text": "任务正在初始化中，请稍候。",
        }

        if analysis_result is not None:
            row.update({"status": "finished", "progress_text": "解析完成"})
            ready_reports.append({
                "index": idx,
                "source_name": paper_name,
                "cache_key": cache_key,
                "analysis_result": analysis_result,
                "result_source": result_source,
            })
        else:
            if job_meta:
                row.update({
                    "status": job_meta.get("status") or "processing",
                    "progress_text": job_meta.get("progress_text") or row["progress_text"],
                })
            elif result_source == "failed":
                row.update({"status": "failed", "progress_text": "后台任务提交或执行失败，请稍后重试。"})
            elif result_source == "submitted":
                row.update({"status": "processing", "progress_text": "后台任务已创建，正在等待离线解析。"})
            elif result_source == "processing_finalize":
                row.update({"status": "processing", "progress_text": "任务已完成，正在同步最终报告。"})

        if row["status"] in {"queued", "processing"}:
            should_poll = True

        batch_rows.append(row)

    if multi_mode:
        render_batch_status_overview(batch_rows)
        pending_exists = any((row.get("status") or "").lower() in {"queued", "processing"} for row in batch_rows)
        if pending_exists:
            finished_count = sum(1 for row in batch_rows if (row.get("status") or "").lower() == "finished")
            if finished_count:
                st.info("已有部分论文解析完成。为避免界面混乱，系统会在全部任务完成后统一展示所有 PDF 的解析报告。")
            if should_poll:
                st_autorefresh(interval=JOB_STATUS_REFRESH_INTERVAL_MS, key="pending_analysis_refresh")
                st.caption("后台任务正在继续运行。您可以关闭页面，稍后重新登录查看；若保持当前页面打开，系统会自动刷新状态。")
        else:
            st.divider()
            st.markdown("### 全部 PDF 解析结果")
            rendered_sections = 0
            total_sections = len(ready_reports) + sum(1 for row in batch_rows if (row.get("status") or "").lower() == "failed")
            for report_item in ready_reports:
                rendered_sections += 1
                st.markdown(f"## 第 {report_item['index']} 篇论文")
                if report_item["result_source"] == "history_cache":
                    st.info(f"检测到当前账号下已存在《{report_item['source_name']}》的历史报告，已直接读取，无需重新解析。")
                render_single_analysis_result(
                    analysis_result=report_item["analysis_result"],
                    cache_key=report_item["cache_key"],
                    source_name=report_item["source_name"],
                    show_paper_title=False,
                )
                if rendered_sections < total_sections:
                    st.divider()
            for row in batch_rows:
                if (row.get("status") or "").lower() != "failed":
                    continue
                rendered_sections += 1
                st.markdown(f"## 第 {row['index']} 篇论文")
                st.error(f"《{row['source_name']}》解析失败：{row.get('progress_text') or '未知错误'}")
                if rendered_sections < total_sections:
                    st.divider()
    else:
        report_item = ready_reports[0] if ready_reports else None
        row = batch_rows[0]
        if report_item:
            if report_item["result_source"] == "history_cache":
                st.info(f"检测到当前账号下已存在《{report_item['source_name']}》的历史报告，已直接读取，无需重新解析。")
            render_single_analysis_result(
                analysis_result=report_item["analysis_result"],
                cache_key=report_item["cache_key"],
                source_name=report_item["source_name"],
                show_paper_title=False,
            )
        else:
            render_pending_job_notice(row, show_title=False)
            if should_poll:
                st_autorefresh(interval=JOB_STATUS_REFRESH_INTERVAL_MS, key="pending_analysis_refresh")
                st.caption("后台任务正在继续运行。您可以关闭页面，稍后重新登录查看；若保持当前页面打开，系统会自动刷新状态。")

    st.divider()
    if st.button("开始全新探索", type="primary", key="start_fresh_workspace_after_analysis"):
        start_fresh_workspace(st.session_state.get("current_user", ""))
        st.rerun()


# ==========================================
# 模块 15：状态初始化
# ==========================================
try:
    ensure_app_storage()
except Exception as e:
    st.error(str(e))
    st.stop()

if "current_user" not in st.session_state:
    st.session_state.current_user = ""
if "selected_history_report_id" not in st.session_state:
    st.session_state.selected_history_report_id = None
if "app_state" not in st.session_state:
    st.session_state.app_state = "IDLE"
if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "final_result" not in st.session_state:
    st.session_state.final_result = ""
if "loop_count" not in st.session_state:
    st.session_state.loop_count = 0
if "has_provided_feedback" not in st.session_state:
    st.session_state.has_provided_feedback = False
if "ui_logs" not in st.session_state:
    st.session_state.ui_logs = []
if "current_pdf_hash" not in st.session_state:
    st.session_state.current_pdf_hash = None
if "final_main_report" not in st.session_state:
    st.session_state.final_main_report = ""
if "temp_images" not in st.session_state:
    st.session_state.temp_images = {}
if "source_markdown" not in st.session_state:
    st.session_state.source_markdown = ""
if "text_report" not in st.session_state:
    st.session_state.text_report = ""
if "vision_summaries" not in st.session_state:
    st.session_state.vision_summaries = ""
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "feedback_start_time" not in st.session_state:
    st.session_state.feedback_start_time = None
if "sidebar_direct_entries" not in st.session_state:
    st.session_state.sidebar_direct_entries = []
if "bottom_direct_entries" not in st.session_state:
    st.session_state.bottom_direct_entries = []

if not st.session_state.current_user:
    render_auth_ui()


# ==========================================
# 模块 16：前端 UI
# ==========================================
st.title("AI 智能论文检索 Agent")
st.markdown("基于大模型的多轮深度挖掘，为您精准匹配 Top 6 核心前沿文献，并为每个账号自动保留历史解析报告。")

with st.sidebar:
    st.success(f"当前账号：{st.session_state.current_user}")
    if st.button("退出登录", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.divider()
    render_history_sidebar(st.session_state.current_user)

    st.divider()
    st.header("检索配置")

    user_topic = st.text_input("研究方向", value="")

    user_requirements = st.text_area(
        "具体筛选要求",
        value="",
        placeholder="建议分点填写",
        help="要求越具体，Agent 挖掘的文献越精准。",
    )

    allow_preprint = st.radio(
        "文献收录标准",
        ("排除预印本 (仅限正规期刊/会议)", "接受预印本 (如 arXiv)"),
    )

    start_button = st.button("开始智能检索", type="primary", use_container_width=True)

    st.divider()

    st.header("文献直读")
    sidebar_pdf = st.file_uploader(
        "上传本地 PDF 进行结构化解析",
        type="pdf",
        key="sb_pdf",
        help="跳过检索步骤，直接对已有文献生成精读报告，并保存到当前账号的历史记录",
        accept_multiple_files=True,
    )

    start_analyze_button = st.button(
        "开始解读",
        type="primary",
        key="start_analyze_btn",
        use_container_width=True,
        disabled=not sidebar_pdf,
    )


# ==========================================
# 模块 17：业务路由与检索循环
# ==========================================
if start_analyze_button and sidebar_pdf:
    reset_user_workspace_view(st.session_state.current_user)
    st.session_state.app_state = "IDLE"
    st.session_state.bottom_direct_entries = []
    st.session_state.sidebar_direct_entries = collect_pdf_entries(sidebar_pdf)
    st.rerun()

if start_button:
    if not user_topic:
        st.warning("请填写研究方向！")
    else:
        reset_user_workspace_view(st.session_state.current_user)
        seen_paper_ids.clear()
        st.session_state.sidebar_direct_entries = []
        st.session_state.bottom_direct_entries = []
        sys_prompt = get_system_prompt(user_requirements, allow_preprint)
        st.session_state.agent = LLMClient(
            sys_prompt=sys_prompt,
            model="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            max_tokens=10000,
        )
        st.session_state.prompt_history = [f"用户请求: {user_topic}"]
        st.session_state.app_state = "RUNNING"
        st.session_state.loop_count = 0
        st.session_state.has_provided_feedback = False
        st.session_state.ui_logs = []
        st.rerun()

if st.session_state.sidebar_direct_entries:
    st.markdown("---")
    st.info("正在启动【直接解析模式】，开始解构文献……")
    render_analysis_ui(st.session_state.sidebar_direct_entries)
    st.stop()

if (
    st.session_state.selected_history_report_id
    and st.session_state.app_state == "IDLE"
    and not st.session_state.bottom_direct_entries
):
    st.markdown("---")
    render_saved_history_report(st.session_state.current_user, st.session_state.selected_history_report_id)
    st.stop()

if st.session_state.app_state == "IDLE":
    st.markdown("""
### 系统使用指南

欢迎使用 AI 智能论文检索 Agent。本系统旨在通过深度信息挖掘与多轮交互，为您精准匹配最具参考价值的前沿文献。每个账号都会自动拥有独立的历史报告空间，重新登录后可直接查看之前的解析结果。为获得最佳体验，请参考以下操作规范：

**一、 智能文献检索**

1. **精准配置检索条件**
   请在左侧边栏填写宏观的“研究方向”。为进一步提升检索精度，建议在“具体筛选要求”中分点详细说明：研究的特定子领域、目标应用场景、核心算法要求或其他限制条件。您还可以根据严谨性需求，勾选是否排除预印本（如 arXiv）文献。

2. **人机协同与动态纠偏**
   首次检索完成后，系统将输出初步筛选的 6 篇高相关性候选文献。本系统支持动态调优：若结果偏离预期，您无需重新开始，只需在反馈对话框中指出理解偏差或追加新的约束条件，Agent 将据此进行下一轮定向纠偏与深度检索。

3. **会话时效管理**
   为保障系统底层计算资源的有效流转，系统在等待用户反馈时设有 30 分钟的静默超时机制。若超过此时限未收到新指令，当前检索任务将自动归档结束。

**二、 既有文献直读**

* **本地 PDF 深度解析**
  若您已有确定的目标文献，可跳过检索环节。直接通过左侧边栏底部的“上传 PDF 立即深度解读”入口提交文件，系统将自动提取文本并生成结构化的文献精读分析报告，同时写入当前账号的历史报告记录，便于下次登录直接查看。
""")

if st.session_state.app_state != "IDLE":
    st.markdown("### Agent 检索执行轨迹")
    for log in st.session_state.ui_logs:
        with st.expander(log["title"], expanded=False):
            st.markdown(log["content"])

if st.session_state.app_state == "RUNNING":
    st.info("Agent 正在自主检索并筛选文献，请稍候……")
    current_step_container = st.container()

    with st.spinner("Agent 正在思考和执行学术工具……"):
        while True:
            st.session_state.loop_count += 1
            i = st.session_state.loop_count

            if i == 1:
                loop_reminder = "系统提示: 第一次循环开始，请直接使用用户的原始研究方向作为query执行search_and_detail_papers。"
            else:
                loop_reminder = (
                    "系统提示: 请继续执行检索。如果你在对比后认为备选池中的Top 6论文已经完美符合用户的全部要求，【警告】：新query必须是纯粹同义词，严禁加入方法论关键词！"
                    "如果找齐了，请输出 Action: Finish: [推荐结果]。否则请继续 search_and_detail_papers。"
                )

            st.session_state.prompt_history.append(loop_reminder)

            output = st.session_state.agent.generate(st.session_state.prompt_history)
            st.session_state.prompt_history.append(output)

            log_entry = {
                "title": f"Agent 运行日志（第 {i} 步）",
                "content": f"**Agent 思考与决策：**\n```text\n{output}\n```",
            }
            st.session_state.ui_logs.append(log_entry)
            with current_step_container.expander(log_entry["title"], expanded=True):
                st.markdown(log_entry["content"])

            action_match = re.search(r"Action:\s*(.*)", output, re.DOTALL)
            if not action_match:
                if i >= 8:
                    st.session_state.final_result = "Agent 未返回有效 Action，请调整筛选条件后重试。"
                    st.session_state.app_state = "COMPLETED"
                    st.rerun()
                continue

            action_str = action_match.group(1).strip()

            if action_str.startswith("Finish"):
                clean_result = re.sub(r"^Finish\s*[:：\[]?\s*", "", action_str).rstrip("]").strip()
                st.session_state.final_result = clean_result
                if not st.session_state.has_provided_feedback:
                    st.session_state.app_state = "WAITING_FEEDBACK"
                    st.session_state.feedback_start_time = time.time()
                else:
                    st.session_state.app_state = "COMPLETED"
                st.rerun()
                break

            tool_match = re.search(r"(\w+)\((.*)\)", action_str)
            if tool_match:
                tool_name, args_str = tool_match.group(1), tool_match.group(2)
                raw_kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
                if tool_name in available_tools:
                    import inspect
                    allowed_keys = inspect.signature(available_tools[tool_name]).parameters.keys()
                    kwargs = {k: v for k, v in raw_kwargs.items() if k in allowed_keys}

                    if "query" in kwargs:
                        observation = available_tools[tool_name](**kwargs)
                    else:
                        observation = "错误：必须提供query参数。"
                else:
                    observation = f"错误:未定义的工具 '{tool_name}'"
            else:
                observation = "错误:Action格式无法解析。"

            st.session_state.prompt_history.append(f"Observation: {observation}")

elif st.session_state.app_state == "WAITING_FEEDBACK":
    if st.session_state.feedback_start_time:
        elapsed_time = time.time() - st.session_state.feedback_start_time
        remaining_time = 1800 - elapsed_time

        if remaining_time <= 0:
            st.session_state.app_state = "COMPLETED"
            st.rerun()
        st_autorefresh(interval=10000, key="feedback_timer")
        mins_left = int(remaining_time // 60)
        st.caption(f"系统将在 {mins_left} 分钟后自动确认结果并结束任务。")

    st.markdown("### 阶段性检索结果展示")
    st.write("请审阅 Agent 挑选出的文献，判断是否符合您的要求：")

    with st.container(border=True):
        st.markdown(st.session_state.final_result)

    st.divider()
    st.markdown("#### 您对当前的文献组合满意吗？")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("满意，结束检索", use_container_width=True):
            st.session_state.app_state = "COMPLETED"
            st.rerun()

    with col2:
        with st.popover("不满意，修改要求", use_container_width=True):
            new_req = st.text_area("请指出不符合要求的地方：")
            if st.button("提交新要求并继续"):
                if new_req.strip():
                    feedback_prompt = f"用户反馈: 【{new_req}】。请基于此继续筛选 Top 6。"
                    st.session_state.prompt_history.append(feedback_prompt)
                    st.session_state.has_provided_feedback = True
                    st.session_state.app_state = "RUNNING"
                    st.rerun()

elif st.session_state.app_state == "COMPLETED":
    st.success("文献检索任务已完成！")
    if st.session_state.has_provided_feedback == False and st.session_state.feedback_start_time:
        elapsed = time.time() - st.session_state.feedback_start_time
        if elapsed > 1800:
            st.warning("提示：由于超过 30 分钟未响应，系统已为您自动确认最终结果。")
    st.markdown("### 最终确认的 Top 6 核心论文推荐")
    with st.container(border=True):
        st.markdown(st.session_state.final_result)

    st.divider()
    st.header("开启深度解读工作流")
    st.info("从上方选定并下载任意一篇或多篇论文的 PDF，在此上传，系统将分别生成完整 7 节精读报告，并自动存入当前账号的历史报告记录。")

    uploaded_pdf = st.file_uploader("上传 PDF 文件以获取精读报告", type="pdf", key="bottom_pdf", accept_multiple_files=True)
    bottom_start_btn = st.button(
        "开始深度解读",
        type="primary",
        disabled=not uploaded_pdf,
        use_container_width=True,
    )
    if bottom_start_btn and uploaded_pdf:
        reset_user_workspace_view(st.session_state.current_user)
        st.session_state.bottom_direct_entries = collect_pdf_entries(uploaded_pdf)
        st.rerun()

    if st.session_state.bottom_direct_entries:
        st.markdown("---")
        render_analysis_ui(st.session_state.bottom_direct_entries)

    if st.button("开启全新检索轮次", type="primary"):
        current_user = st.session_state.get("current_user", "")
        st.session_state.clear()
        if current_user:
            st.session_state.current_user = current_user
        st.rerun()
