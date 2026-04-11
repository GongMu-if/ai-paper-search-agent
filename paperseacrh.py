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
from io import BytesIO
from typing import Any, Dict, List, Tuple, Optional

import requests
import streamlit as st
from openai import OpenAI

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

# 检索时避免重复论文
seen_paper_ids = set()

# 固定的报告章节规范：后续会逐节生成，避免一次生成过长被截断
CORE_SECTION_SPECS = [
    "## 1. 研究问题与核心贡献",
    "## 2. 背景、研究缺口与前人路线",
    "## 3. 方法总览与整体数据流",
    "## 4. 关键模块逐层机制剖析",
    "## 5. 实验设计、关键证据与论点验证",
    "## 6. 复现要点与方法适用边界",
    "## 7. 局限性与未解决问题",
]
RESEARCH_SECTION_SPEC = "## 8. 面向后续研究的可执行创新路线"

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
# 模块 4：检索 Agent Prompt
# ==========================================
def get_system_prompt(requirements, preprint_rule):
    """构造检索 Agent 的系统提示词。"""
    if preprint_rule == "仅限同行评审文献 (排除预印本)":
        preprint_prompt = "严禁选择 Venue 为 'Unknown Venue/Preprint' 的预印本论文。"
    else:
        preprint_prompt = "可以接受预印本论文。"

    return f"""
你是一个科研论文搜索专家。你的任务是根据研究方向，在近一年内的论文中筛选六篇。

# 工作流程
1. 通过 search_and_detail_papers 工具获得相关论文的标题、摘要和 Introduction。
2. 将符合要求的记录在 Thought 中作为“备选池”累加。
3. 学习同义词或近义词，作为下一次查询词。【警告】：严禁直接加入用户要求中的关键词。
4. 如果论文不符合或没有摘要：直接摒弃。
5. 最终选出最好的六篇。

# 用户要求
{requirements}
{preprint_prompt}

# 输出格式
Thought: [你的思考逻辑]
Action: [执行工具或结束]

Action 格式支持：
1. search_and_detail_papers(query="关键词")
2. Finish:
使用 Markdown 直接列出最终 6 篇论文（标题、Venue、DOI、推荐理由）。
不要多余说明。
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
                f"  - Abstract: {final_abstract[:800]}..."
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


def extract_local_context(md_content: str, image_id: str, window: int = 1800) -> str:
    """为单张图片提取局部上下文，供 Vision Agent 使用。"""
    idx = md_content.find(image_id)
    if idx == -1:
        return build_source_pack(md_content, max_chars=5000)

    start = max(0, idx - window)
    end = min(len(md_content), idx + len(image_id) + window)

    prefix = md_content[:idx]
    headings = re.findall(r'^(#{1,6}\s+.+)$', prefix, flags=re.MULTILINE)
    nearest_headings = "\n".join(headings[-3:]) if headings else "未找到最近章节标题"

    local = md_content[start:end]
    return f"【最近章节】\n{nearest_headings}\n\n【图像附近原文】\n{local}"


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
                st.image(base64.b64decode(b64), caption=caption, use_container_width=True)
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
    def __init__(self, sys_prompt, model="deepseek-chat", api_key="", base_url="", max_tokens=6000):
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
    1) 第 1-7 节逐节生成
    2) 第 8 节单独生成
    3) 若缺节，则按缺失章节补生成
    4) 经过审校后再做修订
    """
    base_context = build_analysis_context(md_content, text_report, vision_summaries, image_ids)

    main_agent = LLMClient(
        sys_prompt=MAIN_AGENT_PROMPT,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        max_tokens=4500,
    )

    research_agent = LLMClient(
        sys_prompt=RESEARCH_AGENT_PROMPT,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        max_tokens=3500,
    )

    auditor = LLMClient(
        sys_prompt=REPORT_AUDITOR_PROMPT,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        max_tokens=2000,
    )

    sections = ["# 论文全维度深度透视报告"]
    for spec in CORE_SECTION_SPECS:
        sections.append(generate_section(main_agent, base_context, spec))

    research_prompt = f"""
{base_context}

【已生成的第1-7节主报告】
{chr(10).join(sections)}

请只输出：
{RESEARCH_SECTION_SPEC}
"""
    section8 = research_agent.generate([research_prompt]).strip()
    if RESEARCH_SECTION_SPEC not in section8:
        section8 = RESEARCH_SECTION_SPEC + "\n" + section8
    sections.append(section8)

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
必须保留完整 8 个章节标题。
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
    r'^(?:表|table)\s*\d+(?:\s*[：:.-]|(?:\s+[-–—]\s+)|\s+).+',
    flags=re.I,
)
INLINE_MARKUP_TOKEN_RE = re.compile(
    r'(\*\*[^\n]+?\*\*|__[^\n]+?__|\$[^$\n]+\$|\\\([^\n]+?\\\)|\\\[[^\n]+?\\\]|`[^`\n]+`)',
    flags=re.S,
)
ASCII_TEXT_RE = re.compile(r'([A-Za-z0-9\.\,\:\;\-\+\=\(\)\/_%#@\[\]\{\}<>\'\"\s]+)')


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


def is_table_title_line(text: str) -> bool:
    return bool(TABLE_TITLE_LINE_RE.match(normalize_table_title_line(text)))


def looks_like_formula_text(text: str) -> bool:
    candidate = extract_formula_text(text)
    if not candidate:
        return False

    explicit_tokens = [
        r'\frac', r'\sum', r'\prod', r'\sqrt', r'\alpha', r'\beta', r'\gamma',
        r'\theta', r'\lambda', r'\mu', r'\sigma', r'\omega', r'\mathbf',
        r'\mathrm', r'\mathcal', r'\mathbb', r'\hat', r'\tilde', r'\left',
        r'\right', r'\begin', r'\end', r'\log', r'\exp', r'\arg', r'\max',
        r'\min',
    ]
    if any(token in candidate for token in explicit_tokens):
        return True

    operator_count = sum(ch in candidate for ch in '=<>^_\\+-*/')
    has_letter = bool(re.search(r'[A-Za-zΑ-Ωα-ω]', candidate))
    return has_letter and operator_count >= 1


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

    expr = re.sub(r'\\begin\{[^{}]+\}', '', expr)
    expr = re.sub(r'\\end\{[^{}]+\}', '', expr)
    expr = expr.replace('&', ' ')
    expr = expr.replace('\\\\', ' ')
    expr = expr.replace(r'\displaystyle', '')
    expr = expr.replace('−', '-').replace('–', '-').replace('—', '-')
    expr = re.sub(r'\\label\{.*?\}', '', expr)
    expr = re.sub(r'\\tag\{.*?\}', '', expr)
    expr = re.sub(r'\\nonumber\b', '', expr)
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
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.font_manager import FontProperties
        from matplotlib.mathtext import math_to_image

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


def wrap_plain_text_for_paragraph(text: str) -> str:
    escaped = html.escape(text)
    if not escaped:
        return ''

    parts = []
    for chunk in ASCII_TEXT_RE.split(escaped):
        if not chunk:
            continue
        if ASCII_TEXT_RE.fullmatch(chunk):
            parts.append(f'<font name="Times-Roman">{chunk}</font>')
        else:
            parts.append(f'<font name="STSong-Light">{chunk}</font>')
    return ''.join(parts)


def inline_math_markup(formula_text: str, asset_ctx: Dict[str, Any], font_size: float = 12.0) -> str:
    asset = render_formula_image(formula_text, asset_ctx, font_size=font_size, display=False)
    if not asset:
        fallback = extract_formula_text(formula_text)
        return f'<font name="DejaVuSans">{html.escape(fallback)}</font>'

    return (
        f'<img src="{html.escape(asset["path"], quote=True)}" '
        f'width="{asset["width"]:.2f}" height="{asset["height"]:.2f}" valign="middle"/>'
    )


def mixed_inline_markup(text: str, asset_ctx: Optional[Dict[str, Any]] = None) -> str:
    """
    渲染 ReportLab Paragraph 可识别的行内富文本：
    - 普通文本保留中英文字体切换
    - **bold** / __bold__ 会转为真正的加粗样式
    - $...$ / \\(...\\) / `公式` 会优先渲染为公式图片
    - 非公式代码仍使用等宽字体展示
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
            parts.append(wrap_plain_text_for_paragraph(value[start:match.start()]))

        token = match.group(0)
        if token.startswith('**') and token.endswith('**'):
            parts.append(f'<b>{mixed_inline_markup(token[2:-2], asset_ctx)}</b>')
        elif token.startswith('__') and token.endswith('__'):
            parts.append(f'<b>{mixed_inline_markup(token[2:-2], asset_ctx)}</b>')
        elif token.startswith('$') and token.endswith('$'):
            parts.append(inline_math_markup(token[1:-1], asset_ctx))
        elif token.startswith(r'\(') and token.endswith(r'\)'):
            parts.append(inline_math_markup(token[2:-2], asset_ctx))
        elif token.startswith(r'\[') and token.endswith(r'\]'):
            parts.append(inline_math_markup(token[2:-2], asset_ctx))
        elif token.startswith('`') and token.endswith('`'):
            inner = token[1:-1]
            if looks_like_formula_text(inner):
                parts.append(inline_math_markup(inner, asset_ctx))
            else:
                parts.append(f'<font name="DejaVuSansMono">{html.escape(inner)}</font>')
        start = match.end()

    if start < len(value):
        parts.append(wrap_plain_text_for_paragraph(value[start:]))

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
    - 普通段落：空行分隔
    """
    text = normalize_report_markdown(md_text)
    lines = text.split("\n")
    blocks: List[Tuple[str, object]] = []

    paragraph_buffer: List[str] = []

    def flush_paragraph():
        if paragraph_buffer:
            p = " ".join([x.strip() for x in paragraph_buffer if x.strip()]).strip()
            if p:
                blocks.append(("paragraph", p))
            paragraph_buffer.clear()

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


# ==========================================
# 模块 11：Flowable 构建
# ==========================================
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
    norm_left = normalize_compare_text(left)
    norm_right = normalize_compare_text(right)
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
            fontName='DejaVuSans',
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
                suppress_caption = captions_equivalent(caption, pending_table_title)
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
# 模块 13：论文精读主流程
# ==========================================
def get_pdf_cache_key(pdf_bytes: bytes) -> str:
    """为每篇论文生成稳定缓存键，支持多文件分别缓存。"""
    return hashlib.sha256(pdf_bytes).hexdigest()


def build_analysis_result(pdf_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    单篇论文的完整分析流程：
    1. 远端解析 PDF
    2. Text Agent 输出 FACT_BANK + 文本综述
    3. Vision Agent 输出 FIGURE_CARD
    4. 主报告逐节生成
    5. 服务器端生成高清 PDF
    """
    result = analyze_pdf_with_modal(pdf_bytes)
    if not (result and result.get("status") == "success"):
        return None

    md_content = result["markdown"]
    ordered_images = sort_images_by_doc_order(md_content, result.get("images", {}))
    images_dict = dict(ordered_images)

    with st.spinner("文本专家正在精读全篇文本……"):
        text_agent = LLMClient(
            sys_prompt=TEXT_AGENT_PROMPT,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            max_tokens=6000,
        )
        text_report = text_agent.generate([f"请详尽解析此论文：\n{md_content}"])

    vision_summaries = ""
    if images_dict:
        with st.spinner(f"视觉专家正在分析 {len(images_dict)} 张关键图表……"):
            vision_agent = LLMClient(
                sys_prompt=VISION_AGENT_PROMPT,
                model="qwen3.6-plus",
                api_key=QWEN_API_KEY,
                base_url=QWEN_BASE_URL,
                max_tokens=3000,
            )
            cards = []
            for name, b64 in ordered_images:
                local_context = extract_local_context(md_content, name)
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
        final_main_report = normalize_report_markdown(report)

    if is_report_truncated(final_main_report):
        with st.spinner("检测到报告可能截断，正在补全缺失内容……"):
            report = generate_full_report(
                md_content=md_content,
                text_report=text_report,
                vision_summaries=vision_summaries,
                image_ids=list(images_dict.keys()),
            )
            final_main_report = normalize_report_markdown(report)

    with st.spinner("正在服务器端排版并生成高清 PDF……"):
        final_pdf_bytes = build_pdf_bytes_from_markdown(final_main_report, images_dict)

    return {
        "source_markdown": md_content,
        "text_report": text_report,
        "vision_summaries": vision_summaries,
        "images": images_dict,
        "main_report": final_main_report,
        "pdf_bytes": final_pdf_bytes,
    }


def get_or_create_analysis_result(pdf_bytes: bytes) -> Tuple[str, Optional[Dict[str, Any]]]:
    """读取或创建单篇论文分析结果，支持多文件分别缓存。"""
    cache_key = get_pdf_cache_key(pdf_bytes)
    cache_pool = st.session_state.analysis_results
    if cache_key in cache_pool and cache_pool[cache_key] is not None:
        return cache_key, cache_pool[cache_key]

    analysis_result = build_analysis_result(pdf_bytes)
    if analysis_result is not None:
        cache_pool[cache_key] = analysis_result
    else:
        cache_pool.pop(cache_key, None)
    return cache_key, analysis_result


def build_export_filename(source_name: str, suffix: str) -> str:
    base_name = re.sub(r'(?i)\.pdf$', '', (source_name or '').strip())
    base_name = re.sub(r'[\\/:*?"<>|]+', '_', base_name).strip() or '论文'
    return f"{base_name}{suffix}"


def render_single_analysis_result(
    analysis_result: Dict[str, Any],
    cache_key: str,
    source_name: str,
    show_paper_title: bool = False,
):
    if show_paper_title:
        st.markdown(f"### {source_name}")
    st.success("论文深度透视报告已生成！")
    render_report_with_images(analysis_result["main_report"], analysis_result["images"])

    st.divider()
    st.markdown("### 导出与下载")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "下载报告原文（Markdown）",
            analysis_result["main_report"],
            file_name=build_export_filename(source_name, "_论文全维度深度透视报告.md"),
            mime="text/markdown",
            use_container_width=True,
            key=f"download_md_{cache_key}",
        )
    with col2:
        st.download_button(
            "下载高清 PDF 报告",
            analysis_result["pdf_bytes"],
            file_name=build_export_filename(source_name, "_论文全维度深度透视报告.pdf"),
            mime="application/pdf",
            use_container_width=True,
            key=f"download_pdf_{cache_key}",
        )


def render_analysis_ui(pdf_inputs):
    """
    上传论文后的主工作流：
    - 支持单篇 PDF 分析
    - 支持多篇 PDF 同时上传，并分别生成各自报告与 PDF
    """
    entries: List[Tuple[str, bytes]] = []

    if isinstance(pdf_inputs, bytes):
        entries.append(("论文.pdf", pdf_inputs))
    else:
        for idx, uploaded in enumerate(pdf_inputs or [], start=1):
            if uploaded is None:
                continue
            paper_name = getattr(uploaded, 'name', f'paper_{idx}.pdf')
            if hasattr(uploaded, 'getvalue'):
                entries.append((paper_name, uploaded.getvalue()))
            else:
                entries.append((paper_name, uploaded))

    if not entries:
        return

    multi_mode = len(entries) > 1
    for idx, (paper_name, pdf_bytes) in enumerate(entries, start=1):
        if multi_mode:
            st.markdown(f"## 第 {idx} 篇论文")
        cache_key, analysis_result = get_or_create_analysis_result(pdf_bytes)
        if analysis_result:
            render_single_analysis_result(
                analysis_result=analysis_result,
                cache_key=cache_key,
                source_name=paper_name,
                show_paper_title=multi_mode,
            )
        if multi_mode and idx < len(entries):
            st.divider()


# ==========================================
# 模块 14：前端 UI
# ==========================================
st.title("AI 智能论文检索 Agent")
st.markdown("基于大模型的多轮深度挖掘，为您精准匹配 Top 6 核心前沿文献。")

with st.sidebar:
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
# 模块 15：状态初始化
# ==========================================
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
if "final_pdf_bytes" not in st.session_state:
    st.session_state.final_pdf_bytes = None
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


# ==========================================
# 模块 16：业务路由与检索循环
# ==========================================
if start_analyze_button and sidebar_pdf:
    st.markdown("---")
    st.info("正在启动【直接解析模式】，开始解构文献……")
    render_analysis_ui(sidebar_pdf)
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
            max_tokens=2500,
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
    st.info("Agent 正在自主检索并筛选文献，请稍候……")
    current_step_container = st.container()

    with st.spinner("Agent 正在思考和执行学术工具……"):
        while True:
            st.session_state.loop_count += 1
            i = st.session_state.loop_count

            loop_reminder = "系统提示：正在执行检索……" if i > 1 else "系统提示：第一次循环开始……"
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
                st.session_state.app_state = "WAITING_FEEDBACK" if not st.session_state.has_provided_feedback else "COMPLETED"
                st.rerun()
                break

            tool_match = re.search(r"(\w+)\((.*)\)", action_str)
            if tool_match:
                tool_name, args_str = tool_match.group(1), tool_match.group(2)
                raw_kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
                observation = available_tools[tool_name](**raw_kwargs) if tool_name in available_tools else "Observation: 工具不存在。"
                st.session_state.prompt_history.append(observation)

            if i >= 10:
                st.session_state.final_result = "Agent 超过最大迭代轮数，已停止本轮检索。你可以修改要求后重新检索。"
                st.session_state.app_state = "COMPLETED"
                st.rerun()
                break

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
            new_req = st.text_area("指出不符合要求的地方 / 添加新约束：")
            if st.button("提交纠偏指令"):
                st.session_state.prompt_history.append(f"用户反馈: {new_req}")
                st.session_state.has_provided_feedback = True
                st.session_state.app_state = "RUNNING"
                st.rerun()

elif st.session_state.app_state == "COMPLETED":
    st.success("文献检索任务已完成！")
    st.markdown("### 最终确认的 Top 6 核心论文推荐")
    with st.container(border=True):
        st.markdown(st.session_state.final_result)

    st.divider()
    st.header("开启深度解读工作流")
    st.info("从上方选定并下载任意一篇或多篇论文的 PDF，在此上传，系统将分别生成完整 8 节精读报告，并导出各自的高清 PDF。")

    uploaded_pdf = st.file_uploader("上传 PDF 文件以获取精读报告", type="pdf", key="bottom_pdf", accept_multiple_files=True)
    bottom_start_btn = st.button(
        "开始深度解读",
        type="primary",
        disabled=not uploaded_pdf,
        use_container_width=True,
    )
    if bottom_start_btn and uploaded_pdf:
        render_analysis_ui(uploaded_pdf)

    if st.button("开启全新检索轮次", type="primary"):
        st.session_state.clear()
        st.rerun()
