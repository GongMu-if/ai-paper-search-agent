# ==========================================
# 模块 1: 依赖导入与页面基础配置
# ==========================================

import re
import html as html_lib
import time
import base64
import datetime
from io import BytesIO
from string import Template
from typing import Dict, List, Tuple, Optional

import requests
import streamlit as st
from openai import OpenAI
from PIL import Image
from bs4 import BeautifulSoup, Tag
from weasyprint import HTML
import mistune

# 设置 Streamlit 页面基础信息。
st.set_page_config(page_title="AI 论文检索 Agent", page_icon="📚", layout="wide")


# ==========================================
# 模块 2: 全局变量与 API 密钥配置
# ==========================================

# DeepSeek：用于文本抽取、主报告撰写、研究路线设计、审校。
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Qwen：用于多模态图表分析。
QWEN_API_KEY = st.secrets["QWEN_API_KEY"]
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Modal 云端 PDF 结构解析服务。
MODAL_API_URL = st.secrets["MODAL_API_URL"]

# 用于论文检索阶段去重，避免同一轮中重复读到同一篇论文。
seen_paper_ids = set()

# PDF 导出参数。
# 这些值是本版本针对“图偏大、页首文字裁切、表格分页不稳、留白偏多”重新调过的一组更稳妥的默认值。
PDF_EXPORT_CONFIG = {
    # 正文版芯宽度。略窄于整页可用宽度，可给表格和图留出呼吸感。
    "content_width_mm": 172,
    # A4 页边距。
    "margin_top_mm": 12,
    "margin_right_mm": 14,
    "margin_bottom_mm": 12,
    "margin_left_mm": 14,
    # 正文字号与行高。略紧凑，减少页间稀疏感。
    "body_font_size_px": 13.5,
    "body_line_height": 1.80,
    # 普通图尺寸：进一步缩小，避免架构图过分霸占页面。
    "figure_max_width_pct": 64,
    "figure_max_height_mm": 92,
    # 宽图尺寸：适当放大横向图，但仍控制在稳妥范围内。
    "wide_visual_max_width_pct": 82,
    "wide_visual_max_height_mm": 102,
    # 表格型图片：尽量接近整页宽度，方便一页完整显示。
    "table_visual_max_width_pct": 96,
    "table_visual_max_height_mm": 236,
    # 长图尺寸：收窄宽度，避免高图把一页切得很碎。
    "tall_visual_max_width_pct": 60,
    "tall_visual_max_height_mm": 150,
    # 真 Markdown 表格的字号和内边距。
    "table_font_size_px": 10,
    "table_cell_padding_px": 4,
}

# 章节生成与审校相关配置。
MAX_SECTION_RETRY = 2
MAX_AUDIT_ROUNDS = 2


# ==========================================
# 模块 3: 核心 Agent 提示词库
# ==========================================

# 主报告 Agent：只负责第 1-7 节事实型主报告。
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
4. 你可以提出研究延伸，但只能放在最后一节，并明确说明“这是基于本文机制的研究设想，不是原文结论”。
5. 每个章节必须由多个自然段组成，不能把整章写成一个大段。
6. 每次解释方法或实验时，都要回答四个问题：它要解决什么问题、具体怎么做、证据是什么、边界在哪里。
7. 插图必须使用 Markdown 图片语法，且图片占位符必须从提供的图片ID列表中原样复制，格式为：
   ![图X：学术化图注](图片ID)
8. 表格前后各保留一个空行；表标题必须单独作为一行。
9. 语言风格保持学术、克制、清晰，不做口语化渲染。
10. 当用户只要求你输出某一节或某几节时，你必须只输出对应部分，不要补出其他章节。
11. 请不要省略后续章节，不要输出“略”或“同理可得”。

【默认报告结构】
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

# 文本 Agent：只做事实抽取，不做 speculative 创新扩展。
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

# 视觉 Agent：把每张图或表转成可调用的证据卡。
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

# 研究路线 Agent：只负责第 8 节，不污染事实层。
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
5. 若信息不足，必须明确写“原文未提供足够证据，以下为机制导向的研究设想”。

请只输出：
## 8. 面向后续研究的可执行创新路线
"""

# 审校 Agent：只做核查，不直接改写整篇。
REPORT_AUDITOR_PROMPT = """
你是学术报告审校员。请检查以下报告是否存在：
1. 不被原始 markdown / FACT_BANK / FIGURE_CARD 支持的结论
2. 不存在的图片ID或错误图片占位符
3. 漏掉的关键模块、关键实验、关键局限
4. 将研究设想误写为原文结论
5. 同一章节内容重复、前后矛盾或与图表证据不一致
6. 缺失第1-8节中的任意一节

输出格式：
RESULT: PASS 或 FAIL
ISSUES:
- 若无问题，写“无重大问题”
- 若有问题，逐条列出
"""


# ==========================================
# 模块 4: 论文检索系统提示词
# ==========================================


def get_system_prompt(requirements: str, preprint_rule: str) -> str:
    """根据用户筛选要求，生成论文检索 Agent 的系统提示词。"""
    if preprint_rule == "仅限同行评审文献 (排除预印本)":
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
# 模块 5: 论文检索工具函数
# ==========================================


def reconstruct_abstract(inverted_index: dict) -> str:
    """把 OpenAlex 的倒排摘要恢复为自然文本。"""
    if not inverted_index:
        return ""
    word_index = [(pos, word) for word, positions in inverted_index.items() for pos in positions]
    word_index.sort(key=lambda x: x[0])
    return " ".join([word for _, word in word_index])



def search_and_detail_papers(query: str) -> str:
    """调用 Semantic Scholar / OpenAlex 搜索近一年论文，并补全摘要。"""
    global seen_paper_ids
    api_key = st.secrets["S2_API_KEY"]
    email = "gaoym3@mails.neu.edu.cn"
    current_year = datetime.datetime.now().year
    start_year = max(current_year - 1, 2024)

    s2_url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={query}&limit=100&year={start_year}-{current_year}"
        "&fields=paperId,title,abstract,year,externalIds,venue"
    )
    headers = {"x-api-key": api_key}

    try:
        # 轻微 sleep，降低被上游接口限流的概率。
        time.sleep(2)
        s2_response = requests.get(s2_url, headers=headers, timeout=20)
        s2_response.raise_for_status()
        papers = s2_response.json().get("data", [])

        if not papers:
            return f"Observation: 未找到关于'{query}'的近一年论文。"

        results = []
        for paper in papers:
            paper_id = paper.get("paperId", "No ID")
            if paper_id in seen_paper_ids:
                continue

            title = paper.get("title", "No Title")
            doi = paper.get("externalIds", {}).get("DOI", "")
            venue = paper.get("venue") or "Unknown Venue/Preprint"

            final_abstract = ""
            openalex_mark = ""
            s2_abstract = paper.get("abstract")

            if s2_abstract and s2_abstract.strip():
                final_abstract = s2_abstract.strip()
            else:
                # 若 Semantic Scholar 没摘要，则尝试从 OpenAlex 补全。
                try:
                    if doi:
                        oa_url = f"https://api.openalex.org/works/https://doi.org/{doi}"
                    else:
                        oa_url = f"https://api.openalex.org/works?filter=title.search:{title}"
                    oa_res = requests.get(oa_url, params={"mailto": email}, timeout=20)
                    if oa_res.status_code == 200:
                        oa_json = oa_res.json()
                        work_data = oa_json.get("results", [None])[0] if "results" in oa_json else oa_json
                        if work_data and work_data.get("abstract_inverted_index"):
                            final_abstract = reconstruct_abstract(work_data["abstract_inverted_index"])
                            openalex_mark = " [via OpenAlex]"
                except Exception:
                    pass

            # 没有可用摘要时，直接丢弃。
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


# 检索阶段目前只开放一个工具。
available_tools = {"search_and_detail_papers": search_and_detail_papers}


# ==========================================
# 模块 6: Markdown / 图片 / HTML / PDF 通用工具
# ==========================================

# Mistune Markdown 渲染器。
# 这里启用 table 插件，保证 Markdown 表格能稳定转成 HTML 表格。
MARKDOWN_RENDERER = mistune.create_markdown(
    renderer=mistune.HTMLRenderer(escape=False),
    plugins=["table", "strikethrough"],
)

# 章节规范。
# 每一节都单独生成，以避免长输出被模型截断。
SECTION_SPECS = {
    1: {
        "title": "研究问题与核心贡献",
        "task": "定义本文试图解决的核心问题，准确概括相对前人工作的主要创新，并说明这些创新分别改变了哪一个技术瓶颈。",
        "min_chars": 180,
    },
    2: {
        "title": "背景、研究缺口与前人路线",
        "task": "还原研究背景、主流路线及其局限，说明本文的问题为什么值得解决，以及它切入的位置在哪里。",
        "min_chars": 220,
    },
    3: {
        "title": "方法总览与整体数据流",
        "task": "结合原始文本和图表证据，说明系统从输入到输出的完整链路。若有总架构图，应在这里插入。",
        "min_chars": 220,
    },
    4: {
        "title": "关键模块逐层机制剖析",
        "task": "按照模型真实工作顺序拆解关键模块，说明输入、变换、必要性、耦合关系与预期改善问题。若有模块结构图，应在对应段落处插入。",
        "min_chars": 260,
    },
    5: {
        "title": "实验设计、关键证据与论点验证",
        "task": "交代数据集、评价指标、对照组、主实验和消融实验。每写一个结论，都要明确指出是哪组结果支持它，并解释这项结果验证了哪条方法主张。",
        "min_chars": 260,
    },
    6: {
        "title": "复现要点与方法适用边界",
        "task": "总结复现时最不能忽视的输入条件、训练设置、模块依赖和评测前提，并说明方法适用于什么情形、不适用于什么情形。",
        "min_chars": 180,
    },
    7: {
        "title": "局限性与未解决问题",
        "task": "区分作者明确承认的局限，以及从实验设计中可以直接看出的未解决问题，但后者也必须基于论文证据。",
        "min_chars": 180,
    },
    8: {
        "title": "面向后续研究的可执行创新路线",
        "task": "给出 3 到 5 条可执行研究路线，每条都要说明缺口、关联模块、可改造方案、预期收益、验证方式和技术风险。",
        "min_chars": 220,
    },
}



def infer_image_mime(b64_data: str) -> str:
    """根据 base64 前缀推断图片 MIME 类型。"""
    if b64_data.startswith("/9j/"):
        return "image/jpeg"
    if b64_data.startswith("iVBOR"):
        return "image/png"
    if b64_data.startswith("UklGR"):
        return "image/webp"
    return "image/png"



def strip_code_fences(text: str) -> str:
    """移除大模型常见的 ```markdown ... ``` 包裹。"""
    if not text:
        return ""
    text = text.strip()
    match = re.match(r"^```(?:markdown|md)?\s*(.*?)\s*```$", text, flags=re.S | re.I)
    return match.group(1).strip() if match else text



def normalize_markdown_tables(md_text: str) -> str:
    """保证 Markdown 表格前后有空行，避免渲染器把表格吃坏。"""
    lines = md_text.splitlines()
    normalized = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and stripped.endswith("|")

        if is_table_line and not in_table:
            if normalized and normalized[-1].strip():
                normalized.append("")
            in_table = True

        if not is_table_line and in_table:
            if normalized and normalized[-1].strip():
                normalized.append("")
            in_table = False

        normalized.append(line)

    if in_table and normalized and normalized[-1].strip():
        normalized.append("")

    return "\n".join(normalized)



def normalize_report_markdown(md_text: str) -> str:
    """把最终报告 Markdown 规范化，方便前端和 PDF 共用。"""
    text = strip_code_fences(md_text)
    # 兼容旧式 [REF_IMG: xxx] 占位符，统一转成 Markdown 图片语法。
    text = re.sub(r"\[REF_IMG:\s*(.*?)\]", r"![\1](\1)", text)
    # 确保图片前后都有空行。
    text = re.sub(r"[ \t]*(!\[[^\]]*\]\([^\)]+\))[ \t]*", r"\n\n\1\n\n", text)
    text = normalize_markdown_tables(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()



def build_source_pack(md_content: str, max_chars: int = 52000) -> str:
    """对超长论文 markdown 做截断保留，但尽量按章节保留完整块。"""
    md_content = md_content.strip()
    if len(md_content) <= max_chars:
        return md_content

    sections = re.split(r"(?=^#{1,6}\s)", md_content, flags=re.M)
    collected = []
    total = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue
        chunk = section[:9000]
        if total + len(chunk) > max_chars:
            break
        collected.append(chunk)
        total += len(chunk)

    return "\n\n".join(collected).strip() if collected else md_content[:max_chars]



def sort_images_by_doc_order(md_content: str, images_dict: Dict[str, str]) -> List[Tuple[str, str]]:
    """按图片 ID 在论文 markdown 中出现的先后顺序排序图片，便于视觉分析按文档顺序进行。"""
    def sort_key(item: Tuple[str, str]) -> int:
        position = md_content.find(item[0])
        return position if position >= 0 else 10 ** 9

    return sorted(images_dict.items(), key=sort_key)



def extract_local_context(md_content: str, image_id: str, window: int = 1800) -> str:
    """提取图片附近的局部原文上下文，增强 Vision Agent 对图表作用的判断能力。"""
    index = md_content.find(image_id)
    if index == -1:
        return build_source_pack(md_content, max_chars=6000)

    start = max(0, index - window)
    end = min(len(md_content), index + len(image_id) + window)

    prefix = md_content[:index]
    headings = re.findall(r"^(#{1,6}\s+.+)$", prefix, flags=re.MULTILINE)
    nearest_headings = "\n".join(headings[-3:]) if headings else "未找到章节标题"
    local_snippet = md_content[start:end]

    return f"""【最近章节】
{nearest_headings}

【图像附近原文】
{local_snippet}
"""



def find_matching_image_key(img_key: str, images_dict: Dict[str, str]) -> Optional[str]:
    """兼容图片 ID 的模糊匹配，避免模型输出和真实 ID 有少量差异时找不到图。"""
    if img_key in images_dict:
        return img_key
    for img_name in images_dict:
        if img_key in img_name or img_name in img_key:
            return img_name
    return None



def previous_nonempty_element_sibling(node: Tag) -> Optional[Tag]:
    """找到前一个非空白元素兄弟节点。"""
    sibling = node.previous_sibling
    while sibling is not None:
        if isinstance(sibling, Tag):
            return sibling
        if getattr(sibling, "strip", None) and sibling.strip():
            return None
        sibling = sibling.previous_sibling
    return None



def image_size_from_base64(b64_data: str) -> Tuple[int, int]:
    """读取 base64 图片尺寸，供图像分类时使用。"""
    try:
        image_bytes = base64.b64decode(b64_data)
        with Image.open(BytesIO(image_bytes)) as img:
            return img.size
    except Exception:
        return (0, 0)



def build_figure_html(alt_text: str, b64_data: str) -> str:
    """把 Markdown 图片占位符替换成带尺寸分类信息的 figure HTML。"""
    mime_type = infer_image_mime(b64_data)
    safe_alt = html_lib.escape(alt_text)

    width, height = image_size_from_base64(b64_data)
    ratio = (width / height) if width and height else 1.0

    classes = ["pdf-figure"]
    # 若图注写的是“表X”或“Table X”，则视为表格型图片。
    if re.match(r"^\s*(表|table)\s*\d+", alt_text, flags=re.I):
        classes.append("table-like")
    elif ratio >= 1.45:
        classes.append("wide-visual")
    elif ratio <= 0.85:
        classes.append("tall-visual")

    return (
        "\n"
        f'<div class="{" ".join(classes)}">'
        f'<img src="data:{mime_type};base64,{b64_data}" alt="{safe_alt}" />'
        f'<div class="img-caption">{safe_alt}</div>'
        "</div>\n"
    )



def embed_base64_images(md_text: str, images_dict: Dict[str, str]) -> str:
    """把报告中的图片占位符替换成真正的内嵌 base64 图像。"""

    def replace_markdown_img(match: re.Match) -> str:
        alt_text = match.group(1).strip()
        img_placeholder = match.group(2).strip()
        matched_key = find_matching_image_key(img_placeholder, images_dict)
        if matched_key:
            return build_figure_html(alt_text, images_dict[matched_key])
        return match.group(0)

    def replace_ref_img(match: re.Match) -> str:
        img_placeholder = match.group(1).strip()
        matched_key = find_matching_image_key(img_placeholder, images_dict)
        if matched_key:
            return build_figure_html(img_placeholder, images_dict[matched_key])
        return match.group(0)

    md_text = re.sub(r"!\[(.*?)\]\((.*?)\)", replace_markdown_img, md_text)
    md_text = re.sub(r"\[REF_IMG:\s*(.*?)\]", replace_ref_img, md_text)
    return md_text



def markdown_to_html(md_text: str) -> str:
    """把 Markdown 转成 HTML。"""
    return MARKDOWN_RENDERER(md_text)



def wrap_tables_and_pair_captions(soup: BeautifulSoup) -> None:
    """把 HTML 表格包装成 table-block，并把前一行表标题吸附到表格上方。"""
    for table in soup.find_all("table"):
        if table.parent and isinstance(table.parent, Tag) and "table-wrapper" in (table.parent.get("class") or []):
            continue

        wrapper = soup.new_tag("div", attrs={"class": "table-wrapper"})
        block = soup.new_tag("div", attrs={"class": "table-block"})

        table.wrap(wrapper)
        wrapper.wrap(block)

        prev = previous_nonempty_element_sibling(block)
        if prev and prev.name == "p":
            caption_text = prev.get_text(" ", strip=True)
            if re.match(r"^\s*(表|table)\s*\d+[:：]", caption_text, flags=re.I):
                classes = prev.get("class", [])
                prev["class"] = classes + ["table-caption"]
                prev.extract()
                block.insert(0, prev)



def defer_visuals_by_section(soup: BeautifulSoup) -> None:
    """
    把图表延后到当前节更靠后的位置。

    这样做的目的是：
    - 让文字尽量优先填满页面；
    - 当图表在页末放不下时，文字仍然可以先排进去；
    - 图表可以顺延到下一页，而不是强制和前一段文字紧贴，减少大块空白。
    """
    root = soup.find("div", attrs={"class": "report-shell"})
    if not root:
        return

    children = list(root.children)
    new_children = []
    pending_visuals = []

    def is_section_boundary(node) -> bool:
        return isinstance(node, Tag) and node.name in {"h2", "h3"}

    def is_visual_node(node) -> bool:
        if not isinstance(node, Tag):
            return False
        node_classes = set(node.get("class", []))
        return "pdf-figure" in node_classes or "table-block" in node_classes

    def flush_pending() -> None:
        nonlocal pending_visuals
        new_children.extend(pending_visuals)
        pending_visuals = []

    for node in children:
        if is_section_boundary(node) and new_children:
            flush_pending()
            new_children.append(node.extract())
            continue

        if is_visual_node(node):
            pending_visuals.append(node.extract())
            continue

        new_children.append(node.extract() if isinstance(node, Tag) else node)

    flush_pending()
    root.clear()
    for child in new_children:
        root.append(child)



def build_pdf_html_document(report_md: str, images_dict: Dict[str, str]) -> str:
    """
    生成服务器端 PDF 使用的 HTML 文档。

    这里不再走 html2pdf 的浏览器截图路径，而是直接把 HTML 交给 WeasyPrint。
    好处是：
    - 文本是真实文字，不会在分页处被截图裁切；
    - 分页规则更稳定；
    - 英文/数字与中文混排时字体控制更可靠。
    """
    normalized_md = normalize_report_markdown(report_md)
    md_with_images = embed_base64_images(normalized_md, images_dict)
    html_content = markdown_to_html(md_with_images)

    # 用一个 report-shell 容器包裹整个正文，便于 CSS 控制版芯宽度。
    soup = BeautifulSoup(f'<div class="report-shell">{html_content}</div>', "html.parser")

    # 对 HTML 表格进行包装，并把表标题和表格绑定成一个块。
    wrap_tables_and_pair_captions(soup)

    # 把图表在节内适度延后，减少页末大空白。
    defer_visuals_by_section(soup)

    body_html = str(soup)

    # CSS 说明：
    # 1. font-family 把 Times New Roman 放在最前面，英文和数字优先使用它；
    # 2. 中文字符因 TNR 不支持，会自动回退到中文字体；
    # 3. 图表与表格分别用不同样式控制。
    template = Template(
        """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8" />
    <style>
        @page {
            size: A4;
            margin: ${margin_top_mm}mm ${margin_right_mm}mm ${margin_bottom_mm}mm ${margin_left_mm}mm;
        }

        html, body {
            margin: 0;
            padding: 0;
            background: #ffffff;
            color: #111111;
        }

        body {
            font-family: "Times New Roman", "Nimbus Roman No9 L", "Liberation Serif", "Microsoft YaHei", "PingFang SC", "Noto Serif CJK SC", "SimSun", serif;
            font-size: ${body_font_size_px}px;
            line-height: ${body_line_height};
            text-align: justify;
            text-justify: inter-ideograph;
        }

        .report-shell {
            width: ${content_width_mm}mm;
            max-width: ${content_width_mm}mm;
            margin: 0 auto;
        }

        h1, h2, h3, h4 {
            color: #111111;
            line-height: 1.38;
            break-after: avoid-page;
            page-break-after: avoid;
            page-break-inside: avoid;
        }

        h1 {
            font-size: 29px;
            margin: 0 0 10mm 0;
            text-align: left;
        }

        h2 {
            font-size: 23px;
            margin: 8mm 0 4mm 0;
        }

        h3 {
            font-size: 18px;
            margin: 6mm 0 3mm 0;
        }

        h4 {
            font-size: 16px;
            margin: 5mm 0 2.5mm 0;
        }

        p, li, td, th, blockquote {
            word-break: break-word;
            overflow-wrap: anywhere;
        }

        p {
            margin: 0 0 0.95em 0;
            text-indent: 2em;
            orphans: 3;
            widows: 3;
        }

        ul, ol {
            margin: 0.3em 0 1em 1.2em;
        }

        li {
            margin-bottom: 0.4em;
        }

        pre, code {
            font-family: "Courier New", "Consolas", monospace;
            white-space: pre-wrap;
            word-break: break-word;
        }

        blockquote, pre {
            margin: 4mm 0 5mm 0;
            break-inside: avoid-page;
            page-break-inside: avoid;
        }

        .pdf-figure,
        .table-block {
            width: 100%;
            margin: 4mm 0 5mm 0;
            break-inside: avoid-page;
            page-break-inside: avoid;
        }

        .pdf-figure {
            text-align: center;
        }

        .pdf-figure img {
            display: inline-block;
            width: auto;
            height: auto;
            max-width: ${figure_max_width_pct}%;
            max-height: ${figure_max_height_mm}mm;
            object-fit: contain;
            margin: 0 auto 2mm auto;
        }

        .pdf-figure.wide-visual img {
            max-width: ${wide_visual_max_width_pct}%;
            max-height: ${wide_visual_max_height_mm}mm;
        }

        .pdf-figure.table-like img {
            max-width: ${table_visual_max_width_pct}%;
            max-height: ${table_visual_max_height_mm}mm;
        }

        .pdf-figure.tall-visual img {
            max-width: ${tall_visual_max_width_pct}%;
            max-height: ${tall_visual_max_height_mm}mm;
        }

        .img-caption,
        .table-caption {
            font-size: 12px;
            text-indent: 0;
            text-align: center;
            color: #444444;
            margin: 1.5mm 0 0 0;
            font-weight: 600;
            break-after: avoid-page;
        }

        .table-block {
            margin-top: 4mm;
        }

        .table-caption {
            margin-bottom: 2.2mm;
        }

        .table-wrapper {
            width: 100%;
            overflow: visible;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            font-size: ${table_font_size_px}px;
            line-height: 1.42;
        }

        thead {
            display: table-header-group;
        }

        tfoot {
            display: table-footer-group;
        }

        tr {
            break-inside: avoid;
            page-break-inside: avoid;
        }

        th, td {
            border: 1px solid #111111;
            padding: ${table_cell_padding_px}px ${table_cell_padding_px}px;
            text-align: center;
            vertical-align: middle;
        }

        th {
            background: #f3f3f3;
            font-weight: 700;
        }

        img {
            max-width: 100%;
        }
    </style>
</head>
<body>
    $body_html
</body>
</html>
        """
    )

    return template.substitute(
        body_html=body_html,
        content_width_mm=PDF_EXPORT_CONFIG["content_width_mm"],
        margin_top_mm=PDF_EXPORT_CONFIG["margin_top_mm"],
        margin_right_mm=PDF_EXPORT_CONFIG["margin_right_mm"],
        margin_bottom_mm=PDF_EXPORT_CONFIG["margin_bottom_mm"],
        margin_left_mm=PDF_EXPORT_CONFIG["margin_left_mm"],
        body_font_size_px=PDF_EXPORT_CONFIG["body_font_size_px"],
        body_line_height=PDF_EXPORT_CONFIG["body_line_height"],
        figure_max_width_pct=PDF_EXPORT_CONFIG["figure_max_width_pct"],
        figure_max_height_mm=PDF_EXPORT_CONFIG["figure_max_height_mm"],
        wide_visual_max_width_pct=PDF_EXPORT_CONFIG["wide_visual_max_width_pct"],
        wide_visual_max_height_mm=PDF_EXPORT_CONFIG["wide_visual_max_height_mm"],
        table_visual_max_width_pct=PDF_EXPORT_CONFIG["table_visual_max_width_pct"],
        table_visual_max_height_mm=PDF_EXPORT_CONFIG["table_visual_max_height_mm"],
        tall_visual_max_width_pct=PDF_EXPORT_CONFIG["tall_visual_max_width_pct"],
        tall_visual_max_height_mm=PDF_EXPORT_CONFIG["tall_visual_max_height_mm"],
        table_font_size_px=PDF_EXPORT_CONFIG["table_font_size_px"],
        table_cell_padding_px=PDF_EXPORT_CONFIG["table_cell_padding_px"],
    )



def generate_pdf_bytes_from_markdown(report_md: str, images_dict: Dict[str, str]) -> bytes:
    """用 WeasyPrint 在服务器端直接生成 PDF 字节流。"""
    html_document = build_pdf_html_document(report_md, images_dict)
    return HTML(string=html_document).write_pdf()


@st.cache_data(show_spinner=False)
def build_pdf_bytes_cached(report_md: str, images_items: Tuple[Tuple[str, str], ...]) -> bytes:
    """
    缓存 PDF 生成结果。

    Streamlit 每次 rerun 都可能重新渲染页面，缓存后可避免重复生成 PDF。
    """
    images_dict = dict(images_items)
    return generate_pdf_bytes_from_markdown(report_md, images_dict)



def render_report_with_images(report_md: str, images_dict: Dict[str, str]) -> None:
    """在 Streamlit 页面内渲染报告，并把图片占位符替换成实际图片。"""
    normalized_report = normalize_report_markdown(report_md)
    pattern = r"(\[REF_IMG:\s*.*?\]|!\[.*?\]\(.*?\))"
    sections = re.split(pattern, normalized_report)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        ref_match = re.fullmatch(r"\[REF_IMG:\s*(.*?)\]", section)
        md_match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", section)

        if ref_match:
            alt_text = ref_match.group(1).strip()
            img_key = ref_match.group(1).strip()
        elif md_match:
            alt_text = md_match.group(1).strip()
            img_key = md_match.group(2).strip()
        else:
            st.markdown(section)
            continue

        matched_key = find_matching_image_key(img_key, images_dict)
        if matched_key:
            st.image(base64.b64decode(images_dict[matched_key]), caption=alt_text, use_container_width=True)
        else:
            st.markdown(section)


# ==========================================
# 模块 7: LLM 客户端类
# ==========================================

class LLMClient:
    """对 OpenAI 兼容接口做一个轻量封装，统一普通生成和多模态生成。"""

    def __init__(self, sys_prompt: str, model: str = "deepseek-chat", api_key: str = "", base_url: str = ""):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.sys_prompt = sys_prompt

    def generate(self, prompt_history: List[str]) -> str:
        """普通文本生成。"""
        messages = [{"role": "system", "content": self.sys_prompt}]
        for msg in prompt_history:
            messages.append({"role": "user", "content": msg})

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < 2:
                    time.sleep(3)
                else:
                    raise e

    def generate_with_images(self, user_prompt: str, base64_images: List[str]) -> str:
        """多模态生成：文本 + 图片。"""
        messages = [{"role": "system", "content": self.sys_prompt}]
        content_list = [{"type": "text", "text": user_prompt}]
        for b64 in base64_images:
            mime_type = infer_image_mime(b64)
            content_list.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                }
            )
        messages.append({"role": "user", "content": content_list})

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < 2:
                    time.sleep(3)
                else:
                    raise e


# ==========================================
# 模块 8: 报告生成、完整性校验与审校修补
# ==========================================


def remove_leading_headings(text: str) -> str:
    """去掉模型输出最前面多余的 H1/H2 标题，后续统一由程序补回标准标题。"""
    lines = strip_code_fences(text).strip().splitlines()
    while lines and re.match(r"^#{1,6}\s+", lines[0].strip()):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip()



def remove_extra_top_sections(text: str) -> str:
    """如果模型在当前章节后又顺手写了下一节，这里把多余部分截掉。"""
    parts = re.split(r"^##\s*[1-8]\.\s+.*$", text, maxsplit=1, flags=re.M)
    return parts[0].strip()



def ensure_section_heading(section_number: int, raw_text: str) -> str:
    """强制把某一节包装成标准标题格式。"""
    section_title = SECTION_SPECS[section_number]["title"]
    include_report_title = section_number == 1

    body = remove_leading_headings(raw_text)
    body = re.sub(r"^#\s+论文全维度深度透视报告\s*", "", body, flags=re.M).strip()
    body = remove_extra_top_sections(body)

    prefix = "# 论文全维度深度透视报告\n\n" if include_report_title else ""
    expected_heading = f"## {section_number}. {section_title}"
    final_text = f"{prefix}{expected_heading}\n\n{body}".strip()
    return normalize_report_markdown(final_text)



def section_body_text(section_md: str) -> str:
    """取出章节正文，不含 H1/H2 标题，用于判断内容是否过短。"""
    lines = section_md.splitlines()
    body_lines = []
    for line in lines:
        if re.match(r"^#{1,2}\s+", line.strip()):
            continue
        body_lines.append(line)
    return "\n".join(body_lines).strip()



def section_is_sufficient(section_number: int, section_md: str) -> bool:
    """判断某一节是否足够完整。"""
    expected_heading = f"## {section_number}. {SECTION_SPECS[section_number]['title']}"
    body = section_body_text(section_md)
    return expected_heading in section_md and len(body) >= SECTION_SPECS[section_number]["min_chars"]



def build_previous_sections_excerpt(section_map: Dict[int, str], current_section_number: int, max_chars: int = 16000) -> str:
    """为当前章节提供少量前文上下文，帮助章节衔接自然。"""
    ordered = [section_map[num] for num in sorted(section_map) if num < current_section_number and section_map.get(num)]
    joined = "\n\n".join(ordered)
    return joined[-max_chars:] if len(joined) > max_chars else joined



def assemble_report_from_section_map(section_map: Dict[int, str]) -> str:
    """按 1-8 节顺序组装完整报告。"""
    ordered_sections = [section_map[num].strip() for num in range(1, 9) if section_map.get(num)]
    return normalize_report_markdown("\n\n".join(ordered_sections))



def generate_single_section(
    agent: LLMClient,
    combined_prompt: str,
    section_number: int,
    section_map: Dict[int, str],
    extra_notes: str = "",
    current_draft: str = "",
) -> str:
    """
    单独生成某一节。

    这是防止“只生成前 4-5 节就截断”的关键：
    每一节单独请求模型，保证单次输出不会过长。
    """
    section_title = SECTION_SPECS[section_number]["title"]
    task = SECTION_SPECS[section_number]["task"]
    previous_excerpt = build_previous_sections_excerpt(section_map, section_number)
    expected_heading = f"## {section_number}. {section_title}"

    final_candidate = current_draft.strip()

    for attempt in range(MAX_SECTION_RETRY):
        retry_note = "" if attempt == 0 else "请把本节写得更完整，确保不要遗漏关键证据、关键模块或关键实验。"
        prompt = f"""
{combined_prompt}

【已经生成的前文（仅供衔接参考）】
{previous_excerpt if previous_excerpt else '暂无前文'}

【当前章节必须输出的标准标题】
{expected_heading}

【当前章节写作任务】
{task}

【补充要求】
{extra_notes if extra_notes else '无'}
{retry_note}

【当前章节旧稿】
{current_draft if current_draft else '无'}

请只输出当前章节，不要输出其他章节，不要重复前文，不要提前写下一节。
"""
        raw = agent.generate([prompt])
        candidate = ensure_section_heading(section_number, raw)
        final_candidate = candidate

        if section_is_sufficient(section_number, candidate):
            break

        current_draft = candidate

    return final_candidate



def generate_visual_evidence(md_content: str, ordered_images: List[Tuple[str, str]]) -> str:
    """为每张图片生成 FIGURE_CARD。"""
    if not ordered_images:
        return "无可用图表。"

    global_context = build_source_pack(md_content, max_chars=5000)
    vision_agent = LLMClient(
        sys_prompt=VISION_AGENT_PROMPT,
        model="qwen3.6-plus",
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
    )

    cards = []
    for index, (name, b64) in enumerate(ordered_images, start=1):
        local_context = extract_local_context(md_content, name)
        vision_prompt = f"""
请严格按照系统提示输出 FIGURE_CARD。

【图片ID】
{name}

【图片在论文中的相对顺序】
第 {index} 个图像证据

【论文局部上下文】
{local_context}

【论文整体上下文摘要】
{global_context}
"""
        v_res = vision_agent.generate_with_images(vision_prompt, [b64])
        cards.append(f"--- 图表标识: {name} ---\n{v_res}")

    return "\n\n".join(cards)



def generate_initial_section_map(
    combined_prompt: str,
    main_agent: LLMClient,
    research_agent: LLMClient,
) -> Dict[int, str]:
    """先生成完整的 1-8 节 section_map。"""
    section_map: Dict[int, str] = {}

    # 先生成第 1-7 节事实型主报告。
    for section_number in range(1, 8):
        section_map[section_number] = generate_single_section(
            agent=main_agent,
            combined_prompt=combined_prompt,
            section_number=section_number,
            section_map=section_map,
        )

    # 再生成第 8 节研究路线。
    section_map[8] = generate_single_section(
        agent=research_agent,
        combined_prompt=combined_prompt + "\n\n请明确：第8节属于基于本文机制的研究设想，不是原文结论。",
        section_number=8,
        section_map=section_map,
    )

    return section_map



def ensure_section_map_complete(
    section_map: Dict[int, str],
    combined_prompt: str,
    main_agent: LLMClient,
    research_agent: LLMClient,
) -> Dict[int, str]:
    """
    检查 1-8 节是否都存在且不过短。

    这是针对“为什么只显示前 5 节就结束了”这一问题增加的硬校验。
    即便某一轮模型漏掉了某节，这里也会定向补写缺失章节。
    """
    for section_number in range(1, 9):
        needs_regen = not section_map.get(section_number) or not section_is_sufficient(section_number, section_map[section_number])
        if needs_regen:
            target_agent = research_agent if section_number == 8 else main_agent
            extra_notes = "请确保本节独立完整，不要遗漏。"
            section_map[section_number] = generate_single_section(
                agent=target_agent,
                combined_prompt=combined_prompt,
                section_number=section_number,
                section_map=section_map,
                extra_notes=extra_notes,
                current_draft=section_map.get(section_number, ""),
            )
    return section_map



def extract_target_sections_from_audit(audit_text: str) -> List[int]:
    """从审校意见中提取可能需要修补的章节编号。"""
    targets = set()

    for num in re.findall(r"第\s*([1-8])\s*节", audit_text):
        targets.add(int(num))
    for num in re.findall(r"##\s*([1-8])\.", audit_text):
        targets.add(int(num))

    # 若审校文本没有明确点名章节，就根据关键词做一层弱映射。
    keyword_map = {
        1: ["核心贡献", "研究问题"],
        2: ["背景", "前人路线", "研究缺口"],
        3: ["方法总览", "整体数据流"],
        4: ["关键模块", "机制剖析"],
        5: ["实验", "消融", "关键证据"],
        6: ["复现", "适用边界"],
        7: ["局限", "未解决问题"],
        8: ["研究路线", "创新路线"],
    }
    lowered = audit_text.lower()
    for num, keywords in keyword_map.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            targets.add(num)

    return sorted(targets)



def audit_and_refine_report(
    section_map: Dict[int, str],
    combined_prompt: str,
    source_pack: str,
    text_report: str,
    vision_summaries: str,
    available_img_ids: str,
    main_agent: LLMClient,
    research_agent: LLMClient,
    auditor: LLMClient,
) -> Tuple[Dict[int, str], str]:
    """
    审校并定向修补 section_map。

    这里故意不做“整篇重新生成”，因为那会再次触发长文本截断。
    如果审校失败，只修对应章节。
    """
    last_audit_result = ""

    for _ in range(MAX_AUDIT_ROUNDS):
        section_map = ensure_section_map_complete(section_map, combined_prompt, main_agent, research_agent)
        final_report = assemble_report_from_section_map(section_map)

        audit_prompt = f"""
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
        audit_result = auditor.generate([audit_prompt])
        last_audit_result = audit_result

        if not re.search(r"RESULT\s*:\s*FAIL", audit_result, flags=re.I):
            break

        targets = extract_target_sections_from_audit(audit_result)
        if not targets:
            targets = list(range(1, 9))

        for section_number in targets:
            target_agent = research_agent if section_number == 8 else main_agent
            section_map[section_number] = generate_single_section(
                agent=target_agent,
                combined_prompt=combined_prompt,
                section_number=section_number,
                section_map=section_map,
                extra_notes=f"请针对以下审校意见修订本节，并只输出本节：\n{audit_result}",
                current_draft=section_map.get(section_number, ""),
            )

    section_map = ensure_section_map_complete(section_map, combined_prompt, main_agent, research_agent)
    return section_map, last_audit_result


# ==========================================
# 模块 9: PDF 结构解析服务调用
# ==========================================


def analyze_pdf_with_modal(pdf_file_bytes: bytes):
    """把用户上传的 PDF 发送到云端解析服务，获取结构化 markdown 与图片。"""
    with st.spinner("正在唤醒云端 GPU 引擎，深度解析公式与版面... "):
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
# 模块 10: 论文精读主流程
# ==========================================


def render_analysis_ui(pdf_bytes: bytes) -> None:
    """
    论文精读工作流总入口。

    整个流程：
    1. 解析 PDF -> markdown + 图片
    2. 文本 Agent 做 FACT_BANK 抽取
    3. 视觉 Agent 生成 FIGURE_CARD
    4. 主报告 Agent 分章节生成 1-7 节
    5. 研究路线 Agent 生成第 8 节
    6. 完整性校验 + 审校 + 定向修补
    7. 页面渲染 + Markdown/PDF 下载
    """
    file_hash = hash(pdf_bytes)

    # 只有当上传的 PDF 发生变化时，才重新跑整套解析与生成。
    if st.session_state.get("current_pdf_hash") != file_hash:
        st.session_state.current_pdf_hash = file_hash
        st.session_state.final_main_report = ""
        st.session_state.temp_images = {}
        st.session_state.final_text_report = ""
        st.session_state.final_vision_reports = ""
        st.session_state.source_md_content = ""
        st.session_state.final_audit_result = ""

        result = analyze_pdf_with_modal(pdf_bytes)
        if result and result.get("status") == "success":
            md_content = result.get("markdown", "")
            raw_images = result.get("images", {})
            ordered_images = sort_images_by_doc_order(md_content, raw_images)
            ordered_images_dict = dict(ordered_images)

            st.session_state.source_md_content = md_content
            st.session_state.temp_images = ordered_images_dict

            # --------------------
            # 步骤 1：文本事实抽取
            # --------------------
            with st.spinner("文本专家正在抽取论文事实、方法与实验证据..."):
                text_agent = LLMClient(
                    sys_prompt=TEXT_AGENT_PROMPT,
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL,
                )
                text_prompt = (
                    "请严格按系统提示完成论文事实抽取。以下为论文原始结构化 markdown：\n\n"
                    f"{build_source_pack(md_content, max_chars=60000)}"
                )
                text_report = text_agent.generate([text_prompt])
                st.session_state.final_text_report = text_report

            # --------------------
            # 步骤 2：图表证据抽取
            # --------------------
            with st.spinner(f"视觉专家正在分析 {len(ordered_images)} 张关键图表，并补齐局部上下文..."):
                vision_summaries = generate_visual_evidence(md_content, ordered_images)
                st.session_state.final_vision_reports = vision_summaries

            # 构造主报告阶段的统一上下文包。
            available_img_ids = "\n".join([f"- {name}" for name, _ in ordered_images]) if ordered_images else "无可用图片"
            source_pack = build_source_pack(md_content, max_chars=52000)
            combined_prompt = f"""
【论文原始结构化 markdown】
{source_pack}

【Text Agent 输出】
{st.session_state.final_text_report}

【Vision Agent 输出】
{st.session_state.final_vision_reports}

【当前所有可用的图表真实标识符列表】（你必须从中复制以插入图片）
{available_img_ids}
"""

            # --------------------
            # 步骤 3：初始化大模型客户端
            # --------------------
            main_agent = LLMClient(
                sys_prompt=MAIN_AGENT_PROMPT,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
            )
            research_agent = LLMClient(
                sys_prompt=RESEARCH_AGENT_PROMPT,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
            )
            auditor = LLMClient(
                sys_prompt=REPORT_AUDITOR_PROMPT,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
            )

            # --------------------
            # 步骤 4：分章节生成
            # --------------------
            with st.spinner("主编代理正在分章节生成第 1-8 节研究型报告..."):
                section_map = generate_initial_section_map(
                    combined_prompt=combined_prompt,
                    main_agent=main_agent,
                    research_agent=research_agent,
                )

            # --------------------
            # 步骤 5：完整性校验 + 审校 + 定向修补
            # --------------------
            with st.spinner("审校代理正在核查章节完整性、证据一致性与图表引用..."):
                section_map, audit_result = audit_and_refine_report(
                    section_map=section_map,
                    combined_prompt=combined_prompt,
                    source_pack=source_pack,
                    text_report=st.session_state.final_text_report,
                    vision_summaries=st.session_state.final_vision_reports,
                    available_img_ids=available_img_ids,
                    main_agent=main_agent,
                    research_agent=research_agent,
                    auditor=auditor,
                )
                st.session_state.final_audit_result = audit_result

            final_report = assemble_report_from_section_map(section_map)
            st.session_state.final_main_report = final_report

    # 若已有最终报告，则渲染结果与下载入口。
    if st.session_state.final_main_report:
        st.success("论文全维度深度透视报告已生成！")

        render_report_with_images(
            st.session_state.final_main_report,
            st.session_state.temp_images,
        )

        # 审校日志折叠显示，方便调试，但不打扰主界面阅读。
        if st.session_state.get("final_audit_result"):
            with st.expander("查看审校结果（调试信息）", expanded=False):
                st.code(st.session_state.final_audit_result)

        st.divider()
        st.markdown("### 导出与下载")
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "下载报告原文 (Markdown)",
                st.session_state.final_main_report,
                file_name="Report.md",
                use_container_width=True,
            )

        with col2:
            # PDF 改为服务器端生成。
            # 这样不会再出现 html2canvas 在分页处把文字切掉的问题。
            with st.spinner("正在生成服务器端排版 PDF，请稍候..."):
                pdf_bytes = build_pdf_bytes_cached(
                    st.session_state.final_main_report,
                    tuple(st.session_state.temp_images.items()),
                )

            st.download_button(
                "下载标准版学术 PDF 报告",
                pdf_bytes,
                file_name="论文深度透视报告.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


# ==========================================
# 模块 11: 侧边栏及前端 UI 定义
# ==========================================

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
# 模块 12: 全局应用状态机初始化
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
if "feedback_start_time" not in st.session_state:
    st.session_state.feedback_start_time = None
if "ui_logs" not in st.session_state:
    st.session_state.ui_logs = []
if "current_pdf_hash" not in st.session_state:
    st.session_state.current_pdf_hash = None
if "final_text_report" not in st.session_state:
    st.session_state.final_text_report = ""
if "final_vision_reports" not in st.session_state:
    st.session_state.final_vision_reports = ""
if "parse_success" not in st.session_state:
    st.session_state.parse_success = False
if "temp_images" not in st.session_state:
    st.session_state.temp_images = {}
if "final_main_report" not in st.session_state:
    st.session_state.final_main_report = ""
if "source_md_content" not in st.session_state:
    st.session_state.source_md_content = ""
if "final_audit_result" not in st.session_state:
    st.session_state.final_audit_result = ""


# ==========================================
# 模块 13: 业务路由分发与主循环
# ==========================================

# 入口 A：直接解析本地 PDF。
if start_analyze_button and sidebar_pdf:
    st.markdown("---")
    st.info("正在启动【直接解析模式】，开始解构文献...")
    render_analysis_ui(sidebar_pdf.read())
    st.stop()

# 入口 B：论文检索模式。
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


# ------------------------------
# IDLE：显示系统使用说明
# ------------------------------
if st.session_state.app_state == "IDLE":
    st.markdown(
        """
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
    """
    )


# ------------------------------
# 非 IDLE：展示 Agent 运行日志
# ------------------------------
if st.session_state.app_state != "IDLE":
    st.markdown("### Agent 检索执行轨迹")
    for log in st.session_state.ui_logs:
        with st.expander(log["title"], expanded=False):
            st.markdown(log["content"])


# ------------------------------
# RUNNING：检索 Agent 主循环
# ------------------------------
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


# ------------------------------
# WAITING_FEEDBACK：展示检索结果并允许纠偏
# ------------------------------
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


# ------------------------------
# COMPLETED：展示最终检索结果并支持上传 PDF 精读
# ------------------------------
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
