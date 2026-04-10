# ==========================================
# 模块 1: 依赖导入与页面基础配置
# ==========================================
import base64
import datetime
import hashlib
import re
import time
from typing import Dict, List, Tuple

import markdown
import requests
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

st.set_page_config(page_title="AI 论文检索 Agent", page_icon="📚", layout="wide")

# ==========================================
# 模块 2: 全局变量与 API 密钥配置
# ==========================================

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
    """避免把 </script> 等字符串直接塞入 HTML 模板时破坏脚本结构。"""
    return text.replace("</script>", "<\\/script>")

# ==========================================
# 模块 7: 浏览器端 PDF 导出组件
# ==========================================
def download_pdf_component(md_text: str):
    """在 Streamlit 页面中插入“导出 PDF”按钮与导出逻辑。"""
    html_content = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
    html_content = wrap_markdown_tables(html_content)
    html_content = html_escape_for_component(html_content)

    cfg = PDF_EXPORT_CONFIG

    html_code = f"""
    <html>
    <head>
        <meta charset="utf-8" />
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
        <style>
            * {{ box-sizing: border-box; }}

            html, body {{
                margin: 0;
                padding: 0;
                background: #ffffff;
                color: #000000;
            }}

            body {{
                padding: 12px;
                font-family: "Times New Roman", "Liberation Serif", "Nimbus Roman", "Songti SC", "SimSun", "Noto Serif CJK SC", serif;
            }}

            .download-btn {{
                display: block;
                width: 100%;
                padding: 12px;
                background-color: #2e7d32;
                color: #fff;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 700;
                cursor: pointer;
                margin-bottom: 12px;
            }}

            .download-btn:hover {{
                background-color: #256b2a;
            }}

            /* 这里用 offscreen 而不是 display:none，避免 html2canvas 在隐藏节点上测量出错 */
            #pdf-host {{
                position: fixed;
                left: -100000px;
                top: 0;
                width: 210mm;
                background: #fff;
                z-index: -1;
                opacity: 1;
            }}

            #report-content {{
                width: {cfg['content_width_mm']}mm;
                min-width: {cfg['content_width_mm']}mm;
                max-width: {cfg['content_width_mm']}mm;
                margin: 0 auto;
                background: #fff;
                color: #000;
                font-size: {cfg['body_font_size_px']}px;
                line-height: {cfg['body_line_height']};
                text-align: justify;
                text-justify: inter-ideograph;
                overflow: visible;
                padding-top: 2mm;
            }}

            h1, h2, h3, h4 {{
                color: #000;
                line-height: 1.45;
                margin: 20px 0 12px;
                page-break-after: avoid;
                break-after: avoid;
                font-weight: 700;
            }}

            h1 {{ font-size: 28px; text-align: center; margin-top: 8px; margin-bottom: 18px; }}
            h2 {{ font-size: 24px; margin-top: 26px; }}
            h3 {{ font-size: 20px; }}
            h4 {{ font-size: 18px; }}

            p, li, blockquote, td, th {{
                word-break: break-word;
                overflow-wrap: anywhere;
                white-space: normal;
            }}

            p {{
                margin: 0 0 0.95em;
                text-indent: 2em;
                orphans: 2;
                widows: 2;
            }}

            code, pre {{
                white-space: pre-wrap;
                word-break: break-word;
            }}

            pre, blockquote {{
                page-break-inside: avoid;
                break-inside: avoid;
                margin: 16px 0 20px;
            }}

            .report-section {{
                width: 100%;
            }}

            .pdf-figure, .table-block {{
                width: 100%;
                max-width: 100%;
                margin: 18px 0 24px;
                page-break-inside: avoid;
                break-inside: avoid;
            }}

            .pdf-figure {{
                text-align: center;
            }}

            .pdf-figure img {{
                display: inline-block;
                width: auto;
                height: auto;
                max-width: {cfg['figure_max_width_pct']}%;
                max-height: {cfg['figure_max_height_mm']}mm;
                object-fit: contain;
                margin: 0 auto 6px;
            }}

            .pdf-figure.wide-visual img {{
                max-width: {cfg['wide_visual_max_width_pct']}%;
                max-height: {cfg['wide_visual_max_height_mm']}mm;
            }}

            .pdf-figure.table-visual img {{
                max-width: {cfg['table_visual_max_width_pct']}%;
                max-height: {cfg['table_visual_max_height_mm']}mm;
            }}

            .pdf-figure.tall-visual img {{
                max-width: {cfg['tall_visual_max_width_pct']}%;
                max-height: {cfg['tall_visual_max_height_mm']}mm;
            }}

            .img-caption {{
                text-align: center;
                font-size: 12px;
                color: #444;
                text-indent: 0;
                margin-top: 4px;
                font-weight: 700;
            }}

            .table-block {{
                overflow: hidden;
            }}

            table {{
                width: 100% !important;
                max-width: 100%;
                border-collapse: collapse;
                table-layout: fixed;
                margin: 0;
                font-size: {cfg['table_font_size_px']}px;
                page-break-inside: avoid;
                break-inside: avoid;
            }}

            th, td {{
                border: 1px solid #000;
                padding: {cfg['table_cell_padding_px']}px;
                text-align: center;
                vertical-align: middle;
            }}

            th {{
                background: #f6f6f6;
                font-weight: 700;
            }}
        </style>
    </head>
    <body>
        <button class="download-btn" onclick="generatePDF()">📥 导出标准版学术 PDF 报告</button>
        <div id="pdf-host">
            <div id="report-content">{html_content}</div>
        </div>
        <script>
            function sleep(ms) {{
                return new Promise(resolve => setTimeout(resolve, ms));
            }}

            async function waitForImages(root) {{
                const images = Array.from(root.querySelectorAll('img'));
                await Promise.all(images.map(img => {{
                    if (img.complete) return Promise.resolve();
                    return new Promise(resolve => {{
                        const done = () => resolve();
                        img.onload = done;
                        img.onerror = done;
                    }});
                }}));
            }}

            function wrapSections(root) {{
                const nodes = Array.from(root.childNodes);
                let currentSection = null;
                nodes.forEach(node => {{
                    if (node.nodeType === 1 && node.tagName === 'H2') {{
                        currentSection = document.createElement('div');
                        currentSection.className = 'report-section';
                        root.insertBefore(currentSection, node);
                        currentSection.appendChild(node);
                    }} else if (currentSection) {{
                        currentSection.appendChild(node);
                    }}
                }});
            }}

            function classifyFigures(root) {{
                const figures = Array.from(root.querySelectorAll('.pdf-figure'));
                figures.forEach(fig => {{
                    const img = fig.querySelector('img');
                    if (!img) return;
                    const alt = (img.alt || '').toLowerCase();
                    const w = img.naturalWidth || img.width || 1;
                    const h = img.naturalHeight || img.height || 1;
                    const ratio = w / Math.max(h, 1);

                    fig.classList.remove('wide-visual', 'table-visual', 'tall-visual');

                    if (alt.includes('表') || alt.includes('table') || ratio > 1.85) {{
                        fig.classList.add('table-visual');
                    }} else if (h / Math.max(w, 1) > 1.35) {{
                        fig.classList.add('tall-visual');
                    }} else if (ratio > 1.35) {{
                        fig.classList.add('wide-visual');
                    }}
                }});
            }}

            function moveVisualsToSectionTail(root) {{
                const sections = Array.from(root.querySelectorAll('.report-section'));
                sections.forEach(section => {{
                    const visuals = Array.from(section.querySelectorAll(':scope > .pdf-figure, :scope > .table-block'));
                    visuals.forEach(v => v.remove());
                    visuals.forEach(v => section.appendChild(v));
                }});
            }}

            async function prepareLayout(root) {{
                wrapSections(root);
                await waitForImages(root);
                classifyFigures(root);
                moveVisualsToSectionTail(root);
                if (document.fonts && document.fonts.ready) {{
                    try {{ await document.fonts.ready; }} catch (e) {{}}
                }}
                await sleep(100);
                await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
            }}

            async function generatePDF() {{
                const element = document.getElementById('report-content');
                const clone = element.cloneNode(true);
                const host = document.getElementById('pdf-host');

                const exportNode = document.createElement('div');
                exportNode.id = 'report-content';
                exportNode.innerHTML = clone.innerHTML;
                host.innerHTML = '';
                host.appendChild(exportNode);

                await prepareLayout(exportNode);

                const opt = {{
                    margin: [{cfg['margin_top_mm']}, {cfg['margin_right_mm']}, {cfg['margin_bottom_mm']}, {cfg['margin_left_mm']}],
                    filename: '论文深度透视报告.pdf',
                    image: {{ type: 'jpeg', quality: 0.98 }},
                    html2canvas: {{
                        scale: {cfg['html2canvas_scale']},
                        useCORS: true,
                        scrollX: 0,
                        scrollY: 0,
                        letterRendering: false,
                        windowWidth: Math.ceil(exportNode.scrollWidth)
                    }},
                    jsPDF: {{
                        unit: 'mm',
                        format: 'a4',
                        orientation: 'portrait',
                        compress: true
                    }},
                    pagebreak: {{
                        mode: ['css', 'legacy'],
                        avoid: ['.pdf-figure', '.table-block', 'table', 'tr', 'pre', 'blockquote', 'h1', 'h2', 'h3', 'h4']
                    }}
                }};

                await html2pdf().set(opt).from(exportNode).save();
            }}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=90)

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

    def generate(self, prompt_history: List[str]) -> str:
        """纯文本生成接口。"""
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
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt < 2:
                    time.sleep(3)
                else:
                    raise e

    def generate_with_images(self, user_prompt: str, base64_images: List[str]) -> str:
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
    return normalize_report_markdown(research_agent.generate([prompt]))


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
        final_report = normalize_report_markdown(main_agent.generate([revise_prompt]))

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
            pdf_markdown = embed_base64_images(
                st.session_state.final_main_report,
                st.session_state.temp_images,
            )
            download_pdf_component(pdf_markdown)

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
