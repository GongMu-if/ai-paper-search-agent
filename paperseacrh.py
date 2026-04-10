# ==========================================
# 模块 1: 依赖导入与页面基础配置
# ==========================================
import re
import html as html_lib
import requests
from openai import OpenAI
import time
import streamlit as st
import datetime
import base64
import markdown
from string import Template
import streamlit.components.v1 as components

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

PDF_EXPORT_CONFIG = {
    "content_width_mm": 170,
    "margin_top_mm": 12,
    "margin_right_mm": 14,
    "margin_bottom_mm": 12,
    "margin_left_mm": 14,
    "body_font_size_px": 14,
    "body_line_height": 1.88,
    "figure_max_width_pct": 72,
    "figure_max_height_mm": 105,
    "wide_visual_max_width_pct": 88,
    "wide_visual_max_height_mm": 110,
    "table_visual_max_width_pct": 94,
    "table_visual_max_height_mm": 220,
    "tall_visual_max_width_pct": 66,
    "tall_visual_max_height_mm": 170,
    "table_font_size_px": 11,
    "table_cell_padding_px": 5,
    "html2canvas_scale": 2,
}

# ==========================================
# 模块 3: 核心 Agent 提示词库
# ==========================================
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
10. 对未来研究路线，必须给出可执行方案，而不是泛泛愿景。
11. 当用户只要求你输出某一节或某几节时，你必须只输出对应部分，不要补出其他章节。

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

REPORT_AUDITOR_PROMPT = """
你是学术报告审校员。请检查以下报告是否存在：
1. 不被原始 markdown / FACT_BANK / FIGURE_CARD 支持的结论
2. 不存在的图片ID或错误图片占位符
3. 漏掉的关键模块、关键实验、关键局限
4. 将研究设想误写为原文结论
5. 同一章节内容重复、前后矛盾或与图表证据不一致

输出格式：
RESULT: PASS 或 FAIL
ISSUES:
- 若无问题，写“无重大问题”
- 若有问题，逐条列出
"""


def get_system_prompt(requirements, preprint_rule):
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
# 模块 4: 工具函数与通用处理库
# ==========================================
def reconstruct_abstract(inverted_index: dict) -> str:
    if not inverted_index:
        return ""
    word_index = [(pos, word) for word, positions in inverted_index.items() for pos in positions]
    word_index.sort(key=lambda x: x[0])
    return " ".join([word for _, word in word_index])


def search_and_detail_papers(query: str) -> str:
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
            final_abstract = ""
            openalex_mark = ""

            if s2_abstract and s2_abstract.strip():
                final_abstract = s2_abstract.strip()
            else:
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


def infer_image_mime(b64_data: str) -> str:
    if b64_data.startswith("/9j/"):
        return "image/jpeg"
    if b64_data.startswith("iVBOR"):
        return "image/png"
    if b64_data.startswith("UklGR"):
        return "image/webp"
    return "image/png"


def strip_code_fences(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    match = re.match(r"^```(?:markdown|md)?\s*(.*?)\s*```$", text, flags=re.S | re.I)
    return match.group(1).strip() if match else text


def normalize_markdown_tables(md_text: str) -> str:
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
    text = strip_code_fences(md_text)
    text = re.sub(r"\[REF_IMG:\s*(.*?)\]", r"![\1](\1)", text)
    text = re.sub(r"[ \t]*(!\[[^\]]*\]\([^\)]+\))[ \t]*", r"\n\n\1\n\n", text)
    text = normalize_markdown_tables(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_source_pack(md_content: str, max_chars: int = 52000) -> str:
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


def sort_images_by_doc_order(md_content: str, images_dict: dict):
    def sort_key(item):
        position = md_content.find(item[0])
        return position if position >= 0 else 10 ** 9

    return sorted(images_dict.items(), key=sort_key)


def extract_local_context(md_content: str, image_id: str, window: int = 1800) -> str:
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


def clean_section_output(text: str, keep_h1: bool = False) -> str:
    cleaned = normalize_report_markdown(text)
    if not keep_h1:
        cleaned = re.sub(r"^#\s+.+?(?:\n+|$)", "", cleaned).strip()
    return cleaned.strip()


def find_matching_image_key(img_key: str, images_dict: dict):
    if img_key in images_dict:
        return img_key
    for img_name in images_dict:
        if img_key in img_name or img_name in img_key:
            return img_name
    return None


def render_report_with_images(report_md: str, images_dict: dict):
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
            st.image(
                base64.b64decode(images_dict[matched_key]),
                caption=alt_text,
                use_container_width=True,
            )
        else:
            st.markdown(section)


def build_figure_html(alt_text: str, b64_data: str) -> str:
    mime_type = infer_image_mime(b64_data)
    safe_alt = html_lib.escape(alt_text)
    kind = "table" if re.match(r"^\s*(表|table)\s*\d+", alt_text, flags=re.I) else "figure"
    return (
        "\n"
        f'<div class="pdf-figure" data-kind="{kind}">'
        f'<img src="data:{mime_type};base64,{b64_data}" alt="{safe_alt}" />'
        f'<div class="img-caption">{safe_alt}</div>'
        "</div>\n"
    )


def embed_base64_images(md_text, images_dict):
    def replace_markdown_img(match):
        alt_text = match.group(1).strip()
        img_placeholder = match.group(2).strip()
        matched_key = find_matching_image_key(img_placeholder, images_dict)
        if matched_key:
            return build_figure_html(alt_text, images_dict[matched_key])
        return match.group(0)

    def replace_ref_img(match):
        img_placeholder = match.group(1).strip()
        matched_key = find_matching_image_key(img_placeholder, images_dict)
        if matched_key:
            return build_figure_html(img_placeholder, images_dict[matched_key])
        return match.group(0)

    md_text = re.sub(r"!\[(.*?)\]\((.*?)\)", replace_markdown_img, md_text)
    md_text = re.sub(r"\[REF_IMG:\s*(.*?)\]", replace_ref_img, md_text)
    return md_text


def download_pdf_component(md_text):
    md_text = normalize_report_markdown(md_text)
    html_content = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    html_content = re.sub(
        r"(<table>.*?</table>)",
        r'<div class="table-wrapper">\1</div>',
        html_content,
        flags=re.DOTALL,
    )
    safe_html_content = html_content.replace("$", "$$")

    template = Template(
        """
    <html>
    <head>
        <meta charset="utf-8" />
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
        <style>
            * {
                box-sizing: border-box;
            }

            html, body {
                margin: 0;
                padding: 0;
                background: #ffffff;
                color: #111111;
                font-family: 'Microsoft YaHei', 'PingFang SC', 'Noto Serif CJK SC', 'SimSun', serif;
                line-height: $body_line_height;
            }

            body {
                padding: 12px;
            }

            #report-source {
                display: none;
            }

            #pdf-stage {
                position: fixed;
                left: -240vw;
                top: 0;
                width: ${content_width_mm}mm;
                background: #ffffff;
                z-index: -1;
            }

            .report-shell {
                width: ${content_width_mm}mm;
                min-width: ${content_width_mm}mm;
                max-width: ${content_width_mm}mm;
                margin: 0 auto;
                background: #ffffff;
                color: #111111;
                font-size: ${body_font_size_px}px;
                text-align: justify;
                text-justify: inter-ideograph;
                overflow: visible;
            }

            h1, h2, h3, h4 {
                color: #111111;
                margin: 22px 0 12px;
                line-height: 1.4;
                page-break-after: avoid;
                break-after: avoid;
            }

            h1 {
                margin-top: 4px;
                font-size: 30px;
            }

            h2 {
                margin-top: 26px;
                font-size: 24px;
            }

            h3 {
                margin-top: 20px;
                font-size: 20px;
            }

            p, li, blockquote, td, th {
                word-break: break-word;
                overflow-wrap: anywhere;
                white-space: normal;
            }

            p {
                text-indent: 2em;
                margin: 0 0 0.95em;
                padding-bottom: 1px;
            }

            .pdf-figure,
            .table-block,
            pre,
            blockquote {
                width: 100%;
                max-width: 100%;
                margin: 16px 0 20px;
            }

            .pdf-figure,
            .table-block {
                page-break-inside: avoid;
                break-inside: avoid;
            }

            .pdf-figure {
                text-align: center;
            }

            .pdf-figure img {
                display: inline-block;
                width: auto;
                max-width: ${figure_max_width_pct}%;
                height: auto;
                max-height: ${figure_max_height_mm}mm;
                object-fit: contain;
                margin: 0 auto 6px;
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

            img {
                display: block;
                width: auto;
                height: auto;
                max-width: 100%;
            }

            .img-caption,
            .table-caption {
                text-indent: 0;
                text-align: center;
                font-size: 12px;
                color: #555555;
                margin-top: 4px;
                margin-bottom: 0;
                font-weight: 600;
            }

            .table-block {
                margin-top: 14px;
            }

            .table-caption {
                margin-bottom: 8px;
            }

            .table-wrapper {
                width: 100%;
                overflow: visible;
            }

            table {
                width: 100% !important;
                max-width: 100%;
                border-collapse: collapse;
                table-layout: fixed;
                margin: 0 auto;
                font-size: ${table_font_size_px}px;
                line-height: 1.5;
                page-break-inside: avoid;
                break-inside: avoid;
            }

            thead {
                display: table-header-group;
            }

            tr {
                page-break-inside: avoid;
                break-inside: avoid;
            }

            th, td {
                border: 1px solid #111111;
                padding: ${table_cell_padding_px}px ${table_cell_padding_px}px;
                text-align: center;
                vertical-align: middle;
            }

            th {
                background: #f5f5f5;
                font-weight: 700;
            }

            pre, code {
                white-space: pre-wrap;
                word-break: break-word;
            }

            .download-btn {
                display: block;
                width: 100%;
                padding: 12px;
                background-color: #4CAF50;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                font-size: 16px;
                cursor: pointer;
                font-weight: 700;
            }

            .download-btn:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <button class="download-btn" onclick="generatePDF()">📥 导出标准版学术 PDF 报告</button>

        <div id="report-source">
            <div class="report-shell">$html_content</div>
        </div>
        <div id="pdf-stage"></div>

        <script>
            function sleep(ms) {
                return new Promise(resolve => setTimeout(resolve, ms));
            }

            async function waitForImages(root) {
                const images = Array.from(root.querySelectorAll('img'));
                await Promise.all(images.map((img) => {
                    if (img.complete) {
                        return Promise.resolve();
                    }
                    return new Promise((resolve) => {
                        const done = () => resolve();
                        img.onload = done;
                        img.onerror = done;
                    });
                }));
            }

            function cleanupEmptyParagraphs(root) {
                Array.from(root.querySelectorAll('p')).forEach((p) => {
                    const text = (p.textContent || '').trim();
                    if (!text && !p.querySelector('img')) {
                        p.remove();
                    }
                });
            }

            function classifyFigures(root) {
                const figures = Array.from(root.querySelectorAll('.pdf-figure'));
                figures.forEach((figure) => {
                    const img = figure.querySelector('img');
                    const captionNode = figure.querySelector('.img-caption');
                    const caption = captionNode ? (captionNode.textContent || '').trim() : '';
                    const declaredKind = (figure.dataset.kind || '').toLowerCase();

                    if (declaredKind === 'table' || /^(表|table)\\s*\\d+[:：]/i.test(caption)) {
                        figure.classList.add('table-like');
                    }

                    if (img && img.naturalWidth && img.naturalHeight) {
                        const ratio = img.naturalWidth / img.naturalHeight;
                        if (ratio >= 1.45) {
                            figure.classList.add('wide-visual');
                        }
                        if (ratio <= 0.85) {
                            figure.classList.add('tall-visual');
                        }
                    }
                });
            }

            function pairTableCaptions(root) {
                const wrappers = Array.from(root.querySelectorAll('.table-wrapper'));
                wrappers.forEach((wrapper) => {
                    if (wrapper.parentElement && wrapper.parentElement.classList.contains('table-block')) {
                        return;
                    }

                    const block = document.createElement('div');
                    block.className = 'table-block';

                    const prev = wrapper.previousElementSibling;
                    let captionNode = null;
                    if (prev && prev.tagName === 'P') {
                        const captionText = (prev.textContent || '').trim();
                        if (/^(表|table)\\s*\\d+[:：]/i.test(captionText)) {
                            captionNode = prev;
                            captionNode.classList.add('table-caption');
                        }
                    }

                    if (captionNode) {
                        captionNode.parentNode.insertBefore(block, captionNode);
                        block.appendChild(captionNode);
                    } else {
                        wrapper.parentNode.insertBefore(block, wrapper);
                    }
                    block.appendChild(wrapper);
                });
            }

            function deferVisualsBySection(root) {
                const children = Array.from(root.children);
                const fragment = document.createDocumentFragment();
                let pendingVisuals = [];

                function isSectionBoundary(node) {
                    return node.tagName === 'H2' || node.tagName === 'H3';
                }

                function isVisualNode(node) {
                    return node.classList && (
                        node.classList.contains('pdf-figure') ||
                        node.classList.contains('table-block')
                    );
                }

                function flushVisuals() {
                    pendingVisuals.forEach((node) => fragment.appendChild(node));
                    pendingVisuals = [];
                }

                children.forEach((node) => {
                    if (isSectionBoundary(node) && fragment.childNodes.length > 0) {
                        flushVisuals();
                        fragment.appendChild(node);
                        return;
                    }
                    if (isVisualNode(node)) {
                        pendingVisuals.push(node);
                        return;
                    }
                    fragment.appendChild(node);
                });

                flushVisuals();
                root.innerHTML = '';
                root.appendChild(fragment);
            }

            function buildPdfDom() {
                const stage = document.getElementById('pdf-stage');
                const source = document.getElementById('report-source');
                stage.innerHTML = source.innerHTML;
                const root = stage.querySelector('.report-shell');
                cleanupEmptyParagraphs(root);
                pairTableCaptions(root);
                classifyFigures(root);
                deferVisualsBySection(root);
                cleanupEmptyParagraphs(root);
                return root;
            }

            async function generatePDF() {
                const element = buildPdfDom();
                await sleep(120);

                if (document.fonts && document.fonts.ready) {
                    try {
                        await document.fonts.ready;
                    } catch (e) {}
                }

                await waitForImages(element);
                await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));

                const opt = {
                    margin: [$margin_top_mm, $margin_right_mm, $margin_bottom_mm, $margin_left_mm],
                    filename: '论文深度透视报告.pdf',
                    image: { type: 'jpeg', quality: 0.97 },
                    html2canvas: {
                        scale: $html2canvas_scale,
                        useCORS: true,
                        scrollX: 0,
                        scrollY: 0,
                        windowWidth: Math.ceil(element.scrollWidth + 8)
                    },
                    jsPDF: {
                        unit: 'mm',
                        format: 'a4',
                        orientation: 'portrait',
                        compress: true
                    },
                    pagebreak: {
                        mode: ['css'],
                        avoid: ['.pdf-figure', '.table-block', 'pre', 'blockquote']
                    }
                };

                try {
                    await html2pdf().set(opt).from(element).save();
                } finally {
                    document.getElementById('pdf-stage').innerHTML = '';
                }
            }
        </script>
    </body>
    </html>
        """
    )

    html_code = template.substitute(
        html_content=safe_html_content,
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
        html2canvas_scale=PDF_EXPORT_CONFIG["html2canvas_scale"],
    )
    components.html(html_code, height=90, scrolling=False)


def analyze_pdf_with_modal(pdf_file_bytes):
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
# 模块 5: LLM 客户端类
# ==========================================
class LLMClient:
    def __init__(self, sys_prompt, model="deepseek-chat", api_key="", base_url=""):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.sys_prompt = sys_prompt

    def generate(self, prompt_history):
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

    def generate_with_images(self, user_prompt, base64_images):
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
# 模块 6: 论文精读与渲染管线
# ==========================================
SECTION_TASKS = [
    "请只输出以下内容，不要输出其他章节：\n# 论文全维度深度透视报告\n\n## 1. 研究问题与核心贡献",
    "请只输出以下内容，不要输出其他章节：\n## 2. 背景、研究缺口与前人路线",
    "请只输出以下内容，不要输出其他章节：\n## 3. 方法总览与整体数据流",
    "请只输出以下内容，不要输出其他章节：\n## 4. 关键模块逐层机制剖析",
    "请只输出以下内容，不要输出其他章节：\n## 5. 实验设计、关键证据与论点验证",
    "请只输出以下内容，不要输出其他章节：\n## 6. 复现要点与方法适用边界",
    "请只输出以下内容，不要输出其他章节：\n## 7. 局限性与未解决问题",
]


def generate_visual_evidence(md_content: str, ordered_images):
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


def generate_sectional_main_report(main_agent: LLMClient, combined_prompt: str) -> str:
    parts = []

    for idx, task in enumerate(SECTION_TASKS):
        completed_context = "\n\n".join(parts)
        if len(completed_context) > 18000:
            completed_context = completed_context[-18000:]

        prompt = f"""
{combined_prompt}

【已完成章节（仅供衔接参考）】
{completed_context if completed_context else '尚无已完成章节'}

【当前写作任务】
{task}
"""
        part = main_agent.generate([prompt])
        cleaned = clean_section_output(part, keep_h1=(idx == 0))
        parts.append(cleaned)

    return normalize_report_markdown("\n\n".join([p for p in parts if p.strip()]))


def render_analysis_ui(pdf_bytes):
    file_hash = hash(pdf_bytes)
    if st.session_state.get("current_pdf_hash") != file_hash:
        st.session_state.current_pdf_hash = file_hash
        st.session_state.final_main_report = ""
        st.session_state.temp_images = {}
        st.session_state.final_text_report = ""
        st.session_state.final_vision_reports = ""
        st.session_state.source_md_content = ""

        result = analyze_pdf_with_modal(pdf_bytes)
        if result and result.get("status") == "success":
            md_content = result.get("markdown", "")
            raw_images = result.get("images", {})
            ordered_images = sort_images_by_doc_order(md_content, raw_images)
            ordered_images_dict = dict(ordered_images)

            st.session_state.source_md_content = md_content
            st.session_state.temp_images = ordered_images_dict

            with st.spinner("文本专家正在抽取论文事实、方法与实验证据..."):
                text_agent = LLMClient(
                    sys_prompt=TEXT_AGENT_PROMPT,
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL,
                )
                text_prompt = f"请严格按系统提示完成论文事实抽取。以下为论文原始结构化 markdown：\n\n{build_source_pack(md_content, max_chars=60000)}"
                text_report = text_agent.generate([text_prompt])
                st.session_state.final_text_report = text_report

            with st.spinner(f"视觉专家正在分析 {len(ordered_images)} 张关键图表，并补齐局部上下文..."):
                vision_summaries = generate_visual_evidence(md_content, ordered_images)
                st.session_state.final_vision_reports = vision_summaries

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

            with st.spinner("主编代理正在分章节生成第 1-7 节研究型报告..."):
                main_agent = LLMClient(
                    sys_prompt=MAIN_AGENT_PROMPT,
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL,
                )
                core_report = generate_sectional_main_report(main_agent, combined_prompt)

            with st.spinner("研究路线代理正在生成第 8 节可执行创新路线..."):
                research_agent = LLMClient(
                    sys_prompt=RESEARCH_AGENT_PROMPT,
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL,
                )
                research_prompt = f"""
{combined_prompt}

【已生成的第1-7节主报告】
{core_report}

【当前写作任务】
请只输出：
## 8. 面向后续研究的可执行创新路线
"""
                section8 = research_agent.generate([research_prompt])
                section8 = clean_section_output(section8, keep_h1=False)

            final_report = normalize_report_markdown(core_report + "\n\n" + section8)

            with st.spinner("审校代理正在核查无证据结论、图表ID与章节遗漏..."):
                auditor = LLMClient(
                    sys_prompt=REPORT_AUDITOR_PROMPT,
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL,
                )
                audit_prompt = f"""
【原始 markdown】
{source_pack}

【Text Agent 输出】
{st.session_state.final_text_report}

【Vision Agent 输出】
{st.session_state.final_vision_reports}

【当前报告】
{final_report}

【合法图片ID】
{available_img_ids}
"""
                audit_result = auditor.generate([audit_prompt])

            if re.search(r"RESULT\s*:\s*FAIL", audit_result, flags=re.I):
                with st.spinner("根据审校意见进行最终修订..."):
                    revise_prompt = f"""
{combined_prompt}

【当前报告】
{final_report}

【审校意见】
{audit_result}

请严格根据审校意见修订整篇 Markdown 报告，只输出修订后的最终报告。
"""
                    final_report = normalize_report_markdown(main_agent.generate([revise_prompt]))

            st.session_state.final_main_report = final_report

    if st.session_state.final_main_report:
        st.success("论文全维度深度透视报告已生成！")

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
                file_name="Report.md",
                use_container_width=True,
            )
        with col2:
            pdf_markdown = embed_base64_images(
                st.session_state.final_main_report,
                st.session_state.temp_images,
            )
            download_pdf_component(pdf_markdown)


# ==========================================
# 模块 7: 侧边栏及前端 UI 定义
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
# 模块 8: 全局应用状态机初始化
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


# ==========================================
# 模块 9: 业务路由分发与主循环
# ==========================================
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
