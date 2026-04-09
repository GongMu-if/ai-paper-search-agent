# ==========================================
# 模块 1: 依赖导入与页面基础配置
# ==========================================
import re
import requests
from openai import OpenAI
import time
import streamlit as st
import datetime
import base64
import markdown
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

# ==========================================
# 模块 3: 核心 Agent 提示词库
# ==========================================
MAIN_AGENT_PROMPT = """
你是一个顶级的学术期刊主编。你的任务是将一份来自文本专家的“论文深度文本综述”和来自视觉专家的“多张图表视觉解析”，无缝合并为一篇专业、易读、深度的“论文全维度透视报告”。你的目标是让读者直接通过你的报告彻底读懂文章的全部细节。内容必须极其详尽，拒绝简略。
【撰写原则与核心纪律】
1. 纯段落结构：在正文撰写中，禁止使用生硬的无序列表或数字编号打断阅读节奏。每一部分必须以逻辑严密、承上启下的段落形式呈现。绝不能将一整个章节写成一个巨大的自然段！必须在阐述不同概念、不同模块、不同实验结论时，进行合理的换行分段。利用段首主旨句引导阅读，确保排版疏密有致，富有呼吸感。
2. 深度图文融合：绝不能仅仅是将文本和图片解析简单拼接。你必须将图片解析中的关键发现作为证据，自然地编织进文本的逻辑链条中。如果视觉图表中包含了文本中缺失的细节（例如模型架构中的隐藏层连接、未在正文提及的实验对比分支），你必须用它来补充文本描述。
3. 深度图文融合与原图插入：必须将图片和表格解析中的关键论据，自然嵌入文本逻辑中。在合适的学术段落处，直接插入对应的图片，格式严谨为：![图X：学术化图注](图片占位符)。
4. 严谨客观的学术话语体系：禁止使用任何口语化、感情色彩浓烈的词汇（如“娓娓道来”、“没说透”、“绝不能”）。必须使用标准的学术论述用语（如“本文旨在”、“研究表明”、“机制剖析”、“局限性在于”）。
5. 图表全局统一编号：你必须根据正文逻辑顺序列出图表。所有插入的图片和表格，必须重新规范命名，格式为“图1：XXX”、“表1：XXX”。严禁保留原系统生成的哈希长串或无意义的名称（如“pdf”）。

【报告结构要求】
请严格按照以下 Markdown 结构输出你的报告：
# 论文全维度深度透视报告

## 1. 核心综述
结合原文摘要，用一段详尽且专业的文字，全面定义本文的宏观贡献。准确提炼出该文章区别于前人工作的核心创新点，并说明这些创新在当前领域内的学术价值。

## 2. 问题背景与研究动机
将领域背景娓娓道来，详细还原该研究方向目前面临的核心痛点。分析现有主流技术路线的具体做法及其优缺点。说明这些技术瓶颈如何限制了领域的发展，从而在逻辑上推导出本论文“非解决不可”的紧迫性，并引出作者试图解决该问题的核心动机。（如原文有阐述背景概念的引导图，请在此处插入并结合图文解析）。

## 3. 技术方案全解
这是本报告的重中之重。首先，结合文本描述和视觉解析中的“系统架构图”，详细还原其研究的整体架构和数据流向，并在此处插入整体模型架构图。随后，针对架构图中的每一个关键模块，将文本的理论机制与视觉图表中的结构细节深度结合，详尽解释其内部算法、设计逻辑以及每个模块在整个模型中起到的决定性作用。确保读者能够看着架构图，读懂每一步的运转逻辑。

## 4. 关键证据与实验结果
首先详细交代实验的数据集与运行环境。接着，将视觉专家对图表的解读与文本的实验分析进行深度耦合。在对应的实验描述后插入关键的实验结果图表（如对比折线图、消融实验柱状图、可视化效果图等）。针对每一个实验，不仅要描述对比了哪些基线模型，更要指着插入的图表，给出具体的性能提升数据（定量分析），并解释这些数据究竟证明了架构中哪个模块的有效性（定性分析）。

## 5. 局限性与未来蓝图
结合作者的自述以及你作为主编在审视其实验设计时的批判性思考，客观指出该文章没说透的地方或实验环境中的薄弱环节。最后，基于本文的理论基础和技术细节，为读者提出几个具体、可执行的未来研究方向，并以连贯的段落说明每个方向的潜在创新价值及预期会遇到的难点。
"""

TEXT_AGENT_PROMPT = """
你是一个极其严谨、具有像素级解析能力的资深学术大牛（Text Agent）。你的任务是对提供的论文文本进行深度、全量的信息解构。
【核心纪律】
1. 绝对忠于原文与禁止幻想：你的所有总结、分析和提取必须百分之百基于我提供的文本。严禁使用大模型自带的先验知识进行推理、延伸或脑补。如果原文没有提及某项细节，绝对不能凭空捏造。
2. 拒绝压缩与细节为王：内容描述必须极其详尽，以确保后续处理能够获得全面的文本解读。保留文章中提及的所有关键参数、模型名称和逻辑因果。
3. 纯段落输出：在以下所有章节的输出中，禁止使用任何项目符号（如无序列表或数字编号）。每一个小节的内容必须以逻辑连贯的纯段落形式进行撰写，报告以分段形式呈现。
4. 对每个内容都要描述出该内容为了干什么做了什么事解决了什么问题。

【输出结构】
请严格按照以下 Markdown 结构输出你的分析结果：

# 论文深度文本综述

## 1. 摘要翻译
直接将原文的摘要部分完整、准确地翻译为中文，保持学术表达的严谨性，不加任何额外的解读，以纯段落形式输出。

## 2. 引言与背景
对文章的研究方向和宏观背景进行具备高度可读性的介绍。详细指出该研究方向目前面临的核心问题。深入分析当前存在的主流技术路线，分别阐述这些路线的具体做法，并客观论述它们的优势与缺点。说明在目前的研究方向下，由于什么具体的缺失或瓶颈，导致了什么不良的后果与局限。接着，详细阐明为了克服上述种种问题，本论文提出了什么全新的方法，并顺理成章地引出该文章的核心创新点。

## 3. 方法论与模型架构
首先对整个模型架构和系统的整体运行流程进行宏观且详尽的描述，理清输入到输出的全链路。随后，针对架构中的每一个具体模块进行极其详尽的解构，详细描述各个模块的内部运作机制、算法逻辑，并明确指出每一个模块在整个系统中具体实现了什么不可或缺的功能。请将这些内容写成连贯的段落，通过承上启下的关联词展现各模块间的耦合关系。

## 4. 实验设计与结果分析
首先详细描述实验所采用的数据集特征以及具体的操作、软硬件环境配置。随后，针对文章中进行的每一个具体实验（包括主实验、消融实验等），详细描述该实验具体是怎么操作的，选用了哪些对照组，最终产生了什么具体的定量与定性效果。特别需要强调的是，必须明确指出每一个实验的具体结果究竟证明了架构中哪一个模块的有效性或哪一个理论假设的成立。全篇以严密的分段落形式展开。

## 5. 结论、不足与未来方向
详尽地总结该文章的核心贡献以及在领域内取得的突破。根据原文作者的自述，客观描述该研究目前仍然存在的局限性与不足之处，并详细阐述作者在文中指出的未来研究方向。所有内容以自然段落呈现。
"""

VISION_AGENT_PROMPT = """
你是一个顶级的学术图表解析专家（Vision Agent）。
你的任务是深度解读用户提供的学术论文截图（包括数据图、流程图、系统架构图、数据表格等）。

请严格按以下结构输出：
1. 【图表学术定位】：这是什么类型的图表？（如收敛折线图、消融实验柱状图、网络拓扑图）。如果图表本身带有原文编号（如 Figure 1, Table II），请务必在此提取并指出。
2. 【数据与机制提取】：若是数据图表，使用学术语言精准提取核心趋势、极值对比与显著性差异；若是架构/流程图，解释其核心节点的设计逻辑。
3. 【学术推论】：总结该图表在全文逻辑链中提供了怎样的实证支撑。

注意：如果是无意义的单行公式、排版噪音或页眉页脚，请直接回复：“⚠️ 排版噪音，无实质学术信息。”
"""

def get_system_prompt(requirements, preprint_rule):
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
# 模块 4: 工具函数与通用处理库
# ==========================================
def reconstruct_abstract(inverted_index: dict) -> str:
    if not inverted_index: return ""
    word_index = [(pos, word) for word, positions in inverted_index.items() for pos in positions]
    word_index.sort(key=lambda x: x[0])
    return " ".join([word for _, word in word_index])

def search_and_detail_papers(query: str) -> str:
    global seen_paper_ids
    api_key = st.secrets["S2_API_KEY"]
    email = "gaoym3@mails.neu.edu.cn"
    s2_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=100&year=2025-2026&fields=paperId,title,abstract,year,externalIds,venue"
    headers = {"x-api-key": api_key}

    try:
        time.sleep(2)
        s2_response = requests.get(s2_url, headers=headers, timeout=20)
        s2_response.raise_for_status()
        papers = s2_response.json().get("data", [])

        if not papers: return f"Observation: 未找到关于'{query}'的近一年论文。"

        results = []
        for p in papers:
            paper_id = p.get("paperId", "No ID")
            if paper_id in seen_paper_ids: continue

            title = p.get("title", "No Title")
            doi = p.get("externalIds", {}).get("DOI", "")
            venue = p.get("venue") or "Unknown Venue/Preprint"

            s2_abstract = p.get("abstract")
            final_abstract, openalex_mark = "", ""

            if s2_abstract and s2_abstract.strip():
                final_abstract = s2_abstract.strip()
            else:
                try:
                    oa_url = f"https://api.openalex.org/works/https://doi.org/{doi}" if doi else f"https://api.openalex.org/works?filter=title.search:{title}"
                    oa_res = requests.get(oa_url, params={"mailto": email}, timeout=20)
                    if oa_res.status_code == 200:
                        work_data = oa_res.json().get("results", [None])[0] if "results" in oa_res.json() else oa_res.json()
                        if work_data and work_data.get("abstract_inverted_index"):
                            final_abstract = reconstruct_abstract(work_data["abstract_inverted_index"])
                            openalex_mark = " [via OpenAlex]"
                except:
                    pass

            if not final_abstract or len(final_abstract.strip()) < 10: continue

            seen_paper_ids.add(paper_id)
            results.append(f"Title: {title}\n  - Venue: {venue}\n  - S2_ID: {paper_id} | DOI: {doi}{openalex_mark}\n  - Abstract: {final_abstract[:800]}...")
            if len(results) >= 30: break

        if not results: return f"Observation: 搜索到关于 '{query}' 的论文，但均为已读或无摘要，未能提供新信息。"
        return f"Observation: 提取到 {len(results)} 篇全新论文：\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"Observation: 搜索出错 - {str(e)}"

available_tools = {"search_and_detail_papers": search_and_detail_papers}

def embed_base64_images(md_text, images_dict):
    def replace_img(match):
        alt_text = match.group(1)
        img_placeholder = match.group(2).strip()
        clean_placeholder = re.sub(r'[^a-zA-Z0-9]', '', img_placeholder).lower()
        
        for img_name, b64 in images_dict.items():
            clean_key = re.sub(r'[^a-zA-Z0-9]', '', img_name).lower()、
            if clean_key and (clean_key in clean_placeholder or clean_placeholder in clean_key):、
                return f"![{alt_text}](data:image/jpeg;base64,{b64})\n<div style='text-align: center; font-size: 0.9em; color: #555;'><b>{alt_text}</b></div>"
        
        return match.group(0) 
    return re.sub(r'!\[(.*?)\]\((.*?)\)', replace_img, md_text)

def download_pdf_component(md_text):
    html_content = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
    html_code = f"""
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
        <style>
            body {{
                font-family: 'Times New Roman', 'SimSun', serif; /* 修改为更符合学术规范的字体 */
                color: #000; line-height: 1.6; padding: 20px;
            }}
            /* 核心修复：防止元素在分页处被截断 */
            img, table, pre, code, blockquote {{
                page-break-inside: avoid;
                break-inside: avoid;
            }}
            h1, h2, h3, h4 {{
                color: #000; margin-top: 24px;
                page-break-after: avoid; /* 标题后不强制分页 */
                break-after: avoid;
            }}
            p {{ page-break-inside: avoid; }}
            
            img {{ max-width: 100%; height: auto; margin: 20px auto; display: block; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.9em; }}
            th, td {{ border: 1px solid #000; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            
            .download-btn {{
                display: block; width: 100%; padding: 12px;
                background-color: #4CAF50; color: white; border: none; /* 改为沉稳的绿色 */
                border-radius: 4px; font-size: 16px; cursor: pointer; font-weight: bold;
            }}
            .download-btn:hover {{ background-color: #45a049; }}
        </style>
    </head>
    <body>
        <button class="download-btn" onclick="generatePDF()">📥 导出标准版学术 PDF 报告</button>
        <div id="report-content" style="display: none;">
            {html_content}
        </div>
        <script>
            function generatePDF() {{
                var element = document.getElementById('report-content');
                element.style.display = 'block'; 
                var opt = {{
                    margin:       [15, 15, 15, 15],
                    filename:     '论文深度透视报告.pdf',
                    image:        {{ type: 'jpeg', quality: 0.98 }},
                    html2canvas:  {{ scale: 2, useCORS: true, letterRendering: true }},
                    jsPDF:        {{ unit: 'mm', format: 'a4', orientation: 'portrait' }},
                    pagebreak:    {{ mode: ['css', 'legacy'] }} // 核心修复：启用 CSS 分页规则
                }};
                html2pdf().set(opt).from(element).save().then(() => {{
                    element.style.display = 'none'; 
                }});
            }}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=70)

def analyze_pdf_with_modal(pdf_file_bytes):
    with st.spinner("正在唤醒云端 GPU 引擎，深度解析公式与版面... "):
        try:
            files_payload = {"file": ("paper.pdf", pdf_file_bytes, "application/pdf")}
            response = requests.post(MODAL_API_URL, files=files_payload, timeout=600)
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    return result
                else:
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
                    model=self.model, messages=messages, temperature=0.2
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
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        messages.append({"role": "user", "content": content_list})

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0.2
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
def render_analysis_ui(pdf_bytes):
    file_hash = hash(pdf_bytes)
    if st.session_state.get("current_pdf_hash") != file_hash:
        st.session_state.current_pdf_hash = file_hash
        st.session_state.final_main_report = ""
        st.session_state.temp_images = {}

        result = analyze_pdf_with_modal(pdf_bytes)
        if result and result.get("status") == "success":
            md_content = result["markdown"]
            st.session_state.temp_images = result.get("images", {})

            with st.spinner("文本专家正在精读全篇文本..."):
                text_agent = LLMClient(sys_prompt=TEXT_AGENT_PROMPT, api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
                text_report = text_agent.generate([f"请详尽解析此论文：\n{md_content}"])

            vision_summaries = ""
            if st.session_state.temp_images:
                with st.spinner(f"视觉专家正在分析 {len(st.session_state.temp_images)} 张关键图表..."):
                    vision_agent = LLMClient(sys_prompt=VISION_AGENT_PROMPT, model="qwen3.6-plus", api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
                    for name, b64 in st.session_state.temp_images.items():
                        v_res = vision_agent.generate_with_images(f"解析图表 {name}", [b64])
                        vision_summaries += f"\n--- 图表标识: {name} ---\n{v_res}\n"

            with st.spinner("总策宣官正在融合图文，生成终极报告..."):
                main_agent = LLMClient(sys_prompt=MAIN_AGENT_PROMPT, api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
                combined_prompt = f"【文本综述】：\n{text_report}\n\n【视觉分析库】：\n{vision_summaries}"
                st.session_state.final_main_report = main_agent.generate([combined_prompt])

    if st.session_state.final_main_report:
        st.success("论文深度透视报告已生成！")

        report_sections = re.split(r'(\[REF_IMG: .*?\])', st.session_state.final_main_report)
        for section in report_sections:
            img_match = re.match(r'\[REF_IMG: (.*?)\]', section)
            if img_match:
                img_name = img_match.group(1).strip()
                if img_name in st.session_state.temp_images:
                    st.image(base64.b64decode(st.session_state.temp_images[img_name]), caption=f"引用图表: {img_name}", use_container_width=True)
            else:
                st.markdown(section)

        st.divider()
        st.markdown("### 导出与下载")
        col1, col2 = st.columns(2)
        with col1:
            # 1. Markdown 纯文本下载
            st.download_button(
                "下载报告原文 (Markdown)", 
                st.session_state.final_main_report, 
                file_name="Report.md",
                use_container_width=True
            )
        with col2:
            pdf_markdown = embed_base64_images(
                st.session_state.final_main_report, 
                st.session_state.temp_images
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
        help="要求越具体，Agent 挖掘的文献越精准。"
    )
    
    allow_preprint = st.radio(
        "文献收录标准", 
        ("仅限同行评审文献 (排除预印本)", "接受预印本 (如 arXiv)")
    )
    
    start_button = st.button("开始智能检索", type="primary", use_container_width=True)

    st.divider()

    st.header("文献直读")
    sidebar_pdf = st.file_uploader(
        "上传本地 PDF 进行结构化解析", 
        type="pdf", 
        key="sb_pdf",
        help="跳过检索步骤，直接对已有文献生成精读报告"
    )
    start_analyze_button = st.button(
        "开始解读", 
        type="primary", 
        key="start_analyze_btn",
        use_container_width=True,
        disabled=not sidebar_pdf  # 如果 sidebar_pdf 为空，则禁用按钮
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
    st.session_state.final_vision_reports = {}
if "parse_success" not in st.session_state:
    st.session_state.parse_success = False

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
        st.session_state.agent = LLMClient(sys_prompt=sys_prompt, model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
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
                "content": f"**Agent思考与决策:**\n```text\n{output}\n```"
            }
            st.session_state.ui_logs.append(log_entry)
            with current_step_container.expander(log_entry["title"], expanded=True):
                st.markdown(log_entry["content"])

            action_match = re.search(r"Action:\s*(.*)", output, re.DOTALL)
            if not action_match: continue

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
        use_container_width=True
    )
    if bottom_start_btn and uploaded_pdf:
        render_analysis_ui(uploaded_pdf.read())

    if st.button("开启全新检索轮次", type="primary"):
        st.session_state.clear()
        st.rerun()
