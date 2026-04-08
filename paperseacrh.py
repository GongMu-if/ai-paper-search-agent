import re
import requests
from openai import OpenAI
import time
import streamlit as st
import datetime
import base64

# ==========================================
# 0. 页面基本设置 & 全局配置
# ==========================================
st.set_page_config(page_title="AI 论文检索 Agent", page_icon="📚", layout="wide")
st.title("📚 AI 智能论文检索 Agent")
st.markdown("通过多轮自主检索与阅读，为您精准挖掘 Top 6 前沿文献。")

# 【这里统一定义了所有的 API 密钥和地址！】
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

QWEN_API_KEY = st.secrets["QWEN_API_KEY"]
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MODAL_API_URL = st.secrets["MODAL_API_URL"]

# ==========================================
# 1. 侧边栏：用户输入区
# ==========================================
with st.sidebar:
    st.header("检索配置")
    user_topic = st.text_input("研究方向", value="")

    user_requirements = st.text_area(
        "具体筛选要求 (分点填写)",
        value=""
    )

    allow_preprint = st.radio(
        "是否接受预印本 (如 arXiv)?",
        ("排除预印本 (仅限正规期刊/会议)", "接受预印本")
    )

    start_button = st.button("开始检索", type="primary")
    st.divider()
    st.header("直接解析论文")
    sidebar_pdf = st.file_uploader("上传 PDF 立即深度解读", type="pdf", key="sb_pdf")

# ==========================================
# 2. 提示词定义 (三大 Agent 专属提示词)
# ==========================================

# --- 文本专家提示词 ---
TEXT_AGENT_PROMPT = """
你是一个极其严谨的资深学术大牛（Text Agent）。你的任务是对提供的论文 Markdown 文本进行深度、全面的解构与综述。

【核心纪律】
1. 绝对忠于原文：你的所有总结、分析和提取必须100%基于我提供的文本。严禁使用你自带的先验知识进行推理、延申或“脑补”。如果原文没写，请明确标出“原文未提及”。
2. 细节为王：在分析“方法论”时，绝不能只给宏观概念，必须连贯、详细地还原文章的核心架构和技术细节。

【工作流与输出格式】
请严格按照以下 Markdown 结构输出你的分析结果，不要输出多余的寒暄：

# 论文深度文本综述
## 1. 研究背景 (Background)
## 2. 现有研究的困境 (Current Issues)
## 3. 本文核心目标 (Problem Addressed)
## 4. 方法论与架构设计 (Methodology)
## 5. 作用效果与实验表现 (Effects & Results)
## 6. 研究不足与局限性 (Limitations)
"""

# --- 视觉专家提示词 (新加入) ---
VISION_AGENT_PROMPT = """
你是一个顶级的学术图表解析专家（Vision Agent）。
你的任务是深度解读用户提供的学术论文截图（包括数据图、流程图、系统架构图等）。

请严格按以下结构输出：
1. 【图表定位】：这是什么类型的图？（如折线图、神经网络架构图），它在说明什么核心概念？
2. 【数据/逻辑提取】：如果是数据图，指出明显的趋势、极值、对比差异；如果是流程图，按原理解释核心节点和流转逻辑。
3. 【一句话结论】：总结这张图证明了什么。

注意：如果图片看起来像是无意义的单行公式、极小的图标或排版噪音，请直接回复：“⚠️ 这是一张排版噪音图片，无实质学术信息。”
"""

# --- 搜索专家提示词生成器 ---
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
# 3. 工具与通用大模型客户端 (升级支持图片)
# ==========================================
seen_paper_ids = set()

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
                    
    # 【新加入：专门给 Vision Agent 看图的方法】
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
# 4. 核心功能函数 (解析与渲染，含多模态调用)
# ==========================================
def analyze_pdf_with_modal(pdf_file_bytes):
    with st.spinner("🚀 正在唤醒云端 GPU 引擎，深度解析公式与版面..."):
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

def render_analysis_ui(pdf_bytes):
    result = analyze_pdf_with_modal(pdf_bytes)
    
    if result and result.get("status") == "success":
        st.success("🎉 论文底层结构化解析成功！")
        
        md_content = result["markdown"]
        images_dict = result.get("images", {})
        
        tab1, tab2, tab3 = st.tabs(["📄 原始 Markdown", "🖼️ 提取的图表", "🧠 Text Agent 深度精读"])
        
        with tab1:
            st.markdown("### 论文文本流提取结果")
            st.markdown(md_content)
            st.download_button("下载原始 Markdown", md_content, file_name="raw_analysis.md")
            
        with tab2:
            st.markdown("### 论文多模态图表提取结果")
            if images_dict:
                cols = st.columns(2)
                for i, (img_name, img_base64) in enumerate(images_dict.items()):
                    with cols[i % 2]:
                        with st.container(border=True):
                            st.image(base64.b64decode(img_base64), caption=img_name, use_container_width=True)
                            
                            # 状态保持：为每张图准备一个报告存储位
                            state_key = f"vision_report_{img_name}"
                            if state_key not in st.session_state:
                                st.session_state[state_key] = ""
                            
                            # 【核心交互：唤醒 Vision Agent (Qwen-VL)】
                            if st.button(f"👁️ 唤醒 Vision Agent 解读图表", key=f"btn_{img_name}"):
                                with st.spinner("Qwen-VL 视觉专家正在看图思考..."):
                                    try:
                                        vision_agent = LLMClient(
                                            sys_prompt=VISION_AGENT_PROMPT, 
                                            model="qwen-vl-max", # 阿里通义千问最强视觉模型
                                            api_key=QWEN_API_KEY, 
                                            base_url=QWEN_BASE_URL
                                        )
                                        user_req = f"请详细解读这张图片（它在原论文中的标识为 {img_name}）。"
                                        report = vision_agent.generate_with_images(user_req, [img_base64])
                                        st.session_state[state_key] = report
                                    except Exception as e:
                                        st.error(f"视觉解析失败: {e}")
                            
                            if st.session_state[state_key]:
                                st.markdown("---")
                                st.markdown(f"**🤖 专家解读：**\n{st.session_state[state_key]}")
            else:
                st.info("ℹ️ 本篇论文未提取到图表信息。")
                
        with tab3:
            st.markdown("### 让大模型为您庖丁解牛")
            st.info("点击下方按钮，系统将唤醒专属的 Text Agent，为您瞬间提取全篇最硬核的技术骨架。")
            
            if "text_agent_report" not in st.session_state:
                st.session_state.text_agent_report = ""
                
            if st.button("🚀 唤醒 Text Agent 开始精读", type="primary"):
                with st.spinner("Text Agent 正在逐字精读、提炼方法论与实验细节，请耐心等待 (约 30-60 秒)..."):
                    try:
                        # 【核心交互：唤醒 Text Agent (DeepSeek)】
                        text_agent = LLMClient(sys_prompt=TEXT_AGENT_PROMPT, model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
                        user_request = f"请精读以下论文的完整 Markdown 内容，并严格按要求给出深度综述：\n\n{md_content}"
                        
                        report = text_agent.generate([user_request])
                        st.session_state.text_agent_report = report
                    except Exception as e:
                        st.error(f"Text Agent 运行出错: {e}")
            
            if st.session_state.text_agent_report:
                st.divider()
                st.markdown(st.session_state.text_agent_report)
                st.download_button("下载 AI 精读报告", st.session_state.text_agent_report, file_name="AI_Review_Report.md", type="primary")
                
    else:
        st.error("解析失败，请检查后端状态。")

# ==========================================
# 5. 主程序运行逻辑与状态机
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

# --- 侧边栏交互：开始检索 ---
if start_button:
    if not user_topic:
        st.warning("请填写研究方向！")
    else:
        seen_paper_ids.clear()
        sys_prompt = get_system_prompt(user_requirements, allow_preprint)
        # 【核心交互：唤醒 Search Agent (DeepSeek)】
        st.session_state.agent = LLMClient(sys_prompt=sys_prompt, model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        st.session_state.prompt_history = [f"用户请求: {user_topic}"]
        
        st.session_state.app_state = "RUNNING"
        st.session_state.loop_count = 0
        st.session_state.has_provided_feedback = False
        st.session_state.feedback_start_time = None
        st.session_state.ui_logs = [] 
        st.rerun()

# --- 【逻辑 A】：最高优先级 - 侧边栏快速解析 ---
if sidebar_pdf:
    st.markdown("---")
    st.info("检测到侧边栏上传文件，正在进入【直接解析模式】...")
    render_analysis_ui(sidebar_pdf.read())
    st.stop() 

# --- 【逻辑 B】：正常的 Agent 搜索流程 ---
if st.session_state.app_state == "IDLE":
    st.markdown("""
    ### 系统使用指南
    欢迎使用 AI 智能论文检索分析 Agent。为了获得最佳的文献推荐体验，请参考以下操作规范：

    1.除了在侧边栏填写宏观的研究方向外，请在具体筛选要求中尽量明确研究的子分支、目标应用场景或特定的文章类型。

    2.本系统支持人机协同的动态优化。首轮文献挖掘完成后，Agent 会展示初步筛选的 Top 6 候选文献并征求您的意见。每位用户在单次任务中享有 **1 次**修改要求的机会。若结果偏离预期，您可直接指出大模型理解的偏差或补充新的约束条件，Agent 将据此进行第二轮定向纠偏与深度检索。

    3.为保障系统计算资源的有效流转，在首轮检索结果展示并进入满意度确认环节后，若超过 **30 分钟** 未收到您的反馈指令，系统将默认您对当前文献组合满意，并自动结束本次任务。

    4.您也可直接对已有论文进行分析，从而获取论文精读报告。
    """)

if st.session_state.app_state != "IDLE":
    st.markdown("### Agent 检索轨迹")
    for log in st.session_state.ui_logs:
        with st.expander(log["title"], expanded=False):
            st.markdown(log["content"])

if st.session_state.app_state == "RUNNING":
    st.info("Agent 正在自主检索文献，请稍候...")
    current_step_container = st.container()
    
    with st.spinner("Agent 正在思考和执行工具..."):
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
        if st.button("满意，结束检索", use_container_width=True):
            st.session_state.app_state = "COMPLETED"
            st.rerun()
    with col2:
        with st.popover("不满意，修改要求", use_container_width=True):
            new_req = st.text_area("指出不符合要求的地方：")
            if st.button("提交"):
                st.session_state.prompt_history.append(f"用户反馈: {new_req}")
                st.session_state.has_provided_feedback = True 
                st.session_state.app_state = "RUNNING"
                st.rerun()

elif st.session_state.app_state == "COMPLETED":
    st.success("任务已完成！")
    st.markdown("### 最终确认的 Top 6 论文推荐")
    with st.container(border=True):
        st.markdown(st.session_state.final_result)
    
    st.divider()
    st.header("📄 论文深度解读")
    st.info("上传上述推荐论文的 PDF，系统将进行深度解析。")
    
    uploaded_pdf = st.file_uploader("上传 PDF 文件进行解析", type="pdf", key="bottom_pdf")
    if uploaded_pdf:
        render_analysis_ui(uploaded_pdf.read())

    if st.button("开启全新检索", type="primary"):
        st.session_state.clear()
        st.rerun()
