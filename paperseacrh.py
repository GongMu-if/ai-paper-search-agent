import re
import requests
from openai import OpenAI
import time
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import datetime
import base64

# ==========================================
# 0. 页面基本设置
# ==========================================
st.set_page_config(page_title="AI 论文检索 Agent", page_icon="📚", layout="wide")
st.title("📚 AI 智能论文检索 Agent")
st.markdown("通过多轮自主检索与阅读，为您精准挖掘 Top 6 前沿文献。")

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
# 2. 动态生成系统提示词
# ==========================================
TEXT_AGENT_PROMPT = """
你是一个极其严谨的资深学术大牛（Text Agent）。你的任务是对提供的论文 Markdown 文本进行深度、全面的解构与综述。

【核心纪律】
1. 绝对忠于原文：你的所有总结、分析和提取必须100%基于我提供的文本。严禁使用你自带的先验知识进行推理、延申或“脑补”。如果原文没写，请明确标出“原文未提及”。
2. 细节为王：在分析“方法论”时，绝不能只给宏观概念，必须连贯、详细地还原文章的核心架构和技术细节。

【工作流与输出格式】
请严格按照以下 Markdown 结构输出你的分析结果，不要输出多余的寒暄：

# 论文深度文本综述

## 1. 研究背景 (Background)
[精读摘要和引言部分，总结本研究在什么宏观背景下开展。]

## 2. 现有研究的困境 (Current Issues)
[明确指出目前的领域/前人的研究中存在什么具体的痛点、缺陷或未解之谜。]

## 3. 本文核心目标 (Problem Addressed)
[精炼总结本文究竟要克服上述的哪些问题，提出了什么核心主张。]

## 4. 方法论与架构设计 (Methodology)
[这是最重要的部分。请详细精读文章的中间部分。详细、连贯地剖析本文的整体研究架构、算法设计、实验流程或系统构建步骤。不要遗漏关键的技术细节。]

## 5. 作用效果与实验表现 (Effects & Results)
[提取文章的实验部分，给出客观的效果分析。方法是否生效？在什么指标上取得了什么具体成果？]

## 6. 研究不足与局限性 (Limitations)
[精准提取文章“结论与讨论(Conclusion/Discussion)”部分作者自己承认的不足、局限性或未来的改进方向。]
"""

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
在此处使用 Markdown 格式直接列出选出的 6 篇论文（必须包含 1.标题 2.Venue 3.DOI 4.推荐理由）。不要写多余的话。
"""

MODAL_API_URL = st.secrets["MODAL_API_URL"]
def analyze_pdf_with_modal(pdf_file_bytes):
    """调用 Modal 云端接口解析 PDF"""
    with st.spinner("🚀 正在唤醒云端 GPU 引擎，深度解析公式与版面..."):
        try:
            # 【核心修改】：必须构建一个包含(文件名, 数据, 类型)的元组，
            # 这里的 "file" 这个 key，必须和后端 parse_pdf(file: UploadFile) 中的参数名完全一致！
            files_payload = {
                "file": ("paper.pdf", pdf_file_bytes, "application/pdf")
            }
            
            # 使用 files= 发送，requests 会自动帮你生成正确的 multipart/form-data 标头
            response = requests.post(MODAL_API_URL, files=files_payload, timeout=600)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    return result
                else:
                    st.error(f"解析内部错误: {result.get('message')}")
                    # 如果有 traceback，打印出来方便 Debug
                    if "trace" in result:
                        with st.expander("查看详细报错"):
                            st.code(result["trace"])
            else:
                # 把 422 的具体报错打印出来
                st.error(f"服务器响应错误: {response.status_code}")
                st.write(response.text) # 这一行是 Debug 的金钥匙
                
        except Exception as e:
            st.error(f"连接云端失败: {str(e)}")
            
    return None

def render_analysis_ui(pdf_bytes):
    """统一的解析结果展示区"""
    result = analyze_pdf_with_modal(pdf_bytes)
    if result and result.get("status") == "success":
        st.success("解析成功！")
        st.markdown("### 论文深度解析内容")
        st.markdown(result["markdown"])
        
        # 展示图片
        images = result.get("images", {})
        if images:
            st.divider()
            for img_name, img_base64 in images.items():
                st.image(base64.b64decode(img_base64), caption=img_name)
        st.download_button("下载 Markdown", result["markdown"], file_name="analysis.md")
    else:
        st.error("解析失败，请检查后端。")
# ==========================================
# 3. 工具与 Agent 定义
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

# ==========================================
# 4. 主程序运行逻辑
# ==========================================
API_KEY = st.secrets["DEEPSEEK_API_KEY"]
BASE_URL = "https://api.deepseek.com"

def render_analysis_ui(pdf_bytes):
    """统一的解析结果与多智能体展示区"""
    result = analyze_pdf_with_modal(pdf_bytes)
    
    if result and result.get("status") == "success":
        st.success("论文底层结构化解析成功！")
        
        md_content = result["markdown"]
        images_dict = result.get("images", {})
        
        tab1, tab2, tab3 = st.tabs(["原始 Markdown", "提取的图表", "Text Agent 深度精读"])
        
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
                        st.image(base64.b64decode(img_base64), caption=img_name, use_container_width=True)
            else:
                st.info("本篇论文未提取到图表信息。")
                
        with tab3:
            st.markdown("### 让大模型为您庖丁解牛")
            st.info("点击下方按钮，系统将唤醒专属的 Text Agent，为您瞬间提取全篇最硬核的技术骨架。")
            
            if "text_agent_report" not in st.session_state:
                st.session_state.text_agent_report = ""
                
            if st.button("唤醒 Text Agent 开始精读", type="primary"):
                with st.spinner("Text Agent 正在逐字精读、提炼方法论与实验细节，请耐心等待 (约 30-60 秒)..."):
                    try:
                        text_agent = LLMClient(sys_prompt=TEXT_AGENT_PROMPT, api_key=API_KEY, base_url=BASE_URL)
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

# --- 状态初始化 ---
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

# --- 点击开始按钮 ---
if start_button:
    if not user_topic:
        st.warning("请填写研究方向！")
    else:
        seen_paper_ids.clear()
        sys_prompt = get_system_prompt(user_requirements, allow_preprint)
        st.session_state.agent = LLMClient(sys_prompt=sys_prompt, api_key=API_KEY, base_url=BASE_URL)
        st.session_state.prompt_history = [f"用户请求: {user_topic}"]
        
        st.session_state.app_state = "RUNNING"
        st.session_state.loop_count = 0
        st.session_state.has_provided_feedback = False
        st.session_state.feedback_start_time = None
        st.session_state.ui_logs = [] 
        st.rerun()

# ==========================================
# 5. UI 渲染区
# ==========================================

# --- 【逻辑 A】：最高优先级 - 侧边栏快速解析入口 ---
if sidebar_pdf:
    st.markdown("---")
    st.info("检测到侧边栏上传文件，正在进入【直接解析模式】...")
    # 这里的 sidebar_pdf.read() 会获取二进制流
    render_analysis_ui(sidebar_pdf.read())
    
    # 强制停止后续渲染，确保页面只显示解析结果
    st.stop() 


# --- 【逻辑 B】：正常的 Agent 搜索流程 ---

# 1. 初始/空闲状态：显示欢迎页
if st.session_state.app_state == "IDLE":
    st.markdown("""
    ### 系统使用指南
    欢迎使用 AI 智能论文检索分析 Agent。为了获得最佳的文献推荐体验，请参考以下操作规范：

    1.除了在侧边栏填写宏观的研究方向外，请在具体筛选要求中尽量明确研究的子分支、目标应用场景或特定的文章类型。

    2.本系统支持人机协同的动态优化。首轮文献挖掘完成后，Agent 会展示初步筛选的 Top 6 候选文献并征求您的意见。每位用户在单次任务中享有 **1 次**修改要求的机会。若结果偏离预期，您可直接指出大模型理解的偏差或补充新的约束条件，Agent 将据此进行第二轮定向纠偏与深度检索。

    3.为保障系统计算资源的有效流转，在首轮检索结果展示并进入满意度确认环节后，若超过 **30 分钟** 未收到您的反馈指令，系统将默认您对当前文献组合满意，并自动结束本次任务。

    4.您也可直接对已有论文进行分析，从而获取论文报告。
    """)

# 2. 运行中/完成状态：显示检索轨迹（历史日志）
if st.session_state.app_state != "IDLE":
    st.markdown("### Agent 检索轨迹")
    for log in st.session_state.ui_logs:
        with st.expander(log["title"], expanded=False):
            st.markdown(log["content"])

# 3. 正在搜索状态
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
            
            # 执行工具逻辑 (保持你原有的 search_and_detail_papers 调用)
            tool_match = re.search(r"(\w+)\((.*)\)", action_str)
            if tool_match:
                tool_name, args_str = tool_match.group(1), tool_match.group(2)
                raw_kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
                observation = available_tools[tool_name](**raw_kwargs) if tool_name in available_tools else "错误"
                st.session_state.prompt_history.append(f"Observation: {observation}")

# 4. 等待反馈状态 (满意度确认)
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

# 5. 任务完成状态 (展示结果 + 底部上传解析)
elif st.session_state.app_state == "COMPLETED":
    st.success("任务已完成！")
    st.markdown("### 最终确认的 Top 6 论文推荐")
    with st.container(border=True):
        st.markdown(st.session_state.final_result)
    
    st.divider()
    st.header("论文深度解读")
    st.info("上传上述推荐论文的 PDF，系统将进行深度解析。")
    
    # 底部上传入口
    uploaded_pdf = st.file_uploader("上传 PDF 文件进行解析", type="pdf", key="bottom_pdf")
    if uploaded_pdf:
        # 同样调用统一的 UI 函数
        render_analysis_ui(uploaded_pdf.read())

    if st.button("开启全新检索", type="primary"):
        st.session_state.clear()
        st.rerun()
