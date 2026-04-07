import re
import requests
from openai import OpenAI
import time
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import datetime

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

# ==========================================
# 2. 动态生成系统提示词
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
# 5. UI 渲染区 (完美实现滚轮向下继承效果)
# ==========================================
# ---------------- 新增：初始界面的使用说明 ----------------
if st.session_state.app_state == "IDLE":
    st.markdown("""
    ### 系统使用指南
    欢迎使用 AI 智能论文检索 Agent。为了获得最佳的文献推荐体验，请参考以下操作规范：

    1.除了在侧边栏填写宏观的研究方向外，请在具体筛选要求中尽量明确研究的子分支、目标应用场景或特定的文章类型。

    2.本系统支持人机协同的动态优化。首轮文献挖掘完成后，Agent 会展示初步筛选的 Top 6 候选文献并征求您的意见。每位用户在单次任务中享有 **1 次**修改要求的机会。若结果偏离预期，您可直接指出大模型理解的偏差或补充新的约束条件，Agent 将据此进行第二轮定向纠偏与深度检索。
    
    3.为保障系统计算资源的有效流转，在首轮检索结果展示并进入满意度确认环节后，若超过 **30 分钟** 未收到您的反馈指令，系统将默认您对当前文献组合满意，并自动结束本次任务。
   
    """)
    
    st.info("👈 请在左侧边栏配置您的检索参数，并点击“开始检索”启动 Agent。")
# 第一部分：自上而下，永远先渲染历史思考过程
if st.session_state.app_state != "IDLE":
    st.markdown("### Agent 检索轨迹")
    for log in st.session_state.ui_logs:
        # 为了不让页面太长，过去的步骤自动折叠，标题清晰
        with st.expander(log["title"], expanded=False):
            st.markdown(log["content"])

# 第二部分：紧接着在下方，根据当前状态渲染最新内容
if st.session_state.app_state == "RUNNING":
    st.info("Agent 正在自主检索文献，请稍候...")
    
    # 用一个空的容器来实时显示当前正在进行的一步
    current_step_container = st.container()
    
    with st.spinner("Agent 正在思考和执行工具..."):
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
            
            # 1. 大模型生成回答
            output = st.session_state.agent.generate(st.session_state.prompt_history)
            st.session_state.prompt_history.append(output)
            
            # 2. 存入日志并实时在网页最下方画出来
            log_entry = {
                "title": f"Agent 运行日志 (第 {i} 步)", 
                "content": f"**Agent思考与决策:**\n```text\n{output}\n```"
            }
            st.session_state.ui_logs.append(log_entry)
            with current_step_container.expander(log_entry["title"], expanded=True):
                st.markdown(log_entry["content"])
            
            # 3. 解析动作
            action_match = re.search(r"Action:\s*(.*)", output, re.DOTALL)
            if not action_match:
                st.session_state.prompt_history.append("Observation: 错误：未找到Action格式，请严格按照要求输出。")
                continue
            
            action_str = action_match.group(1).strip()
            
            # 4. 判断是否结束 (过滤掉所有前缀，只留纯 Markdown 结果)
            if action_str.startswith("Finish"):
                clean_result = re.sub(r"^Finish\s*[:：\[]?\s*", "", action_str).rstrip("]").strip()
                st.session_state.final_result = clean_result 
                
                if not st.session_state.has_provided_feedback:
                    st.session_state.app_state = "WAITING_FEEDBACK"
                    st.session_state.feedback_start_time = time.time()
                else:
                    st.session_state.app_state = "COMPLETED"
                
                st.rerun() # 这里一刷新，代码就会流转到下面的 WAITING_FEEDBACK
                break
            
            # 5. 执行工具
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
    # --- 超时检测逻辑 ---
    if st.session_state.feedback_start_time:
        elapsed_time = time.time() - st.session_state.feedback_start_time
        remaining_time = 1800 - elapsed_time
        
        if remaining_time <= 0:
            st.session_state.app_state = "COMPLETED"
            st.rerun()
        st_autorefresh(interval=10000, key="feedback_timer")    
        mins_left = int(remaining_time // 60)
        secs_left = int(remaining_time % 60)
        st.caption(f"系统将在 {mins_left} 分钟后自动确认结果并结束任务。")

    # --- 紧接着思考过程的下方，渲染满意度测试 ---
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
    # --- 紧接着思考过程的下方，渲染最终完成界面 ---
    st.success("任务已完成！")
    
    if st.session_state.has_provided_feedback == False and st.session_state.feedback_start_time:
        elapsed = time.time() - st.session_state.feedback_start_time
        if elapsed > 1800:
            st.warning("提示：由于超过 30 分钟未响应，系统已为您自动确认最终结果。")

    st.markdown("### 最终确认的 Top 6 论文推荐")
    with st.container(border=True):
        st.markdown(st.session_state.final_result)
    
    st.write("") 
    if st.button("开启全新检索", type="primary"):
        st.session_state.clear()
        st.rerun()
