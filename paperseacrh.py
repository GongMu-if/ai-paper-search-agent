import re
import requests
from openai import OpenAI
import time
import streamlit as st

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
    # 根据用户选择动态生成预印本规则
    if preprint_rule == "排除预印本 (仅限正规期刊/会议)":
        preprint_prompt = "严禁选择 Venue 为 'Unknown Venue/Preprint' 的预印本论文。"
    else:
        preprint_prompt = "可以接受预印本论文。"

    return f"""
你是一个科研论文搜索专家。你的任务是根据研究方向，在近一年内的论文中筛选六篇。

# 你的工作流程：
1. 通过`search_and_detail_papers`工具来获得相关论文的标题、摘要和Introduction。
2. 每次搜索后，阅读标题、摘要和Introduction。如果符合用户的具体要求，将其记录在你的 Thought 中作为“备选池”累加。
3. 每次阅读完一批论文后，学习同义词或近义词，作为下一次的 `search_and_detail_papers` 查询词。
4. 如果不符合或无法获取摘要：摒弃该论文。
5. 最终在备选池中选出最好的六篇来作为结果。

# 用户的具体筛选要求：
{requirements}
{preprint_prompt}

# 输出格式要求：
Thought: [你的思考逻辑，包括学习到的新关键词]
Action: [执行工具或结束]

Action格式：
1. search_and_detail_papers(query="关键词")
2. Finish[最终选出的六篇论文及其推荐理由]
"""


# ==========================================
# 3. 工具与 Agent 定义 (保持你的逻辑基本不变)
# ==========================================
seen_paper_ids = set()
#转码
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
                        work_data = oa_res.json().get("results", [None])[
                            0] if "results" in oa_res.json() else oa_res.json()
                        if work_data and work_data.get("abstract_inverted_index"):
                            final_abstract = reconstruct_abstract(work_data["abstract_inverted_index"])
                            openalex_mark = " [via OpenAlex]"
                except:
                    pass

            if not final_abstract or len(final_abstract.strip()) < 10: continue

            seen_paper_ids.add(paper_id)
            results.append(
                f"Title: {title}\n  - Venue: {venue}\n  - S2_ID: {paper_id} | DOI: {doi}{openalex_mark}\n  - Abstract: {final_abstract[:800]}...")
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

        # 增加重试机制防断线
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
# 4. 主程序运行逻辑 (与 Web UI 绑定)
# ==========================================
API_KEY = st.secrets["DEEPSEEK_API_KEY"]
BASE_URL = "https://api.deepseek.com"

# 初始化系统状态变量
if "app_state" not in st.session_state:
    st.session_state.app_state = "IDLE"  # 状态：IDLE, RUNNING, WAITING_FEEDBACK, COMPLETED
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
        st.session_state.has_provided_feedback = False # 每次全新检索重置机会
        st.session_state.feedback_start_time = None # 重置时间
        st.rerun()

# ---------------- B. Agent 运行逻辑 ----------------
if st.session_state.app_state == "RUNNING":
    st.info(f"Agent 正在自主检索文献，请稍候...")
    log_container = st.container()
    
    with st.spinner("Agent 正在思考和执行工具..."):
        while True:
            st.session_state.loop_count += 1
            i = st.session_state.loop_count
            
            if i == 1:
                loop_reminder = "系统提示: 第一次循环开始，请直接使用用户的原始研究方向作为query执行search_and_detail_papers。"
            else:
                loop_reminder = (
                    "系统提示: 请继续执行检索。如果你在对比后认为备选池中的Top 6论文已经完美符合用户的全部要求，"
                    "请输出 Action: Finish[最终选出的Top6论文及推荐理由]。否则请继续 search_and_detail_papers。"
                )

            st.session_state.prompt_history.append(loop_reminder)
            
            # 1. 思考
            output = st.session_state.agent.generate(st.session_state.prompt_history)
            st.session_state.prompt_history.append(output)
            
            with log_container.expander(f"Agent 运行日志 (第 {i} 步)", expanded=False):
                st.markdown(f"**Agent思考与决策:**\n```text\n{output}\n```")
            
            # 2. 解析
            action_match = re.search(r"Action:\s*(.*)", output)
            if not action_match:
                st.session_state.prompt_history.append("Observation: 错误：未找到Action格式，请严格按照要求输出。")
                continue
            action_str = action_match.group(1).strip()
            
            # 3. 结束判断 (核心逻辑修改点)
            if action_str.startswith("Finish"):
                final_answer = re.match(r"Finish\[(.*)\]", action_str, re.DOTALL)
                if final_answer:
                    st.session_state.final_result = final_answer.group(1)
                else:
                    st.session_state.final_result = output 
                
                # 判断是否还有反馈机会
                if not st.session_state.has_provided_feedback:
                    # 还有机会，进入等待反馈状态
                    st.session_state.app_state = "WAITING_FEEDBACK"
                else:
                    # 已经反馈过一次了，直接强制结束，防止多次修改导致幻觉
                    st.session_state.app_state = "COMPLETED"
                
                st.rerun()
                break
            
            # 4. 执行工具
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

# ---------------- C. 用户满意度测试 (仅限一次) ----------------
elif st.session_state.app_state == "WAITING_FEEDBACK":
    # 核心修改：超时检测逻辑
    if st.session_state.feedback_start_time:
        elapsed_time = time.time() - st.session_state.feedback_start_time
        remaining_time = 1800 - elapsed_time # 1800秒 = 30分钟
        
        if remaining_time <= 0:
            # 已经超过 30 分钟，自动切换到完成状态
            st.session_state.app_state = "COMPLETED"
            st.rerun()
        
        # 可选：在界面显示倒计时
        mins_left = int(remaining_time // 60)
        st.caption(f"系统将在 {mins_left} 分钟后自动确认结果并结束任务。")

    st.markdown("### 阶段性检索结果")
    st.info(st.session_state.final_result)
    
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

# ---------------- D. 任务彻底完成 ----------------
elif st.session_state.app_state == "COMPLETED":
    st.balloons()
    st.success("任务已完成！")
    # 如果是因为超时自动完成的，可以加个提示
    if st.session_state.has_provided_feedback == False and st.session_state.feedback_start_time:
        elapsed = time.time() - st.session_state.feedback_start_time
        if elapsed > 1800:
            st.warning("提示：由于超过 30 分钟未响应，系统已为您自动确认最终结果。")

    st.markdown("### 最终确认的 Top 6 论文推荐")
    st.info(st.session_state.final_result)
    
    if st.button("开启全新检索"):
        st.session_state.clear()
        st.rerun()
