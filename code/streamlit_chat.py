import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client=OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
              base_url="https://api.deepseek.com")
st.title("多模型对话|System Prompt切换")
# 侧边栏配置
with st.sidebar:#st.sidebar代表网页左侧侧边栏，with语法表示下面所有组件都放在侧边栏内。
    st.header("⚙️ 参数配置")
    # 模型下拉选择
    selected_model=st.selectbox(
        "选择模型",
        options=["deepseek-chat","deepseek-code"],
        index=0#默认选中第一个deepseek-chat
    )#用户选中的模型名称，保存到变量selected_model，后续传给 API。
    # System Prompt输入框
    system_prompt=st.text_area(
        "System Prompt 系统角色",
        value="你是一个乐于助人的AI助手,回答简洁清晰。",#文本框默认内容，系统提示词
        height=130#文本框高度
    )
    st.divider()#画一条水平分割线，美化侧边栏界面。
#历史对话
#初始化历史
if "history" not in st.session_state:
    st.session_state.history=[]
#输入
user_input=st.chat_input("请输入问题")
if user_input:
        # 用户消息加入对话历史
    st.session_state.history.append({"role": "user", "content": user_input})
    messages=[
        {"role":"system","content":system_prompt},
        *st.session_state.history#解包运算符，把历史对话列表全部展开，拼接在 system 后面。
    ]
    resp=client.chat.completions.create(
        model=selected_model,
        messages=messages
    )
    ai_content=resp.choices[0].message.content
    st.session_state.history.append({"role":"assistant","content":ai_content})
#显示
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):#创建聊天气泡。
        st.write(msg["content"])#在气泡内输出消息文本。
#清空对话按钮
if st.button("清空对话"):
    st.session_state.history=[]
    st.rerun()#页面强制重新运行脚本，页面刷新，对话界面清空。