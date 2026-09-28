import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# 1. 配 API Key（走 .env，不写死）
load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com")

# 2. 页面
st.title("我的第一个 AI 聊天机器人")

# 3. 历史对话
if "history" not in st.session_state:#网页刷新不会丢失里面的数据；每个浏览器窗口独立一份对话
    st.session_state.history = []#初始记忆列表为空

# 4. 输入
user_input = st.chat_input("说点什么吧")
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=st.session_state.history#把全部历史对话一次性发给大模型，实现记忆上下文
    )
    st.session_state.history.append(
        {"role": "assistant", "content": resp.choices[0].message.content})#拿到模型返回的第一条回答文本

# 5. 显示
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):#生成聊天气泡
        st.write(msg["content"])#在气泡内渲染消息文本
# 清空对话按钮
if st.button("清空对话"):
    st.session_state.history = []
    st.rerun()