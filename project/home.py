import streamlit as st

#sidebar
st.sidebar.title("여기는 사이드바")
st.sidebar.write("여기는 사이드바 설명")

st.sidebar.selectbox(
    '옵션을 선택하세요',
    ['opt1','opt2','opt3']
)

st.title("Hi, I'm Movie Recommender! 🎬🍿")
st.write("""
    welcome to movie recommender
""")
