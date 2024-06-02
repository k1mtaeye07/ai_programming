import streamlit as st
import pandas as pd
import numpy as np

with st.echo(code_location="below"):

    # 데이터 로드
    movies = pd.read_csv('/home/ubuntu/project/ml-latest-small/movies.csv')
    ratings = pd.read_csv('/home/ubuntu/project/ml-latest-small/ratings.csv')

    # 제목에서 연도를 추출하여 새로운 열 생성
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)', expand=False)
    movies['year'] = pd.to_numeric(movies['year'], errors='coerce')

    # 제목 열에서 연도 제거
    movies['title'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()

    # 평점 데이터와 영화 데이터를 병합하여 제목과 연도 포함
    ratings_merged = pd.merge(ratings, movies, on='movieId')
    ratings_merged = ratings_merged.drop(columns=['timestamp'])

    # Streamlit
    st.title("Collaborative Filtering 결과 출력")
    st.write("CF 알고리즘을 활용하여 결과를 출력합니다.")

    # 사용자 아이디 입력
    user_id = st.sidebar.text_input("사용자 아이디를 입력하세요:", "1")
    user_id = int(user_id)
    button_clicked = st.sidebar.button("결과 갱신")

    # 입력된 사용자 아이디에 따른 평점 필터링
    user_ratings = ratings_merged[ratings_merged['userId'] == user_id]
    rated_movies = user_ratings[['title', 'rating', 'genres', 'year']].sort_values(by='rating', ascending=False)

    # 데이터셋과 사용자 평점을 위한 탭
    tabs = st.tabs(["Dataset", "Your Ratings"])
    with tabs[0]:
        st.write("데이터셋 예제 입니다.")
        st.write(ratings_merged.head(10))

    with tabs[1]:
        st.write("사용자님이 평점을 매긴 영화 리스트 입니다.")
        st.write(rated_movies)
