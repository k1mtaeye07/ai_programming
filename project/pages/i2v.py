import streamlit as st
import pandas as pd
import numpy as np

from konlpy.tag import Okt
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import joblib

movies = pd.read_csv('../project/dataset/movie_with_nouns.csv')
sel_col = ['title','plot','plot_nouns']

# Streamlit
st.title("Word2Vec 결과 출력")
st.write("결과를 출력합니다.")

#text_input
text = st.sidebar.text_input("키워드를 입력하세요.", "어벤져스 같은 슈퍼히어로 영화")
button_clicked = st.sidebar.button("결과 갱신")

tabs = st.tabs(["Dataset", "Preprocessing","Analytics","Result"])
with tabs[0]:
    st.write("데이터셋 예제 입니다.")
    st.write(movies[sel_col].sample(10))

with tabs[1]:
    st.write("전처리 과정 입니다.")
    code = '''	
        # 불용어 리스트
		stopwords = ['하다','위해','영화','이야기','로부터','맞닥뜨리','과연']

		okt = Okt()

		# 학습에 사용하지 않을 불용어 적용 및 토크나이저
		tokenized_data = []
		for sentence in (data['plot_nouns']):
		tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
                # 불용어 제거
		stopwords_removed_sentence = 
                [word for word in tokenized_sentence if not word in stopwords]
		tokenized_data.append(stopwords_removed_sentence)

		cores = multiprocessing.cpu_count()

		start_time = time.time()
		model = Word2Vec(tokenized_data, #토큰
				vector_size=400, #임베딩 된 벡터의 차원
				window=10, #컨텍스트 윈도우 크기
				hs=1, #계층적 softmax 사용
				min_count=7, #단어 최소 빈도 수 제한 (빈도가 적은 단어 학습X)
				sg=1, #0은 CBOW, 1은 Skip-gram.
				workers=cores-1, # CPU수
				epochs=25) #학습 반복 횟수

		model.save("./model/word2vec_movie.model")
        '''
    st.code(code,language='python')
with tabs[2]:
    st.write("분석")
    code = '''
    model = Word2Vec.load("./model/word2vec_movie.model")
    st.write(model.wv.most_similar(positive=['어벤져스'], negative=['멜로']))
    st.write(model.wv.similarity('어린이','어른'))
    '''
    model = Word2Vec.load("./model/word2vec_movie.model")
    model.wv.most_similar(positive=['어벤져스'], negative=['멜로'])
    model.wv.similarity('어린이','어른')

    st.code(code,language='python')

with tabs[3]:
    st.write("결과 입니다.")
    file_path = './dataset/movie_with_nouns.csv'
    data = pd.read_csv(file_path)
    model = Word2Vec.load("./model/word2vec_movie.model")

    #print(model.wv.most_similar(positive=['어벤져스'], negative=['멜로']))
    #print(model.wv.similarity('어린이','어른'))

    okt = Okt()
    def extract_nouns(text):
	    return ' '.join(okt.nouns(text))

    # 각 줄거리를 벡터로 변환
    def vectorize_summary(summary, model):
	    words = summary.split()
	    word_vectors = [model.wv[word] for word in words if word in model.wv]
	    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

    #벡터 생성
    data['summary_vector'] = data['plot_nouns'].apply(lambda x: vectorize_summary(x, model))

    # 입력 문장의 벡터 변환
    def get_recommendations(input_text, model, data):
	    input_nouns = extract_nouns(input_text)
	    input_vector = vectorize_summary(input_nouns, model).reshape(1, -1)

	    # 코사인 유사도 계산
	    similarities = cosine_similarity(input_vector, np.vstack(data['summary_vector'].values))
	    data['similarity'] = similarities[0]
	    #print(similarities)

	    # 유사도가 높은 순으로 정렬하여 상위 10개 추천
	    recommendations = data.sort_values(by='similarity', ascending=False).head(10)
	    return recommendations[['title', 'similarity','page_url']]

    input_text = "어벤져스 같은 슈퍼히어로 영화"
    #input_text = "무서운 공포 병원 생체실험 배경"


    recommendations = get_recommendations(input_text, model, data)

    #print(recommendations)
    st.write(recommendations)


