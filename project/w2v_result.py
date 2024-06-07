import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from konlpy.tag import Okt
from sklearn.metrics.pairwise import cosine_similarity

file_path = './dataset/movie_vectors.csv'
movies_df = pd.read_csv(file_path)

# 벡터 데이터 리스트로 변환
movies_df['vector'] = movies_df['vector'].apply(lambda x: np.array(eval(x)))

# Word2Vec 모델 로드
model_path = "./model/word2vec_movie.model"
model = Word2Vec.load(model_path)

okt = Okt()

def extract_nouns(text):
    return okt.nouns(text)

# 벡터화 함수
def vectorize_summary(summary, model):
    words = summary
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

# 추천 함수
def get_recommendations(input_text, model, data):
    input_nouns = extract_nouns(input_text)
    input_vector = vectorize_summary(input_nouns, model).reshape(1, -1)

    # 코사인 유사도 계산
    similarities = cosine_similarity(input_vector, np.vstack(data['vector'].values))
    data['similarity'] = similarities[0]

    # 유사도가 높은 순으로 정렬하여 상위 10개 추천
    recommendations = data.sort_values(by='similarity', ascending=False).head(10)
    return recommendations[['title','similarity','page_url']]

# 입력 텍스트에 대한 영화 추천
input_text = "아이언맨"
recommendations = get_recommendations(input_text, model, movies_df)

# 추천 결과 출력
print("추천 영화")
pd.set_option('display.max_colwidth', None)
print(recommendations)

