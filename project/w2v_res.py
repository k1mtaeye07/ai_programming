import pandas as pd
import numpy as np
from konlpy.tag import Okt
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import joblib

file_path = '../dataset/movie_with_nouns.csv'
data = pd.read_csv(file_path)
model = Word2Vec.load("../model/word2vec_movie.model")

#print(model.wv.most_similar("크리스마스"))
#print(model.wv.similarity())

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
    print(f" input_nouns : {input_nouns}")
    input_vector = vectorize_summary(input_nouns, model).reshape(1, -1)
    
    # 코사인 유사도 계산
    similarities = cosine_similarity(input_vector, np.vstack(data['summary_vector'].values))
    data['similarity'] = similarities[0]
    
    # 유사도가 높은 순으로 정렬하여 상위 10개 추천
    recommendations = data.sort_values(by='similarity', ascending=False).head(10)
    return recommendations[['title', 'similarity']]


#input_text = "크리스마스 어린이 가족"
input_text = "무서운 공포 병원배경"

recommendations = get_recommendations(input_text, model, data)

print(recommendations)

