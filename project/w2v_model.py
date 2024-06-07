import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import json

# 불용어 리스트
stopwords = ['하다', '위해', '영화', '이야기', '로부터', '맞닥뜨리', '과연', '심지어', '직접']

# 명사 리스트 로드
with open('./dataset/movie_nouns.json', 'r', encoding='utf-8') as f:
    movies_data = json.load(f)
movies_df = pd.DataFrame(movies_data)

# 불용어 제거
movies_df['nouns'] = movies_df['nouns'].apply(lambda nouns: [noun for noun in nouns if noun not in stopwords])

# Word2Vec 모델 학습
nouns_list = movies_df['nouns'].tolist()
model = Word2Vec(nouns_list,
                vector_size=200,
                window=10,
                min_count=2,
                workers=4,
                sg=1)
model.save("./model/word2vec_movie.model")

# 벡터화 함수
def nouns_to_vector(nouns, model):
    words = [word for word in nouns if word in model.wv]
    if not words:
        return np.zeros(model.vector_size)
    word_vectors = np.array([model.wv[word] for word in words])
    return word_vectors.mean(axis=0)

# 줄거리 벡터화 및 저장
movies_df['vector'] = movies_df['nouns'].apply(lambda nouns: nouns_to_vector(nouns, model))
movies_df['vector'] = movies_df['vector'].apply(lambda x: x.tolist())
movies_df.to_csv('./dataset/movie_vectors.csv', index=False)
print("벡터 저장 완료")

