import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

from konlpy.tag import Okt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import multiprocessing

file_path = '../dataset/movie_with_nouns.csv'

data = pd.read_csv(file_path)

# 불용어 리스트
stopwords = ['하다','위해']

okt = Okt()

# 학습에 사용하지 않을 불용어 적용 및 토크나이저
tokenized_data = []
for sentence in (data['plot_nouns']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    tokenized_data.append(stopwords_removed_sentence)

cores = multiprocessing.cpu_count()

model = Word2Vec(tokenized_data, #토큰
                    vector_size=200, #임베딩 된 벡터의 차원
                    window=5, #컨텍스트 윈도우 크기
                    hs=1, #계층적 softmax 사용
                    min_count=10, #단어 최소 빈도 수 제한 (빈도가 적은 단어 학습X)
                    sg=1, #0은 CBOW, 1은 Skip-gram.
                    workers=cores-1, # CPU수
                    epochs=5) #학습 반복 횟수

model.save("../model/word2vec_movie.model")

print(model.wv.vectors.shape)


'''
# 각 줄거리를 벡터 변환
def vectorize_summary(summary, model):
    words = summary.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

data['summary_vector'] = data['plot_nouns'].apply(lambda x: vectorize_summary(x, model))

# 데이터 분할 (train 0.8, test 0.2)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 벡터와 라벨 분리
X_train = np.vstack(train_data['summary_vector'].values)
y_train = train_data['genre']
X_test = np.vstack(test_data['summary_vector'].values)
y_test = test_data['genre']

# 로지스틱 회귀 모델 학습
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# 테스트 데이터로 예측
y_pred = classifier.predict(X_test)

# 성능 평가
print(classification_report(y_test, y_pred))
'''
