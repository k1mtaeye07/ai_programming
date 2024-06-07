import numpy as np
import pandas as pd
from gensim.models import Word2Vec

# Word2Vec 모델 로드
model_path = './model/word2vec_movie.model'
model = Word2Vec.load(model_path)

# 벡터와 메타데이터 저장을 위한 파일 경로
vectors_file = './model/vectors.tsv'
metadata_file = './model/metadata.tsv'

# 단어와 벡터 추출
words = model.wv.index_to_key
vectors = [model.wv[word] for word in words]

# 벡터를 .tsv 파일로 저장
with open(vectors_file, 'w', encoding='utf-8') as f:
    for vector in vectors:
        f.write('\t'.join([str(x) for x in vector]) + '\n')

# 메타데이터를 .tsv 파일로 저장
with open(metadata_file, 'w', encoding='utf-8') as f:
    for word in words:
        f.write(word + '\n')

print("tsv finished.")

