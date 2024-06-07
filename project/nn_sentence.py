import pandas as pd
import numpy as np
from konlpy.tag import Okt
import json

file_path = './dataset/movie.csv'
movies_df = pd.read_csv(file_path)

# 결측값이 있는 행 제거 ('plot' 열에서만 결측값 제거)
movies_df = movies_df.dropna(subset=['plot'])

okt = Okt()

# 명사 추출 함수
def extract_nouns(plot):
    return okt.nouns(plot)

# copus 생성
sentences = movies_df['plot'].apply(extract_nouns).tolist()
# 빈 문장 필터링
sentences = [sentence for sentence in sentences if sentence]
print(sentences[:2])

movies_df['nouns'] = sentences
movies_df[['title', 'nouns', 'page_url']].to_json('./dataset/movie_nouns.json', orient='records', force_ascii=False)

