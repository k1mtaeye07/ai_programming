import pandas as pd
from konlpy.tag import Okt
import re

file_path = '../dataset/movie.csv'
movies_df = pd.read_csv(file_path)

#명사 추출 
okt = Okt()

def extract_unique_nouns(text):
    nouns = okt.nouns(text)
    #중복 제거
    unique_nouns = list(set(nouns))
    #단어 정규화
    filtered_nouns = [word for word in unique_nouns if len(word) > 1]
    return ' '.join(filtered_nouns)

movies_df['plot_nouns'] = movies_df['plot'].apply(extract_unique_nouns)

output_file_path = '../dataset/movie_with_nouns.csv'
movies_df.to_csv(output_file_path, index=False)



