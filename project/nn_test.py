import pandas as pd
from konlpy.tag import Okt
import re

okt = Okt()

plot = "제1차 세계대전이 한창인 1917년. 독일군에 의해 모든 통신망이 파괴된 상황 속에서 영국군 병사 '스코필드'(조지 맥케이)와 '블레이크'(딘-찰스 채프먼)에게 하나의 미션이 주어졌다. 함정에 빠진 영국군 부대의 수장 '매켄지' 중령(베네딕트 컴버배치)에게 '에린무어' 장군(콜린 퍼스)의 공격 중지 명령을 전하는 것! 둘은 16 명의 아군과 '블레이크'의 형(리차드 매든)을 구하기 위해 전쟁터 한복판을 가로지르며 사투를 이어가는데.."

# 오타 정규화
print(okt.normalize(plot))

def extract_unique_nouns(text):
    nouns = okt.nouns(text)
    print(f"nouns : {nouns}")
    # 두 글자 이상인 명사만 추출
    unique_nouns = list(set(nouns))
    filtered_nouns = [word for word in unique_nouns if len(word) > 1]
    return ' '.join(filtered_nouns)

extracted_nouns = extract_unique_nouns(plot)
print(extracted_nouns)

