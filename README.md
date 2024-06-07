# 한국어 영화 추천시스템 

- 데이터셋은 네이버영화 줄거리를 크롤링한 파일을 사용합니다.
  - (출처 : https://github.com/jbose038/naver-movie-recommendation/tree/master/dataset/movies04293.csv)
- 한국어 형태소분석기 `konlpy` 를 사용합니다.
  - https://konlpy.org/ko/latest/

이 모델을 테스트 하기 위해 `konlpy`, `jdk` 설치가 필요합니다.

## 설치

```
pip install konlpy
sudo apt-get install openjdk-11-jdk
```

## 환경변수
vi ~/.bashrc 으로 파일을 열어 아래 세줄을 추가 합니다.
```
export JAVA_HOME=/usr/lib/jvm/ava-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
export CLASSPATH=$CLASSPATH:$JAVA_HOME/jre/lib/ext:$JAVA_HOME/lib/tools.jar
```
source ~/.bashrc 명령어를 입력하여 환경변수를 적용 합니다.

## 사용법

colab 예제 : `w2v.ipynb`

아래 절차대로 실행합니다.
1. `nn_sentence.py` : 줄거리를 `okt.nouns()` 함수를 사용해 명사추출 후 copus를 생성합니다.
2. `w2v_model.py` : 불용어를 제거 후 모델을 생성하고, 코사인 유사도로 비교하기 위해 줄거리말뭉치를 벡터화 하여 `movie_vectors.csv` 따로 저장합니다.
3. `w2v_result.py` : 입력텍스트를 코사인유사도로 비교하여, 유사도 높은 순으로 상위10개를 출력합니다.

## 파일설명
- `dataset` : `.csv`, `.json` 형식의 데이터 디렉토리 
  - `project/dataset/movie.csv` : 네이버영화 크롤링데이터
  - `project/dataset/movie_nouns.json` : 줄거리를 통해 명사 추출 후, copus 생성
  - `project/dataset/movie_vectors.csv` : 줄거리 벡터화 데이터
- `model` : word2vec 모델 디렉토리
  -  `project/model/word2vec_movie.model` : word2vec 줄거리 학습모델


## good case
- input_text = "어벤져스 토니 스타크"

![g1](https://github.com/k1mtaeye07/ai_programming/assets/106365897/a14a5955-3353-4fa7-a11a-85844df643c5)


- input_text = "공포 병원 배경"


![g2](https://github.com/k1mtaeye07/ai_programming/assets/106365897/f65ea38b-39de-4809-b205-521fb0a3fc0e)


## bad case
- input_text = "야호" ("야호" 키워드벡터값이 모델에 존재 하지 않는 경우)


![n1](https://github.com/k1mtaeye07/ai_programming/assets/106365897/2675243d-f56e-46b7-aa45-0c9507bfff8a)

- input_text = "컴퓨터 바이러스" (키워드에 적합하지 않은 아쉬운결과)

![a1](https://github.com/k1mtaeye07/ai_programming/assets/106365897/fd989327-cc10-494d-8af7-42df4d842f32)


## 임베딩 프로젝터를 사용하여 시각화하기
- 링크 : https://projector.tensorflow.org/
- label = "음악"

![e1](https://github.com/k1mtaeye07/ai_programming/assets/106365897/a8a734c4-b3fa-4ba8-b8a9-d615ee2a06ec)

- label = "병원"

![e3](https://github.com/k1mtaeye07/ai_programming/assets/106365897/9254c00c-4489-445e-a1ec-05e95eb695da)


