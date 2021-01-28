Ch02. 추천 서비스 준비

01. 파이썬이 없다면, 추천 시스템이 가능했을까?

(1) 파이썬 라이브러리 주도로 개발하게 됨.

- 다양한 라이브러리를 포함하여 개발할 수 있음.

개발 도구는 크게 주피터나 구글 colab이 있음. 둘 중 선택하면 됨.

- github guide
https://guides.github.com/activities/hello-world/
- python tutorial
https://colab.research.google.com/github/cs231n/cs231n.github.io/blob/master/python-colab.ipynb


02. Numpy, Pandas와 친해지기 (feat. Tensor)
github 
https://github.com/thj0309/thj0309.bigdata.github.io/blob/main/2_fastcampus/1_02_02_pandas.ipynb
https://github.com/thj0309/thj0309.bigdata.github.io/blob/main/2_fastcampus/1_02_02_numpy.ipynb

* 추천 시스템을 구현하기 위해서는 많은 복습과 노력이 필요함

* 간단 요약 (PANDAS)

import pandas as pd

pd_series = pd.Series(index=['a','b','c'], data=[1,2,3]) # 1차원 데이터 집합

* padnas series의 기초정보

print('차원 : ', pd_series.ndim)

print('형태 : ', pd_series.shape)

print('총 원소의 수 : ', pd_series.size)

print('값 : ', pd_series.values)

print('index : ', pd_series.index) # padnas index 활용하여 데이터 찾기

print(pd_series.iloc[0]) # iloc는 integer location을 찾아서 조회함

print(pd_series.loc['e']) # index를 알아야만함

* 데이터 삭제하기 (영구 삭제 X)

pd_series.drop('d')

* 데이터 삭제하기 (완전)

pd_series.drop('d', inplace=True)

* padnas dataframe 만들기( 2차원 집합 만들기)

data = {
    'A' : np.arange(15)
    , 'B' : np.random.randint(low =0,high=15,size=(15))
    , 'C' : np.random.rand(15)
}

data_df= pd.DataFrame(data) # 데이터 프레임 선언

data_df.head(7) # 최초 7건

data_df.tail(6) # 마지막 6건

data_df.shape # 집합 형태 보기

data_df[1:3] # indexing & slicing

data_df.loc[1] # 1번 index 데이터 보기

data_df.drop('D', axis=1, inplace=True) # D열삭제

data_df.sort_index(axis=0, ascending=True) # 정렬하기

-----------------------------------------------------------------------------------------

* 간단 요약 (NUMPY)

import numpy as np
import warnings
warnings.filterwarnings("ignore") # 버전차이로 생기는 오류 무시

* 리스트, 행렬 선언

a = [1,2, 3] --> [1, 2, 3]

b = ['a', 'b', 'c']  --> ['a', 'b', 'c']

c = [['a'], ['a','b'], ['a','c']] --> [['a'], ['a', 'b'], ['a', 'c']]

d = np.array([1,2, 3]) --> [1 2 3]

e = np.array(['a', 'b', 'c']) --> ['a' 'b' 'c']

f = np.array([['a'], ['a','b'], ['a','c']]) --> [list(['a']) list(['a', 'b']) list(['a', 'c'])]

* append 하기

c.append(['new item'])

* 특정값 채우기

print(np.ones((1,5)) (1,5) 행렬에 1로 채우기

print(np.zeros((1,5) 행렬에 0로 채우기


* 배열 또는 행렬 안에 순차적으로 증가하는 리스트 만들기

print(np.arange(10))

print(np.arange(3, 7, dtype=np.float))

print(np.arange(3, 7, 2))

* 행렬 선언하기

mat1 = np.array([[1,2,3],[4,5,6]])

--> [[1 2 3]
 [4 5 6]]
 
*  random으로 matrix 만들기

mat2 = np.random.randint(low=1, high=10, size=(3,2))

--> [[1 8]
 [8 8]
 [2 7]]

mat3 = np.random.rand(3,2)

--> [[0.85236058 0.90423549]
 [0.48843569 0.77266385]
 [0.70511257 0.38879966]]
 
 
* indexing
a = [1,3,5,7,9,11]

* print(a[2], a[5], a[-1])

--> 5 11 11


* slicing

b= [2,4,6,8,10]

print(b[2:])

--> [6, 8, 10]

print(b[:2])

--> [2, 4]

print(b[:])

--> [2, 4, 6, 8, 10]

* numpy의 reshape(pytorch의 view와 비교)

matt1 = np.random.rand(6,3) <-- (row, col)
print(matt1)

--> [[0.33168586 0.85913014 0.62195975]
 [0.15767386 0.07727356 0.71972504]
 [0.59797132 0.16730903 0.8281012 ]
 [0.14123067 0.64382609 0.65508103]
 [0.41432323 0.65193856 0.81270469]
 [0.87221708 0.45118063 0.51891783]]

* reshape <-- -1: all
print(matt1.reshape(1,-1).shape)

--> (1, 18)

print(matt1.reshape(1,-1))

--> [[0.33168586 0.85913014 0.62195975 0.15767386 0.07727356 0.71972504
  0.59797132 0.16730903 0.8281012  0.14123067 0.64382609 0.65508103
  0.41432323 0.65193856 0.81270469 0.87221708 0.45118063 0.51891783]]

print("---------------------------------")

print(matt1.reshape(-1,1).shape)

--> (18, 1)

print(matt1.reshape(-1,1))

--> [[0.33168586]
 [0.85913014]
 [0.62195975]
 [0.15767386]
 [0.07727356]
 [0.71972504]
 [0.59797132]
 [0.16730903]
 [0.8281012 ]
 [0.14123067]
 [0.64382609]
 [0.65508103]
 [0.41432323]
 [0.65193856]
 [0.81270469]
 [0.87221708]
 [0.45118063]
 [0.51891783]]
 
 * reshape 기능
 
print(matt1.reshape(2,9).shape)

--> (2, 9)

print(matt1.reshape(2,9))

--> [[0.33168586 0.85913014 0.62195975 0.15767386 0.07727356 0.71972504
  0.59797132 0.16730903 0.8281012 ]
 [0.14123067 0.64382609 0.65508103 0.41432323 0.65193856 0.81270469
  0.87221708 0.45118063 0.51891783]]
  
print(matt1.reshape(2,5).shape)

print(matt1.reshape(2,5))

--> matt1는 3,6으로 구성된 행렬이므로 에러가 발생함


* tensor 형태로 나타내기

print(matt1.reshape(3,2,3).shape)

-->  (3, 2, 3)

print(matt1.reshape(3,2,3))

--> [[[0.33168586 0.85913014 0.62195975]
  [0.15767386 0.07727356 0.71972504]]

 [[0.59797132 0.16730903 0.8281012 ]
  [0.14123067 0.64382609 0.65508103]]

 [[0.41432323 0.65193856 0.81270469]
  [0.87221708 0.45118063 0.51891783]]]

* matrix or tensor 형태에서도 slicing 가능하다.

matt2 = np.arange(24).reshape(-1,4)

print(matt2)

--> [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]
 [16 17 18 19]
 [20 21 22 23]]
 
 
-----------------------------------------------------------------------------------------


03. PyTorch와 Tensor 미리보기
pytorch는 과정 중간 쯤 깊게 이해하는 시간이 있음.
Tensor는 3차원 이상의 데이터를 활용하는 방법을 가르쳐줌.



04. MovieLens 데이터 분석 1 (유저, 영화, 평점 데이터 분석)
영화 평점과 관련된 dataset (추천시스템에서 많이 활용함)
numpy와 pandas를 활용하여 'MovieLens' dataset를 핸들링하는 것을 이해할 수 있음.
(0) 데이터 수로 small datas / full dataset으로 나누어 제공함. (https://grouplens.org/datasets/movielens/)



05. MovieLens 데이터 분석 2 (영화 정보)



06. 행렬이 없다면 추천이 가능했을까?





