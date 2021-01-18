https://www.notion.so/Tensorflow-52e2fb60bfd24c9e92212acd8517639f

1강. Orientation

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ad868336-d19a-44a4-9073-a93b845b1db8/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ad868336-d19a-44a4-9073-a93b845b1db8/Untitled.png)

- Tensorflow 등 다양한 라이브러리 프레임워크를 이용하여, 다양한 알고리즘을 활용하여 수많은 문제를 해결할 수 있음.

2강. 목표와 전략

- Tensorflow1를 활용하여 수업함

문제를 해결하기 위하여 많은 솔루션을 고민하는 과정을 통해

효율적이고 보다 생산적인 문제 풀이방법을 찾는 것이 해당 과정에 대한 목표이자 앞으로의 전략.

3강. 지도학습의 빅픽쳐

*지도학습의 전체 과정을 4단계로 구분하여 이해하고 그림과 설명으로 프로세스를 이해함

강의 정리

1. 과거의 데이터를 준비합니다.
2. 모델의 구조를 만듭니다.
3. 데이터로 모델을 학습(FIT) 시킵니다.
4. 모델을 이용합니다.

위의 4STEP이 익숙해질 때까지~

4강. 실습환경 - Google Colaboratory

- 구글 툴을 소개하면서 실습 환경을 설정

[https://colab.research.google.com/](https://colab.research.google.com/)

5강. 표를 다루는 도구 '판다스'

판다스를 통하여 데이터 핸들링을 배우는 수업.

파일로부터 데이터를 읽어들이고 독립/종속 변수를 분리하여 데이터를 준비

Variable(변수) 에는 독립/종속 변수가 있음.

보통, 원인이 되는 변수를 독립변수로 말하며, 원인으로 인하여 결과를 만드는 변수를 종속변수라 부름. (인과/상관관계 범위 내에서)

판다스를 통해 엑셀파일을 불러들일 수 있음.

import pandas as pd

6강. 첫번째 딥러닝- 레모네이드 판매 예측

model,fit(독립, 종속, epochs=10)

// 10번을 model을 반복해라

// LOSS가 0에 가까워질 수록, 학습이 잘 된 것이다.

Loss = Error ^ 2 = (예측 - 결과) ^ 2

7강. 두번째 딥러닝 - 보스턴 집값 예측

공식을 만들어줌. 

예를 들어, y = ax + b 라는 식이 있는데

y와 x는 변수이므로, 상수인 a와 b의 값을 찾아야만한다.

이 값은 단순한 식에서는 찾기 쉽지만,

다차항이거나 고차항의 식에서는 찾기어렵다.

하지만, 앞으로 배우는 딥러닝으로 고/다차항의 식에서 상수를 찾기 쉽다.

8강. 보스턴 집값 예측

독립과 종속변수를 분리한 후, 모델의 구조를 만들고 학습시킨다

- 수식과 퍼셉트론
    - X = tf.keras.layers.Input(shape=[13])

        독립변수 13개

    - Y = tf.keras.layers.Dense(1)(X)
    종속변수 1개. 13개의 input으로 1개의 output을 만듦.
    ex) y = w1x1 + w2x2 + - - - + w13x13 + b
    b (bias) 편향
    w (weight) 가중치
    - model = tf.keras.models.Model(X, Y)
    X, Y로 모델을 생성
    - model.compile(loss='mse')

    - 퍼셉트론 2개
        - X = tf.keras.layers.input(shape=[12])
        - Y = tf.keras.layers.Dense(2)(X)
        - y1 = w1x1 + w2x2 + w12x12 + b
        - y2 = w1x2 + w2x2 + w12x12 + b
        - 총 26개 숫자를 찾아야함
    - 

9강. 아이리스 품종 분류

양적 데이터는 회귀 (Regression)

범주형 데이터는 분류 (Classification)

- 원핫인코딩
    - 숫자가 아닌 범주형 데이터인 경우 사용함
    - 컴퓨터는 2진수 숫자만 인식하기 때문에, 프로그래밍에서는 범주형 데이터를 처리할 수 없다. 그래서 컴퓨터가 이해하고 처리할 수 있는 숫자로 변환해준다. 이때, 원핫인코딩은 0과 1 2가지의 값으로 인코딩해주는 방법이다.

10강. 신경망의 완성 : 히든레이어

지금까지는 퍼셉트론은 input이 곧 output으로 표현이 쉬웠다.

하지만, 점점 input과 output이 많아지며, 정교하고 정확한 결과값을 예측하기 어려워져다. input과 output 사이에 hidden layer를 만들어 연결해주어 조금 더 깊이있고 정확한 결과값을 예측 가능하도록 한다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d9d727e7-5a63-43cf-b69a-665c70f7cfa1/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d9d727e7-5a63-43cf-b69a-665c70f7cfa1/Untitled.png)

- 히든레이어를 사용하지 않은 예

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/db8df226-0c1c-411f-bb04-1300ea8327a6/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/db8df226-0c1c-411f-bb04-1300ea8327a6/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a3ace143-ebd5-4587-be79-0ee4dbc9d977/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a3ace143-ebd5-4587-be79-0ee4dbc9d977/Untitled.png)

- 히든레이어를 사용한 예 (1개)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b4a7df94-6182-4b91-af8d-b2144631dd87/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b4a7df94-6182-4b91-af8d-b2144631dd87/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b9dd40ce-953d-4b16-840e-396b28626ddc/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b9dd40ce-953d-4b16-840e-396b28626ddc/Untitled.png)

- 히든레이어를 사용한 예 (멀티)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0b98f56f-2b2e-423f-bb03-7a2063769f3a/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0b98f56f-2b2e-423f-bb03-7a2063769f3a/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d6b5461c-5e43-4006-8d48-a845219c7524/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d6b5461c-5e43-4006-8d48-a845219c7524/Untitled.png)

(부록1) 데이터를 위한 팁

- 데이터 타입의 문제
    - 컬럼의 데이터 타입을 우선 확인한 후
    CASE1) 데이터가 일반 문자형이 아닌 이상하게 보여줄 때에는, 범주형 데이터 등으로 변환할 필요가 있다. 이 책의 사례에서는 int형 데이터 타입을 범주형 데이터 타입으로 변경하면 됨.
    ex) 인코딩 = pd.get_dummies(아이리스)
    인코딩.head() // 품종이 이상하게 나옴
    아이리스['품종'] = 아이리스['품종'].astype('category')
    print(아이리스.dtypes)
- NA 값의 처리
    - NA값은 mean으로 처리하여, 데이터 전처리 가능

(부록2) 모델을 위한 팁

- BatchNormalization Layer
    - ex) tf.keras.layers.BatchNormalization()(H)
    -
