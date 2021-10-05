# 문제 : 10개의 수를 입력 받아, 그 중에서 홀수만 더한 값을 출력하는 프로그램을 작성하라.

#3
#3 17 1 39 8 41 2 32 99 2
#22 8 5 123 7 2 63 7 3 46
#6 63 2 3 58 76 21 33 8 1


length = int(input()) + 1                       #정수형 변수 1개 입력 받는 예제
for step in range(1, length):
    a,b,c,d,e,f,g,h,i,j =  map(int, input().split())   #실수형 변수 3개 입력 받는 예제
    sum = 0

    if a % 2 == 1:
        sum = sum + a
    if b % 2 == 1:
        sum = sum + b
    if c % 2 == 1:
        sum = sum + c
    if d % 2 == 1:
        sum = sum + d
    if e % 2 == 1:
        sum = sum + e
    if f % 2 == 1:
        sum = sum + f
    if g % 2 == 1:
        sum = sum + g
    if h % 2 == 1:
        sum = sum + h
    if i % 2 == 1:
        sum = sum + i
    if j % 2 == 1:
        sum = sum + j
    print('#%d %d'%(step, sum))