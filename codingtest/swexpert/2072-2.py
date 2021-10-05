# 문제 : 10개의 수를 입력 받아, 그 중에서 홀수만 더한 값을 출력하는 프로그램을 작성하라.

#3
#3 17 1 39 8 41 2 32 99 2
#22 8 5 123 7 2 63 7 3 46
#6 63 2 3 58 76 21 33 8 1

length = int(input())                       #정수형 변수 1개 입력 받는 예제

arr =  [list(map(int, input().split()))  for _ in range(length)]  #실수형 변수 3개 입력 받는 예제
print(arr)

for i in range(len(arr)):
    #print(arr[i])
    sum = 0
    for j in range(10):
        if(arr[i][j]%2 == 1):
            sum = sum + arr[i][j]
    print("#%d %d"%(i+1, sum))
