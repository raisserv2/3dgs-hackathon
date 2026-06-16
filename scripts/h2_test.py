N, K = map(int, input().split())

arr = []
for _ in range(N):
    arr.append(list(map(int, input().split())))
temp = list(map(int, input().split()))

arr = [temp] + arr 

#init
dp = [[[float('inf')]*(3) for _ in range(K+1)] for _ in range(N+1)]

#base 
# a = arr[1][0]
# b = arr[1][1]
# c = arr[2][0]
# d = arr[2][1]

# dp[3][1][0] = arr[3][0]
# dp[3][2][0] = arr[3][0] + min(a, b, c)
# dp[3][3][0] = arr[3][0] + c + a

# dp[3][1][1] = arr[3][1]
# dp[3][2][1] = arr[3][1] + min(a, b, d)
# dp[3][3][1] = arr[3][1] + d + b

# dp[3][1][2] = min(a, b, c, d)
# dp[3][2][2] = min(a+c, b+d)
dp[1][0][2] = 0
dp[1][1][0] = arr[1][0]
dp[1][1][1] = arr[1][1]

for n in range(1, N+1):

    dp[n][0][2] = 0

#print('hi')

#transition
for n in range(2, N+1):
    for k in range(0, min(K, n) + 1):
        dp[n][k][0] = float('inf')
        dp[n][k][1] = float('inf')
        dp[n][k][2] = float('inf')
        if k >= 1:
            dp[n][k][0] = arr[n][0] + min(dp[n-1][k-1][0], dp[n-1][k-1][2])
            dp[n][k][1] = arr[n][1] + min(dp[n-1][k-1][1], dp[n-1][k-1][2])
        dp[n][k][2] = min(dp[n-1][k][0], dp[n-1][k][1], dp[n-1][k][2])

#print(dp)

print(sum(sum(row) for row in arr[1:]) - min(dp[N][K]))
                







