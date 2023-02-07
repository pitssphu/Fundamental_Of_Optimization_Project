import time
import numpy as np
import copy

N = 6
K = 2

fmin = 1e9
f1min = 1e9
d_min = 1e9
f_current = 0
remaining_edge = N+K #number of edge that still have to choose
count = 0
opt_route = list()
'''
#Test 1
d = np.array([[ 0,  1, 12,  9,  9,  3,  8,  6, 19, 10],
              [ 3,  0,  5,  8, 16, 16, 10,  3, 10, 14],
              [ 2, 11,  0, 12, 10,  7,  3,  7, 11, 16],
              [ 2,  4, 14,  0,  3, 16, 18, 10,  1, 19],
              [16, 15,  6,  8,  0, 17,  2, 20, 16, 16],
              [15, 10, 19,  6, 14,  0,  8,  9,  7, 12],
              [12,  2,  1,  9, 20, 15,  0,  1,  2,  7],
              [12,  2, 19,  6, 12, 17, 14,  0,  7,  2],
              [11, 15,  8,  4, 15, 15, 20, 19,  0, 10],
              [11, 19,  7, 18,  8, 20, 17,  9,  7,  0]])
'''
#Test 2
d = np.array([[ 0, 46, 39, 20, 21, 39, 50, 45, 43, 44],
              [41,  0, 33, 45, 34, 36, 32, 30, 45, 36],
              [34, 30,  0, 23, 28, 39, 26, 26, 42, 27],
              [21, 35, 26,  0, 46, 21, 23, 35, 46, 20],
              [49, 42, 39, 40,  0, 48, 47, 36, 37, 41],
              [39, 21, 28, 42, 28,  0, 33, 33, 38, 42],
              [33, 28, 43, 48, 42, 40,  0, 23, 48, 28],
              [42, 26, 50, 40, 41, 50, 25,  0, 22, 49],
              [39, 35, 29, 31, 43, 43, 36, 46,  0, 36],
              [27, 31, 27, 41, 28, 40, 33, 50, 47,  0]])

for i in range(N+1):
    for j in range(N+1):
        if (d[i][j] != 0) and (d[i][j] < d_min):
            d_min = d[i][j]

#Notation
B = list()
F1 = list()
F2 = list()
F3 = list()
F4 = list()
A = list()
A_plus = list()
A_minus = list()

for i in range(1, N+2*K+1):
    for j in range(1, N+2*K+1):
        B.append((i, j))

for i in range(1, N+2*K+1):
    for k in range(1, K+1):
        F1.append((i, N+k))
        
for i in range(1, N+2*K+1):
    for k in range(1, K+1):
        F2.append((N+K+k, i))
        
for i in range(1, N+2*K+1):
    F3.append((i, i))
    
for i in range(N+1, N+K+1):
    for j in range(N+K+1, N+2*K+1):
        F4.append((i, j))
    
for edge in B:
    if (edge not in F1) and (edge not in F2) and (edge not in F3) and (edge not in F4):
        A.append(edge)

A_plus.append([])
for i in range(1, N+2*K+1):
    sub_lst = list()
    for edge in A:
        if edge[0] == i:
            sub_lst.append(edge[1])
    A_plus.append(sub_lst)

A_minus.append([])
for i in range(1, N+2*K+1):
    sub_lst = list()
    for edge in A:
        if edge[1] == i:
            sub_lst.append(edge[0])
    A_minus.append(sub_lst)
    

#Variables and constraints
x = [[[0 for i in range(N+2*K+1)] for j in range(N+2*K+1)] for k in range(K+1)]
u = [[0 for i in range(N+2*K+1)] for k in range(K+1)]

def check_constraint_1(x, a, b, c):
    global K, N
    for k in range(1, K+1):
        sum_1 = sum_2 = 0
        for j in range(1, N+1):
            sum_1 += x[k][k+N][j]
            sum_2 += x[k][j][N+K+k]
        if (a == k) and (b == N+k) and (c == N):
            if (sum_1 != 1) or (sum_2 != 1):
                return False
        elif (sum_1 > 1) or (sum_2 > 1):
            return False
    return True

def check_constraint_2(x, a, b, c):
    global K, N, A_plus, A_minus
    for i in range(1, N+1):
        sum_1 = sum_2 = 0
        for k in range(1, K+1):
            for j in A_plus[i]:
                sum_1 += x[k][i][j]
            for j in A_minus[i]:
                sum_2 += x[k][j][i]
        if (a == K) and (b == N+K) and (c == N):
            if (sum_1 != 1) or (sum_2 != 1):
                return False
        elif (sum_1 > 1) or (sum_2 > 1):
            return False
    return True

def check_constraint_3(x, a, b, c):
    global K, N, A_plus, A_minus
    for i in range(1, N+1):
        for k in range(1, K+1):
            sum_1 = sum_2 = 0
            for j in A_plus[i]:
                sum_1 += x[k][i][j]
            for j in A_minus[i]:
                sum_2 += x[k][j][i]
            if (b == N+K) and (c == N):
                if (sum_1 != sum_2):
                    return False
            elif (sum_1 > 1) or (sum_2 > 1):
                return False
    return True

def check_while_generate(v, x, a, b, c):
    x_copy = copy.deepcopy(x)
    x_copy[a][b][c] = v
    if check_constraint_1(x_copy, a, b, c) and check_constraint_2(x_copy, a, b, c) and check_constraint_3(x_copy, a, b, c):
        return True
    return False

def check(x, a, b, c):
    if check_constraint_1(x, a, b, c) and check_constraint_2(x, a, b, c) and check_constraint_3(x, a, b, c):
        return True
    return False

def find_path(x, m, k, u):
    global N, K
    for i in range(1, N+2*K+1):
        if (x[k][m][i] == 1):
            u[k][i] = u[k][m] + 1
            if (i == N+K+k):
                u[k][i] = 0
                return
            return(find_path(x, i, k, u))
            
def check_MTZ_constraint(x, u):
    global K, N
    for k in range(1, K+1):
        find_path(x, N+k, k, u)
    for k in range(1, K+1):
        for (i, j) in A:
            if (i < N+1) and (j < N+1):
                if (u[k][i] - u[k][j] + N*x[k][i][j] > N-1):
                    return False
    return True

def cal_f_current_when_choosing(x, a, b, c):
    global f_current
    if (b > N):
        f_current += d[0][c] * x[a][b][c]
    elif (c > N):
        f_current += d[b][0] * x[a][b][c]
    else:
        f_current += d[b][c] * x[a][b][c]

def cal_f_current_when_backtracking(x, a, b, c):
    global f_current
    if (b > N):
        f_current -= d[0][c] * x[a][b][c]
    elif (c > N):
        f_current -= d[b][0] * x[a][b][c]
    else:
        f_current -= d[b][c] * x[a][b][c]

def return_route(x):
    route = list()
    for k in range(1, K+1):
        temp = list()
        for i in range(N+2*K+1):
            for j in range(N+2*K+1):
                if (x[k][i][j] == 1) and (i > N) and (j <= N):
                    temp.append((0,j))
                elif (x[k][i][j] == 1) and (j > N) and (i <= N):
                    temp.append((i, 0))
                elif (x[k][i][j] == 1) and (j <= N) and (i <= N):
                    temp.append((i, j))
        route.append(temp)
    return route

def give_result():
    global x, d, fmin, f1min, K, N, route
    f1 = f2 = f = 0
    for k in range(1, K+1):
        for edge in A:
            if (edge[0] > N):
                f1 += d[0][edge[1]] * x[k][edge[0]][edge[1]]
            elif (edge[1] > N):
                f1 += d[edge[0]][0] * x[k][edge[0]][edge[1]]
            else:
                f1 += d[edge[0]][edge[1]] * x[k][edge[0]][edge[1]]
    for k in range(1, K+1):
        temp = 0
        for edge in A:
            if (edge[0] in range(N+1, N+2*K+1)):
                temp += d[0][edge[1]] * x[k][edge[0]][edge[1]]
            elif (edge[1] in range(N+1, N+2*K+1)):
                temp += d[edge[0]][0] * x[k][edge[0]][edge[1]]
            else:
                temp += d[edge[0]][edge[1]] * x[k][edge[0]][edge[1]]
        if temp >= f2:
            f2 = temp

    f = (f1 + K*f2)
    if f <= fmin:
        fmin = f
        f1min = f1
        print('update best: f =', f, 'f1 =', f1, 'f2 =', f2)
        
        new_route = return_route(x)
        new_route.append(fmin)
        opt_route.append(new_route)
        
        f_min = 1e9
        for route in opt_route:
            if f_min > route[K]:
                f_min = route[K]
        for i in range(5):
            for route in opt_route:
                if route[K] > f_min:
                    opt_route.remove(route)
        print(opt_route)


#Brannch and bound
#N = 4, K = 2 chay het 18s
#N = 5, K = 2 chay het 602s
#N = 6, K = 2 chay het 

def Try(a, b, c):
    global A, u, count, f_current, remaining_edge, fmin, f1min
    if ((b, c) in A) and ((b <= N) or (b == N+a)) and ((c <= N) or (c == N+K+a)):
        for v in [1, 0]:
            if check_while_generate(v, x, a, b, c) == True:
                x[a][b][c] = v
                remaining_edge -= v
                cal_f_current_when_choosing(x, a, b, c)
                if f_current + remaining_edge*d_min <= f1min:
                    if (a == K) and (b == N+K) and (c == N):
                        if check(x, a, b, c) and check_MTZ_constraint(x, u):
                            count += 1
                            give_result()
                            u = [[0 for i in range(N+2*K+1)] for k in range(K+1)]
                    elif (b == N+K) and (c == N):
                        Try(a+1, 1, 1)
                    elif (c == N+2*K):
                        Try(a, b+1, 1)
                    else:
                        Try(a, b, c+1)
                cal_f_current_when_backtracking(x, a, b, c)
                x[a][b][c] = 0
                remaining_edge += v

    else:
        if (a == K) and (b == N+K) and (c == N):
            if check(x, a, b, c) and check_MTZ_constraint(x, u):
                give_result()
                u = [[0 for i in range(N+2*K+1)] for k in range(K+1)]
        elif (b == N+K) and (c == N):
            Try(a+1, 1, 1)
        elif (c == N+2*K):
            Try(a, b+1, 1)
        else:
            Try(a, b, c+1)

if __name__ == '__main__':
    t1 = time.time()
    Try(1, 1, 1)
    t2 = time.time()
    print(t2 - t1)