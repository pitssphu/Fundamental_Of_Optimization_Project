#generate test 
import time
import numpy as np

N = 6
K = 3

remaining_edge = N+K
fmin = 1e9
f1min = 1e9
d_min = 1e9
f_current = 0
d = [[0 for i in range(N+1)] for j in range(N+1)]
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

#Notations
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
    
A_copy = A[:] #a copy of the original A as A will change when branch and bound

#Variables
X = [[[0 for i in range(N+2*K+1)] for j in range(N+2*K+1)] for k in range(K+1)]
u = [[0 for i in range(N+2*K+1)] for k in range(K+1)]

#Constraints and objective
def check_constraint_1():
    global X, K, N
    for k in range(1, K+1):
        sum_1 = sum_2 = 0
        for j in range(1, N+1):
            sum_1 += X[k][k+N][j]
            sum_2 += X[k][j][N+K+k]
        if (sum_1 != 1) or (sum_2 != 1):
            return False
    return True

def check_constraint_2():
    global X, K, N, A_plus, A_minus
    for i in range(1, N+1):
        sum_1 = sum_2 = 0
        for k in range(1, K+1):
            for j in A_plus[i]:
                sum_1 += X[k][i][j]
            for j in A_minus[i]:
                sum_2 += X[k][j][i]
        if (sum_1 != 1) or (sum_2 != 1):
            return False
    return True

def check_constraint_3():
    global X, K, N, A_plus, A_minus
    sum_1 = sum_2 = 0
    for i in range(1, N+1):
        for k in range(1, K+1):
            for j in A_plus[i]:
                sum_1 += X[k][i][j]
            for j in A_minus[i]:
                sum_2 += X[k][j][i]
            if (sum_1 != sum_2):
                return False
    return True
            
def find_path(X, m, k, u):
    global N, K
    for i in range(1, N+2*K+1):
        if (X[k][m][i] == 1):
            u[k][i] = u[k][m] + 1
            if (i == N+K+k):
                u[k][i] = 0
                return
            return(find_path(X, i, k, u))
            
def check_MTZ_constraint(X, u):
    global K, N
    for k in range(1, K+1):
        find_path(X, N+k, k, u)
    for k in range(1, K+1):
        for i in range(1, N+1):
            for j in range(1, N+1):
                if (i != j) and (u[k][i] - u[k][j] + N*X[k][i][j] > N-1):
                    return False
    return True
    
def cal_f_current_when_choosing(X, a, b, c):
    global f_current
    if (b > N):
        f_current += d[0][c] * X[a][b][c]
    elif (c > N):
        f_current += d[b][0] * X[a][b][c]
    else:
        f_current += d[b][c] * X[a][b][c]

def cal_f_current_when_backtracking(X, a, b, c):
    global f_current
    if (b > N):
        f_current -= d[0][c] * X[a][b][c]
    elif (c > N):
        f_current -= d[b][0] * X[a][b][c]
    else:
        f_current -= d[b][c] * X[a][b][c]

def return_route(X):
    route = list()
    for k in range(1, K+1):
        temp = list()
        for i in range(N+2*K+1):
            for j in range(N+2*K+1):
                if (X[k][i][j] == 1) and (i > N) and (j <= N):
                    temp.append((0,j))
                elif (X[k][i][j] == 1) and (j > N) and (i <= N):
                    temp.append((i, 0))
                elif (X[k][i][j] == 1) and (j <= N) and (i <= N):
                    temp.append((i, j))
        route.append(temp)
    return route
    
        
def give_result():
    global X, d, fmin, f1min, K, N, route
    #Objective function
    f1 = f2 = f = 0
    for k in range(1, K+1):
        for edge in A_copy:
            if (edge[0] > N):
                f1 += d[0][edge[1]] * X[k][edge[0]][edge[1]]
            elif (edge[1] > N):
                f1 += d[edge[0]][0] * X[k][edge[0]][edge[1]]
            else:
                f1 += d[edge[0]][edge[1]] * X[k][edge[0]][edge[1]]
    for k in range(1, K+1):
        temp = 0
        for edge in A_copy:
            if (edge[0] in range(N+1, N+2*K+1)):
                temp += d[0][edge[1]] * X[k][edge[0]][edge[1]]
            elif (edge[1] in range(N+1, N+2*K+1)):
                temp += d[edge[0]][0] * X[k][edge[0]][edge[1]]
            else:
                temp += d[edge[0]][edge[1]] * X[k][edge[0]][edge[1]]
        if temp >= f2:
            f2 = temp

    f = (f1 + K*f2)
    if f <= fmin:
        fmin = f
        f1min = f1
        print('update best: f =', f, 'f1 =', f1, 'f2 =', f2)
        
        #save route of fmin
        new_route = return_route(X)
        new_route.append(fmin)
        opt_route.append(new_route)
        
        #only save the optimal route
        f_min = 1e9
        for route in opt_route:
            if f_min > route[K]:
                f_min = route[K]
        for i in range(5):
            for route in opt_route:
                if route[K] > f_min:
                    opt_route.remove(route)
        print(opt_route)
        

def Try(a, b, c):
    global count, A, u, remaining_edge
    if ((b, c) in A) and ((b <= N) or (b == N+a)) and ((c <= N) or (c == N+K+a)):
        for v in range(2):
            if v == 0:
                X[a][b][c] = 0
                if (a == K) and (b == N+K) and (c == N):
                    if (check_constraint_1() and check_constraint_2() and check_constraint_3()):
                        if check_MTZ_constraint(X, u):
                            give_result()
                            u = [[0 for i in range(N+2*K+1)] for k in range(K+1)]
                elif (b == N+K) and (c == N):
                    Try(a+1, 1, 1)
                elif (c == N+2*K):
                    Try(a, b+1, 1)
                else:
                    Try(a, b, c+1)

            elif v == 1:
                A_mark = A[:]
                for i in range(5):
                    for edge in A:
                        if (edge[0] == b) or (edge[1] == c):
                            A.remove(edge)
                X[a][b][c] = 1
                remaining_edge -= 1
                cal_f_current_when_choosing(X, a, b, c)
                if f_current + d_min * remaining_edge <= f1min:
                    if (a == K) and (b == N+K) and (c == N):
                        if (check_constraint_1() and check_constraint_2() and check_constraint_3()):
                            if check_MTZ_constraint(X, u):
                                give_result()
                                u = [[0 for i in range(N+2*K+1)] for k in range(K+1)]
                    elif (b == N+K) and (c == N):
                        Try(a+1, 1, 1)
                    elif (c == N+2*K):
                        Try(a, b+1, 1)
                    else:
                        Try(a, b, c+1)
                A = A_mark[:]
                cal_f_current_when_backtracking(X, a, b, c)
                remaining_edge += 1

    else:
        X[a][b][c] = 0
        if (a == K) and (b == N+K) and (c == N):
            if (check_constraint_1() and check_constraint_2() and check_constraint_3()):
                if check_MTZ_constraint(X, u):
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
    
   