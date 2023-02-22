import time
import random
import numpy as np
import copy

N = 15
K = 3
initial_sol = list()

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
'''
#Test 3
d = np.array([[0, 50, 28, 24, 30, 34, 41, 28, 47, 45, 25, 23, 39, 26, 36, 24, 32, 50, 49, 21], 
              [22, 0, 24, 33, 48, 39, 31, 44, 20, 41, 24, 45, 24, 33, 40, 44, 46, 22, 30, 28], 
              [47, 37, 0, 49, 22, 50, 30, 32, 43, 38, 34, 39, 36, 37, 49, 49, 47, 28, 43, 28], 
              [40, 43, 33, 0, 34, 37, 46, 48, 32, 48, 31, 43, 48, 23, 35, 22, 45, 28, 46, 31], 
              [21, 35, 50, 27, 0, 47, 50, 37, 29, 36, 31, 31, 35, 25, 27, 20, 42, 44, 35, 47], 
              [24, 37, 27, 44, 44, 0, 45, 27, 21, 27, 46, 45, 50, 36, 26, 46, 29, 28, 48, 28], 
              [32, 46, 48, 27, 44, 28, 0, 43, 28, 35, 47, 29, 28, 48, 50, 37, 37, 27, 36, 38], 
              [36, 50, 37, 26, 29, 45, 37, 0, 39, 38, 37, 50, 21, 28, 42, 50, 47, 43, 42, 34], 
              [48, 24, 23, 26, 30, 28, 30, 20, 0, 44, 20, 36, 20, 44, 40, 23, 46, 35, 34, 38], 
              [31, 50, 24, 41, 23, 23, 46, 32, 28, 0, 22, 38, 22, 42, 42, 43, 47, 41, 34, 34], 
              [33, 23, 20, 32, 49, 50, 47, 40, 31, 33, 0, 27, 50, 29, 29, 21, 47, 25, 39, 37], 
              [41, 25, 50, 29, 20, 39, 20, 43, 41, 21, 46, 0, 34, 39, 45, 29, 40, 37, 24, 30], 
              [43, 44, 41, 49, 24, 30, 33, 35, 50, 32, 21, 20, 0, 27, 34, 23, 27, 40, 30, 21], 
              [25, 26, 48, 22, 24, 50, 41, 33, 26, 34, 49, 38, 37, 0, 20, 44, 23, 26, 40, 30], 
              [24, 35, 27, 21, 50, 43, 36, 43, 27, 29, 33, 41, 40, 37, 0, 50, 38, 30, 33, 49], 
              [31, 27, 50, 33, 47, 22, 36, 45, 47, 47, 32, 41, 21, 32, 37, 0, 42, 43, 21, 27], 
              [23, 44, 44, 37, 36, 43, 42, 44, 37, 48, 37, 27, 42, 38, 29, 42, 0, 27, 25, 38], 
              [48, 40, 20, 37, 37, 40, 20, 36, 20, 32, 22, 41, 39, 44, 34, 33, 20, 0, 49, 50], 
              [21, 26, 21, 48, 30, 25, 22, 47, 24, 36, 41, 22, 42, 49, 40, 20, 24, 21, 0, 39], 
              [37, 30, 27, 28, 21, 32, 22, 24, 43, 42, 41, 31, 39, 29, 24, 35, 42, 35, 46, 0]])

def generate_solution():
    global K, N
    solution = [[0] for k in range(K+1)]
    visited = [False for i in range(N+1)]
    visited[0] = True
    count = 0

    while count < N:
        city = random.randint(1, N)
        if visited[city] == False:
            visited[city] = True
            mailman_index = random.randint(1, K)
            solution[mailman_index].append(city)
            count += 1

    for i in range(1, len(solution)):
        solution[i].append(0)

    return solution


def check_sol(solution):
    for i in range(1, len(solution)):
        if len(solution[i]) <= 2:
            return False
    return True


def cal_obj_func(solution):
    global d
    dis_travel = [0 for k in range(K+1)]
    for k in range(1, K+1):
        temp = 0
        for i in range(len(solution[k])-1):
            temp += d[solution[k][i]][solution[k][i+1]]
        dis_travel[k] = temp
    f1 = sum(dis_travel)
    f2 = max(dis_travel)
    f = f1 + K*f2
    return (f, f1, f2)

def find_better_neighbor(solution):
    neighbor_sol = copy.deepcopy(solution)
    temp_lst = list()
    for k in range(1, K+1):
        for i in range(1, len(neighbor_sol[k])-1):
            temp_lst.append([neighbor_sol[k][i], [k, i]])
    for i in range(len(temp_lst)):
        for j in range(len(temp_lst)):
            if i != j:
                temp_lst[i][0], temp_lst[j][0] = temp_lst[j][0], temp_lst[i][0]
                neighbor_sol[temp_lst[i][1][0]][temp_lst[i][1][1]] = temp_lst[i][0]
                neighbor_sol[temp_lst[j][1][0]][temp_lst[j][1][1]] = temp_lst[j][0]                
                if cal_obj_func(neighbor_sol)[0] < cal_obj_func(solution)[0]:
                    return neighbor_sol
                else:
                    temp_lst[i][0], temp_lst[j][0] = temp_lst[j][0], temp_lst[i][0]
                    neighbor_sol[temp_lst[i][1][0]][temp_lst[i][1][1]] = temp_lst[i][0]
                    neighbor_sol[temp_lst[j][1][0]][temp_lst[j][1][1]] = temp_lst[j][0]
    return

def Iterated_Local_Search():
    solutions = list()
    final_sol = list()
    count = 0
    fmin = 1e9
    while count <= 20:
        solution = generate_solution()
        if check_sol(solution):
            solutions.append(solution)
            count += 1

    for i in range(len(solutions)):
        while (find_better_neighbor(solutions[i]) != None):
            solutions[i] = find_better_neighbor(solutions[i])
    
    for i in range(len(solutions)):
        res = cal_obj_func(solutions[i])
        solutions[i].append(res)
        if res[0] < fmin:
            fmin = res[0]
    
    for i in range(len(solutions)):
        if solutions[i][-1][0] == fmin:
            final_sol.append(solutions[i])
    
    print(final_sol)

if __name__ == '__main__':
    t1 = time.time()
    Iterated_Local_Search()
    t2 = time.time()
    print(t2 - t1)


