import numpy as np
import copy
import time

N = 19
K = 2

initial_sol = list()
solution = list()

visited = list()
visited_0 = [True for k in range(K+1)]
visited.append(visited_0) #visited[0][k] will represent the time that truck k can return to the depot
for i in range(N):
    visited.append(False)

current = [0 for i in range(K+1)] #the current city of trucks
done = [0 for k in range(K+1)] #truck k have done its route or not
route = list()
for k in range(K+1):
    route.append([0])
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


def find_next_city(current, visited):
    d_min = 1e9
    res = (K, 0)
    if (sum(done) == K-1):
        for i in range(1, N+1):
            if visited[i] == False:
                for k in range(1, K+1):
                    if done[k] == 0:
                        if d[current[k]][i] < d_min:
                            d_min = d[current[k]][i]
                            res = (k, i)
    else:
        for k in range(1, K+1):
            if done[k] == 0:
                if visited[0][k] == False:
                    if d[current[k]][0] < d_min:
                        d_min = d[current[k]][0]
                        res = (k, 0)
        for i in range(1, N+1):
            if visited[i] == False:
                for k in range(1, K+1):
                    if done[k] == 0:
                        if d[current[k]][i] < d_min:
                            d_min = d[current[k]][i]
                            res = (k, i)
    return res

def time_can_return(route):
    global K
    for k in range(1, K+1):
        if N > 2:
            if len(route[k]) >= int(N/K)+1:
                visited[0][k] = False
        else:
            if len(route[k]) >= 2:
                visited[0][k] = False

def check(route):
    for i in range(1, K+1):
        if (route[i][-1] != 0) or (route[i][-1] == 0 and len(route[i]) == 1):
            return False
    return True

def cal_obj_func(solution):
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
                
def Greedy():
    global current, visited, route, done, initial_sol
    for k in range(K+1):
        if (route[k][-1] == 0) and (len(route[k]) > 2):
            done[k] = 1

    time_can_return(route)
    (salesman_k, next_city) = find_next_city(current, visited)
    current[salesman_k] = next_city
    route[salesman_k].append(next_city)

    if next_city == 0:
        visited[0][salesman_k] = True
    else:
        visited[next_city] = True

    if check(route):
        initial_sol = copy.deepcopy(route)
        print(initial_sol[1:])
        print(cal_obj_func(initial_sol))
    else:
        Greedy()

def Hill_Climbing():
    global solution
    solution = initial_sol
    while (find_better_neighbor(solution) != None):
        solution = find_better_neighbor(solution)
    print(solution[1:])
    print(cal_obj_func(solution))

if __name__ == '__main__':
    t1 = time.time()
    Greedy()
    Hill_Climbing()
    t2 = time.time()
    print(t2 - t1)