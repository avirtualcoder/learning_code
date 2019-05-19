import random
import datetime
import matplotlib.pyplot as plt

import pandas as pd
from random import shuffle

#用遗传算法解实验3单任务单分配问题


task = pd.read_csv('tasks1.csv')
user= pd.read_csv('usersAndRoutes1.csv')
X = task[['id', 'tx1', 'ty1']].values             #X为tasks1的数据的numpy数组，数据类型为（298,3），X为任务数据
# print(X)
Y=user[['id','name','x1','y1','x2','y2']].values  #Y为用户数据



#轮盘赌函数
def roulette_wheel_selection(N,p):                #N为种群大小，*p为种群个体适应值同群体适应值总和的比
    m=0
    r=random.random()

    for i in range(1,N+1):
        m=m+p[i-1]
        if r<=m:
            return i
            break




def detour(x1,y1,x2,y2,tx,ty):
    return abs(x1-tx)+abs(y1-ty)+abs(tx-x2)+abs(ty-y2)-(abs(x1-x2)+abs(y1-y2))

#定义评估函数
def fitness(R):
    dis_sum=0
    for i in range(len(R)):
        uid=R[i]
        dis=detour(Y[uid][2],Y[uid][3],Y[uid][4],Y[uid][5],X[i][1],X[i][2])
        dis_sum +=dis
    return dis_sum



#取得最优个体
def best(N,b,a=[x for x in range(len(X))]):
    #初始化a为历史最优个体，元素个数与种群个体一致，N为种群规模，b为包含种群所有个体的染色体的列表
    Best =a
    c=b.copy()
    for i in range(N):
        bval =fitness(Best)

        v=fitness(c[i])

        if v<bval:
            Best=c[i]
        else:
            pass
    return Best


#通过选择得到新种群
def select(N,a):              #N为种群规模，a为包含种群所有个体的染色体的列表
    b=a.copy()
    sum_v=0                   #适应值总和
    for i in range(N):
        sum_v=sum_v+(1/fitness(b[i]))
    p=[]
    for i in range(N):
        p.append((1/fitness(b[i]))/sum_v)


    new_a=[]                  #new_a为通过选择后的新种群
    for i in range(N):
        c=roulette_wheel_selection(N,p)
        new_a.append(b[c-1])

    return new_a


#交配后得到新种群
def crossover(N,a):           #N为种群规模，a为包含种群所有个体的染色体的列表
    b=[]                      #获取需要交配的染色体序列号
    c=a.copy()
    for i in range(N):
        pc = random.random()
        if pc<= 0.5:
            b.append(i)

    while(len(b)>1):
        cut=random.randint(0,len(c[0]))
        t1=[]
        t2=[]
        t1.extend(c[b[0]][:cut])
        t1.extend(c[b[1]][cut:])
        t2.extend(c[b[1]][:cut])
        t2.extend(c[b[0]][cut:])
        if len(set(t1))==len(a[0]):
            c[b[0]]=t1
        else:
            pass
        if len(set(t2))==len(a[0]):
            c[b[1]] = t2
        else:
            pass

        b.pop(0)
        b.pop(0)

    return c                     #返回交配后的新种群



#变异产生新种群
def mutation(N,a):               #N为种群规模，a为包含种群所有个体的染色体的列表
    b=a.copy()

    for i in range(N):
        pm = random.random()

        if pm <0.3:
            for i in range(10):
                update = random.randint(0, len(b[0]) - 1)
                # print(update)
                left = [x for x in range(0, 338) if x not in b[i]]
                index = random.randint(0, len(left) - 1)
                b[i][update] = left[index]

    return b



if __name__=='__main__':

    # 开始运行
    start = datetime.datetime.now()

    #初始化种群
    a =[]
    for i in range(39):
        aa=[x for x in range(i,len(X)+i)]
        shuffle(aa)
        a.append(aa)
    for i in range(11):
        aa = [x for x in range(i, len(X) + i)]
        shuffle(aa)
        a.append(aa)


    Old_best_value=1000000000000000000
    Best = best(50, a)
    i = 1
    result = {}                     # 代数和最优值的字典                                                      PLT
    while (fitness(Best) > 17280):
        Old_best_value=fitness(Best)
        new_a = select(50, a)
        new_a = crossover(50, new_a)
        new_a = mutation(50, new_a)
        a = new_a.copy()
        Best = best(50, new_a, Best)
        print('第', i, '代后种群最优个体：', Best)
        best_v = fitness(Best)
        print("最优值为：", best_v)
        result[i] = best_v  # 代数和最优值的字典                                            PLT
        i = i + 1

    end = datetime.datetime.now()
    print('运行时间为：', end - start)

    # 画出算法的效率曲线                                                                      PLT
    plt.plot(result.keys(), result.values(), label='useing time: ' + str(end - start))
    plt.xlabel('GENERATION')
    plt.ylabel('BestValue')
    plt.title('EFFICIENCY CHART OF GA')
    plt.legend(loc='best')
    plt.show()






