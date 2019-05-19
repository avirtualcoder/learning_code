import random
import numpy as np
import time

"""
16计算机7班
20160310412
李宏佳
"""

def adapt(x):       #适应值函数
    ls=[float(i)**2 for i in x]
    return sum(ls)


def init():         #种群初始化构造函数
    size = 100            #种群个体数
    n = 10           #染色体基因数
    population = np.random.uniform(-10,10,[size,n])        #种群染色体列
    print('初始群体如下\n{}'.format(population))
    return population       #返回种群


def Eval(population):           #评估种群
    eval = [adapt(i) for i in population]       #种群适应值
    best_eval=min(eval)                     #最优适应值及下标（个体号）
    best_index=eval.index(best_eval)
    # print('种群适应值为 {}    最优个体的适应值为 {}     最优个体号 {}'.format(eval,best_eval,best_index))
    return eval,best_eval,best_index        #返回适应值列表，最优值，最优个体


def chose(population):          #轮盘选择出新种群，优胜略汰
    eval,best_eval,best_index=Eval(population)
    P1=[sum(eval)/i for i in eval]      #因求最小适应值，求倒数后在划分轮盘，P2为大小概率
    P2=[i/sum(P1) for i in P1]
    circle_probability=np.cumsum(P2)        #轮盘上各个体概率
    chose_probability=np.random.rand(len(eval))         #随机概率用于选择
    # print('各个体的适应值概率 {}\n轮盘上的个体的概率占位 {}\n轮盘选择概率 {}'.format(P2,circle_probability,chose_probability))

    left=[]     #用于存放留下的个体号
    for i in range(len(population)):    #迭代样本个数(选择次数)，迭代选择概率(寻找命中)，逐一个体逐一轮盘概率与选择概率进行比较
        for j in range(len(circle_probability)):
            if j==0:
                if chose_probability[i]<=circle_probability[j]:
                    left.append(j)
                    break
            elif circle_probability[j-1]<chose_probability[i]<=circle_probability[j]:
                left.append(j)
                break
    out = [i for i in range(len(P2)) if i not in left]      #淘汰的个体号
    # print('留下的个体为 {}\n淘汰的个体为 {}'.format(left,out))

    for i in out:       #将淘汰的个体替换为最优个体的染色体
        population[i]=population[best_index]
    # print('选择后的新种群\n{}'.format(population))
    return population       #返回选择后的种群


def mating(population):         #种群交配
    mat_probability=0.88        #发生交配的概率
    mat_chose=np.random.rand(population.shape[0])   #随机概率用于判断是否参与交配
    mat_unit=[]             #交配单元用于存放要发生交配的个体
    for i in range(len(mat_chose)):     #根据交配概率求出交配个体存入交配单元
        if mat_chose[i]<=mat_probability:
            mat_unit.append(i)
    # print('交配个体     ',mat_unit)
    for i in range(int(len(mat_unit)/2)):       #根据个体数求交配对数
        mat_site=random.randint(1,len(population[0])-1)

        temp1=list(population[mat_unit[i]][mat_site:])      #去交配单元列表的两头开始配对染色体交叉
        temp2=population[mat_unit[-i-1]][mat_site:]
        population[mat_unit[i]][mat_site:]=temp2
        population[mat_unit[-i - 1]][mat_site:]=temp1
        # print('交配个体 {}与{}  因子 {}'.format(mat_unit[i],mat_unit[-i-1],mat_site))
    # print('交配出的新种群\n{}'.format(population))
    return population

def variation(population):          #变异
    for i in range(len(population)):        #迭代每个个体染色体的基因接受变异机会
        for j in range(len(population[0])):
            v_p=0.05
            k=random.random()
            if k<=v_p:
                population[i][j]=random.uniform(-10,10)
                # print('变异因子位 ({},{})  及概率 {}'.format(i,j,k))
    # print('变异后的种群为\n{}'.format(population))
    return population

g_ls=[]
time_ls=[]
for i in range(10):
    star=time.clock()
    population=init()       #初始化产生种群
    eval,best_eval,best_index=Eval(population)      #评估函数求最优值
    g=0
    while best_eval>0.000003:       #当达标时退出
        population=chose(population)    #轮盘选择，优胜劣汰
        population=mating(population)   #新种群交配，染色体交叉
        population=variation(population)    #基因随机突变
        eval,best_eval,best_index=Eval(population)  #再次评估出最优
        g+=1            #计算当前代数
        print('第 {} 代，  最优适应值为 {}'.format(g,best_eval))
    end=time.clock()
    time_ls.append(end-star)
    g_ls.append(g)
    print('计算时间 {} s'.format(time_ls))
print('总迭代为 {}次，平均迭代为 {}次'.format(sum(g_ls),sum(g_ls)/len(g_ls)))
print('总时间为 {}s，平均时间为 {}s'.format(round(sum(time_ls),2),round(sum(time_ls)/len(time_ls),2)))