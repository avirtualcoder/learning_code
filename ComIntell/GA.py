import random
import numpy as np
import time

def adapt(x):
    out=[float(i)**2 for i in x]
    return sum(out)

# def init():
#     size=4
#     n=2
#     population=np.random.randint(-10,10,size,n)
#     print(population)
#
g=0
size=4
n=10
b=[[random.randrange(-10,10) for i in range(n)]for j in range(size)]
print('初始种群   ',b)
# 初始化染色体

Eval=[adapt(i) for i in b]
best_eval=min(Eval)
best_index=Eval.index(best_eval)
print('适应值   ',Eval)
print('最好适应值的下标 ',best_index)
# 计算适应值并评估出最优适应值

while best_eval>0.0000003:
    sum1=[round(sum(Eval)/i,2)  for i in Eval]
    P=[round(i/sum(sum1),2) for i in sum1]
    print('适应值占比    ',P)
    chose=[]
    p=0
    for i in P:
        p+=i
        chose.append(p)
    print('轮盘选择上的各个体概率     ',chose)
    circle_chose=[round(random.random(),2) for i in range(len(chose))]
    print('随机概率     ',circle_chose,'       开始选择  ')
    left=[]
    for i in range(0,size):
        for j in range(0,size):
            if j==0:
                if  circle_chose[i]<=chose[j]:
                    left.append(j)
            elif chose[j-1]<circle_chose[i]<=chose[j]:
                left.append(j)
    print('轮盘选择优胜个体    ',left)
    out=[i for i in range(size) if i not in left]
    print('轮盘选择淘汰个体     ',out)
    for i in out:
        b[i]=b[best_index]
    print('选择后的新种群  ',b)
    # 轮盘选择，优胜略汰

    mating=[i for i in range(size) if random.random()<0.88]
    print('交配概率下命中的交配个体   ',mating)
    for i in range(0,int(len(mating)/2)):
        j=random.randint(0,n)
        print('交配因子位',j)
        for j in range(j,n):
            b[mating[i]][j],b[mating[2*i]][j]=b[mating[-i]][j],b[mating[i]][j]
    print('交配后的种群   ',b)
    # 交配出新种群

    for i in range(size):
        for j in range(n):
            k=random.random()
            if  k<=0.05:
                b[i][j]=random.randint(-10,10)
                print('变异概率为{}  第{}个体的{}号基因变异了'.format(k,i,j))
    print('变异后的种群   ',b)
    #每个个体的染色体的每个基因变异的概率为0.05

    Eval=[adapt(i) for i in b]
    if min(Eval)<best_eval:
        best_eval=min(Eval)
        best_index=Eval.index(best_eval)
    g+=1
    print('新一代的种群',b)
    print('最优值为',best_eval)
    print('当前代数 ',g)
print('遗传代数为 ',g,'得出的最优值为 ',best_eval)
print(time.clock())


