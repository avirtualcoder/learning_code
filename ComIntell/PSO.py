import random
import time

def adapt(x):       #适应值函数
    ls=[float(i)**2 for i in x]
    return sum(ls)

def init():     #初始化构造函数
    size=100      #粒子个数
    n=10         #空间维度
    x=[[random.uniform(-10,10) for i in range(n)] for i in range(size)]     #粒子群坐标集
    v=[[random.uniform(-10,10) for i in range(n)] for i in range(size)]     #粒子群速度集
    print('初始坐标    ',x)
    print('初始速度    ',v)
    return x,v

def assess(x):      #评估函数
    Pbest=x         #局部最优为粒子个体本身坐标
    # print('各个体局部最优坐标 Pbest   ',Pbest)
    Eval = [adapt(Pbest[i]) for i in range(len(x))]     #各粒子适应值成列表
    # print('各个体的适应值  ', Eval)
    Gbest=Pbest[Eval.index(min(Eval))]          #列表中得全局最优粒子坐标
    # print('全局最优坐标 Gbest  ',Gbest)
    return Pbest,Gbest

def update(x,v,Pbest,Gbest):        #更新粒子群坐标及速度
    w=0.5       #惯性权重
    c1=c2=2.0       #加速系数
    for i in range(len(x)):     #迭代粒子数
        for j in range(len(x[0])):      #迭代维数
            v[i][j]=w*v[i][j]+c1*random.random()*(Pbest[i][j]-x[i][j])+c2*random.random()*(Gbest[j]-x[i][j])
            x[i][j] += v[i][j]      #根据自身局部最优和全局最优向当前最优坐标前进更新
    return x,v


time_ls=[]
g_ls=[]
for i in range(10):
    star=time.clock()
    x,v=init()
    Pbest,Gbest=assess(x)
    g=0
    while adapt(Gbest)>0.000003:
        x,v=update(x,v,Pbest,Gbest)
        Pbest,Gbest=assess(x)
        g+=1
        print('第{}代     最优值为{}'.format(g,adapt(Gbest)))
    print('一共更新{}代     最优值为{}'.format(g,adapt(Gbest)))
    end=time.clock()
    print('计算时间 {}s'.format(end-star))
    time_ls.append(round(end-star,2))
    g_ls.append(g)
print('10次迭代为 {}次 平均迭代为 {}次'.format(sum(g_ls),sum(g_ls)/len(g_ls)))
print('各代计算时间   ',time_ls)
print('10次总时间为 {}s 平均时间为 {}s'.format(round(sum(time_ls),2),round(sum(time_ls)/len(time_ls),2)))



