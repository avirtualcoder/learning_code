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

def init():             #初始化构造函数，初始各值
    size=100             #个体数量
    n=10                #个体变量维数
    probability=[]      #变量取值概率
    for i in range(n):  #遍历维数构造单个变量组成变量取值矩阵
        part_probability=np.linspace(-10,10,11).tolist()
        probability.append(part_probability)
    x=[[random.uniform(-10,10) for i in range(n)] for i in range(size)] #区间随机生成有n维变量的个体数量
    for i in x:            #遍历个体集求适应值
        i.append(adapt(i))
    x=np.array(x)
    print('初始概率模型   ',probability)
    print('初始个体取值\n',x)
    return probability,x,size,n


def chose_better(x):        #择优函数，选取优秀个体留下
    se=20                    #优秀个体数量
    x=x[np.lexsort(x.T)]    #根据最后的适应值排序
    x=x[0:se]               #切片截取
    # print('优选个体\n',x)
    return x                #返回择优后的个体集（个体集由n个变量和一个适应值组成）


def update(probability,x):  #根据变量取值区间和择优个体集更新个体集
    new=[]                  #创建新的列表用于存放新的个体集
    for j in range(len(probability)):   #遍历维度求新的界限
        x_min=min([x[i][j] for i in range(len(x))])
        x_max=max([x[i][j] for i in range(len(x))])
        # print(x_min,x_max)
        part_new=np.linspace(x_min,x_max,11).tolist()   #根据新的限制分解区间
        # print(part_new)
        new.append(part_new)            #添加至新取值区间
    # print('更新后的概率模型 ',new)
    return new      #返回新取值区间

def build(probability,size,n):      #创建函数，根据新取值区间和数量，维度创建新个体集
    x = [[random.uniform(probability[i][0],probability[i][-1]) for i in range(n)] for i in range(size)]
    for i in x:
        i.append(adapt(i))
    x=np.array(x)
    # print('新个体  ',x)
    return x            #返回新个体集


g_ls=[]     #代数列表用于计算代数和
time_ls=[]  #时间列表用于计算时间和
for i in range(10): #进行10次实验
    star=time.clock()   #计时开始
    probability,x,size,n=init()     #初始化构建
    x_aver=sum([x[i][-1] for i in range(len(x))])/len(x)    #计算平均适应值
    g=0             #代数值
    print('初始适应值    ',x_aver)
    test=[]         #测试列表，用于测试是否陷入局部最优
    while x_aver>0.000003:         #精度设置
        x=chose_better(x)   #选取更优个体
        probability=update(probability,x)#根据优个体出新的变量生成概率模型
        x=build(probability,size,n)      #根据新概率模型和个体数量要求及变量维度生成新个体集
        x_aver = sum([x[i][-1] for i in range(len(x))]) / len(x)    #计算平均适应值
        g+=1                   #代数计算
        print('第{}代 x的适应值为{}'.format(g,x_aver))
        test.append(x_aver)     #将适应值放入测试列表
        flag=False              #用于结束未到精度要求的局部最优适应值
        if g>20:                #当代数大于60时
            if test[g-1]==test[g-15]:   #如果15代适应值无变化
                flag=True       #将节点置为True
        if flag:
            break
    end = time.clock()      #结束计时
    time_ls.append(end - star)  #计算单次总时
    g_ls.append(g)              #入代数列表
    print('计算时间     ',time.process_time())
print('总迭代为 {}次，平均迭代为 {}次'.format(sum(g_ls), sum(g_ls) / len(g_ls)))
print('总时间为 {}s，平均时间为 {}s'.format(round(sum(time_ls), 2), round(sum(time_ls) / len(time_ls), 2)))









'''关于分布估计算法的简化偷懒，我很抱歉'''