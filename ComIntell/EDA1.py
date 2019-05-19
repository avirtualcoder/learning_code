import random as rd
import numpy as np

class Eda():
    def __init__(self,goal,n,size,X_limit=(-10,10)):    #初始函数（）初始化各值
        self.n=n                #变量维度个数
        self.m=size             #个体总数大小
        self.goal=goal          #目标输出精度
        self.limit=X_limit      #变量限制区间
        ls=np.linspace(0,20,11)
        self.C_V=[[(ls[i]-10,ls[i+1]-10) for i in range(len(ls)-1)] for i in range(n)]
        self.P=[[0.1 for i in range(10) ]for j in range(n)]
        # print(self.C_V)     #初始变量选择区间
        # print(self.P)       #初始变量区间选择概率
        # print(self.limit)

    def build(self,C_V,P):      #个体建立函数（），根据区间和概率生成个体集
        X=[]
        for k in range(self.m):         #迭代生成个体数，生成X
            value = []
            for i in range(self.n):     #迭代变量维数生成变量个数 1 to n
                c=0
                for j in range(10):     #轮盘赌选出一个变量的下标
                    c+=P[i][j]
                    if rd.uniform(0,1)<=c:
                        value.append(C_V[i][j])
                        break

            X.append([(value[i][0],value[i][1]) for i in range(self.n)])
        # print('变量取值区间\n',self.C_V)     #变量取值区间矩阵
        # print('变量选择概率\n',self.P)       #变量选择概率矩阵
        print('个体集\n',X)            #根据区间矩阵和概率矩阵出的个体集矩阵维度为n，个数为m
        return X

    def adapt(self,X):      #输入个体集矩阵，输出带有适应值的新个体集矩阵
        for i in range(self.m):
            sum = 0
            for j in range(self.n):
                sum+=((X[i][j][0]+X[i][j][1])/2)**2
            X[i].append(sum)
        # print('加入适应值的个体集\n',X)
        return X

    def better(self,X_eval):
        self.se=self.m-2
        X_sort=np.array(X_eval)
        X_better=X_sort[np.lexsort(X_sort.T)][0:self.se]     #根据适应值排序并且截取se个个体
        # print('择优个体集\n',X_better)
        return X_better

    def update(self,X_better):
        limit=[np.min(X_better[0:self.se],axis=0).tolist(),np.max(X_better[0:self.se],axis=0).tolist()]
        # print('新的取值区间大小\n',limit)
        for i in range(self.n):
            ls=np.linspace(limit[0][i][0]+10,limit[1][i][1]+10,11)
            print(ls)
            self.C_V=[[(ls[k]-10,ls[k+1]-10) for k in range(len(ls)-1)] for i in range(self.n)]
        print(self.C_V)
        self.P=[[0 for i in range(10)] for j in range(self.n)]
        for i in range(len(X_better[0])-1):
            for j in range(self.se):
                for k in range(10):         #分区数
                    if self.C_V[i][k][0]<(X_better[j][i][0]+X_better[j][i][1])/2<self.C_V[i][k][1]:
                        self.P[i][k]+=1/self.se
        # print('新的概率模型  ',self.P)

goals=100
test=Eda(0.3,2,10)
X=test.build(test.C_V,test.P)
while goals>0.3:
    X_eval=test.adapt(X)
    X_better=test.better(X_eval)
    test.update(X_better)
    X=test.build(test.C_V,test.P)
    X_eval=test.adapt(X)
    OUT=np.array(X_eval)
    goals=OUT[np.lexsort(OUT.T)][0][-1]
    print(goals)