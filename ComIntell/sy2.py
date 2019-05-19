import random
import math
import matplotlib.pyplot as plt

class PSO():
    def __init__(self,n=2,size=3):
        self.n=n        #维度
        self.size=size  #种群大小
        self.PI=3.1415926535    #圆周率，用于度与角的转换
        self.X=[[random.uniform(0.1,10),random.uniform(1,90)]for i in range(size)]  #按限制条件创建变量
        self.V=[[random.uniform(-1,1),random.uniform(-10,10)]for i in range(size)]  #初始化速度
        self.bset_x_value=self.adapt(self.X[0])     #记录最优适应值

    def adapt(self,x):      #求面积函数
        area=1/2*(40-4*x[0]+2*x[0]*math.cos(x[1]*self.PI/180))*math.sin(x[1]*self.PI/180)*x[0]
        return  area

    def assess(self):  # 评估函数
        self.Pbest = self.X  # 局部最优为粒子个体本身坐标
        # print('各个体局部最优坐标 Pbest   ',Pbest)
        Eval = [self.adapt(self.Pbest[i]) for i in range(self.size)]  # 各粒子适应值成列表
        # print('各个体的适应值  ', Eval)
        self.Gbest = self.Pbest[Eval.index(max(Eval))]  # 列表中得全局最优粒子坐标
        self.bset_x_value=max(Eval)
        # print('全局最优坐标 Gbest  ',Gbest)

    def update(self):  # 更新粒子群坐标及速度
        w = 0.5  # 惯性权重
        c1 = 2
        c2 = 2  # 加速系数
        for i in range(self.size):  # 迭代粒子数
            for j in range(self.n):  # 迭代维数
                self.V[i][j] = w * self.V[i][j] + c1 * random.random() * (self.Pbest[i][j] - self.X[i][j]) + c2 * random.random() * (
                            self.Gbest[j] - self.X[i][j])
                self.X[i][j] += self.V[i][j]  # 根据自身局部最优和全局最优向当前最优坐标前进更新
x=[]
b=[]
s=[]
# for i in range(3):
a=PSO()
g=0     #进化代数
while g<500:
        g+=1
        a.assess()
        a.update()
        s.append(a.bset_x_value)
        print('第 {} 代粒子群进化  最优位置即变量值为 {}  面积为 {}'.format(g,a.Gbest,a.bset_x_value))
print('最终的梯形面积为 {}  腰边长为 {}  a倾角为 {}度'.format(round(a.bset_x_value,2),round(a.Gbest[0],1),round(a.Gbest[1],2)))
plt.plot(s)
plt.title('多次面积图', fontproperties='SimHei')
plt.ylabel('面积大小', fontproperties='SimHei')
plt.xlabel('次数大小', fontproperties='SimHei')
plt.show()
x.append(round(a.Gbest[0],1))
b.append(round(a.Gbest[1],2))
# print('3次PSO的平均边长为{}米  倾角为{}度 '.format(round(sum(x)/len(x),1),round(sum(b)/len(b))))
