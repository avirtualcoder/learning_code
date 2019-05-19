import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

task = pd.read_csv('tasks2.csv')
user= pd.read_csv('usersAndRoutes2.csv')
X = task[['tx1', 'ty1']].values             #X为tasks1的数据的numpy数组，数据类型为（298,3），X为任务数据
# print(X[0:5])
Y=user[['x1','y1','x2','y2']].values  #Y为用户数据
# print(Y[0:5])

class EM_allocate():                #贪心算法
    def __init__(self,Tasks,Users ):
        self.T=Tasks        #任务集
        self.U=Users        #用户集
        self.u=[0 for i in range(len(self.U))]      #用户次数集

    def dis(self,x1, y1, x2, y2,tx,ty): #计算曼哈顿距离函数
        return abs(x1-tx)+abs(y1-ty)+abs(tx-x2)+abs(ty-y2)-(abs(x1-x2)+abs(y1-y2))

    def chose_can(self):        #判断可分配用户返回可分配列表
        self.u_can=[]
        for i in range(len(self.u)):
            if self.u[i]<5:
                self.u_can.append(i)
        return self.u_can

    def chose_user(self,u_can,r):   #根据距离和分配列表进行分配
        count=0             #每个任务不可过5人
        factor=[]           #记录当前任务分配的用户
        for i in u_can:
            a=self.dis(self.U[i][0],self.U[i][1],self.U[i][2],self.U[i][3],r[0],r[1])
            if a==0 and count<5:
                factor.append(i)
                self.u[i]+=1        #分配次加一，一个用户不可超5个任务
                count+=1

        return factor


if __name__=='__main__':
    star=datetime.datetime.now()    #记录时间
    a=EM_allocate(X,Y)              #输入用户坐标和任务坐标
    count=0
    show=[]
    R=[]                            #分配列表
    for i in range(len(X)):         #逐个任务进行分配
        u_can=a.chose_can()         #出可分配列表
        fator=a.chose_user(u_can,X[i])  #分配可执行当前任务的用户
        R.append(fator)
        count+=len(fator)           #累计分配数
        show.append(len(fator))
        print("任务{}  分配的人为:  {} ".format(i,fator))
fitness=count/(5*len(R))    #分配率
end=datetime.datetime.now()
print("分配率为",fitness)
print("时间为",end-star)









# show=np.array(show)
#
#
# n, bins, patches = plt.hist(show.T, bins=len(show), normed=0, facecolor='black', edgecolor='black',alpha=1,histtype='bar')
# plt.show()