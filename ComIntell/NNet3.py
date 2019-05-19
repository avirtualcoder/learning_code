import pandas
import numpy as np
import matplotlib.pyplot as plt

iris = pandas.read_csv("iris.csv")
# shuffle rows
shuffled_rows = np.random.permutation(iris.index)          #根据iris的index进行 行顺序 打乱,使得数据更具有代表性
iris = iris.loc[shuffled_rows,:]          #切片操作

# 添加一个值全为1的属性iris["ones"]，截距
iris["ones"] = np.ones(iris.shape[0])    #iris.shape为（100,5）
X = iris[['ones', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
# 将Iris-versicolor类标签设置为1，Iris-virginica设置为0
y = (iris.species == 'Iris-versicolor').values.astype(int)


class NNet3:
    #初始化必要参数
    def __init__(self,learn_rate=0.5,maxepochs=10000,convergence_thre=0.000001,hidden_layer=4):
        self.learn_rate=learn_rate      #学习率
        self.maxepochs=int(maxepochs)   #最大训练次数
        self.convergence_thres=convergence_thre  #收敛阈值，判断误差改变是否过小
        self.hidden_layer=int(hidden_layer)    #隐含层层数
        self.theta0=np.random.normal(0,0.01,size=(5,4))  #初始化hide层权重
        self.theta1=np.random.normal(0,0.01,size=(5,1))  #初始化out层权重

    #激活函数
    def sigmoid(self,X,theta):
        X=np.array(X)           #隐含层输入即特征集转置(5,n)/输出层输入转置(5,n)
        theta=np.array(theta)   #隐含层权重(5,4)/输出层权重（5,1）
        return 1/(1+np.exp(-np.dot(theta.T,X)))  #隐含层神经元输出(4,n)，输出层神经元输出(1,n)

    #前向传播计算出最终值
    def feedforward(self,X):
        a=self.sigmoid(X.T,self.theta0).T  #特征集X(n,5),theta0(5,4),返回隐含层输出(4,n)再转置为(n,4)
        l1=np.column_stack([np.ones(a.shape[0]),a])  #合并两个行数相同的矩阵(n,5)，第一个为神经元偏置输入参数全置为1(n,1)，第二个为(n,4)
        l2=self.sigmoid(l1.T,self.theta1)       #隐含层输出(n,5),theta1输出神经元权重(5,1)
        return  l1,l2     #返回隐含层输出l1(n,5),n个个体的预测结果集l2(1,n)

    #计算预测结果误差
    def multiple_cost(self,X,y):
        l1,l2=self.feedforward(X)     #hide和out层的预测
        inner=y*np.log(l2)+(1-y)*np.log(1-l2)      #计算l2误差，最终误差(1,n)
        return -np.mean(inner)      #当前数据集的均误差，因为log（）恒为负，所以加-

    #传入未知样本进行预测
    def predict(self,X):
        h,y=self.feedforward(X)     #输入特征集返回预测结果集
        print(y)
        return y    #(1,n)

    #
    def learn(self,X,y):
        obs,cols=X.shape        #读出特征集的行列
        self.costs=[]           #存放误差的列表
        cost=self.multiple_cost(X,y)    #第一次总误差均值
        self.costs.append(cost)         #入列表
        costprev=cost+self.convergence_thres+1  #误差预测
        counter=0

        for i in range(self.maxepochs):     #不超最大次数
            l1,l2=self.feedforward(X)       #数据集一次前向传播各层结果
            l2_delta=(y-l2)*l2*(1-l2)       #根据最终误差求out层神经元梯度
            l1_delta=l2_delta.T.dot(self.theta1.T)*l1*(1-l1)    #根据out层的梯度与权重求hide层神经元梯度
            self.theta1+=l1.T.dot(l2_delta.T)/obs*self.learn_rate   #更新hide层的权重
            self.theta0+=X.T.dot(l1_delta)[:,1:]/obs*self.learn_rate#更新out层的权重
            counter+=1
            costprev=cost           #前一次误差
            cost=self.multiple_cost(X,y)    #当前误差
            self.costs.append(cost)
            print('当前为第{}次训练，训练集此次误差为{}'.format(counter,cost))
            if np.abs(costprev-cost)<self.convergence_thres and counter>self.maxepochs:       #误差变化小或代数过出循环
                break

X_train=X[0:90]     #训练输入集
y_train=y[0:90]     #训练集目标输出
X_test=X[90:]       #测试集输入
y_test=y[90:]       #测试集目标输出


model=NNet3(learn_rate=0.5,maxepochs=5000,convergence_thre=0.000001)       #加载神经网络类
model.learn(X_train,y_train)    #用训练集学习出权重集
y_pre=model.predict(X_test)           #用学习后的模型参数预测结果集

TP=0        #真正类数
TP_FP=0     #真正类数与假正类数，预测正类数
TP_FN=0     #真正类数与假负类数，实际正类数
for i in range(len(y_pre[0])):
    if y_test[i]==1:
        TP_FN+=1
    if y_pre[0][i]>0.5 and y_test[i]==1:
        TP+=1
    if y_pre[0][i]>0.5:
        TP_FP+=1
Precision=TP/TP_FP  #精确率机器分1对数占机器分1总数的比     查准率
Recall=TP/TP_FN     #机器分1对数占实际为1数的比     查全率
F_measure=2*Precision*Recall/(Precision+Recall) #预测效果评估

print('机器分1正确数 {}，机器分1数 {}， 实际1数 {}'.format(TP,TP_FP,TP_FN))
print('最终预测误差为 {}'.format(model.costs[-1]))
print('准确率 {}，召唤率 {},评估 {}'.format(Precision,Recall,F_measure))


plt.plot(model.costs)
plt.title("损失函数图",fontproperties='SimHei')
plt.ylabel("损失函数值/误差",fontproperties='SimHei')
plt.xlabel("训练次数",fontproperties='SimHei')
plt.show()