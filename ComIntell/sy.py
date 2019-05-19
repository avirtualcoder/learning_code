#-*- coding:utf-8 -*-
#-*- coding:utf-8 -*-
import pandas
import datetime
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

#定义激活函数，返回输出值
def sigmoid_activation(x, theta):     #x为shape为（n，5）的2D张量，
                                      # theta为shape为（input_shape[1],output_shape[1]
    x = np.asarray(x)                 #将输入数据转化为张量
    theta = np.asarray(theta)
    return 1 / (1 + np.exp(-np.dot(theta.T, x)))

#定义前向传播函数，返回计算出的每层输出值
def feedforward(X, theta0, theta1):
    # 激活函数中X.T为(5,N), theta0为(5,4)->(4,N)
    a = sigmoid_activation(X.T, theta0).T             
    # a中每一行是一个样本产生的中间层的四个输入
    a = np.column_stack([np.ones(a.shape[0]), a])# 添加一个列值为1的截距向量，
                                                 # column_stack将将行数相同的两个数组纵向合并
    out = sigmoid_activation(a.T, theta1)
    return a,out

def predict(X, theta0_init, theta1_init):             #定义输出预测值函数
    _, y =feedforward(X, theta0_init, theta1_init)
    return y

# h = predict(X, theta0_init, theta1_init)
# print(h)


# 定义损失函数
def multiplecost(X, y, theta0, theta1):   #X为输入，y为目标输出
    _,h = feedforward(X, theta0, theta1)
    # compute error
    inner = y * np.log(h) + (1-y) * np.log(1-h)
    # negative of average error
    return -np.mean(inner)


def learn(X_train, y_train,X_test,y_test,theta0, theta1,learning_rate, maxepochs, convergence_thres):
    costs = []              #训练集每轮的误差
    val_costs=[]            #测试集每轮的误差
    cost =multiplecost(X_train, y_train, theta0, theta1)
    costs.append(cost)
    costprev = cost + convergence_thres+1
    counter = 0
    for counter in range(maxepochs):
    # 计算中间层和输出层的输出
        l1, l2 =feedforward(X_train, theta0, theta1)
        # 首先计算输出层的梯度，再计算中间层的梯度
        l2_delta = (y_train-l2) * l2 * (1-l2)
        l1_delta = l2_delta.T.dot(theta1.T) * l1 * (1-l1)
        # 更新参数
        theta1 += l1.T.dot(l2_delta.T) / X_train.shape[0] * learning_rate
        theta0 += X_train.T.dot(l1_delta)[:,1:] / X_train.shape[0] * learning_rate
        counter += 1
        costprev = cost
        cost = multiplecost(X_train, y_train, theta0, theta1) # get next cost
        val_cost=multiplecost(X_test,y_test,theta0,theta1)      
        costs.append(cost)
        val_costs.append(val_cost)
        print('第',counter,'次更新后误差为：',cost)
        if np.abs(costprev-cost) < convergence_thres and counter >5000:  #这里的原来为500，若数字再大结果越精确
                                                                        
            break

    return theta0,theta1


# 测试用的代码
# theta0_init = np.random.normal(0,0.01,size=(5,4))#theta0_init是输入层到中间层的权值，因为中间层是4个神经元，所有这里第二个是4列，5是输入数据加上偏置就5个
# theta1_init = np.random.normal(0,0.01,size=(5,1))#theta1_init是中间层到输出层的权值。5是隐含层4个神经元，输出数据4个加上偏置就为5个输入
# learning_rate = 0.1                                 #定义学习率
# maxepochs = 10000                                  
# convergence_thres = 0.0001           #定义阈值
# theta0,theta1=learn(X, y, theta0_init,theta1_init,learning_rate, maxepochs, convergence_thres)              #theta0,theta1为最终学习好的参数


#开始运行
start = datetime.datetime.now()
X_train = X[:70]
y_train = y[:70]
X_test = X[-30:]
y_test = y[-30:]
theta0_init = np.random.normal(0,0.01,size=(5,4))#theta0_init是输入层到中间层的权值，因为中间层是4个神经元，
                                                 # 所有这里第二个是4列，5是输入数据加上偏置就5个
theta1_init = np.random.normal(0,0.01,size=(5,1))#theta1_init是中间层到输出层的权值。5是隐含层4个神经元，
                                                 # 输出数据4个加上偏置就为5个输入
learning_rate = 0.1                                 #定义学习率
maxepochs = 10000                                   #定义轮次，在所有训练数据上迭代一次叫做一个轮回
convergence_thres = 0.0001           #定义阈值
theta0,theta1=learn(X_train, y_train,X_test,y_test,theta0_init,theta1_init,learning_rate, maxepochs, convergence_thres)
yhat = predict(X_test,theta0,theta1)          #X_text为n:5,且X_text.dim要为2，不可以直接X_text[n],这样得出的是一个一维的
print('预测结果：',yhat)



n=0
a=[]
for i in range(len(yhat[0])):
    if yhat[0][i]<=0.5:
        n=n+1
        a.append(i)            #a为包含被分类为0的数据的索引

print('机器分类为0的个数有：',n)
m=0
for i in range(len(y_test)):
    if y_test[i]==0:
        m+=1
print('实际分类为0的个数有：',m)
z=0
for i in a:
    if y_test[i]==0:
        z+=1
print('被机器分类为0且实际为0的数据个数有：',z)
TP=z
FN=m-z
FP=n-TP
TN=len(yhat[0])-FP-TP-FN
print('故TP，FP，FN，TN分别为：',TP,FP,FN,TN)
Precision=TP/(TP+FP)
Recall=TP/(TP+FN)
F_measure=(2*Precision*Recall)/(Precision+Recall)
print('精确率、召回率、F-measure分别为：',Precision,Recall,F_measure)


end = datetime.datetime.now()
print ('运行时间为：',end-start)
