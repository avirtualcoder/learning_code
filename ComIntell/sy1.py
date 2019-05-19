import pandas
import matplotlib.pyplot as plt
import numpy as np


iris = pandas.read_csv("iris.csv")
# shuffle rows
shuffled_rows = np.random.permutation(iris.index)
iris = iris.loc[shuffled_rows,:]
print(iris.head())
'''
 sepal_length sepal_width petal_length petal_width species
80 7.4 2.8 6.1 1.9 Iris-virginica
84 6.1 2.6 5.6 1.4 Iris-virginica
33 6.0 2.7 5.1 1.6 Iris-versicolor
81 7.9 3.8 6.4 2.0 Iris-virginica
93 6.8 3.2 5.9 2.3 Iris-virginica
'''
# There are 2 species
print(iris.species.unique())
'''
['Iris-virginica' 'Iris-versicolor']
'''
# iris.hist()
# plt.show()

# 添加一个值全为1的属性iris["ones"]，截距
iris["ones"] = np.ones(iris.shape[0])
X = iris[['ones', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
# 将Iris-versicolor类标签设置为1，Iris-virginica设置为0
y = (iris.species == 'Iris-versicolor').values.astype(int)
# The first observation
x0 = X[0]
# 随机初始化一个系数列向量
theta_init = np.random.normal(0,0.01,size=(5,1))
def sigmoid_activation(x, theta):
    x = np.asarray(x)
    theta = np.asarray(theta)
    return 1 / (1 + np.exp(-np.dot(theta.T, x)))
a1 = sigmoid_activation(x0, theta_init)
print('X的0行特征与权重的隐含层输出:',a1)


# First observation's features and target
x0 = X[0]
y0 = y[0]
theta_init = np.random.normal(0,0.01,size=(5,1))
def singlecost(X, y, theta):
 # Compute activation
    h = sigmoid_activation(X.T, theta)
 # Take the negative average of target*log(activation) + (1-target) * log(1-activatio
    cost = -np.mean(y * np.log(h) + (1-y) * np.log(1-h))
    return cost
first_cost = singlecost(x0, y0, theta_init)
print('0行特征与0行已知标签得出的第一次损失误差：',first_cost)

# Initialize parameters
theta_init = np.random.normal(0,0.01,size=(5,1))
# Store the updates into this array
grads = np.zeros(theta_init.shape) # (5,1)梯度
# Number of observations
n = X.shape[0]
for j, obs in enumerate(X):
 # 计算预测值h(xi)
    h = sigmoid_activation(obs, theta_init)
 # 计算参数偏导δi
    delta = (y[j] - h) * h * (1 - h) * obs
# 对δi求平均
    grads += delta[:, np.newaxis] / X.shape[0]
print('x中个体特征的误差更新梯度：',grads)

theta_init = np.random.normal(0,0.01,size=(5,1))
# set a learning rate
learning_rate = 0.1
# maximum number of iterations for gradient descent
maxepochs = 10000
# costs convergence threshold, ie. (prevcost - cost) > convergence_thres
convergence_thres = 0.0001      #收敛阈值
def learn(X, y, theta, learning_rate, maxepochs, convergence_thres):
    costs = []
 # 计算一个样本产生的误差损失
    cost = singlecost(X, y, theta)
 # 0.01+阈值是为了在第一次迭代后与前一次（初始化为第一个样本的误差）误差的差值大于阈值
    costprev = cost + convergence_thres + 0.01
    counter = 0
    for counter in range(maxepochs):
        grads = np.zeros(theta.shape) # 初始化梯度为全0向量
    for j, obs in enumerate(X): # for循环计算总样本的平均梯度
        h = sigmoid_activation(obs, theta)
        delta = (y[j]-h) * h * (1-h) * obs
        grads += delta[:,np.newaxis]/X.shape[0]
 # 更新参数，由于J(Θ)前面加了负号，因此求最大值，所有此处是+
        theta += grads * learning_rate
        counter += 1
        costprev = cost # 存储前一次迭代产生的误差
        cost = singlecost(X, y, theta) # compute new cost
        costs.append(cost)
        if np.abs(costprev-cost) < convergence_thres: # 两次迭代误差大于阈值退出
            break
    plt.plot(costs)
    plt.title("Convergence of the Cost Function")
    plt.ylabel("J($\Theta$)")
    plt.xlabel("Iteration")
    plt.show()
    return theta
theta = learn(X, y, theta_init, learning_rate, maxepochs, convergence_thres)

theta0_init = np.random.normal(0,0.01,size=(5,4))
theta1_init = np.random.normal(0,0.01,size=(5,1))
# sigmoid_activation函数是前面已经写好的，在这作参考
def sigmoid_activation(x, theta):
    x = np.asarray(x)
    theta = np.asarray(theta)
    return 1 / (1 + np.exp(-np.dot(theta.T, x)))
def feedforward(X, theta0, theta1):
 # 逻辑函数中X.T(5,N), theta0(5,4)->(4,N)
    a = sigmoid_activation(X.T, theta0).T
 # a中每一行是一个样本产生的中间层的四个输入
 # 添加一个列值为1的截距向量
 # column_stack将将行数相同的两个数组纵向合并
    a = np.column_stack([np.ones(a.shape[0]), a])
 # activation units are then inputted to the output layer
    out = sigmoid_activation(a.T, theta1)
    return out
h = feedforward(X, theta0_init, theta1_init)

theta0_init = np.random.normal(0,0.01,size=(5,4))
theta1_init = np.random.normal(0,0.01,size=(5,1))
# X and y are in memory and should be used as inputs to multiplecost()
def multiplecost(X, y, theta0, theta1):
 # feed through network
    h = feedforward(X, theta0, theta1)
 # compute error
    inner = y * np.log(h) + (1-y) * np.log(1-h)
 # negative of average error
    return -np.mean(inner)
c = multiplecost(X, y, theta0_init, theta1_init)
# 将模型的函数凝结为一个类，这是很好的一种编程习惯
class NNet3:
 # 初始化必要的几个参数
 def __init__(self, learning_rate=0.5, maxepochs=1e4, convergence_thres=1e-5, hidden_layer=4):
    self.learning_rate = learning_rate
    self.maxepochs = int(maxepochs)
    self.convergence_thres = 1e-5
    self.hidden_layer = int(hidden_layer)
 # 计算最终的误差
 def _multiplecost(self, X, y):
 # l1是中间层的输出，l2是输出层的结果
    l1, l2 = self._feedforward(X)
 # 计算误差，这里的l2是前面的h
    inner = y * np.log(l2) + (1-y) * np.log(1-l2)
 # 添加符号，将其转换为正值
    return -np.mean(inner)
 # 前向传播函数计算每层的输出结果
 def _feedforward(self,  X):
     l1 = sigmoid_activation(X.T, self.theta0).T
     # 为中间层添加一个常数列
     l1 = np.column_stack([np.ones(l1.shape[0]), l1])
     # 中间层的输出作为输出层的输入产生结果l2
     l2 = sigmoid_activation(l1.T, self.theta1)
     return l1, l2

     # 传入一个结果未知的样本，返回其属于1的概率
 def predict(self, X):
     _, y = self._feedforward(X)

     return y

     # 学习参数，不断迭代至参数收敛，误差最小化
 def learn(self, X, y):
     nobs, ncols = X.shape
     self.theta0 = np.random.normal(0, 0.01, size=(ncols, self.hidden_layer))
     self.theta1 = np.random.normal(0, 0.01, size=(self.hidden_layer + 1, 1))
     self.costs = []
     cost = self._multiplecost(X, y)
     self.costs.append(cost)
     costprev = cost + self.convergence_thres + 1
     counter = 0
     for counter in range(self.maxepochs):
     # 计算中间层和输出层的输出
        l1, l2 = self._feedforward(X)
     # 首先计算输出层的梯度，再计算中间层的梯度
        l2_delta = (y - l2) * l2 * (1 - l2)
        l1_delta = l2_delta.T.dot(self.theta1.T) * l1 * (1 - l1)
     # 更新参数
        self.theta1 += l1.T.dot(l2_delta.T) / nobs * self.learning_rate
        self.theta0 += X.T.dot(l1_delta)[:, 1:] / nobs * self.learning_rate
        counter += 1
        costprev = cost
        cost = self._multiplecost(X, y)  # get next cost
        self.costs.append(cost)
        if np.abs(costprev - cost) < self.convergence_thres and counter > 500:
            break

 # Set a learning rate
 learning_rate = 0.5
 # Maximum number of iterations for gradient descent
 maxepochs = 10000
 # Costs convergence threshold, ie. (prevcost - cost) > convergence_thres
 convergence_thres = 0.00001
 # Number of hidden units
hidden_units = 4
# Initialize model
model = NNet3(learning_rate=learning_rate, maxepochs=maxepochs,
 convergence_thres=convergence_thres, hidden_layer=hidden_units)
# Train model
model.learn(X, y)
# Plot costs
plt.plot(model.costs)
plt.title("Convergence of the Cost Function")
plt.ylabel("J($\Theta$)")
plt.xlabel("Iteration")
plt.show()

# First 70 rows to X_train and y_train
# Last 30 rows to X_train and y_train
X_train = X[:70]
y_train = y[:70]
X_test = X[-30:]
y_test = y[-30:]

from sklearn.metrics import roc_auc_score
# Set a learning rate
learning_rate = 0.5
# Maximum number of iterations for gradient descent
maxepochs = 10000
# Costs convergence threshold, ie. (prevcost - cost) > convergence_thres
convergence_thres = 0.00001
# Number of hidden units
hidden_units = 4
# Initialize model
model = NNet3(learning_rate=learning_rate, maxepochs=maxepochs,convergence_thres=convergence_thres, hidden_layer=hidden_units)
model.learn(X_train, y_train)
# 因为predict返回的是一个二维数组，此处是(1,30)，取第一列作为一个列向量
yhat = model.predict(X_test)[0]
auc = roc_auc_score(y_test, yhat)



