import csv
import math
filename='./iris_sepal_length.csv'
with open(filename,'r') as f:
    data=csv.reader(f)
    # data=sorted(data)
    # print(data)
    data_str=[str(i).strip('[''\''']') for i in data]
    data_float= list(map(float, data_str))
    data=sorted(data_float)
print(data)     #原始数据排序
print('去重',set(data_float))
print('累积和为',sum(data_float))
aver=sum(data_float)/len(data)
print('均值',aver)
var=sum([(i-aver)**2 for i in data])/len(data)
print('方差',var)
print('标准差',math.sqrt(var))
print('最小值',min(data_float))
print('最大值',max(data_float))