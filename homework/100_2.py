# 判断输入字符串的数据类型
str=input().split(' ')
intCount=0
strCount=0
otherCount=0
for i in str:
    if i.isdigit():
        intCount+=1
    elif i.isalpha():
        strCount+=1
    else:
        otherCount+=1
print('字符类型有%d个，整型有%d个，其他类型有%d个'%(strCount,intCount,otherCount))
