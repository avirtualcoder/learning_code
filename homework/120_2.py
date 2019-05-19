#用lambda表达式实现函数用map内置函数求平方
power=lambda x:x**2
print(list(map(power,range(9))))