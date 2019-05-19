import random
ls=[random.randint(0,100) for i in range(20)]
lb=sorted(ls[0:10])
lc=sorted(ls[10:20],reverse=1)
print( lb,lc)