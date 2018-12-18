import random
ls=[random.randint(0,100) for i in range(1000)]
lb=set(ls)
d={}
for i in lb:
    d[i]=ls.count(i)
print(d)

