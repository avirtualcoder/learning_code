import random
import csv
import math
ls=[(random.randint(0,100),random.randint(0,100)) for x in range(10) for y in range(10)]
print(len(ls))
point1=ls[random.randint(0,99)]
point2=ls[random.randint(0,99)]
distance=round(math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2),2)
filename='./distance.csv'
print(point1,point2,distance)
with open(filename,'a',newline='') as f:
    f.writelines(str(point1).replace(',',' ')+','+str(point2).replace(',',' ')+','+str(distance)+'\n')
with open(filename,'r') as f:
    dict=csv.DictReader(f)
    print([i for i in dict])