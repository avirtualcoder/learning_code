# 自定义函数求列表中位数
def middle(*number):
    ls=list(number)
    ls.sort()
    if len(ls)%2==0:
        return (float((ls[int(len(ls)/2)]+ls[int(len(ls)/2)+1])/2))
    else:
        return (float(ls[int(len(ls)/2)]))

print(middle(1,2,3,4,6))
