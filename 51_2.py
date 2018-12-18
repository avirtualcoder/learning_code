import math

flag='1'

while flag=='1':

    NumberTest = input('please input the number you wanna test:')
    Number = NumberTest.split(' ')
    sum = 0;n = len(Number); i = 0
    # 输入数据，初始化所需的参数

    while i<n:
        Number[i]=int(Number[i])
        sum+=Number[i]
        i=i+1
    u=sum/n;sum=0;i=0
    #while循环求均值

    while i<n:
        sum+=(i-u)*(i-u);i=i+1
    pp=sum/n;p=math.sqrt(pp)
    # while循环求方差，标准差

    print('''the number\'s average is {0:<20,.2f}  
             its fancha is {1:>20,.2f}  
             its biaozhuncai is {2:<20,.2f}'''.format(u,pp,p))
    # 格式化输出

    flag=input('if you wanna test again input \'1\' , else input anything without \'1\'')
    # 判断是否继续