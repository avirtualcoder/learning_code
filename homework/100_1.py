#猜数字游戏
import random
guess=1
number=random.randint(0,100)
while True:
    numberGuess = eval(input('输入一个0和100间的整数： '))
    print(numberGuess,end='    ')
    if numberGuess!=number:
        print('第{}次猜测，猜错了，结果偏{}'.format(guess,'大' if numberGuess>number else "小"))
        guess+=1
    else:
        print('弟%d次，猜测猜对了！！！'%(guess))
        break