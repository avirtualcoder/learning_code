import random
print([i for i in [random.randint(0,100) for i in range(50)] if i%2==0])