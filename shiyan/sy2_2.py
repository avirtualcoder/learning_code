def chose_big(*number):
    new=sorted(list(number))
    print(new[-1],sum(new))
chose_big(1,2,3,23,1121,12,31)