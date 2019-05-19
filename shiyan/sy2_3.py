def sorted(*args,reverse=0):
    if len(args)>1:
        ls=list(args)
    else:
        ls=list(args[0])
    print(ls)
    if reverse==0:
        for i in range(len(ls)-1):
            if ls[i]>ls[i+1]:
                ls[i],ls[i+1]=ls[i+1],ls[i]
    else:
        for i in range(len(ls)-1):
            if ls[i]<ls[i+1]:
                ls[i],ls[i+1]=ls[i+1],ls[i]
    return ls
print(sorted([2,1,43,23]))

