number=int(input('输入一个小于1000'))
ls=[]
lb=[number]
def m(i,x):
    number=0
    if x%i==0:
        number=x/i
        ls.append(i)
        m(i,number)
    else:
        lb.append(x)
        return
j=2
while j<=lb[-1]:
    m(j,lb[-1])
    j+=1
out=str(ls)
print(number,'=',out.replace(',','*').strip('['']'))
