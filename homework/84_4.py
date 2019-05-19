s1={1,2,3,4,6,8,12,24}
s2={1,2,3,4,6,9,12,18,36}
s5=set()
s6=set()
i=1
while i<6:
    x=str(i*24)
    y=str(i*36)
    s5.add(x)
    s6.add(y)
    i=i+1
s3=s2-s1
s4=s1-s2
print('24和36的最大公约数为 '+str(max(s1&s2))+"         24和36的最小公倍数为 "+str(min(s5&s6)))
print('24的独有约数集为 '+str(s3)+'    36的独有约数集为 '+str(s4))
