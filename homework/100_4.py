# 20以内的所有数组成的列表
print([i for i in range(1,21)])

# 所有水仙花数组成的列表
print([i*100+j*10+k for i in range(1,10) for j in range(10) for k in range(10) if (i**3+j**3+k**3)==(i*100+j*10+k) ])

#100以内的所有素数组成的集合
print([num for num in range(1,101) if num not in [x for x in range(2,101) for j in range(2,x) if x%j==0 ]])

#由26个英文字母的每个字母及其Unicode编码为键值对构成的字典
print({chr(i):i for i in range(ord('a'),ord('z')+1) })