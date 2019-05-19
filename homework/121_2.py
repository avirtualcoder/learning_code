#统计小说的单词频数并排序
import  time
# time_star=time.time()

f=open('Walden.txt','r')
str=f.read()
# for line in f.readlines():
#     line=line.strip('\n')
f.close()
str=str.lower()
# for j in str:
#     for i in '\n!?,:\'.;':
#         if j==i:
#             j=' '
for i in "`/-\".,:;!()?":
    str.replace(i,' ')
word=str.split()
# str1=[]
# for i in word:
#     if i not in str1:
#         str1.append(i)
word1=set(word)
count=0
dic={}
for i in word1:
    dic[i]=word.count(i)
    count+=dic[i]
print (sorted(dic.items(),key=lambda x:x[1],reverse=True))
# print(sorted(zip(dic.values(),dic.key()),reverse=True))
print(count)
# time_end=time.time()

print(time.process_time())