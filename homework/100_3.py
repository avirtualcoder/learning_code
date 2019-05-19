# 读文件数字数出次数
f=open('hamlet.txt','r')
str=f.read()
f.close()
str=str.lower()
count=0
dic={}
for i in '.,;:?\'|':
    str=str.replace(i,'')
words=str.split(' ')
word=set(words)
for i in word:
    dic[i]=words.count(i)
    count+=dic[i]
    if (dic[i]>=300):
        print('{} 出现 {}次'.format(i,dic[i]))
print('文章有%d字'%(count))
