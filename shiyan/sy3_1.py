# filename='./1.txt'
# flag = eval(input('存词输入1，查词输入输入2：'))
# with open(filename,'a') as f:
#     while flag==1:
#         word=input('输入所存中文和英文：')
#         print(tuple(word.split(' ')))
#         f.writelines(word)
#         f.write('\n')
#         flag = eval(input('存词输入1，查词输入输入2：'))
# with open(filename,'r') as f:
#     while flag==2:
#         word = input('输入查询的单词：')
#         dict={}
#         for line in f:
#             dict_tuple=tuple(line.rstrip().split(' '))
#             dict[dict_tuple[0]]=dict_tuple[1]
#         if word not in dict.keys():
#             print('单词不存在')
#         else:
#             print(dict[word])
#         flag = eval(input('存词输入1，查词输入输入2：'))


filename='./1.txt'
dict={}
with open(filename,'r',encoding='utf-8') as f:
    for line in f:
        dict_tuple = tuple(line.rstrip().split(' '))
        # print(dict_tuple)
        dict[dict_tuple[0]] = dict_tuple[1]

flag = eval(input('存词输入1，查词输入2,输入3退出：'))
with open(filename,'a',encoding='utf-8') as f:
    while flag==1:
        word=input('输入所存中文和英文：')
        test=tuple(word.split(' '))
        if test[0] in dict.keys():
            print(test[0],'该单词已添加到字典库')
        else:
            f.writelines(word)
            f.write('\n')
            print('存入成功')
        flag = eval(input('存词输入1，查词输入2,输入3退出：'))

    while flag == 2:
        word = input('输入查询的单词：')
        test=tuple(word.split(' '))
        if test[0] not in dict.keys():
            print('单词不存在字典库中未找到这个单词')
        else:
            print(word,dict[word])
        flag = eval(input('存词输入1，查词输入2,输入3退出：'))
    if flag==3:
        print('再见，不送！！！！')
    while flag!=1 and flag!=2 and flag!=3:
        print('输入有误')
        break
