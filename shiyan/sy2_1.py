def count(str):
    big_count=0
    small_count=0
    number_count=0
    other_count=0
    str_new=str
    for i in range(len(str_new)):
        if str_new[i]<='Z' and  str_new[i]>='A':
            big_count+=1
        elif str_new[i]<='z' and str_new[i]>='a':
            small_count+=1
        elif str_new[i]<='9' and str_new[i]>='0':
            number_count+=1
        elif str_new[i]!=" ":
            other_count+=1
    return tuple(["big",big_count,"small",small_count,"number",number_count,"other",other_count])


str=input()
print(count(str))

