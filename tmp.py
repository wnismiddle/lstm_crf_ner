
'''
y = tf.abs(x) the presentation of x = a+bj , y = 根号下的（a方+b方）
ex：
    x = tf.constant([[-2.25 + 4.75j], [-3.25 + 5.75j]])
    tf.abs(x)  # [5.25594902, 6.60492229]
y = tf.sign(x) if x<0, y =-1 ;if x=0, y=0; if x>0, y=1
y = reduce_sum(x) 计算输入tensor元素的和，或者按照reduction_indices指定的轴求和
ex :
    x is [[1,1,1],[1,1,1]]
    tf.reduce_sum(x) = 6
    tf.reduce_sum(x, 0) = [2,2,2] 纵向相加
    tf.reduce_sum(x, 1) = [3,3] 横向相加
    tf.reduce_sum(x, 1, keep_dims = true) = [[3],[3]]
    tf.reduce_sum(x, [0,1]) = 6

'''

# def count_zhishu(number):
#     count = 0
#     numbers = []
#     for i in range(2,number):
#         flag = True
#         for j in range(2,i):
#             if i%j==0:
#                 flag = False
#                 break
#         if flag == True:
#             numbers.append(i)
#             count +=1
#     return count, numbers
# n ,numbers = count_zhishu(10)
# print("N=",str(n), "质数有", numbers)
#
# for i in range(0,5):
#     print(i)

# for i, s in enumerate("今天也是晴天o(*￣︶￣*)o"):
#     print(i, '***', s)

# a = [{},{'a':'hhh'}]
# for i in a:
#     if i :
#         print('***{}***'.format(i))


def result_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
        elif tag[0] == "B":
            entity_name = char
            entity_start = idx
        elif tag[0] == "I":
            if entity_start == -1:
                entity_start = idx
            entity_name += char
        elif tag[0] == "E":
            if entity_start != -1:
                entity_name += char
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
                entity_start = -1
            entity_name = ""
        else:
            entity_name = ""
            entity_start = -1
        idx += 1
    return item

string = '今天真是个好日子呢虽然我又熬夜了'
tags = ['B-per','O','B-per','O','I-per','E-per','E-per','O','I-per','O','B-per','O','S-per','E-per','O','I-per',]
item = result_to_json(string, tags)
print(item)
