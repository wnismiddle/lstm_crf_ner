f = open('data/result.txt','r',encoding='utf-8')
check_list = []
check_set = []
count = 1
for line in f.readlines():
    # line = line.strip()
    index = line.split(',')[0]
    check_list.append(index)
    check_set.append(index)
    entities = line.split(',')[1].split(';')
    entities.pop()
    for entity in entities:
        split = entity.split('\t')
        word = split[0]
        start = int(split[1])
        end = int(split[2])
        # print(split)
        if not len(split) == 4:
            print('实体异常')
            print(entity)
            print(index, count, split)
        elif not isinstance(start, int):
            print('起始下标异常')
            print(entity)
            print(index, count, split)
        elif not isinstance(end, int):
            print('结束下标异常')
            print(entity)
            print(index, count, split)
        elif not end - start == len(word):
            print('--------------------\n实体长度与下标不一致')
            print(entity)
            print(index, count, split)
        count += 1

assert len(check_list) == 400
assert len(check_set) == 400