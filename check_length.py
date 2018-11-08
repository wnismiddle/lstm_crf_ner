

def check_index(i, word):
    org_file = 'data/testdata/入院记录现病史-' + str(i) + '.txtoriginal.txt'
    line = ''.join(open(org_file, 'r', encoding='utf-8').readlines())
    print(line)
    index = line.index(word)
    print(index, index+len(word))

if __name__ == '__main__':
    check_index(31, '56G/28')