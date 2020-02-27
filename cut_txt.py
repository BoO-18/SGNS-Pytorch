
if __name__ == '__main__':
    infile = 'data/zhwiki.txt'
    outfile = 'data/zhwiki_part.txt'
    with open(outfile, 'a', encoding='utf-8') as fout:
        fin = open(infile, 'r', encoding='utf-8')
        for i in range(1000):
            line = fin.readline()
            fout.write(line)
        fin.close()
    print('cut txt finished')

