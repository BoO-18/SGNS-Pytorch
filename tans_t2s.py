import os
import sys
import logging
from opencc import OpenCC
import re

def trans_t2s(infile, outfile):
    # read the traditional Chinese corpus
    tra_corpus = []
    with open(infile, 'r', encoding='utf-8') as f:
        for lines in f:
            lines = lines.replace('\n', '').replace('\t', '')
            tra_corpus.append(lines)
        logger.info('read traditional file finished!')

    # tranform the traditional Chinese corpus to simplified Chinese
    sim_corpus = []
    cc = OpenCC('t2s')
    logger.info('the total length of file is {}'.format(len(tra_corpus)))
    i = 0
    for lines in tra_corpus:
        sim_corpus.append(cc.convert(lines))
        i = i + 1
        if i % 1000 == 0:
            logger.info('{} lines have finished'.format(i))
    logger.info('transform finished')

    # write the simplified corpus to outfile
    with open(outfile, 'w', encoding='utf-8') as f:
        for lines in sim_corpus:
            f.writelines(lines + '\n')
    logger.info('write the simplified file finished!')


def remove_English(infile, outfile):
    with open(infile, 'r', encoding='utf-8') as fin:
        fout = open(outfile, 'w', encoding='utf-8')
        for lines in fin.readlines():
            res = re.sub("[ A-Za-z]", "", lines)
            fout.write(str(res))
        fout.close()
        logger.info('finished to remove english corpus')


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(program)
    logger.info('running ' + program + ': transform traditional Chinese to simplified Chinese')

    # define the filename of input and output
    infile = 'data/zhwiki.txt'
    outfile = 'data_total/zhwiki_simplified.txt'
    outfile2 = 'data_total/zhwiki_removed.txt'

    trans_t2s(infile, outfile)
    remove_English(outfile, outfile2)
