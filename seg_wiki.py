import os
import sys
import jieba
import logging

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(program)
    logger.info('running' + program + ': segment the corpus')

    # define the filename of input and output
    infile = 'data_total/zhwiki_removed.txt'
    outfile = 'data_total/zhwiki_seg.txt'

    # segment sorpus
    with open(infile, 'r', encoding='utf-8') as fin:
        fout = open(outfile, 'w', encoding='utf-8')

        for lines in fin:
            seg_result = jieba.cut(lines)
            seg_result = ' '.join(seg_result)
            fout.write(seg_result)

        fout.close()
    logger.info('segment finished')
