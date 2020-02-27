import os
import sys
import logging
import gensim

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(program)
    logger.info('running ' + program + ': finish the test txt')

    infile = 'data/pku_sim_test.txt'
    outfile = 'data/2019110717.txt'

    model = gensim.models.Word2Vec.load('model/zhwiki_model')
    with open(infile, 'r', encoding='utf-8') as fin:
        fout = open(outfile, 'w', encoding='utf-8')
        for line in fin:
            total_input = line.split()
            if total_input[0] in model and total_input[1] in model:
                similarity = model.wv.similarity(total_input[0], total_input[1])
                fout.writelines([total_input[0], '\t', total_input[1], '\t', str(similarity), '\n'])
            else:
                similarity = 'OOV'
                fout.writelines([total_input[0], '\t', total_input[1], '\t', similarity, '\n'])

        fout.close()

    logger.info('finish write output file')
    # req_count = 5
    # for key in model.wv.similar_by_word('世纪', topn=100):
    #     req_count -= 1
    #     print(key[0], key[1])
    #     if req_count == 0:
    #         break
    #
    # temp = model.wv.similarity('猫', '狗')
    # print(temp)
    #
    # print(model.wv['猫'])
