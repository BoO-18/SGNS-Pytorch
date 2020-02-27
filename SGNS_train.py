import os
import sys
import logging
import gensim

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(program)
    logger.info('running ' + program + ': training a SGNS model')

    # define the filename of input and output
    infile = 'data_total/zhwiki_seg.txt'
    outmodel = 'model/zhwiki_model'
    outvec = 'model/zhwiki_vector'

    vector_size = 100
    window_size = 5

    sentence = gensim.models.word2vec.LineSentence(infile)
    model = gensim.models.Word2Vec(sentences=sentence, size=vector_size, window=window_size, min_count=5, workers=4)

    # save model
    model.save(outmodel)
    model.wv.save_word2vec_format(outvec, binary=False)
