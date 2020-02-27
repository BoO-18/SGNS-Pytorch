import logging
import os.path
import sys
from gensim.corpora import WikiCorpus

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(program)
    logger.info('running ' + program + ': parse the chinese corpus')

    # define the filename of input and output
    infile = 'data/zhwiki-20191120-pages-articles-multistream.xml.bz2'
    outfile = 'data/zhwiki.txt'

    # transform .XML to .txt
    i = 0
    output = open(outfile, 'w')
    wiki = WikiCorpus(infile, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        output.write(" ".join(text) + "\n")
        i = i + 1
        if i % 10000 == 0:
            logging.info("Save " + str(i) + " articles")
    output.close()
    logging.info("Finished saved " + str(i) + "articles")
