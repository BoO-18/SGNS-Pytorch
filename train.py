import os
import sys
import time
import logging
import torch
from torch.autograd import Variable
import torch.optim as optim
from model import SkipGram
from data_preprocessing import Preprocess


def train(input_file, vocabulary_size,
                 embedding_dim, epoch, batch_size, windows_size, neg_sample_size):
    model = SkipGram(vocabulary_size, embedding_dim)
    Pre_data = Preprocess(input_file, vocabulary_size)

    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.2)

    for epoch in range(epoch):
        start = time.time()
        Pre_data.process = True
        batch_num = 0
        batch_new = 0

        while Pre_data.process:
            pos_u, pos_v, neg_v = Pre_data.Generate_batch(windows_size, batch_size, neg_sample_size)

            pos_u = Variable(torch.LongTensor(pos_u))
            pos_v = Variable(torch.LongTensor(pos_v))
            neg_v = Variable(torch.LongTensor(neg_v))

            if torch.cuda.is_available():
                pos_u = pos_u.cuda()
                pos_v = pos_v.cuda()
                neg_v = neg_v.cuda()

            optimizer.zero_grad()
            loss = model(pos_u, pos_v, neg_v, batch_size)
            loss.backward()
            optimizer.step()

            if batch_num % 3000 == 0:
                end = time.time()
                logger.info('epoch, batch = %2d %5d:   pair/sec = %4.2f  loss = %4.3f\r'
                      % (epoch, batch_num, (batch_num - batch_new) * batch_size / (end - start), loss.data.item()))
                batch_new = batch_num
                start = time.time()
            batch_num += 1

    model.save_embeddings(Pre_data.idx2word, 'word_embdding.txt', torch.cuda.is_available())


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(program)
    logger.info('running ' + program + ': train our SGNS model')

    infile = 'zhwiki_seg.txt'
    train(input_file=infile, vocabulary_size=100000,
                 embedding_dim=100, epoch=5, batch_size=256, windows_size=2, neg_sample_size=10)
