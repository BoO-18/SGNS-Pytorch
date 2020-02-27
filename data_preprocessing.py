import math
import numpy as np
import collections
import random

data_index = 0


class Preprocess(object):
    def __init__(self, infile, vocab_size):
        self.vocab_size = vocab_size
        self.data_file = infile
        self.vocab = self.Read_file()
        self.data_idx, self.word_count, self.idx2word = self.Build_dataset(self.vocab, self.vocab_size)
        self.train_data = self.Subsampling(self.data_idx)
        self.sample_data = self.Negative_sampling()

    def Read_file(self):
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = f.read().split()
            data = [x for x in data if x != 'eoood']
        return data

    def Build_dataset(self, vocabulary, size):
        count_word = [['UNK', -1]]
        count_word.extend(collections.Counter(vocabulary).most_common(size - 1))
        dictionary = dict()

        for word, _ in count_word:
            dictionary[word] = len(dictionary)
        data_idx = list()
        unk_count = 0
        # print(dictionary)

        for word in vocabulary:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count += 1
            data_idx.append(index)
        # print(data_idx)
        count_word[0][1] = unk_count
        reversed_dict = dict(zip(dictionary.values(), dictionary.keys()))
        return data_idx, count_word, reversed_dict

    def Subsampling(self, dataidx):
        count = [c[1] for c in self.word_count]
        freqs = count / np.sum(count)
        p_drop = dict()

        # t = 1e-5
        # threshould = 0.9
        # for idx, x in enumerate(freqs):
        #     y = 1 - math.sqrt(t / x)
        #     p_drop[idx] = y
        # print(p_drop)
        # subsampled_data = list()
        # for word in dataidx:
        #     if p_drop[word] < threshould:
        #         subsampled_data.append(word)

        sample = 0.001
        for idx, x in enumerate(freqs):
            y = (math.sqrt(x / sample) + 1) * sample / x
            p_drop[idx] = y
        # print(p_drop)
        subsampled_data = list()
        for word in dataidx:
            if random.random() < p_drop[word]:
                subsampled_data.append(word)
        print('length:', len(subsampled_data))
        return subsampled_data

    def Negative_sampling(self):
        count = [c[1] for c in self.word_count]
        pow_freq = np.array(count) ** 0.75
        pow_sum = np.sum(pow_freq)
        ratio = pow_freq / pow_sum

        table_size = 1e8
        count = np.round(ratio * table_size)
        sample_table = []

        for idx, x in enumerate(count):
            sample_table += [idx] * int(x)

        # print(sample_table)
        return np.array(sample_table)

    def Generate_batch(self, window_size, batch_size, neg_sample_size):
        data = self.train_data
        global data_index

        span = 2 * window_size + 1
        context = np.ndarray(shape=(batch_size, 2 * window_size), dtype=np.int64)
        labels = np.ndarray(shape=(batch_size), dtype=np.int64)
        if data_index + span > len(data):
            data_index = 0
            self.process = False

        buffer = data[data_index: data_index + span]
        pos_u = []
        pos_v = []
        for i in range(batch_size):
            context[i, :] = buffer[:window_size] + buffer[window_size+1:]
            labels[i] = buffer[window_size]

            for j in range(span - 1):
                pos_u.append(labels[i])
                pos_v.append(context[i, j])

            data_index += 1
            if data_index + span > len(data):
                buffer = data[:span]
                data_index = 0
                self.process = False
            else:
                buffer = data[data_index: data_index + span]
        neg_v = np.random.choice(self.sample_data, size=(batch_size * 2 * window_size, neg_sample_size))
        return np.array(pos_u), np.array(pos_v), neg_v


if __name__ == '__main__':
    infile = 'zhwiki_seg.txt'
    vocab_size = 10000
    my_data = Preprocess(infile, vocab_size)
    # pos_u, pos_v, neg_v = my_data.Generate_batch(2, 1, 10)
    # print(pos_u)
    # print(pos_v)
    # print(neg_v)
