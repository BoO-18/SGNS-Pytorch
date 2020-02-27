import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.embedding_dim = embedding_dim
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, u_pos, v_pos, v_neg, batch_size):
        embed_u = self.u_embeddings(u_pos)
        pos_embed_v = self.v_embeddings(v_pos)
        pos_score = torch.sum(torch.mul(embed_u, pos_embed_v), dim=1)
        pos_output = F.logsigmoid(pos_score).squeeze()

        neg_embed_v = self.v_embeddings(v_neg)
        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        neg_output = F.logsigmoid(-1 * neg_score).squeeze()

        cost = pos_output + neg_output
        return -1 * cost.sum() / batch_size

    def save_embeddings(self, id2word, file_name, use_cuda):
        if use_cuda:
            embedding = self.u_embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.u_embeddings.weight.data.numpy()

        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(id2word), self.embedding_dim))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))
