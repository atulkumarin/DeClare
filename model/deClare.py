import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class DeClareModel(nn.Module):
    def __init__(self, glove_embeddings, claim_source_vocab_size, article_source_vocab_size, nb_lstm_units, device='cpu'):
        super(DeClareModel, self).__init__()

        self.device = device

        self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(glove_embeddings), freeze=True)
        self.claim_source_embeddings = nn.Embedding(claim_source_vocab_size, 4)
        self.article_source_embeddings = nn.Embedding(article_source_vocab_size, 8)

        self.embedding_dim = glove_embeddings.shape[1]
        self.nb_lstm_units = nb_lstm_units

        self.attention_dense = nn.Linear(2*self.embedding_dim, 1)
        self.attention_dropout = nn.Dropout(0.5)

        self.dense_1 = nn.Linear(2*self.nb_lstm_units + 4 + 8, 64)
        self.dense_1_dropout = nn.Dropout(0.5)

        self.dense_2 = nn.Linear(64, 64)
        self.dense_2_dropout = nn.Dropout(0.5)

        self.output_layer = nn.Linear(64, 1)

        self.biLSTM = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.nb_lstm_units,
                              batch_first=True, bidirectional=True)

        self.to(device)

    def forward(self, claim, claim_len, article, article_len, claim_source, article_source):

        batch_size = claim.shape[0]
        self.hidden = self.init_hidden(batch_size)

        claim_mean_embedding = torch.div(torch.sum(self.word_embeddings(claim).float(), 1), claim_len.float().unsqueeze(1))
        #shape : (batch, embedding_dim)

        article_embeddings = self.word_embeddings(article).float()
        #shape : (batch, seq_len, embedding_dim)


        # ATTENTION BRANCH
        claim_article_concat = torch.cat([article_embeddings,
                                          claim_mean_embedding.unsqueeze(1).expand(-1, article_embeddings.shape[1], -1)], 2)
        #shape : (batch, seq_len, 2*embedding_dim)

        claim_article_concat = torch.tanh(self.attention_dense(claim_article_concat)).squeeze()
        claim_article_concat = self.attention_dropout(claim_article_concat)
        #shape : (batch, seq_len)

        attentions = F.softmax(claim_article_concat, dim=1)
        #shape : (batch, seq_len)

        # create masks based on sequence lengths
        max_len = article.shape[1]
        idxes = torch.arange(0,max_len,out=torch.LongTensor(max_len)).unsqueeze(0).to(self.device) # some day, you'll be able to directly do this on cuda
        mask = Variable((idxes<article_len.unsqueeze(1)).float()).to(self.device)

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(1).view(-1,1).expand_as(attentions)  # sums per row
        attentions = masked.div(_sums)
        #shape : (batch, seq_len)


        # LSTM BRANCH
        article_sequence = nn.utils.rnn.pack_padded_sequence(article_embeddings, article_len, batch_first=True)

        article_sequence_representation, self.hidden = self.biLSTM(article_sequence, self.hidden)

        article_sequence_representation, _ = torch.nn.utils.rnn.pad_packed_sequence(article_sequence_representation, batch_first=True, padding_value=0)
        #shape : (batch_size, seq_len, num_dir*hidden_units)

        article_sequence_representation = article_sequence_representation*attentions.unsqueeze(-1).expand_as(article_sequence_representation)
        attention_focused_article_rep = torch.div(article_sequence_representation.sum(1), article_len.float().unsqueeze(1))
        #shape : (batch_size, num_dir*hidden_units)


        # FINAL INFERENCE LAYERS
        claim_source_embedding = self.claim_source_embeddings(claim_source)
        article_source_embedding = self.article_source_embeddings(article_source)
        #shape : (batch_size, embedding_dim)
        full_feature = torch.cat([claim_source_embedding, attention_focused_article_rep, article_source_embedding], 1)
        #shape : (batch_size, full_feature_dim)

        out = self.dense_1_dropout(F.relu(self.dense_1(full_feature)))
        out = self.dense_2_dropout(F.relu(self.dense_2(out)))
        out = torch.sigmoid(self.output_layer(out))

        return out

    def init_hidden(self, batch_size):
        # the weights are of the form (num_layers*num_directions, batch, hidden_size)
        hidden_a = torch.randn(1*2, batch_size, self.nb_lstm_units).to(self.device)
        hidden_b = torch.randn(1*2, batch_size, self.nb_lstm_units).to(self.device)

        #hidden_a = Variable(hidden_a)
        #hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)
