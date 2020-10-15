# -*- coding: utf-8 -*-#
# Name:         model
# Description:  
# Author:       Fali Wang
# Date:         2020/8/26 21:20
import torch
from torch import nn
import torch.nn.functional as F
from utils import *


'''Convolutional Neural Networks for Sentence Classification'''


class CNNModel(nn.Module):
    def __init__(self, n_vocab, embed_size, num_classes, num_filters, filter_sizes, device, dropout=0.2):
        super(CNNModel, self).__init__()

        self.embedding = nn.Embedding(n_vocab, embed_size, padding_idx=0)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed_size)) for k in filter_sizes])

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(num_filters * len(filter_sizes),
                             num_filters * len(filter_sizes))
        self.fc2 = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids, attention_mask, token_type_ids, labels, lengths):
        x = input_ids
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return [F.cross_entropy(out, labels), out]


class LSTMModel(nn.Module):
    """LSTM text classification model.

    Here is the overall architecture:
      1) Rotate word vectors
      2) Feed to bi-LSTM
      3) Max/mean pool across all time
      4) Predict with MLP

    """
    def __init__(self, n_vocab, embed_size, num_classes, hidden_size, device, pool='mean', dropout=0.2, no_wordvec_layer=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.pool = pool
        self.no_wordvec_layer = no_wordvec_layer
        self.device = device
        self.embs = nn.Embedding(n_vocab, embed_size, padding_idx=0)
        if no_wordvec_layer:
            self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        else:
            self.linear_input = nn.Linear(embed_size, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(2*hidden_size, hidden_size)
        self.fc_output = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids, labels, lengths):
        x = input_ids  # (B, n, 1)
        mask = attention_mask  # (B, n)

        B = x.shape[0]
        x_vecs = self.embs(x).squeeze(-2)  # (B, n, d)
        if not self.no_wordvec_layer:
            x_vecs = self.linear_input(x_vecs)
            z = F.relu(x_vecs)  # B, n, h
        else:
            z = x_vecs  # B, n, d
        output, (h, c) = self.lstm(z)  # output (B, n, 2*h)
        output_mask = output * mask.unsqueeze(2)
        if self.pool == 'mean':
            fc_in = torch.sum(output_mask/lengths.to(dtype=torch.float).view(-1, 1, 1), dim=1)  # (B, 2*h)
        else:
            raise NotImplementedError()
        fc_in = self.dropout(fc_in)
        fc_hidden = F.relu(self.fc_hidden(fc_in))  # (B, h)
        fc_hidden = self.dropout(fc_hidden)
        output = self.fc_output(fc_hidden)  # (B, class_num)

        return [F.cross_entropy(output, labels), output]


'''Char-CNN small version, Character-level Convolutional Networks for Text Classification'''


class CharCNN(nn.Module):
    def __init__(self, num_features, num_classes, dropout=0.5):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_features, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8704, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.fc3 = nn.Linear(1024, num_classes)
        # self.log_softmax = nn.LogSoftmax()

    def forward(self, x, labels):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # collapse
        x = x.view(x.size(0), -1)
        # linear layer
        x = self.fc1(x)
        # linear layer
        x = self.fc2(x)
        # linear layer
        x = self.fc3(x)

        return [F.cross_entropy(x, labels), x]


'''SNLI'''


class BOWModel(nn.Module):
    def __init__(self, word_mat, n_vocab, embed_size, hidden_size, num_classes, drop_prob=0.1, no_wordvec_layer=False):
        super(BOWModel, self).__init__()
        # self.embedding = nn.Embedding(n_vocab, embed_size, padding_idx=0)
        self.embedding = nn.Embedding(n_vocab, embed_size, padding_idx=0, _weight=torch.tensor(word_mat).float())
        self.rotation = nn.Linear(embed_size, hidden_size)
        # self.dropout = nn.Dropout(drop_prop)

        # self.fc1 = nn.Linear(2*hidden_size, 2*hidden_size)
        # self.fc2 = nn.Linear(2*hidden_size, num_classes)

        self.layers = nn.Sequential(
            nn.Linear(2*hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(2*hidden_size, num_classes),
        )

    def _encode(self, sequence, mask):
        vecs = self.embedding(sequence)  # (B, seq_len, embed_size)
        vecs = self.rotation(vecs)  # (B, seq_len, hidden_size)
        z1 = F.relu(vecs)
        z1_masked = z1 * mask.unsqueeze(-1)
        z1_pooled = torch.sum(z1_masked, -2)  # (B, hidden_size)
        return z1_pooled

    def forward(self, x1, x1_mask, x2, x2_mask, labels):
        prem_encoded = self._encode(x1, x1_mask)
        hypo_encoded = self._encode(x2, x2_mask)
        input_encoded = torch.cat([prem_encoded, hypo_encoded], dim=-1)  # (B, 2*hidden_size)
        logits = self.layers(input_encoded)  # (B, num_classes)

        return [F.cross_entropy(logits, labels), logits]


class DecompAttentionModel(nn.Module):
    def __init__(self, word_mat, n_vocab, embed_size, hidden_size, num_classes, drop_prop=0.1, no_wordvec_layer=False):
        super(DecompAttentionModel, self).__init__()
        num_layers = 2
        hidden_size = embed_size
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(word_mat).float(), padding_idx=0)
        self.dropout = nn.Dropout(p=0.3)
        self.null = nn.Parameter(torch.normal(mean=torch.zeros(embed_size)))
        self.rotation = nn.Linear(embed_size, hidden_size)
        ff_layers = self._get_feedforward_layers(num_layers, embed_size, hidden_size, 1, drop_prop)
        self.feedforward = nn.Sequential(*ff_layers)
        compare_layers = self._get_feedforward_layers(num_layers, 2*hidden_size, hidden_size, hidden_size, drop_prop)
        self.compare_ff = nn.Sequential(*compare_layers)
        output_layers = self._get_feedforward_layers(num_layers, 2*hidden_size, hidden_size, hidden_size, drop_prop)
        output_layers.append(nn.Linear(hidden_size, num_classes))
        # output_layers.append(nn.LogSoftmax(dim=1))
        self.output_layer = nn.Sequential(*output_layers)

    def _get_feedforward_layers(self, num_layers, input_size, hidden_size, output_size, dropout_prob):
        layers = []
        for i in range(num_layers):
            layer_in_size = input_size if i == 0 else hidden_size
            layer_out_size = output_size if i == num_layers - 1 else hidden_size
            if dropout_prob:
                layers.append(nn.Dropout(dropout_prob))
            layers.append(nn.Linear(layer_in_size, layer_out_size))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        return layers

    def _encode(self, sequence, mask):
        # vecs = self.dropout(self.embedding(sequence))  # (B, seq_len, embed_size)
        vecs = self.embedding(sequence)  # (B, seq_len, embed_size)
        null = torch.zeros_like(vecs[0])
        null[0] = self.null
        vecs = vecs + null
        vecs = self.rotation(vecs)
        return F.relu(vecs) * mask.unsqueeze(-1)

    def attend_on(self, source, target, attention):
        """
        Args:
          - source: (bXsXe)
          - target: (bXtXe)
          - attention: (bXtXs)
        """
        attention_logsoftmax = F.log_softmax(attention, 1)
        attention_normalized = torch.exp(attention_logsoftmax)
        attended_target = torch.matmul(attention_normalized, source)  # (bXtXe)
        return torch.cat([target, attended_target], dim=-1)

    def forward(self, x1, x1_mask, x2, x2_mask, labels):
        prem_encoded = self._encode(x1, x1_mask)
        hypo_encoded = self._encode(x2, x2_mask)
        prem_weights = self.feedforward(prem_encoded) * x1_mask.unsqueeze(-1)  # (bXpX1)
        hypo_weights = self.feedforward(hypo_encoded) * x2_mask.unsqueeze(-1)  # (bXhX1)
        attention = torch.matmul(prem_weights, hypo_weights.permute(0, 2, 1))  # (bXpX1) X (bX1Xh) => (bXpXh)
        attention_mask = x1_mask.unsqueeze(-1) * x2_mask.unsqueeze(1)
        attention_masked = attention + (1 - attention_mask) * -1e20
        attended_prem = self.attend_on(hypo_encoded, prem_encoded, attention_masked)  # (bXpX2e)
        attended_hypo = self.attend_on(prem_encoded, hypo_encoded, attention_masked.permute(0, 2, 1))  # (bXhX2e)
        compared_prem = self.compare_ff(attended_prem) * x1_mask.unsqueeze(-1)  # (bXpXhid)
        compared_hypo = self.compare_ff(attended_hypo) * x2_mask.unsqueeze(-1)  # (bXhXhid)
        prem_aggregate = torch.sum(compared_prem, dim=1)  # (bXhid)
        hypo_aggregate = torch.sum(compared_hypo, dim=1)  # (bXhid)
        aggregate = torch.cat([prem_aggregate, hypo_aggregate], dim=-1)  # (bX2hid)
        logits = self.output_layer(aggregate)  # (b)

        return [F.cross_entropy(logits, labels), logits]


class BertBasedModel(nn.Module):
    def __init__(self, bert_model, num_classes, drop_prob=0.1):
        super(BertBasedModel, self).__init__()
        self.bert_model = bert_model
        hidden_size = 768
        self.layers = nn.Sequential(
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(2 * hidden_size, num_classes),
        )

    def forward(self, x1, x1_mask, x2, x2_mask, labels):
        prem_encoded = torch.sum(self.bert_model(input_ids=x1, attention_mask=x1_mask)[0], dim=-2)
        hypo_encoded = torch.sum(self.bert_model(input_ids=x2, attention_mask=x2_mask)[0], dim=-2)
        input_encoded = torch.cat([prem_encoded, hypo_encoded], dim=-1)  # (B, 2*hidden_size)
        logits = self.layers(input_encoded)  # (B, num_classes)

        return [F.cross_entropy(logits, labels), logits]


class ESIM(nn.Module):
    """
    Implementation of the ESIM model presented in the paper "Enhanced LSTM for
    Natural Language Inference" by Chen et al.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_size, embeddings=None, padding_idx=0, dropout=0.5, num_classes=3, device="cpu"):
        """
        Args:
            vocab_size: The size of the vocabulary of embeddings in the model.
            embedding_dim: The dimension of the word embeddings.
            hidden_size: The size of all the hidden layers in the network.
            embeddings: A tensor of size (vocab_size, embedding_dim) containing
                pretrained word embeddings. If None, word embeddings are
                initialised randomly. Defaults to None.
            padding_idx: The index of the padding token in the premises and
                hypotheses passed as input to the model. Defaults to 0.
            dropout: The dropout rate to use between the layers of the network.
                A dropout rate of 0 corresponds to using no dropout at all.
                Defaults to 0.5.
            num_classes: The number of classes in the output of the network.
                Defaults to 3.
            device: The name of the device on which the model is being
                executed. Defaults to 'cpu'.
        """
        super(ESIM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        self._word_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx,
                                            _weight=embeddings)

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            # self._rnn_dropout = nn.Dropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        self.hidden_size,
                                        bidirectional=True)

        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))

        # Initialize all weights and biases in the model.
        self.apply(_init_esim_weights)

    def forward(self, premises, premises_lengths, hypotheses, hypotheses_lengths, labels):
        """
        Args:
            premises: A batch of varaible length sequences of word indices
                representing premises. The batch is assumed to be of size
                (batch, premises_length).
            premises_lengths: A 1D tensor containing the lengths of the
                premises in 'premises'.
            hypothesis: A batch of varaible length sequences of word indices
                representing hypotheses. The batch is assumed to be of size
                (batch, hypotheses_length).
            hypotheses_lengths: A 1D tensor containing the lengths of the
                hypotheses in 'hypotheses'.
        Returns:
            logits: A tensor of size (batch, num_classes) containing the
                logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing
                the probabilities of each output class in the model.
        """
        premises_mask = get_mask(premises, premises_lengths).to(self.device)
        hypotheses_mask = get_mask(hypotheses, hypotheses_lengths)\
            .to(self.device)

        embedded_premises = self._word_embedding(premises)
        embedded_hypotheses = self._word_embedding(hypotheses)

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)

        encoded_premises = self._encoding(embedded_premises,
                                          premises_lengths)
        encoded_hypotheses = self._encoding(embedded_hypotheses,
                                            hypotheses_lengths)

        attended_premises, attended_hypotheses =\
            self._attention(encoded_premises, premises_mask,
                            encoded_hypotheses, hypotheses_mask)

        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses -
                                         attended_hypotheses,
                                         encoded_hypotheses *
                                         attended_hypotheses],
                                        dim=-1)

        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_hypotheses = self._rnn_dropout(projected_hypotheses)

        v_ai = self._composition(projected_premises, premises_lengths)
        v_bj = self._composition(projected_hypotheses, hypotheses_lengths)

        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1)
                                                  .transpose(2, 1), dim=1)\
            / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)
        # probabilities = nn.functional.softmax(logits, dim=-1)
        #
        # return logits, probabilities
        return [F.cross_entropy(logits, labels), logits]


def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0


# Class widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/modules/input_variational_dropout.py
class RNNDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs.
    Apply the same dropout mask to all the elements of the same sequence in
    a batch of sequences of size (batch, sequences_length, embedding_dim).
    """

    def forward(self, sequences_batch):
        """
        Apply dropout to the input batch of sequences.
        Args:
            sequences_batch: A batch of sequences of vectors that will serve
                as input to an RNN.
                Tensor of size (batch, sequences_length, emebdding_dim).
        Returns:
            A new tensor on which dropout has been applied.
        """
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0],
                                             sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training,
                                             inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch


class Seq2SeqEncoder(nn.Module):
    """
    RNN taking variable length padded sequences of vectors as input and
    encoding them into padded sequences of vectors of the same length.
    This module is useful to handle batches of padded sequences of vectors
    that have different lengths and that need to be passed through a RNN.
    The sequences are sorted in descending order of their lengths, packed,
    passed through the RNN, and the resulting sequences are then padded and
    permuted back to the original order of the input sequences.
    """

    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.0,
                 bidirectional=False):
        """
        Args:
            rnn_type: The type of RNN to use as encoder in the module.
                Must be a class inheriting from torch.nn.RNNBase
                (such as torch.nn.LSTM for example).
            input_size: The number of expected features in the input of the
                module.
            hidden_size: The number of features in the hidden state of the RNN
                used as encoder by the module.
            num_layers: The number of recurrent layers in the encoder of the
                module. Defaults to 1.
            bias: If False, the encoder does not use bias weights b_ih and
                b_hh. Defaults to True.
            dropout: If non-zero, introduces a dropout layer on the outputs
                of each layer of the encoder except the last one, with dropout
                probability equal to 'dropout'. Defaults to 0.0.
            bidirectional: If True, the encoder of the module is bidirectional.
                Defaults to False.
        """
        assert issubclass(rnn_type, nn.RNNBase),\
            "rnn_type must be a class inheriting from torch.nn.RNNBase"

        super(Seq2SeqEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self._encoder = rnn_type(input_size,
                                 hidden_size,
                                 num_layers=num_layers,
                                 bias=bias,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        """
        Args:
            sequences_batch: A batch of variable length sequences of vectors.
                The batch is assumed to be of size
                (batch, sequence, vector_dim).
            sequences_lengths: A 1D tensor containing the sizes of the
                sequences in the input batch.
        Returns:
            reordered_outputs: The outputs (hidden states) of the encoder for
                the sequences in the input batch, in the same order.
        """
        sorted_batch, sorted_lengths, _, restoration_idx =\
            sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch,
                                                         sorted_lengths,
                                                         batch_first=True)

        outputs, _ = self._encoder(packed_batch, None)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True)
        reordered_outputs = outputs.index_select(0, restoration_idx)

        return reordered_outputs


class SoftmaxAttention(nn.Module):
    """
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the soft attention between their elements.
    The dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the premises for each element of the hypotheses, and
    conversely for the elements of the premises.
    """

    def forward(self,
                premise_batch,
                premise_mask,
                hypothesis_batch,
                hypothesis_mask):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            premise_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            hypothesis_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            hypothesis_mask: A mask for the sequences in the hypotheses batch,
                to ignore padding data in the sequences during the computation
                of the attention.
        Returns:
            attended_premises: The sequences of attention vectors for the
                premises in the input batch.
            attended_hypotheses: The sequences of attention vectors for the
                hypotheses in the input batch.
        """
        # Dot product between premises and hypotheses in each sequence of
        # the batch.
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1)
                                                              .contiguous())

        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2)
                                                        .contiguous(),
                                       premise_mask)

        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(hypothesis_batch,
                                         prem_hyp_attn,
                                         premise_mask)
        attended_hypotheses = weighted_sum(premise_batch,
                                           hyp_prem_attn,
                                           hypothesis_mask)

        return attended_premises, attended_hypotheses


class InferSent(nn.Module):

    def __init__(self, embed_size, hidden_size, num_classes, dropout_prob, device):
        super(InferSent, self).__init__()
        self.word_emb_dim = embed_size
        self.enc_lstm_dim = hidden_size
        self.pool_type = 'mean'
        self.device = device

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1,
                                bidirectional=True, dropout=dropout_prob)
        self.layers = nn.Sequential(
            nn.Linear(4 * hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(4 * hidden_size, num_classes),
        )

    def _encode(self, sent, sent_len):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: ( seq_len x batch_size x embed_size)

        # Sort by length (keep idx)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).to(self.device)
        sent = sent.index_select(1, idx_sort)

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).to(self.device)
        sent_output = sent_output.index_select(1, idx_unsort)

        # Pooling
        if self.pool_type == "mean":
            sent_len = torch.FloatTensor(sent_len.copy()).unsqueeze(1).to(self.device)
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            if not self.max_pad:
                sent_output[sent_output == 0] = -1e9
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2
        else:
            emb = None

        return emb  # batch x 2*nhid

    def forward(self, x1, x1_length, x2, x2_length, labels):
        prem_encoded = self._encode(x1, x1_length)
        hypo_encoded = self._encode(x2, x2_length)
        input_encoded = torch.cat([prem_encoded, hypo_encoded], dim=-1)  # (B, 2*hidden_size)
        logits = self.layers(input_encoded)  # (B, num_classes)

        return [F.cross_entropy(logits, labels), logits]

