import torch
#from torchtext import data
#from torchtext import datasets
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.nn import functional as f

class HierarchicalWordAttention(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout_prob, embedding_weights):
		super().__init__()
		self.it =175
		#generally i think self.it = 2*hidden_dim
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		#self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
		self.bilstms = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = True, dropout=dropout_prob)
		#self.Ws1 = nn.Linear(self.da, 2*hidden_dim)
		self.Ww = nn.Linear(2*hidden_dim, self.it)
		self.Uw = nn.Linear(self.it, 1, bias=False)
		self.fc = nn.Linear(hidden_dim*2, hidden_dim)
		self.label = nn.Linear(hidden_dim, output_dim)
		self.dropout = nn.Dropout(dropout_prob)
		self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
		#self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
		self.hidden_dim = hidden_dim

	def forward(self, input_sequences, cf):
		#print('inside forward')
		#x = [sent len, batch size]
		#embedded = self.dropout(self.embedding(x))

		#shape of x:
		#batch_size, max_sequence_length
		embeddings = self.dropout(self.embedding(input_sequences))
		#print(embeddings.shape)
		#shape output of embedding layer:
		#batch_size, max_sequence_length, embedding_length
		embeddings = embeddings.permute(1, 0, 2)
		#after permute shape becomes
		#max_sequence_length, batch_size, embedding_length
		output, (hidden, cell) = self.bilstms(embeddings)
		#output would have values for all of the hidden states throughout the sequence
		#output = [max_sequence_length, batch_size, hidden_dim * num_directions(bi-lstm)]

		#hidden would have only for the most recent hidden state
		#hidden = [2, batch_size, hidden_dim]
		#[num_layers * num_directions, batch_size, hidden_size]
		#cell = [2, batch_size, hidden_dim]
		#[num_layers * num_directions, batch_size, hidden_size]

		output = output.permute(1,0,2)
		#now the dimension becomes batch_size, max_sequence_length, hidden_dim*2
		#print(output.shape)

		s = self.attention_layer(output)
		#batch_size, 2d
		#print(att_weights.shape)
		
		fc = self.fc(s)
		#print(fc.shape)
		fc=self.dropout(fc)
		linear = self.label(fc)
		#linear output shape: batch_size, 2
		#print(linear.shape)
		return linear
		#return self.fc(hidden.squeeze(0))

	'''def attention_layer(self, output_bilstm):
		u = self.Ww(output_bilstm)
		#(2d, it)(bts, n, 2d) = (bts, n, it)
		#print(a.shape)
		u_tanh = torch.tanh(u)
		att = self.Uw(u_tanh)
		#(it,1)(bts,n,it) = (bts, n, 1)
		#print(att.shape)
		att = att.permute(0,2,1)
		#(bts, 1, n)
		att = f.softmax(att, dim=2) #along n 
		s = torch.bmm(att, output_bilstm)
		#(bts, 1, n)(bts, n, 2d) = (bts, 1, 2d)
		#batch_size, 1, 2*hidden_dim
		#print(M.shape)
		s = s.squeeze(1)
		#(bts,2d)
		return s'''

	def attention_layer(self, output_bilstm):
		u = self.Ww(output_bilstm)
		#(2d, it)(bts, n, 2d) = (bts, n, it)
		#print(a.shape)
		#u_tanh = torch.tanh(u)
		u_tanh = f.relu(u)
		att = self.Uw(u_tanh)
		#(it,1)(bts,n,it) = (bts, n, 1)
		#print(att.shape)
		att = att.permute(0,2,1)
		#(bts, 1, n)
		att = f.softmax(att, dim=2) #along n
		att = att.squeeze(1)
		#print(att.shape)
		#bts,n
		output_bilstm = output_bilstm.permute(2,0,1)
		#2*hidden_dim, bts,n
		#print(output_bilstm.shape)
		si = torch.mul(att, output_bilstm)
		#print(si.shape)
		#2d, bts, n
		si = si.transpose(0,1)
		#print(si.shape)
		#bts,2d,n
		si = torch.sum(si, dim=2)
		#print(si.shape)
		#bts,2d
		return si

class HierarchicalAttention(nn.Module):
	def __init__(self, vocab_size, embedding_dim, word_hidden_dim, sentence_hidden_dim, output_dim, n_layers, dropout_prob, embedding_weights):
		super().__init__()
		self.it = 50
		#generally i think self.it = 2*hidden_dim
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		#self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
		self.bilstm_word = nn.LSTM(embedding_dim, word_hidden_dim, num_layers = n_layers, bidirectional = True, dropout=dropout_prob)
		self.bilstm_sentence = nn.LSTM(2*word_hidden_dim, sentence_hidden_dim, num_layers = n_layers, bidirectional = True, dropout=dropout_prob)
		#since embedding_length for the case of sentence is 2*word_hidden_dim(d)
		#self.Ws1 = nn.Linear(self.da, 2*hidden_dim)
		self.Ww = nn.Linear(2*word_hidden_dim, self.it)
		self.Uw = nn.Linear(self.it, 1, bias=False)
		self.Ws = nn.Linear(2*sentence_hidden_dim, self.it)
		self.Us = nn.Linear(self.it, 1, bias=False)
		self.fc = nn.Linear(sentence_hidden_dim*2, 200) #200 is randomnly selected for now, maybe we don't even need this layer
		self.label = nn.Linear(200, output_dim)
		#self.dropout = nn.Dropout(dropout)
		self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
		#self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
		self.hidden_dim = hidden_dim

	def forward(self, input_sequences):

		#shape of input_sequences would be (batch_size, max_sentences, max_sequence_length(words))

		#first need to convert the shape to have max_sentences as the 0th dimension
		#since we need to iterate over it
		input_sequences = input_sequences.permute(1,0,2)
		#(max_sentences, batch_size, max_sequence_length)
		s_concat = None
		for i in range(input_sequences.shape[0]):
			x = input_sequences[i]
			#shape of x:
			#batch_size, max_sequence_length
			embeddings = self.embedding(x)
			#print(embeddings.shape)
			#shape output of embedding layer:
			#batch_size, max_sequence_length, embedding_length
			embeddings = embeddings.permute(1, 0, 2)
			#after permute shape becomes
			#max_sequence_length, batch_size, embedding_length
			output, (hidden, cell) = self.bilstm_word(embeddings)
			#output would have values for all of the hidden states throughout the sequence
			#output = [max_sequence_length, batch_size, hidden_dim * num_directions(bi-lstm)]

			#hidden would have only for the most recent hidden state
			#hidden = [2, batch_size, hidden_dim]
			#[num_layers * num_directions, batch_size, hidden_size]
			#cell = [2, batch_size, hidden_dim]
			#[num_layers * num_directions, batch_size, hidden_size]

			output = output.permute(1,0,2)
			#now the dimension becomes batch_size, max_sequence_length, hidden_dim*2
			#print(output.shape)
			s = self.word_attention_layer(output)
			#batch_size, 2w_d
			#print(att_weights.shape)
			s = s.unsqueeze(0)
			#(1, bts, 2w_d(word_hidden_dim))
			#keep concatinating also
			if s_concat is None:
				s_concat = s
			else:
				s_concat = torch.cat((s_concat, s), 0)
		#s_concat = (max_sentences, bts, 2w_d)
		s_output, (s_hidden, s_cell) = self.bilstm_sentence(s_concat)
		s_output = s_output.permute(1,0,2)
		#max_sentences, batch_size, 2*sentence_hidden_dim
		v = self.sentence_attention_layer(s_output)
		#v = (batch_size, 2*s_d(sentence_hidden_dim))
		fc = self.fc(s)
		#print(fc.shape)
		linear = self.label(fc)
		#linear output shape: batch_size, 2
		#print(linear.shape)
		return linear

	def word_attention_layer(self, output_bilstm):
		u = self.Ww(output_bilstm)
		#n means max_word_sequences_length
		#(2w_d, it)(bts, n, 2w_d) = (bts, n, it)
		#print(a.shape)
		u_tanh = torch.tanh(u)
		att = self.Uw(u_tanh)
		#(it,1)(bts,n,it) = (bts, n, 1)
		#print(att.shape)
		att = att.permute(0,2,1)
		#(bts, 1, n)
		att = f.softmax(att, dim=2) #along n
		att = att.squeeze(1)
		#print(att.shape)
		#bts,n
		output_bilstm = output_bilstm.permute(2,0,1)
		#2*hidden_dim, bts,n
		#print(output_bilstm.shape)
		si = torch.mul(att, output_bilstm)
		#print(si.shape)
		#2d, bts, n
		si = si.transpose(0,1)
		#print(si.shape)
		#bts,2d,n
		si = torch.sum(si, dim=2)
		#print(si.shape)
		#bts,2d
		return si

	def sentence_attention_layer(self, output_bilstm):
		u = self.Ws(output_bilstm)
		#n here means max_sentences
		#(2s_d, it)(bts, n, 2s_d) = (bts, n, it)
		#print(a.shape)
		u_tanh = torch.tanh(u)
		att = self.Us(u_tanh)
		#(it,1)(bts,n,it) = (bts, n, 1)
		#print(att.shape)
		att = att.permute(0,2,1)
		#(bts, 1, n)
		att = f.softmax(att, dim=2) #along n
		att = att.squeeze(1)
		#print(att.shape)
		#bts,n
		output_bilstm = output_bilstm.permute(2,0,1)
		#2*sentence_hidden_dim(s_d), bts,n
		#print(output_bilstm.shape)
		vi = torch.mul(att, output_bilstm)
		#print(si.shape)
		#2s_d, bts, n
		vi = vi.transpose(0,1)
		#print(si.shape)
		#bts,2s_d,n
		vi = torch.sum(vi, dim=2)
		#print(si.shape)
		#bts,2s_d
		return vi
