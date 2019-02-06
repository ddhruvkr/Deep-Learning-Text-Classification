import torch
#from torchtext import data
#from torchtext import datasets
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.nn import functional as f

class RNN(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout_prob, embedding_weights, character_features_shape, max_pool=False):
		super().__init__()
		self.max_pool = max_pool

		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		#self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
		self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, dropout=dropout_prob)
		#self.fc = nn.Linear(hidden_dim*2, output_dim)
		#self.fc = nn.Linear(hidden_dim+character_features_shape[1], output_dim)
		self.fc = nn.Linear(hidden_dim, output_dim)
		#self.dropout = nn.Dropout(dropout)
		#self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
		self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weights).float(), requires_grad=True)
		self.dropout = nn.Dropout(dropout_prob)
        
	def embedded_dropout(self, embedding, words, dropout=0.1, scale=None):
		#print(embedding.weight)
		#print(embedding.state_dict())
		#print(embedding)
		mask = embedding.weight.data.new().resize_((embedding.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embedding.weight) / (1 - dropout)
		masked_embedding_weight = mask * embedding.weight
		if scale:
			masked_embedding_weight = scale.expand_as(masked_embedding_weight) * masked_embedding_weight

		padding_idx = embedding.padding_idx
		if padding_idx is None:
			padding_idx = -1

		X = f.embedding(words, masked_embedding_weight,
			padding_idx, embedding.max_norm, embedding.norm_type,
			embedding.scale_grad_by_freq, embedding.sparse)
		return X
	#we don't really need to initialize our hidden states for the lstm's since pythorch
	#by default initializes it with zeros
	#we may do it if we have to feed something else as a starting state
	#for example: output of one hidden state of some lstm to some other lstm
	#like could be a case for BIDAF paper
	def forward(self, input_sequences, char_features):
		#print('inside forward')
		#x = [sent len, batch size]
		#embedded = self.dropout(self.embedding(x))
		#print(character_features.shape)
		#shape of x:
		#batch_size, max_sequence_length
		#print(self.embedding.weight)
		embeddings = self.embedded_dropout(self.embedding, input_sequences)
		#embeddings = self.embedding(input_sequences)
		#print(embeddings.shape)
		#shape output of embedding layer:
		#batch_size, max_sequence_length, embedding_length
		embeddings = embeddings.permute(1, 0, 2)
		#print(embeddings.shape)
		#after permute shape becomes
		#max_sequence_length, batch_size, embedding_length
		#output, (hidden, cell) = self.rnn(embeddings)
		output, hidden = self.rnn(embeddings)
		#print(output.shape)
		#print(hidden.shape)
		#print(cell.shape)
		#output would have values for all of the hidden states throughout the sequence
		#output = [max_sequence_length, batch_size, hidden_dim * num_directions(bi-lstm)]

		#hidden would have only for the most recent hidden state
		#hidden = [1, batch_size, hidden_dim]
		#[num_layers * num_directions, batch_size, hidden_size]
		#cell = [1, batch_size, hidden_dim]
		#[num_layers * num_directions, batch_size, hidden_size]

		if self.max_pool:
			output = torch.transpose(output, 1, 0)
			x = f.relu(torch.transpose(output, 1, 2))
			#batch_size, hidden_dim, max_sequence_length
			#x = f.relu(output)
			#print(x.shape)
			x = f.max_pool1d(x, x.size(2))
			#(batch_size, hidden_dim, 1)
			#print(x.shape)
			x = x.squeeze(2)
			x = self.dropout(x)

		else:
			x = hidden[-1]
			#print(x.dtype)
			#print(character_features.dtype)
			#print(x.shape)
			#x = self.dropout(x)
			#print(hidden[-1].shape)
			#hidden[-1] : batch_size, hidden_dim
		#x = torch.cat((x, char_features), dim=1)
		linear = self.fc(x)
		#linear = self.fc(hidden)
		#linear output shape: batch_size, 2
		#print(linear.shape)
		return linear
		#return self.fc(hidden.squeeze(0))
