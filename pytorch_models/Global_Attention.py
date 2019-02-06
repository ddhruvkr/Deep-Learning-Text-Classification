import torch
#from torchtext import data
#from torchtext import datasets
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.nn import functional as f

class GlobalAttention(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout_prob, embedding_weights, concat=True):
		super().__init__()
		self.concat = concat
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		#self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
		self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, dropout=dropout_prob)
		#self.fc = nn.Linear(hidden_dim*2, output_dim)
		self.new_ht = nn.Linear(2*hidden_dim,128, bias=False)
		#new_ht = concat of original and attended
		self.label_concat = nn.Linear(128, output_dim)
		#the above is for the case when we would want to send the concat of original and attended sentence
		self.label = nn.Linear(hidden_dim, output_dim)
		#this is for when we only want to send attended sentence, seems to give worse results, but check
		#self.dropout = nn.Dropout(dropout)
		self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
		#self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
	#we don't really need to initialize our hidden states for the lstm's since pythorch
	#by default initializes it with zeros
	#we may do it if we have to feed something else as a starting state
	#for example: output of one hidden state of some lstm to some other lstm
	#like could be a case for BIDAF paper

	def forward(self, input_sequences):
		#print('inside forward')
		#x = [sent len, batch size]
		#embedded = self.dropout(self.embedding(x))

		#shape of x:
		#batch_size, max_sequence_length
		embeddings = self.embedding(input_sequences)
		#print(embeddings.shape)
		#shape output of embedding layer:
		#batch_size, max_sequence_length, embedding_length
		embeddings = embeddings.permute(1, 0, 2)
		#print(embeddings.shape)
		#after permute shape becomes
		#max_sequence_length, batch_size, embedding_length
		#output, (hidden, cell) = self.rnn(embeddings)
		output, hidden = self.rnn(embeddings)
		#output would have values for all of the hidden states throughout the sequence
		#output = [max_sequence_length, batch_size, hidden_dim]
		#print(hidden.shape)
		#hidden = [2, batch_size, hidden_dim]
		#[num_layers * num_directions, batch_size, hidden_size]
		#cell = [1, batch_size, hidden_dim]
		#[num_layers, batch_size, hidden_size]
		hidden = hidden[-1]
		output = output.permute(1,0,2)
		#now the dimension becomes batch_size, max_sequence_length, hidden_dim*2
		#print(output.shape)

		attended_context = self.attention_layer(output, hidden)
		#batch_size, hidden_dim
		
		if self.concat:
			#concatenation of a attended hidden state and original hidden state
			ht_modified = self.new_ht(torch.cat((attended_context, hidden), dim=1))
			#first is concat, output is (bs, 2d)
			#then linear operation #(bs,100) = (2d,100)(bs,2d) [Ws,ht_modified]
			#could be written as ht_mod.Ws (bs,2d)(2d,100)

			#tanh = nn.Tanh()
			relu = nn.ReLU()
			ht_modified = relu(ht_modified)
			#ht_modified = tanh(ht_modified)
			#experiment with removing tanh, may give better results on test and validation data
			#print(fc.shape)

			linear = self.label_concat(ht_modified)
		else:
			#just the attended hidden state
			linear = self.label(attended_context)
		#linear output shape: batch_size, 2
		#print(linear.shape)
		return linear

	def attention_layer(self, output, final_hidden_state):
		#final_hidden_state = final_hidden_state[-1]
		#print(final_hidden_state.shape)
		#(batch_size, hidden_dim)
		att_weights = torch.bmm(output, final_hidden_state.unsqueeze(2)).squeeze(2)
		#(dot operation from the original paper), could also use the other 2 operations
		#(bs,n,1) = (bs,n,d)(bs,d,1).  n->max_sequence_length, d->hidden_dim
		#then we apply squeeze(2) which converts to (bs,n)
		softmax_att = f.softmax(att_weights, dim=1)
		attended_context = torch.bmm(output.permute(0,2,1), softmax_att.unsqueeze(2)).squeeze(2)
		#(bs,d,1) = (bs,d,n)(bs,n,1)
		#again then we squeeze it
		
		return attended_context