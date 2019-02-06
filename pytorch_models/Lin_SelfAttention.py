import torch
#from torchtext import data
#from torchtext import datasets
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.nn import functional as f

class SelfAttention(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout_prob, embedding_weights):
		super().__init__()
		self.da = 30
		self.r = 5
		self.par = hidden_dim
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		#self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
		self.bilstms = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = True, dropout=dropout_prob)
		#self.Ws1 = nn.Linear(self.da, 2*hidden_dim)
		self.Ws1 = nn.Linear(2*hidden_dim, self.da, bias=False)
		self.Ws2 = nn.Linear(self.da, self.r, bias=False)
		self.fc = nn.Linear(self.r*hidden_dim*2, self.par)
		#so we would have r*2d matrix and we would want to fully connect that whole matrix that's why
		self.label = nn.Linear(self.par, output_dim)
		self.dropout = nn.Dropout(dropout_prob)
		self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weights).float(), requires_grad=True)
		#self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
		#self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
		self.hidden_dim = hidden_dim
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
		output, hidden = self.bilstms(embeddings)
		#print(output.shape)
		#print(hidden.shape)
		#print(cell.shape)
		#output would have values for all of the hidden states throughout the sequence
		#output = [max_sequence_length, batch_size, hidden_dim * num_directions(bi-lstm)]

		#hidden would have only for the most recent hidden state
		#hidden = [2, batch_size, hidden_dim]
		#[num_layers * num_directions, batch_size, hidden_size]
		#cell = [2, batch_size, hidden_dim]
		#[num_layers * num_directions, batch_size, hidden_size]
		
		#hidden_concat = torch.cat((hidden[0], hidden[1]),dim=1)
		#this would work correctly here but would not be very clean when it has multiple layers,
		#since the 1 dim then would be num_layers * num_directions
		#one thing to notice is that the output of forward rnn is of time-step = max_sequence_length
		#while for reverse rnn is of time-step = 0
		#which makes sense
		#although for the case of output, it is gives as it is
		#that is the reason output[-1] doesn't work since it would give last(correct) layer
		#for the case of forward rnn but also last timestep output(which is the first sequence, 
		#therefore incorrect) for the case of reverse rnn.
		
		#output_needed = output[-1]
		#incorrect

		#layer_concat = torch.cat((output[-1, :, :self.hidden_dim], output[0, :, self.hidden_dim:]), dim=1)
		#a better option since output of output doesn't care about the num_layers,
		#therefore would work seemlessly with multiple layer bi-lstms as well
		output = output.permute(1,0,2)
		#now the dimension becomes batch_size, max_sequence_length, hidden_dim*2
		#print(output.shape)

		att_weights = self.attention_layer(output)
		#batch_size, r, max_sequence_length
		#print(att_weights.shape)
		M = torch.bmm(att_weights, output)
		#(bts, r, n)(bts, n, 2d) = (bts, r, 2d)
		#batch_size, r, 2*hidden_dim
		#print(M.shape)
		#So this is r*2d matrix is a sentence embedding
		#usually sentence embedding would be a 1d vector, but this gives a richer 2d matrix
		#this is the first USP of this paper
		#the second is the penalization/regularization method, frobenious norm

		#convert M to a better view, basically concatinating
		M_b = M.view(-1, M.size()[1]*M.size()[2])
		#(bts, r*2d)
		#print(M_b.shape)
		fc = self.fc(M_b)
		fc = f.relu(fc)
		fc = self.dropout(fc)
		#print(fc.shape)
		linear = self.label(fc)
		#linear output shape: batch_size, 2
		#print(linear.shape)
		


		'''batch_I = torch.eye(self.r)
		batch_size = att_weights.shape[0]
		batch_I = batch_I.reshape((1, self.r, self.r))
		batch_I = batch_I.repeat(batch_size,1,1)
		#batch_size, r, r
		loss_matrix = torch.bmm(att_weights, att_weights.transpose(1,2)) - batch_I'''
		#return linear, loss_matrix
		return linear
		#return self.fc(hidden.squeeze(0))

	def attention_layer(self, output_bilstm):
		a = self.Ws1(output_bilstm)
		#(2d, da)(bts, n, 2d) = (bts, n, da)
		#print(a.shape)
		#a_tanh = torch.tanh(a)
		a_tanh = f.relu(a)
		att = self.Ws2(a_tanh)
		#(da,r)(bts,n,da) = (bts, n, r)
		#print(att.shape)
		att = att.permute(0,2,1)
		#(bts, r, n)
		att = f.softmax(att, dim=2) #along n 
		return att
