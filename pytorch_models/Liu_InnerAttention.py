import torch
#from torchtext import data
#from torchtext import datasets
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.nn import functional as f

class InnerAttention(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout_prob, embedding_weights):
		super().__init__()
		self.par = hidden_dim
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		#self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
		self.bilstms = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = True, dropout=dropout_prob)
		#self.Ws1 = nn.Linear(self.da, 2*hidden_dim)
		self.Wy = nn.Linear(2*hidden_dim, 2*hidden_dim, bias=False)
		self.Wh = nn.Linear(2*hidden_dim, 2*hidden_dim, bias=False)
		self.w = nn.Linear(2*hidden_dim, 1, bias=False)
		self.logits = nn.Linear(hidden_dim*2, output_dim)
		#self.fc = nn.Linear(self.r*hidden_dim*2, self.par)
		#so we would have r*2d matrix and we would want to fully connect that whole matrix that's why
		#self.label = nn.Linear(self.par, output_dim)
		self.dropout = nn.Dropout(dropout_prob)
		self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weights).float(), requires_grad=False)
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
		#print(input_sequences.shape)
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
		hidden_pool = self.pool(output)
		#print('after pooling layer')
		#print(hidden_pool.shape)
		#bts, 2d
		att_weights = self.attention_layer(output, hidden_pool)
		#batch_size, max_sequence_length, 1
		#print('after attention layer')
		#print(att_weights.shape)
		att_weights = att_weights.permute(0,2,1)
		Rattn = torch.bmm(att_weights, output)
		#print(Rattn.shape)
		#(bts, 1, n)(bts, n, 2d) = (bts, 1, 2d)
		Rattn = Rattn.squeeze(1)
		#(bts,2d)

		logits = self.logits(Rattn)
		#print(logits.shape)
		return logits

	def pool(self, output):
		x = f.relu(torch.transpose(output, 1, 2))
		#batch_size, 2*hidden_dim, max_sequence_length
		#x = f.relu(output)
		#print(x.shape)
		x = f.avg_pool1d(x, x.size(2))
		#(batch_size, 2*hidden_dim, 1)
		#print(x.shape)
		x = x.squeeze(2)
		x = self.dropout(x)
		return x

	def attention_layer(self, output_bilstm, hidden_maxpool):
		#print('output bilstm')
		#print(output_bilstm.shape)
		a = self.Wy(output_bilstm)
		#print('a')
		#print(a.shape)
		#(2d, 2d)(bts, n, 2d) = (bts, n, 2d)
		x = self.Wh(hidden_maxpool)
		#print('x')
		#print(x.shape)
		#(2d,2d)(bts,2d) = (bts,2d)
		x = x.unsqueeze(2)
		#print(x.shape)
		#(bts, 2d, 1)
		x = x.repeat(1,1,output_bilstm.shape[1])
		#print(x.shape)
		#(bts, 2d, n)
		x = x.permute(0,2,1)
		#print(x.shape)
		M = torch.add(a,x)
		#print(M.shape)
		M = torch.tanh(M)
		att = self.w(M)
		#print('att')
		#(2d,1)(bts,n,2d) = (bts, n, 1)
		#print(att.shape)
		att = f.softmax(att, dim=1) #along n
		#print(att.shape)
		return att