import torch
#from torchtext import data
#from torchtext import datasets
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.nn import functional as f

class BiRNN(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout_prob, embedding_weights, max_pool=False):
		super().__init__()
		self.max_pool = max_pool
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		#self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
		self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = True, dropout=dropout_prob)
		#self.fc = nn.Linear(hidden_dim*2, output_dim)
		self.fc = nn.Linear(hidden_dim*2, output_dim)
		#self.dropout = nn.Dropout(dropout)
		self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weights).float(), requires_grad=True)
		#self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
		#self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
		self.hidden_dim = hidden_dim
		self.dropout = nn.Dropout(dropout_prob)
	#we don't really need to initialize our hidden states for the lstm's since pythorch
	#by default initializes it with zeros
	#we may do it if we have to feed something else as a starting state
	#for example: output of one hidden state of some lstm to some other lstm
	#like could be a case for BIDAF paper
	def forward(self, inp, cf):
		#print('inside forward')
		#x = [sent len, batch size]
		#print(input_sequences.dtype)
		#print(input_sequences.shape)
		#input_sequences = self.dropout(input_sequences)
		#print(inp.shape)
		#z = torch.zeros([inp.shape[0],inp.shape[1]], dtype=torch.int64).cuda()
		#probs = torch.empty([inp.shape[0], inp.shape[1]]).uniform_(0,1).cuda()
		#inp = torch.where(probs > 0.05, inp, z).cuda()
		
		#print(inp.shape)
		#embeddings = self.dropout(self.embedding(input_sequences))
		#print(embedded.shape)
		#probs = torch.empty(input_sequences.size(0)).uniform_(0,1)
		#input_sequences = 
		#shape of x:
		#batch_size, max_sequence_length
		embeddings = self.embedding(inp)
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
		if self.max_pool:
			output = torch.transpose(output, 1, 0)
			x = f.relu(torch.transpose(output, 1, 2))
			#batch_size, 2*hidden_dim, max_sequence_length
			#x = f.relu(output)
			#print(x.shape)
			x = f.max_pool1d(x, x.size(2))
			#(batch_size, 2*hidden_dim, 1)
			#print(x.shape)
			x = x.squeeze(2)
			x = self.dropout(x)
		else:
			x = torch.cat((output[-1, :, :self.hidden_dim], output[0, :, self.hidden_dim:]), dim=1)
			#x = self.dropout(x)
			#print(x.shape)
			#a better option since output of output doesn't care about the num_layers,
			#therefore would work seemlessly with multiple layer bi-lstms as well
		
		linear = self.fc(x)
		#linear output shape: batch_size, 2
		#print(linear.shape)
		return linear
		#return self.fc(hidden.squeeze(0))
