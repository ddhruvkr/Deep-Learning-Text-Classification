import torch
#from torchtext import data
#from torchtext import datasets
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.nn import functional as f

class IntraAttention(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout_prob, embedding_weights):
		super().__init__()
		self.par = 10
		self.da = 150
		self.r = 1
		self.bidirectional = False
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.Ws1 = nn.Linear(2*hidden_dim, self.da, bias=False)
		self.Ws2 = nn.Linear(self.da, self.r, bias=False)
		#self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
		#self.lstm = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, dropout=dropout_prob)
		self.bilstms = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = self.bidirectional, dropout=dropout_prob)
		self.Wq = nn.Linear(2*embedding_dim, self.par)
		self.Wp = nn.Linear(self.par, 1)
		self.w = nn.Linear(2*embedding_dim, 1)
		self.fc1 = nn.Linear(hidden_dim*2+embedding_dim, hidden_dim)
		self.fc2 = nn.Linear(self.r*hidden_dim*2+embedding_dim, hidden_dim)
		self.logit = nn.Linear(hidden_dim*2+embedding_dim, output_dim)
		self.logits = nn.Linear(hidden_dim, output_dim)
		self.dropout = nn.Dropout(dropout_prob)
		self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weights).float(), requires_grad=True)
		#self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
		#self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
		self.hidden_dim = hidden_dim
		self.self_attention=False
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
		#print(embeddings.shape)
		s = self.simplified_intra_attention_layer(embeddings)
		embeddings = embeddings.permute(1, 0, 2)
		#print(embeddings.shape)
		#after permute shape becomes
		#max_sequence_length, batch_size, embedding_length
		output, hidden = self.bilstms(embeddings)
		#output would have values for all of the hidden states throughout the sequence
		#output = [max_sequence_length, batch_size, hidden_dim * num_directions(bi-lstm)]
		if self.self_attention:
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
			r = torch.cat((s,x), dim=1)
			#(bts, emb_dim+r*2d)
			fc = self.fc2(r)

		else:
			if self.bidirectional:
				x = torch.cat((output[-1, :, :self.hidden_dim], output[0, :, self.hidden_dim:]), dim=1)
			else:
				x = hidden[-1]
			r = torch.cat((s,x), dim=1)
			#(bts, emb_dim+r*2d)
			fc = self.fc1(r)

		fc = f.relu(fc)
		fc = self.dropout(fc)
		logits = self.logits(fc)
		#print(logits.shape)
		return logits

	def attention_layer(self, output):
		output = output.permute(1,0,2)
		a = self.Ws1(output)
		#(2d, da)(bts, n, 2d) = (bts, n, da)
		#print(a.shape)
		a_tanh = torch.tanh(a)
		att = self.Ws2(a_tanh)
		#(da,r)(bts,n,da) = (bts, n, r)
		#print(att.shape)
		att = att.permute(0,2,1)
		#(bts, r, n)
		att = f.softmax(att, dim=2) #along n 
		return att

	def simplified_intra_attention_layer(self, embeddings):
		max_sequence_length = embeddings.shape[1]
		#print(no_of_words)
		'''for index_batch,batch in enumerate(embeddings):
			for i in range(no_of_words):
				for j in range(no_of_words):
					#print(j)
					if i!=j:
						emb_concat = torch.cat((batch[i], batch[j]))
						self.s[index_batch][i][j] = self.w(emb_concat)'''
		mask = torch.ones(max_sequence_length,max_sequence_length)
		mask = mask - torch.diag(torch.diag(mask))
		s = torch.bmm(embeddings, embeddings.permute(0,2,1))
		#(bts, max_sequence_length, max_sequence_length)
		#print(s.shape)
		s = s*mask
		#doing masking to make values where word pairs are same(i == j), zero
		#print(s.shape)
		s = f.max_pool1d(s, s.size()[2]).squeeze(2)
		#print(s.shape)
		#(bts, max_sequence_length)
		s = f.softmax(s, dim=1)
		s = s.unsqueeze(dim=1)
		#print(s.shape)
		#(bts, 1, max_sequence_length)
		s = torch.bmm(s, embeddings)
		#print(s.shape)
		#(bts, 1, max_sequence_length)(bts, max_sequence_length, embedding_length) = (bts, 1, embedding_length)
		s = s.squeeze(1)
		#print(s.shape)
		return s


	def singular_intra_attention_layer(self, embeddings):
		max_sequence_length = embeddings.shape[1]
		batch_size = embeddings.shape[0]
		embedding_dim = embeddings.shape[2]
		#print(no_of_words)
		'''for index_batch,batch in enumerate(embeddings):
			for i in range(no_of_words):
				for j in range(no_of_words):
					#print(j)
					if i!=j:
						emb_concat = torch.cat((batch[i], batch[j]))
						self.s[index_batch][i][j] = self.w(emb_concat)'''

		#(bts, msl, dim)
		b = embeddings
		d = embeddings
		b = b.repeat(1,1,max_sequence_length)
		#(bts, max_sequence_length, max_sequence_length*embedding_dim)
		#print(b.shape)
		b = b.view(batch_size, max_sequence_length, max_sequence_length, embedding_dim)
		#print(b.shape)
		d = d.unsqueeze(1)
		#(bts, 1, max_sequence_length, embedding_dim)
		#print(d.shape)
		d = d.repeat(1,max_sequence_length,1,1)
		#(batch_size, max_sequence_length, max_sequence_length, embedding_dim)
		#print(d.shape)
		concat = torch.cat((b,d), dim=3)
		#batch_size, max_sequence_length, max_sequence_length, 2*embedding_dim
		#print(concat.shape)

		'''Lets assume that there are 2 words and each word embedding has 3 dimension
		So embedding matrix would look like this
		[[----1----]
		[----2----]].  (2*3)

		b =
		----1----,----1----
		----2----,----2----


		d=
		----1----,----2----
		----1----,----2----

		Now if you concatenate both, we get all the combinations of word pairs'''


		s = self.w(concat)
		#batch_size, max_sequence_length, max_sequence_length, 1
		#print(s[0])
		s = s.squeeze(3)
		#batch_size, max_sequence_length, max_sequence_length
		#print(s[0])
		#print(s.shape)
		mask = torch.ones(max_sequence_length,max_sequence_length)
		mask = mask - torch.diag(torch.diag(mask))
		#s = torch.bmm(embeddings, embeddings.permute(0,2,1))
		#(bts, max_sequence_length, max_sequence_length)
		#print(s.shape)
		s = s*mask
		#print(s[0])
		#doing masking to make values where word pairs are same(i == j), zero
		#print(s.shape)
		s = f.max_pool1d(s, s.size()[2]).squeeze(2)
		#print(s.shape)
		#(bts, max_sequence_length)
		s = f.softmax(s, dim=1)
		s = s.unsqueeze(dim=1)
		#print(s.shape)
		#(bts, 1, max_sequence_length)
		s = torch.bmm(s, embeddings)
		#print(s.shape)
		#(bts, 1, max_sequence_length)(bts, max_sequence_length, embedding_length) = (bts, 1, embedding_length)
		s = s.squeeze(1)
		#print(s.shape)
		return s


	def multi_dimensional_intra_attention_layer(self, embeddings):
		max_sequence_length = embeddings.shape[1]
		batch_size = embeddings.shape[0]
		embedding_dim = embeddings.shape[2]
		#print(no_of_words)
		'''for index_batch,batch in enumerate(embeddings):
			for i in range(no_of_words):
				for j in range(no_of_words):
					#print(j)
					if i!=j:
						emb_concat = torch.cat((batch[i], batch[j]))
						self.s[index_batch][i][j] = self.w(emb_concat)'''

		#(bts, msl, dim)
		b = embeddings
		d = embeddings
		b = b.repeat(1,1,max_sequence_length)
		#(bts, max_sequence_length, max_sequence_length*embedding_dim)
		#print(b.shape)
		b = b.view(batch_size, max_sequence_length, max_sequence_length, embedding_dim)
		#print(b.shape)
		d = d.unsqueeze(1)
		#(bts, 1, max_sequence_length, embedding_dim)
		#print(d.shape)
		d = d.repeat(1,max_sequence_length,1,1)
		#(batch_size, max_sequence_length, max_sequence_length, embedding_dim)
		#print(d.shape)
		concat = torch.cat((b,d), dim=3)
		#batch_size, max_sequence_length, max_sequence_length, 2*embedding_dim
		#print(concat.shape)

		'''Lets assume that there are 2 words and each word embedding has 3 dimension
		So embedding matrix would look like this
		[[----1----]
		[----2----]].  (2*3)

		b =
		----1----,----1----
		----2----,----2----


		d=
		----1----,----2----
		----1----,----2----

		Now if you concatenate both, we get all the combinations of word pairs

		----1--------1----,----1--------2----
		----2--------1----,----2--------2----

		'''


		s = self.Wq(concat)
		#batch_size, max_sequence_length, max_sequence_length, par
		#print(s.shape)
		s = f.relu(s)
		#print(s.shape)
		s = self.Wp(s)
		#batch_size, max_sequence_length, max_sequence_length, 1
		#print(s.shape)
		#print(s[0])
		s = s.squeeze(3)
		#print(s.shape)
		#batch_size, max_sequence_length, max_sequence_length
		#print(s[0])
		#print(s.shape)
		mask = torch.ones(max_sequence_length,max_sequence_length)
		mask = mask - torch.diag(torch.diag(mask))
		#s = torch.bmm(embeddings, embeddings.permute(0,2,1))
		#(bts, max_sequence_length, max_sequence_length)
		#print(s.shape)
		s = s*mask
		#print(s[0])
		#doing masking to make values where word pairs are same(i == j), zero
		#print(s.shape)
		s = f.max_pool1d(s, s.size()[2]).squeeze(2)
		#print(s.shape)
		#(bts, max_sequence_length)
		s = f.softmax(s, dim=1)
		s = s.unsqueeze(dim=1)
		#print(s.shape)
		#(bts, 1, max_sequence_length)
		s = torch.bmm(s, embeddings)
		#print(s.shape)
		#(bts, 1, max_sequence_length)(bts, max_sequence_length, embedding_length) = (bts, 1, embedding_length)
		s = s.squeeze(1)
		#print(s.shape)
		return s




'''

a
tensor([[0.1000, 0.2000, 0.3000],
        [0.4000, 0.5000, 0.6000]], dtype=torch.float64)

b = a
>>> b.shape
torch.Size([2, 3])
>>> b = b.repeat(1,2)
>>> b
tensor([[0.1000, 0.2000, 0.3000, 0.1000, 0.2000, 0.3000],
        [0.4000, 0.5000, 0.6000, 0.4000, 0.5000, 0.6000]], dtype=torch.float64)
>>> b = b.view(2,2,3)
>>> b
tensor([[[0.1000, 0.2000, 0.3000],
         [0.1000, 0.2000, 0.3000]],

        [[0.4000, 0.5000, 0.6000],
         [0.4000, 0.5000, 0.6000]]], dtype=torch.float64)



d.shape
torch.Size([2, 3])
>>> d = d.unsqueeze(0)
tensor([[[0.1000, 0.2000, 0.3000],
         [0.4000, 0.5000, 0.6000]]], dtype=torch.float64)
>>> d.shape
torch.Size([1, 2, 3])
>>> d=d.repeat(2,1,1)
>>> d
tensor([[[0.1000, 0.2000, 0.3000],
         [0.4000, 0.5000, 0.6000]],

        [[0.1000, 0.2000, 0.3000],
         [0.4000, 0.5000, 0.6000]]], dtype=torch.float64)
>>> d.shape
torch.Size([2, 2, 3])


>>> torch.cat((b,d),dim=2)
tensor([[[0.1000, 0.2000, 0.3000, 0.1000, 0.2000, 0.3000],
         [0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000]],

        [[0.4000, 0.5000, 0.6000, 0.1000, 0.2000, 0.3000],
         [0.4000, 0.5000, 0.6000, 0.4000, 0.5000, 0.6000]]],
       dtype=torch.float64)
>>> concat = torch.cat((b,d),dim=2)
>>> concat.shape
torch.Size([2, 2, 6])
'''
