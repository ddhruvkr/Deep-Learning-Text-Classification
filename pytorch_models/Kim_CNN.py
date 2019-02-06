import torch
#from torchtext import data
#from torchtext import datasets
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.nn import functional as f

class Kim_CNN(nn.Module):
	def __init__(self, vocab_size, embedding_dim, out_channels, dropout, embedding_weights, output_dim, multichannel):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.embedding_multichannel = nn.Embedding(vocab_size, embedding_dim)
		if multichannel:
			in_channels = 2
		else:
			in_channels = 1
		#in_channels would be 1 in this case since the input is just 2d
		#out_channels is the hyperparameter which represents the number of kernels doing the convolutions
		kernel_heights = [3,4,5]
		self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_dim), padding=(2,0))
		self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_dim), padding=(3,0))
		self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_dim), padding=(4,0))
		self.fc = nn.Linear(len(kernel_heights)*out_channels, output_dim)
		self.dropout = nn.Dropout(dropout)
		#self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
		self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weights).float(), requires_grad=True)
		self.embedding_multichannel.weight = nn.Parameter(torch.from_numpy(embedding_weights).float(), requires_grad=False)
		self.multichannel = multichannel

	def forward(self, input_sequences):
		#print('inside forward')
		#x = [sent len, batch size]
		#embedded = self.dropout(self.embedding(x))

		#shape of x:
		#batch_size, max_sequence_length
		embeddings = self.embedding(input_sequences)
		#print(embeddings.shape)
		#shape output of embedding layer:
		#batch_size, max_sequence_length, embedding_dim
		embeddings = embeddings.unsqueeze(1)
		#(batch_size, 1, num_seq, embedding_dim)
		if self.multichannel:
			embeddings_multichannel = self.embedding_multichannel(input_sequences)
			embeddings_multichannel = embeddings_multichannel.unsqueeze(1)
			embeddings = torch.cat((embeddings, embeddings_multichannel), dim=1)
			#(batch_size, 2, num_seq, embedding_dim)
			#print(embeddings.shape)
		max_pool1 = self.conv_block(embeddings, self.conv1)
		max_pool2 = self.conv_block(embeddings, self.conv2)
		max_pool3 = self.conv_block(embeddings, self.conv3)
		
		concat_pool = torch.cat((max_pool1, max_pool2, max_pool3), 1)
		#print(concat_pool.shape)
		# all_out.size() = (batch_size, num_kernels*out_channels)
		fc = self.dropout(concat_pool)
		# fc_in.size()) = (batch_size, num_kernels*out_channels)
		linear = self.fc(fc)
		
		return linear


	def conv_block(self, input, conv_layer):
		conv_out = conv_layer(input)
		#print(conv_out.shape)
		# conv_out.size() = (batch_size, out_channels, embedding_dim, 1)
		activation = f.relu(conv_out.squeeze(3))
		#print(activation.shape)
		# activation.size() = (batch_size, out_channels, embedding_dim)
		max_out = f.max_pool1d(activation, activation.size()[2]).squeeze(2)
		#print(max_out.shape)
		# maxpool_out.size() = (batch_size, out_channels)
		
		return max_out