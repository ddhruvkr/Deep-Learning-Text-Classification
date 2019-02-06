This is a repository for different deep learning based text classification that I implemented. It does not have a data layer
as of now since I initially coded these models for a different project. I'll be adding a data layer in the future.

All of them are coded in Pytorch 1.0 and have the dimensions commented next to each vector operation. I found this very
useful for coming back to models at a later time and for quickly debugging them.

Below is the list of the models with link to original papers attached.

1) RNN.py
RNN, RNN+max_pooling
A standard Recurrent Neural Network. Also has an option to use max_pooling.
All the results that I got in my report was using this(RNN)

2) BiRNN.py
BiRNN, BiRNN + max_pooling
Bi-RNN unlike RNN, processes the data in both the forward and reverse directions(from to rear as well rear to front) thus being able to capture both the previous and future context of the word. It also has a max-pooling option.

3) Kim_CNN.py
Convolutional Neural Networks for Sentence Classification, Kim, 2014
https://arxiv.org/pdf/1408.5882.pdf
CNN for sentence classification

4) Global_Attention.py 
Effective Approaches to Attention-based Neural Machine Translation, Luong et al, 2015
https://arxiv.org/pdf/1508.04025.pdf
Coded the Global Attention model from this paper. Although it was implemented for NMT, seems like we could use the concept. Not using Local Attention for now.

5) HAN.py
Hierarchical Attention Networks for Document Classification, Yang et al, 2016
https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
Coded the word level attention model as well as the whole model as well. Therefore 2 different models.

6) Lin_Self_Attention.py 
A Structured Self-attentive Sentence Embedding, Lin et al, 2017
https://arxiv.org/abs/1703.03130
Coded their self attention model. They originally proposed the model to get embedding for a sentence. We could use for the whole comment(containing multiple sentences) or use it in HAN maybe.
Instead of using a vector, we use a 2-D matrix to represent the embedding, with each row of the matrix attending on a different part of the sentence. We also propose a self-attention mechanism and a special regularization term for the model.

7) Liu_InnerAttention.py
Learning Natural Language Inference using Bidirectional LSTM model and Inner-Attention, Liu et al, 2016
https://arxiv.org/pdf/1605.09090.pdf
This paper is originally proposes a method for Recognizing text entailment. It proposes an inner attention model which is based on the outputs of averaged pooled output of BiLSTM and all the hidden states of BiLSTM. The attention model gives the sentence vector but as mentioned above it should be useful for classification as well. The attention model is inspired from REASONING ABOUT ENTAILMENT WITH NEURAL ATTENTION(Rocktaschel et al).

8) ConvRNN.py 
A Hybrid Framework for Text Modeling with Convolutional RNN, Wang et al, 2017
http://library.usc.edu.ph/ACM/KKD%202017/pdfs/p2061.pdf

9) Tay_IntraAttention.py
Reasoning with Sarcasm by Reading In-between, Tay et al, 2018
https://arxiv.org/pdf/1805.02856.pdf
Self-targetted co-attention on embedding layer
SOTA in sarcasm detection. Use word-word attention layer for intra-attention and an LSTM encoder separately.
I implemented multiple variations of the original model presented by the authors. I will detail each of them here shortly.

10) Self_IntraAttention.py
Combined models presented in 6 and 9

11) BIDAF.py
Applied self-targetted co-attention and it's variant similar to the ones in 9, on top of LSTM hidden layers instead of the embeddings as applied in BIDAF paper.
