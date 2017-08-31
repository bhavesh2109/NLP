import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from xml.dom import minidom
import nltk
import math
import pickle

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class LSTM_Single_Layer(nn.Module):

    def __init__(self,embedding_dim,hidden_dim,vocab_size):
        super(LSTM_Single_Layer,self).__init__()
        self.word_embeddings=nn.Embedding(vocab_size,embedding_dim)
        self.hidden_dim=hidden_dim
        self.lstm=nn.LSTM(embedding_dim,hidden_dim)
        self.linear=nn.Linear(hidden_dim,vocab_size)
        self.hidden=self.init_hidden()

    def init_hidden(self):

        return (autograd.Variable(torch.zeros(1,1,self.hidden_dim)),autograd.Variable(torch.zeros(1,1,self.hidden_dim)))

    def forward(self,sentence):
        embeds=self.word_embeddings(sentence)
        lstm_out,self.hidden=self.lstm(embeds.view(len(sentence),1,-1),self.hidden)
        log_probs=F.log_softmax(self.linear(lstm_out.view(len(sentence),-1)))
        return log_probs



model=LSTM_Single_Layer(64,100,43131)
model.load_state_dict(torch.load('model_params.pkl'))
testList=pickle.load(open("testList","rb"))
word_to_index=pickle.load(open("word_to_index","rb"))
index_to_word=pickle.load(open("index_to_word","rb"))


def sent2id (sentence,word_to_index):
        idxs=[word_to_index[word] for word in sentence]
        tensor=torch.LongTensor(idxs)
        return autograd.Variable(tensor)


def generate_sentence():
    
    new_sentence=autograd.Variable(torch.LongTensor([random.randint(0,43130)]))
    while new_sentence[-1].data[0]!=word_to_index["SENT_END"]:
        if len(new_sentence)==10:
            break
        y_pred=model(new_sentence)
        max_prob,index=torch.max(y_pred, 1)
        print new_sentence
        new_sentence=torch.cat((new_sentence,index[-1]),0)
        

    gen_sentence=[]
    for s in new_sentence:
        gen_sentence.append(index_to_word[s.data[0]])

    print " ".join(gen_sentence)
    
def perplexity():
    X_data=[]
    Y_data=[]
    for s in testList:
        X_data.append([word_to_index[w] for w in s[:-1]])
        Y_data.append([word_to_index[w] for w in s[1:]])

    test_data = [(x, y) for x, y in zip(X_data, Y_data)]
    log_prob_sum = 0
    total_seq_len = 0
    j = 0
    for seq_in, seq_out in test_data[:]:

        words_probs = model(autograd.Variable(torch.LongTensor(seq_in)))
        temp_prob = 0
        if j % 100 == 0:
            print 'Sentence number', j

        for i, next_word in enumerate(seq_out):
        # print i, next_word
        # print (words_probs.data)[i][next_word]
            temp_prob += (words_probs.data)[i][next_word]

        log_prob_sum += temp_prob
        total_seq_len += len(seq_out)
    # print total_seq_len, len(seq_out)
        j += 1 
    log_prob_mean = -1.0*log_prob_sum/total_seq_len
    perplexity = math.exp(log_prob_mean)
    print 'Perplexity:', perplexity

    
if __name__=="__main__":
    perplexity()



