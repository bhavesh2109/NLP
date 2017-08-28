import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from xml.dom import minidom
import nltk
import math
import pickle

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#torch.manual_seed(1)
if __name__=="__main__":

    trainingList=pickle.load(open("trainingList","rb"))
    testList=pickle.load(open("testList","rb"))
    word_to_index=pickle.load(open("word_to_index","rb"))
    index_to_word=pickle.load(open("index_to_word","rb"))



    def sent2id (sentence,word_to_index):
        idxs=[word_to_index[word] for word in sentence]
        tensor=torch.LongTensor(idxs)
        return autograd.Variable(tensor)



    class LSTM_Single_Layer(nn.Module):

        def __init__(self,embedding_dim,hidden_dim,vocab_size):
            super(LSTM_Single_Layer,self).__init__()
            self.word_embeddings=nn.Embedding(vocab_size,embedding_dim)
            self.hidden_dim=hidden_dim
            self.lstm=nn.LSTM(embedding_dim,hidden_dim)
            self.linear=nn.Linear(hidden_dim,vocab_size)
            self.hidden=self.init_hidden()
            self.layers=2

        def init_hidden(self):

            return (autograd.Variable(torch.zeros(1,1,self.hidden_dim)),
                autograd.Variable(torch.zeros(1,1,self.hidden_dim)))

        def forward(self,sentence):
            embeds=self.word_embeddings(sentence)
            lstm_out=embeds.view(len(sentence),1,-1)
            for i in range(self.layers):
                lstm_out,self.hidden=self.lstm(lstm_out,self.hidden)
            log_probs=F.log_softmax(self.linear(lstm_out.view(len(sentence),-1)))
            return log_probs

    HIDDEN_DIM=128
    EMBEDDING_DIM=128

    model=LSTM_Single_Layer(EMBEDDING_DIM,HIDDEN_DIM,len(word_to_index))

    loss_function=nn.NLLLoss()
    optimizer=optim.SGD(model.parameters(),lr=0.1)
    
    i=0
    for epoch in range(1):
        for sentence in trainingList:
            model.zero_grad()
            model.hidden=model.init_hidden()
            sentence_in=sent2id(sentence,word_to_index)
            y_actual=torch.cat((sentence_in[1:],torch.LongTensor([word_to_index["SENT_END"]])),0)
            y_preds=model(sentence_in)
            loss=loss_function(y_preds,y_actual)
            loss.backward()
            optimizer.step()
            print(loss)
            print i
            i=i+1
    torch.save(model.state_dict(),'model_params_2_layers.pkl')
    pickle.dump(trainingList,open("trainingList","wb"))
    pickle.dump(testList,open("testList","wb"))
    pickle.dump(word_to_index,open("word_to_index","wb"))
    pickle.dump(index_to_word,open("index_to_word","wb"))

    


