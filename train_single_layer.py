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
    START_TOKEN = 'SENT_START'
    END_TOKEN = 'SENT_END'

# Load data
    print 'Loading XML file...'
    xmldoc = minidom.parse('ted_en-20160408.xml')
    print 'Getting English text...'
    textList = xmldoc.getElementsByTagName('content')
    print 'Number of transcripts:', len(textList)
# print textList[0].childNodes[0].nodeValue

    sentenceList = []
    NUM_TRANSCRIPTS = 1000

# Get tokenized sentences from French transcripts
    for s in textList[:NUM_TRANSCRIPTS]:

        text = s.childNodes[0].nodeValue
    
    # Split text into sentences
        sentences = nltk.sent_tokenize(text.decode('utf-8').lower())

    # Split sentences into words
        tokenized_sents = [nltk.word_tokenize(s) for s in sentences]
        tokenized_sents = [[START_TOKEN] + t + [END_TOKEN] for t in tokenized_sents]
    
    # i = 1

        sentenceList += tokenized_sents

# Use only some of the sentences in vocabulary
    NUM_SENTENCES = 40000
    TESTING_SENTENCES=10000
    print 'Taking only', NUM_SENTENCES, 'sentences...'
    trainingList = sentenceList[:NUM_SENTENCES]
    testList=sentenceList[NUM_SENTENCES:NUM_SENTENCES+TESTING_SENTENCES]


# Create word_to_index dict from transcripts
    word_to_index = {}
    index_to_word = []
    for s in sentenceList:
        for word in s:
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)
                index_to_word.append(word)
    print 'Size of vocabulary:', len(word_to_index)



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

        def init_hidden(self):

            return (autograd.Variable(torch.zeros(1,1,self.hidden_dim)),
                autograd.Variable(torch.zeros(1,1,self.hidden_dim)))

        def forward(self,sentence):
            embeds=self.word_embeddings(sentence)
            lstm_out,self.hidden=self.lstm(embeds.view(len(sentence),1,-1),self.hidden)
            log_probs=F.log_softmax(self.linear(lstm_out.view(len(sentence),-1)))
            return log_probs

    HIDDEN_DIM=100
    EMBEDDING_DIM=64

    model=LSTM_Single_Layer(EMBEDDING_DIM,HIDDEN_DIM,len(word_to_index))

    loss_function=nn.NLLLoss()
    optimizer=optim.SGD(model.parameters(),lr=0.1)
    '''
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
    torch.save(model.state_dict(),'model_params.pkl')
    '''

    pickle.dump(testList,open("testList","wb"))
    pickle.dump(word_to_index,open("word_to_index","wb"))
    pickle.dump(index_to_word,open("index_to_word","wb"))

    


