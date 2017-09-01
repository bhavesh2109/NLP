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

use_cuda = torch.cuda.is_available()




trainingList=pickle.load(open("trainingList","rb"))
testList=pickle.load(open("testList","rb"))
word_to_index=pickle.load(open("word_to_index","rb"))
index_to_word=pickle.load(open("index_to_word","rb"))
MAX_LENGTH=15


def sent2id (sentence,word_to_index):
    idxs=[word_to_index[word] for word in sentence]
    tensor=torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.LSTM = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.LSTM(output, hidden)
        return output, hidden

    def initHidden(self):
        result = (autograd.Variable(torch.zeros(1, 1, self.hidden_size)),autograd.Variable(torch.zeros(1,1,self.hidden_size)))
        if use_cuda:
            return (result[0].cuda(),result[1].cuda())
            print  (result[0].cuda(),result[1].cuda())
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.LSTM = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.LSTM(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = (autograd.Variable(torch.zeros(1, 1, self.hidden_size)),autograd.Variable(torch.zeros(1,1,self.hidden_size)))
        if use_cuda:
            return result.cuda()
        else:
            return result

teacher_forcing_ratio=0.5

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    
    encoder_outputs = autograd.Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
   
    temp_prob = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = autograd.Variable(torch.LongTensor([word_to_index["SENT_START"]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True 

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            temp_prob+=(decoder_output.data)[0,target_variable[di].data[0]]   

            decoder_input = target_variable[di]  # Teacher forcing





    return temp_prob





def perplexity(encoder, decoder, TESTING_SIZE, print_every=10,learning_rate=0.01):
    plot_losses = [] # Reset every print_every
    print_perplexity_total = 0 
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)


    for i in range(TESTING_SIZE):
        
        input_variable = sent2id(testList[i],word_to_index)
        input_variable = input_variable.cuda() if use_cuda else input_variable
        target_variable =torch.cat((sent2id(testList[i],word_to_index)[1:],torch.LongTensor([word_to_index["SENT_END"]])),0)
        target_variable = target_variable.cuda() if use_cuda else target_variable 
        temp_prob = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer,target_variable.size()[0])
        print_perplexity_total += temp_prob
        print print_perplexity_total
        print i

    total_length=(sum(len(x) for x in testList[:1000]))
    print_perplexity_total=print_perplexity_total/total_length
    perplexity_total=-perplexity_total.data[0]
    print math.exp(print_perplexity_total)





if __name__ == '__main__':
    hidden_size = 256
    encoder1=EncoderRNN(len(word_to_index),hidden_size)
    decoder1=DecoderRNN(hidden_size,len(word_to_index))
    encoder1.load_state_dict(torch.load('encoder1_1_layer.pkl'))
    decoder1.load_state_dict(torch.load('decoder1_1_layer.pkl'))
    if use_cuda:
        encoder1 = encoder1.cuda()
        decoder1 = decoder1.cuda()

    perplexity(encoder1, decoder1, 1000 , print_every=10)
