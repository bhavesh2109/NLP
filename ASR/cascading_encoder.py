import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from mfcc import mfcc_sliced_list

class EncoderRNN(nn.Module):

    def __init__(self,input_size,hidden_dim,n_layers=1):

        super(EncoderRNN,self).__init__()
        self.hidden_dim=hidden_dim
        self.lstm_0=nn.LSTM(input_size,hidden_dim,bidirectional=True)
        self.lstm_1=nn.LSTM(hidden_dim,2*hidden_dim,bidirectional=True)
        self.lstm_2=nn.LSTM(2*hidden_dim,4*hidden_dim,bidirectional=True)

        self.hidden_0=(autograd.Variable(torch.zeros(2,1,hidden_dim)),autograd.Variable(torch.zeros(2,1,hidden_dim)))
        self.hidden_1=(autograd.Variable(torch.zeros(2,1,2*hidden_dim)),autograd.Variable(torch.zeros(2,1,2*hidden_dim)))
        self.hidden_2=(autograd.Variable(torch.zeros(2,1,4*hidden_dim)),autograd.Variable(torch.zeros(2,1,4*hidden_dim)))

    def forward(self,mfcc_input):
        

        output,hidden_0=self.lstm_0(mfcc_input,self.hidden_0)
        output=output.view(output.size(0)/2,1,40)

        output,hidden_1=self.lstm_1(output,self.hidden_1)
        output=output.view(output.size(0)/2,1,-1)

        output,hidden_2=self.lstm_1(output,self.hidden_2)
        output=output.view(output.size(0)/2,1,-1)

        return output


    def initHidden(self):

        self.hidden_0=(autograd.Variable(torch.zeros(2,1,self.hidden_dim)),autograd.Variable(torch.zeros(2,1,self.hidden_dim)))
        self.hidden_1=(autograd.Variable(torch.zeros(2,1,2*self.hidden_dim)),autograd.Variable(torch.zeros(2,1,2*self.hidden_dim)))
        self.hidden_2=(autograd.Variable(torch.zeros(2,1,4*self.hidden_dim)),autograd.Variable(torch.zeros(2,1,4*self.hidden_dim)))

if __name__=="__main__":
    
    mfcc_input=mfcc_sliced_list[0]
    mfcc_input=autograd.Variable(torch.FloatTensor(mfcc_input))
    mfcc_input=torch.t(mfcc_input)
    mfcc_input=mfcc_input.contiguous().view(mfcc_input.size(0),1,mfcc_input.size(1))
    MFCC_Coefficients=20
    HIDDENSIZE=10
    encoder=EncoderRNN(MFCC_Coefficients,HIDDENSIZE,1)

    encoder.initHidden()

    encoder_output=encoder(mfcc_input)




