# this is for a deep reinforcement generative model


# the adoption of pytorch to enable online training of the VC
from collections import namedtuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



from .context import WORD_VEC_SIZE, ACTION_POOL_SIZE, Transition
# we use a deep-Q network with the input as a 300*3

# thanks for the tutorial
"""
https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb
"""





# this works as the replay memory
class Dejavu(object):
    def __init__(self, capacity):
        self.capacity= capacity;
        self.memory=[];
        self.position=0;

    def push(self, *args):
        if(len(self.memory)<self.capacity):
            self.memory.append(None); # to prevent memory not allocated [cold start]
        self.memory[self.position] = Transition(*args);
        self.position= (self.position+1) % self.capacity;

    # you should note that the random sample cannot be processed if the len of the memory is bigger than the batch size
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size);

    # this overrides the length definition
    def __len__(self):
        return len(self.memory);


class Cortex(nn.Module):
    def __init__(self):
        super(Cortex, self).__init__();
        # some constants for the neural network

        self.code_size=50;

        self.encode_layer = nn.Linear(in_features=WORD_VEC_SIZE*3, out_features=self.code_size);
        self.decode_layer= nn.Linear(in_features=self.code_size, out_features=WORD_VEC_SIZE*3);
        self.batch_normalizer=nn.BatchNorm1d(num_features=self.code_size);
        self.predict_layer= nn.Linear(in_features=self.code_size, out_features=ACTION_POOL_SIZE);
        self.code_regularization = nn.MSELoss();

    def auto_encoder_loss(self, x):
        x_prime = F.sigmoid(self.decode_layer(F.sigmoid(self.encode_layer(x))));
        return self.code_regularization(x_prime, x);

    def forward(self, x):
        x= F.relu(self.encode_layer(x));
        x= F.dropout(x, p=0.1);
        x= self.batch_normalizer(x);
        x= F.dropout(x, p=0.05);
        x= self.predict_layer(x);
        return F.tanh(x); # which can be considered as the expected reward value

    # this is works as an interface for client to use to do predict
    def choice(self, np_vec):
        return self.forward(Variable(torch.FloatTensor(np_vec))).max(1)[1].view(1).data;





if(__name__=='__main__'):
    batch_size=10;
    batch_data= Variable(torch.randn(batch_size, WORD_VEC_SIZE*3));
    model=Cortex();
