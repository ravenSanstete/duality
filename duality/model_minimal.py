# the minimal part for the sentiment analysis and the command extraction
# without a neural net model

import os
import pickle
import re
import difflib
import queue

# https://github.com/cjhutto/vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


import spacy
import numpy as np
import Levenshtein as leven
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from .context import voice_name_dict, song_name_dict, WORD_VEC_SIZE, NOISE_LEVEL, Transition, SING_DANCE_NO, DEFAULT_OPTION_NO;
from .cortex import Cortex, Dejavu;




def wrap(np_arr):
    return torch.FloatTensor(np_arr);



"""
We can consider a document as (S_i), and we assume we are able to evaluate the
score of each simple sentence without any outer information. To model the seq, we choose
a simple RNN, since the number of sentence is not too big, but it's variable.


Although a markov chain model can be choosen, a rnn is more powerful to capture the dependency

@param: doc, a roughly preprocessed raw text used in the dataset
@param: a spacy english instance
@param: analyzer, a vader analyzer instance
"""
def doc_eval(doc, nlp, analyzer):
    components= list();
    commands=list();
    _doc= nlp(doc);
    for sent in _doc.sents:
        commands.extend(_imperative_classify(sent)); # extend the command list a.t. the single sentences
        components.append(sent_eval(sent.text, nlp, analyzer)); # evaluate each single sentence's sentiment values
    return _doc_eval(components), commands;

"""
@param: components, a list of sentence sentiment evaluations
"""
def _doc_eval(components):
    return np.average(components);


"""
Evaluation process for a single sentence, which has been roughly cleaned and classified(verb extracted for imperative sentence)

@param: alpha, a linear combination parameter
@param: sent, the raw text, without a deep clean
@param: nlp, spacy english instance
@param: analyzer, vader sentiment instance
@param:

"""
def sent_eval(sent, nlp, analyzer):
    # deep clean done here
    lexicon_values = analyzer.polarity_scores(sent);
    return lexicon_values['compound'];




"""
To evaluate whether a sentence is an imperative sentence, which is almost based on whether
there contains a verb with no subject
@return: the verb phase contained in this sentence, a list is better

"""
def _imperative_classify(soup):
    verbs = list();
    # collect the verb with no tag
    # extract the verb
    for word in soup:
        is_you=-1; # default -1, means no subject
        if(word.pos_=='VERB'):
            # check whether the subject is you
            for c in word.children:
                if(c.dep_=='nsubj'):
                    is_you= 1 if c.text=='you' else 0;
                    break;
            # deal with conjugated words with no explicit subject
            if(word.dep_=='conj' and is_you==-1):
                if(word.head in verbs):
                    verbs.append(word);
                continue;
            # deal with common situations
            if(word.tag_=='VB' and is_you==-1):
                verbs.append(word);
            elif((word.tag_=='VB' or word.tag_=='VBP') and is_you==1):
                verbs.append(word);
    return verbs; # currently, only filter out the move, in future, more things will be parsed out in the tree



"""
a wrapper for the above method
"""
def imperative_classify(sent, nlp):
    return _imperative_classify(nlp(sent));



# auxiliary method for song selection
def _find_song(name):
    max_similarity=-0.1;
    ind=-1;
    threshold=0.5;
    print("match with name %s" % (name));
    for i in song_name_dict.keys():
        sim=leven.jaro(song_name_dict[i], name)
        print("similarity %f" % (sim));
        if(sim>=max_similarity):
            ind=i;
            max_similarity=sim;

    return ind if(max_similarity>=threshold) else None;

def _random_song():
    return -1;



"""
1. sing the song named blissy land [sing -> 1 dobj  [acl.] -> (conj) ->   ]
2. sing blissy land [song -> (1 dobj)]
3. sing the song blissy land for me [sing -> 2 dobj]
"""

"""
1. dance to happy birthday
2. dance to happy birthday for me
3. dance to the song named happy birthday [the same case for]
"""



"""
this method will be invoked when miku finds the command to sing or dance

if it can not find a proper song with a potential name, just return something random

@param: sent, the sentence from where comes the command
@param: command, the recognized command
"""
def _recognize_song_name(sent, command):
    song_name=None;
    prep_to=command; # it is by default the word song
    if(command.text=="dance"):
        for c in command.children:
            if(c.dep_=="prep" and c.text=="to"):
                prep_to=c;
                break;
        if(prep_to==None):
            return _random_song();

    # only deal with the dance sentence with the prep 'to' and sing-, which is a conventional english usage
    if(prep_to!=None):
        beg_pos=prep_to.i+1;
        for c in prep_to.children:
            if(c.dep_=='pobj' or c.dep_=='dobj'):
                # first try with the direct object
                potential= _find_song(sent[beg_pos:c.i+1].text);
                beg_pos=c.i+1;
                if(potential==None):
                    # second trial, find the pp-clause
                    for k in c.children:
                        if(k.dep_=='acl' and k.i+1<len(sent)):
                            print(k.text);
                            potential=_find_song(sent[k.i+1:].text);
                            break;
                    # the final chance,
                    potential=_find_song(sent[beg_pos:].text);
        return potential if(potential!=None) else _random_song();
    # can't find a proper song
    return _random_song();






"""
@param: sentiment_param, a float
@param: is_command, a boolean
"""
# to select a voice a.t. the sentiment param and whether it is a command. from the voice dictionary
def select_voice(sentiment_param, is_command):
    # is command determines whether the voice selection  in a partially random bev
    ind= 0;
    if(sentiment_param>0.2):
        ind=1;
    elif(sentiment_param<-0.2):
        ind=2;
    else:
        ind=3;

    if(is_command and np.random.rand(1)[0]<=0.5):
        # then do a random choice from the permission and the current sentiment voice
        ind=4;
    return ind;

# first merge the vectors of the command and the obj, if there is no obj, just merge with a zero-vector



# return a refined 900*1 vector
def extract_env_vec(command):
    # take both the adv and the obj as the input vectors candidate
    if(command==None):
        return NOISE_LEVEL*np.random.randn(1,3*WORD_VEC_SIZE);
    # only take the first command(and it can be assumed)
    verb_vec=command.vector.reshape((1,WORD_VEC_SIZE));
    obj_vec= NOISE_LEVEL*np.random.randn(1,WORD_VEC_SIZE);
    adv_vec= NOISE_LEVEL*np.random.randn(1,WORD_VEC_SIZE);

    # obj='NULL';
    # adv='NULL';

    for c in command.children:
        # if the word
        if(c.dep_=='advmod'):
            adv_vec=c.vector.reshape((1,WORD_VEC_SIZE));
            # adv=c.text;
        if(c.dep_=='dobj'):
            obj_vec=c.vector.reshape((1,WORD_VEC_SIZE));
            # obj=c.text;
        # if there exists a prep, just find the pobj
        if(c.dep_=='prep'):
            for k in c.children:
                if(k.dep_=='pobj'):
                    obj_vec=k.vector.reshape((1,WORD_VEC_SIZE));
                    # obj=k.text;
                    break;
    return np.concatenate((verb_vec,obj_vec,adv_vec), axis=1);













# which is a soft waterlevel queue
class MemorySeq:
    def __init__(self, max_size):
        self.q=list();
        self.max_size=max_size;

    def put(self, item):
        if(len(self.q)+1>self.max_size):
            self.get();
        self.q.append(item);

    def get(self):
        if(len(self.q)==0):
            return None;
        else:
            result=self.q[0];
            self.q=self.q[1:]; # reshape
            return result;

    def __len__(self):
        return len(self.q);





# the interface
class MikuCore:
    def __init__(self, capacity=1000):
        self.analyzer= SentimentIntensityAnalyzer();
        print('記憶体ロード、待ってくださいね');
        self.nlp= spacy.load('en');
        print('完成です');
        self.batch_size=50;
        self.reg_1=0.01;
        self.gamma=0.01;
        self.max_memory_size=20;
        self.dejavu=Dejavu(capacity);
        self.cortex=Cortex();
        self.optimizer=torch.optim.RMSprop(self.cortex.parameters());
        self.probed_env=wrap(NOISE_LEVEL*np.random.randn(1,3*WORD_VEC_SIZE));
        self.action_selected=torch.LongTensor([0]);

        self.memory_pool={
            "action_ids": MemorySeq(self.max_memory_size),
            "voice_ids":MemorySeq(self.max_memory_size),
            "options":MemorySeq(self.max_memory_size)
        }
    # when a command reaches in, and thus the current state has been changed to the

    # CURRENT_STATE(COMMAND REACHED) -> [ACTION] -> NEXT_STATE(COMMAND UPDATED) -> REWARD
    # each time a COMMENT COMES, PUSH IT INTO THE MEMORY
    def process(self, comment):
        sentiment_value, commands = doc_eval(comment, self.nlp, self.analyzer);

        # ignore the potential second command
        command = commands[0] if(len(commands)>=1) else None;

        # select the voice
        voice_id= select_voice(sentiment_value, command!=None);

        # first check whether the comment means to sing or dance
        if(command!=None):
            potential_dance_no = self.recognize_long_bev(comment, command);
            if(potential_dance_no!=None):
                # enqueue result.
                action_id= SING_DANCE_NO;
                option= potential_dance_no;
                self.put_memory(action_id, voice_id, option);
                return;

        # else, some actions that needed to learn or some comments work as indicator of satisfaction
        former_state=self.probed_env;
        action=self.action_selected;
        # here is only a simulation
        # update both the env and the action adopted
        self.probed_env= wrap(extract_env_vec(command));
        self.action_selected=self.cortex.choice(self.probed_env); # next action it wants to do
        # with no option, thus -1;
        reward = torch.FloatTensor([sentiment_value]); # inferred from the current comment
        self.dejavu.push(former_state, action, self.probed_env, reward);

        # finally, push response to the queues
        self.put_memory(self.action_selected[0], voice_id, DEFAULT_OPTION_NO);
        return self.action_selected[0];

    # this modifies its own parameters to improve the entertaining skills
    def step(self):
        if(len(self.dejavu)<self.batch_size):
            return 5.0; # which means the dejavu is not enough for miku to modify the weight
        # fo the sample
        transitions= self.dejavu.sample(self.batch_size);

        batch = Transition(*zip(*transitions));

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)));
        non_final_next_states_t = torch.cat(tuple(s for s in batch.next_state if s is not None)).type(torch.FloatTensor);
        non_final_next_states = Variable(non_final_next_states_t, volatile=True);

        # unpack the data
        state_batch = Variable(torch.cat(batch.state));
        action_batch = Variable(torch.cat(batch.action));
        reward_batch = Variable(torch.cat(batch.reward));

        # forward to compute the Q value
        state_action_values = self.cortex(state_batch).gather(1,action_batch.view(-1,1));

        # compute the next state value
        next_state_values = Variable(torch.zeros(self.batch_size))
        next_state_values[non_final_mask] = self.cortex(non_final_next_states).max(1)[0];

        next_state_values.volatile = False;

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch;
        # compute the loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values); # self.reg_1* self.cortex.auto_encoder_loss(state_batch);

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.cortex.parameters():
            param.grad.data.clamp_(-1, 1) # a truncate of the data
        self.optimizer.step()
        return torch.mean(loss).data;

    # this wraps the logic: first recognize whether the command is a sing/dance command if so
    def recognize_long_bev(self, sent, command):
        threshold=0.9;
        if(leven.jaro(command.text, 'sing')>=threshold or leven.jaro(command.text, 'dance')>=threshold):
            return  _recognize_song_name(sent, command);
        else:
            return None;

    # find the action that occurs most frequently from the candidate
    def respond(self):
        pass;

    def put_memory(self, action_id, voice_id, option):
        self.memory_pool['action_ids'].put(action_id);
        self.memory_pool['voice_ids'].put(voice_id);
        self.memory_pool['options'].put(option);









docs=[
'go and get me something',
'you go and get me something',
'you should go and get me something',
'you go and i may give you something',
'i go and you should tell me something',
'tell me something about you',
'never thank me',
'i want you to hug me', # a complex case
'sing the song named blissy land',
'sing blissy land',
'sing the song blissy land for me',
'to dance to happy birthday',
'to dance to happy birthday for me',
'to dance to the song happy birthday for me',
'to dance to the song named happy birthday',
'walk forward',
'take the apple sincerely',
'close your eyes'
]




if(__name__=='__main__'):

    # this will finally be initiated when starting the server
    # which works as a global instance, based on some lexicon methods
    miku= MikuCore();


    # for i, doc in enumerate(docs):
    #     miku.process(doc);
    # miku.respond();
    MAX_STEP=1000000;
    LOG_STEP=1000;
    LEARN_STEP=10;
    initial_statement="wave your hands";
    good_statement="fantastic";
    bad_statement="stupid";
    average_loss=0.0;
    hit_time=0;
    for i in range(MAX_STEP):
        action_no=miku.process(initial_statement);
        if(int(action_no)==1):
            miku.process(good_statement);
            hit_time+=1;
        else:
            miku.process(bad_statement);
        if(np.mod(i+1, LEARN_STEP)==0):
            loss=miku.step();
            average_loss+=loss*LEARN_STEP/LOG_STEP;
        if(np.mod(i+1, LOG_STEP)==0):
            print(average_loss);
            print("HIT RATE: %f" % ((hit_time/i)*100));
            average_loss=0.0;
