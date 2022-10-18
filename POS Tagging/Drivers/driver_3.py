import torch
import torch.nn as nn
from torchtext import data
import torch.optim as optim
from math import log, isfinite
from collections import Counter, defaultdict
import numpy as np
import sys, os, time, platform, nltk, random
from tagger import *

tagged_sentences = load_annotated_corpus("en-ud-train.upos.tsv")
sentence = [a for a in "NLP course is very hard and demanding".split(" ") if a]
model_param = {'max_vocab_size': 300,
               'min_frequency': 5,
               'input_rep': 0,
               'embedding_dimension': 100,
               'num_of_layers': 1,
               'output_dimension': 18,
               'epochs': 1,
               'pretrained_embeddings_fn': "glove.6B.100d.txt",
               'data_fn': "en-ud-train.upos.tsv"}

# Evaluate HMM
# [allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B] = learn_params(
#     tagged_sentences=tagged_sentences)

# baseline tagging
# tag1=baseline_tag_sentence(sentence=sentence, perWordTagCounts=perWordTagCounts, allTagCounts=allTagCounts)
# tokens = [word for (word, tag) in tagged_sentences[0]]
# tag1 = tag_sentence(tokens, {'baseline': [perWordTagCounts, allTagCounts]})
# print("baseline tagging")
# c = count_correct(gold_sentence=tagged_sentences[0], pred_sentence=tag1)
# print("correct: {}, correctOOV: {}, OOV: {}".format(c[0], c[1], c[2]))

# hmm tagging
# tag2=hmm_tag_sentence(sentence=sentence, A=A,B=B)
# tag2 = tag_sentence(tokens, {'hmm': [A, B]})
# print("hmm tagging")
# c = count_correct(gold_sentence=tagged_sentences[0], pred_sentence=tag2)
# print("correct: {}, correctOOV: {}, OOV: {}".format(c[0], c[1], c[2]))

# prob evalluation
# prob1 = joint_prob(tag1, A=A, B=B)
# print("prob1")
# print(prob1)
# prob2 = joint_prob(tag2, A=A, B=B)
# print("prob2")
# print(prob2)

# Evaluate LSTMs
# vanila
model = initialize_rnn_model(model_param)
train_rnn(model, tagged_sentences)
tokens = [word for (word, tag) in tagged_sentences[0]]

tag3 = tag_sentence(tokens, {'blstm': [model]})
c = count_correct(gold_sentence=tagged_sentences[0], pred_sentence=tag3)
print("vanila lstm")
print("correct: {}, correctOOV: {}, OOV: {}".format(c[0], c[1], c[2]))

# case sensitive
model_param["input_rep"] = 1
model = initialize_rnn_model(model_param)
train_rnn(model, tagged_sentences)
tag4 = tag_sentence(tokens, {'cblstm': [model]})
c = count_correct(gold_sentence=tagged_sentences[0], pred_sentence=tag4)
print("case sensitive lstm")
print("correct: {}, correctOOV: {}, OOV: {}".format(c[0], c[1], c[2]))

# best
model_param = get_best_performing_model_params()
model = initialize_rnn_model(model_param)
train_rnn(model, tagged_sentences)
if model_param["input_rep"] == 0:
    tag5 = tag_sentence(sentence, {'blstm': [model]})
else:
    tag5 = tag_sentence(sentence, {'cblstm': [model]})
print("best")
print("correct: {}, correctOOV: {}, OOV: {}".format(c[0], c[1], c[2]))
