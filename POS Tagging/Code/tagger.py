"""
nlp, assignment 4, 2021

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""

import torch
import torchtext
import torch.nn as nn
from torchtext import data
import torch.optim as optim
from math import log, isfinite
from collections import Counter, OrderedDict
import numpy as np

import sys, os, time, platform, nltk, random
from VanillaBiLSTM import VanillaBiLSTM
from CaseBasedBLSTM import CaseBasedBLSTM

# With this line you don't need to worry about the HW  -- GPU or CPU
# GPU cuda cores will be used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# You can call use_seed with other seeds or None (for complete randomization)
# but DO NOT change the default value.
def use_seed(seed=2512021):
    """
    Define seed value to avoid total randomization (restore previous results).
    :param seed: default seed to use
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.set_deterministic(True)
    except Exception as e:
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # torch.backends.cudnn.deterministic = True


# utility functions to read the corpus
def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    # TODO edit the dictionary to have your own details
    return {'name': 'Amit Damri', 'id': '312199698', 'email': 'amitdamr@post.bgu.ac.il'}


def read_annotated_sentence(f):
    """Read only one annotated sentence (a new line is a separator between sentences).

    Args:
        f (object): file object
    Return:
        str: annotated sentence. list of pairs (w,t)
    """
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append((word, tag))
        line = f.readline()
    return sentence


def load_annotated_corpus(filename):
    """Load annotated corpus from the given path (filename)

    Args:
        filename (str): the corpus file path

    Return:
        list: a list of tagged sentences, each tagged sentence is a
        list of pairs (w,t)
    """
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_annotated_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_annotated_sentence(f)
    return sentences


# Global variables
START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"

allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}

# log probability distributions: do NOT use Counters inside these because
# missing Counter entries default to 0, not log(0)
A = {}  # transitions probabilities
B = {}  # emmissions probabilities


def learn_params(tagged_sentences):
    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
     and emissionCounts data-structures.
    allTagCounts and perWordTagCounts should be used for baseline tagging and
    should not include pseudocounts, dummy tags and unknowns.
    The transitionCounts and emissionCounts
    should be computed with pseudo tags and should be smoothed.
    A and B should be the log-probability of the normalized counts, based on
    transisionCounts and  emmisionCounts

    Args:
      tagged_sentences: a list of tagged sentences, each tagged sentence is a
       list of pairs (w,t), as retunred by load_annotated_corpus().

   Return:
      [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)
  """
    # TODO complete the code
    if tagged_sentences is not None and len(tagged_sentences) > 0:
        for sentence in tagged_sentences:
            __populate_all_tag_counts(sentence)
            __populate_per_word_tag_counts(sentence)
            __populate_transition_counts(sentence)

            # all words that exist only once, I assumed that they are OOVs - I changed
            # each word that exists only once to UNK to have a little idea about oov words (for the test phase).
            oov_min = 1
            oov_words = [word for word in perWordTagCounts.keys() if sum(perWordTagCounts[word].values()) <= oov_min]
            __populate_emission_counts(sentence, oov_words)

        __populate_log_prob(A, transitionCounts, smooth=True)
        __populate_log_prob(B, emissionCounts, smooth=True)
    return [allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B]


def __populate_all_tag_counts(sentence):
    """Populates the allTagCounts Counter. Counts for each tag the number of
    times it exists in the given sentence.

    Args:
     sentence (list): annotated sentence. list of pairs (w,t).
    """
    allTagCounts.update([tag.upper() for (word, tag) in sentence])


def __populate_per_word_tag_counts(sentence):
    """Populates the perWordTagCounts dict. Counts for each word the number of
    times it exists with each tag in the given sentence.

    Args:
     sentence (list): annotated sentence. list of pairs (w,t).
    """
    # get all the unique words in the given sentence
    words_set = set([word.lower() for (word, tag) in sentence])

    for word in words_set:

        # update counter if word already exists
        if word in perWordTagCounts:
            perWordTagCounts[word].update([tag.upper() for (w, tag) in sentence if w.lower() == word])

        # append the word if it does not exist in the dictionary yet
        else:
            perWordTagCounts[word] = Counter([tag.upper() for (w, tag) in sentence if w.lower() == word])


def __populate_transition_counts(sentence):
    """Populates the transitionCounts dict. Counts for each tag the number of
    times it exists with each consecutive tag in the given sentence.

    Args:
     sentence (list): annotated sentence. list of pairs (w,t).
    """
    # get all consecutive tags pairs
    bigram_tags_set = __get_bigram_tags(sentence)
    tags = set([tag for (word, tag) in sentence] + [START])
    for tag in tags:
        if tag in transitionCounts:
            transitionCounts[tag].update([second for (first, second) in bigram_tags_set if first == tag])
        else:
            transitionCounts[tag] = Counter([second for (first, second) in bigram_tags_set if first == tag])
            # add UNK value for each tag to find sequences that do not exist in the training data.
            transitionCounts[tag][UNK] = 0


def __get_bigram_tags(sentence):
    """Returns pairs of consecutive tags which exist in the given sentence -> (t, t+1)
    The first pair is starts with (START,t) and the last pair ends with (t,END).

    Args:
     sentence (list): annotated sentence. list of pairs (w,t).

    Return:
        bigram_tags (list): list of consecutive tags pairs -> [(t,t+1), (t+1,t+2), etc.]
    """
    bigram_tags = []
    prev_tag = None
    for index, (w, tag) in enumerate(sentence):
        if index == 0:
            bigram_tags.append((START, tag))
        else:
            bigram_tags.append((prev_tag, tag))
        prev_tag = tag
    bigram_tags.append((prev_tag, END))
    return bigram_tags


def __populate_emission_counts(sentence, oov_words):
    """Populates the emissionCounts dict. Counts for each tag the number of
    times it exists with each word in the given sentence.

    Args:
     sentence (list): annotated sentence. list of pairs (w,t).
     oov_words (list): list of out of vocabulary words.
    """
    # get all unique tags of the sentence
    tags_set = set([tag for (word, tag) in sentence])
    # change unknown words to UNK
    sentence = [(word, tag) if word not in oov_words else (UNK, tag) for (word, tag) in sentence]

    for tag in tags_set:
        if tag in emissionCounts:
            emissionCounts[tag].update([word for (word, t) in sentence if t == tag])
        else:
            emissionCounts[tag] = Counter()
            # add UNK value for each tag to find sequences that do not exist in the training data
            emissionCounts[tag][UNK] = 0
            emissionCounts[tag].update([word for (word, t) in sentence if t == tag])


def __populate_log_prob(obj, counter, smooth=True):
    """Populates A/B dictionaries. Calculate the log probability for each pair of tags (if obj=A)
    or for each pair of tag and word (if obj=B). I used smoothing to let pairs of tags (or tag+word)
    that were not exist in our train set, but can be in the test set, a chance to be choose while testing.
    (added 1 to each count and divided by the number of total values because I added 1 for each value (count).

    Args:
     obj (dict): A or B.
     counter (dict): transitionCounts or emissionCounts.
     smooth (bool): default to true. if we want to use smoothing.
    """
    for key in counter.keys():
        summation = sum(counter[key].values())
        if key not in obj:
            obj[key] = {}
        for (val, count) in counter[key].items():
            if smooth:
                prob = (count + 1) / (summation + len(counter.values()))
            else:
                prob = count / summation

            if isfinite(prob):
                obj[key][val] = log(prob)
            else:
                obj[key][val] = 0  # if log(0)


def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Each word is tagged by the tag most
    frequently associated with it. OOV words are tagged by sampling from the
    distribution of all tags.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        Return:
        tagged_sentence (list): list of pairs
    """

    # TODO complete the code
    tagged_sentence = []
    if sentence is not None and len(sentence) > 0:

        for word in sentence:
            low_word = word.lower()
            if low_word in perWordTagCounts:
                word_dict = perWordTagCounts[low_word]
                # if the word has tags, get the tag that is most frequently associated with.
                if len(word_dict.keys()) > 0:
                    tag = max(word_dict, key=word_dict.get)
                # sample oov word from dist of all tags
                else:
                    tag = random.choices(list(allTagCounts.keys()), weights=list(allTagCounts.values()), k=1)[0]
            # sample oov word from dist of all tags
            else:
                tag = random.choices(list(allTagCounts.keys()), weights=list(allTagCounts.values()), k=1)[0]

            tagged_sentence.append((word, tag))

    return tagged_sentence


# ===========================================
#       POS tagging with HMM
# ===========================================


def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterby
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): The HMM emission probabilities.

    Return:
        tagged_sentence (list): list of pairs
    """

    # TODO complete the code
    last_item = viterbi(sentence, A, B)
    backtrace = retrace(last_item)
    tagged_sentence = list(zip(sentence, backtrace))

    return tagged_sentence


def viterbi(sentence, A, B):
    """Creates the Viterbi matrix, column by column. Each column is a list of
    tuples representing cells. Each cell ("item") is a tuple (t,r,p), were
    t is the tag being scored at the current position,
    r is a reference to the corresponding best item from the previous position,
    and p is a log probability of the sequence so far).

    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): The HMM emission probabilities.

    Return:
        obj: the last item, tagged with END. should allow backtraking.

        """
    # Hint 1: For efficiency reasons - for words seen in training there is no
    #      need to consider all tags in the tagset, but only tags seen with that
    #      word. For OOV you have to consider all tags.
    # Hint 2: start with a dummy item  with the START tag (what would it log-prob be?).
    #         current list = [ the dummy item ]
    # Hint 3: end the sequence with a dummy: the highest-scoring item with the tag END

    # TODO complete the code
    dummy_item = (START, None, log(1))  # START prob equals 1 because it is the only tag as the first tag.
    viterbi_list = [[dummy_item]]

    # all words and tags in the training data
    words_set = set([w for words_dic in B.values() for w in words_dic.keys()])
    tags_set = set(tag for tag in A.keys() if tag != START and tag != END)

    for word in sentence:
        column = []

        # for words seen in training consider only tags seen with them.
        if word in words_set:
            word_tags = set([t for t in B.keys() if word in B[t]])

        # for OOV consider all tags
        else:
            word_tags = tags_set

        # for each tag get the best probability based on the previous tags and words.
        for tag in word_tags:
            column.append(predict_next_best(word, tag, viterbi_list[-1], A, B))

        viterbi_list.append(column)

    # get the sequence with the maximum probability
    best_seq = sorted(viterbi_list[-1], key=lambda x: x[2], reverse=True)[0]

    # tag the best sequence with the END tag for backtracking
    v_last = (END, best_seq, best_seq[2])

    return v_last


# a suggestion for a helper function. Not an API requirement
def retrace(end_item):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).

    Args:
        end_item (tuple): three elements tuple (t,r,p). see the viterby function for more information.

    Return:
        tags (list): list of tags.
    """
    tags = []
    curr_item = end_item[1]
    while curr_item[1] is not None:
        tags.append(curr_item[0])
        curr_item = curr_item[1]

    tags.reverse()
    return tags


# a suggestion for a helper function. Not an API requirement
def predict_next_best(word, tag, predecessor_list, A, B):
    """Returns the best next tuple, based on the previous tag and prob.

    Args:
        word (str): current word
        tag (str):  tag to test
        predecessor_list (list): list of tuples (t,r,p) see the viterby function for more information.
        A (dict):   The HMM Transition probabilities
        B (dict):   The HMM Emission probabilities

    Return:
        best tuple (tuple): tuple of (t,r,p)
    """
    next_opt = []
    for pre in predecessor_list:
        prev_tag = pre[0]
        prob = pre[2]

        trans = A[prev_tag][tag] if tag in A[prev_tag] else A[prev_tag][UNK]
        emiss = B[tag][word] if word in B[tag] else B[tag][UNK]

        next_prob = prob + trans + emiss
        next_opt.append((tag, pre, next_prob))

    sorted_opt = sorted(next_opt, key=lambda x: x[2], reverse=True)

    return sorted_opt[0]


def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): the HMM emmission probabilities.

    Return:
        p (float): joint probability of the given sequence of words and tags.
     """
    p = 0  # joint log prob. of words and tags

    # TODO complete the code
    prev = START

    for i in range(len(sentence)):
        w = sentence[i][0]
        t = sentence[i][1]

        if w in B[t]:
            p += B[t][w]
        else:
            p += B[t][UNK]

        p += A[prev][t]
        prev = t

    assert isfinite(p) and p < 0  # Should be negative. Think why! because we are using summation of log probabilities.
    return p


# ===========================================
#       POS tagging with BiLSTM
# ===========================================

""" You are required to support two types of bi-LSTM:
    1. a vanilla biLSTM in which the input layer is based on simple word embeddings
    2. a case-based BLSTM in which input vectors combine a 3-dim binary vector
        encoding case information, see
        https://arxiv.org/pdf/1510.06168.pdf
"""


# Suggestions and tips, not part of the required API
#
#  1. You can use PyTorch torch.nn module to define your LSTM, see:
#     https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
#  2. You can have the BiLSTM tagger model(s) implemented in a dedicated class
#     (this could be a subclass of torch.nn.Module)
#  3. Think about padding.
#  4. Consider using dropout layers
#  5. Think about the way you implement the input representation
#  6. Consider using different unit types (LSTM, GRU, LeRU)


def initialize_rnn_model(params_d):
    """Returns a dictionary with the objects and parameters needed to run/train_rnn
       the lstm model. The LSTM is initialized based on the specified parameters.
       the returned dict is may have other or additional fields.

    Args:
        params_d (dict): a dictionary of parameters specifying the model. The dict
                        should include (at least) the following keys:
                        {'max_vocab_size': max vocabulary size (int),
                        'min_frequency': the occurence threshold to consider (int),
                        'input_rep': 0 for the vanilla and 1 for the case-base (int),
                        'embedding_dimension': embedding vectors size (int),
                        'num_of_layers': number of layers (int),
                        'output_dimension': number of tags in tagset (int),
                        'pretrained_embeddings_fn': str,
                        'data_fn': str
                        }
                        max_vocab_size sets a constraints on the vocab dimension.
                            If its value is smaller than the number of unique
                            tokens in data_fn, the words to consider are the most
                            frequent words. If max_vocab_size = -1, all words
                            occurring more that min_frequency are considered.
                        min_frequency provides a threshold under which words are
                            not considered at all. (If min_frequency=1 all words
                            up to max_vocab_size are considered;
                            If min_frequency=3, we only consider words that appear
                            at least three times.)
                        input_rep (int): sets the input representation. Values:
                            0 (vanilla), 1 (case-base);
                            <other int>: other models, if you are playful
                        The dictionary can include other keys, if you use them,
                             BUT you shouldn't assume they will be specified by
                             the user, so you should spacify default values.
    Return:
        a dictionary with at least the following key-value pairs:
                                       {'lstm': torch.nn.Module object,
                                       input_rep: [0|1]}
        #Hint: you may consider adding the embeddings and the vocabulary
        #to the returned dict
    """

    # TODO complete the code
    global perWordTagCounts

    # additional tokens
    unk_token = UNK
    pad_token = '<PAD>'

    # initialize max vocabulary size and minimum frequency
    max_vocab_size = params_d['max_vocab_size'] if 'max_vocab_size' in params_d else -1
    min_frequency = params_d['min_frequency'] if 'min_frequency' in params_d else 2

    # load vocabulary and tags
    corpus = load_annotated_corpus(params_d['data_fn'])
    vocab = __get_vocab(corpus, max_vocab_size, min_frequency, pad_token, unk_token)
    tags_vocab = __get_tags(corpus, pad_token)

    # initialize perWordTagCounts with the current vocabulary
    perWordTagCounts = dict((word, 0) for word in vocab.get_itos() if word != unk_token and word != pad_token)

    # initialize model parameters
    vocab_size = len(vocab.get_itos())
    tagset_size = params_d['output_dimension'] if 'output_dimension' in params_d else len(tags_vocab.get_itos())
    embedding_dim = params_d['embedding_dimension'] if 'embedding_dimension' in params_d else 100
    hidden_dim = params_d['hidden_dim'] if 'hidden_dim' in params_d else 64
    n_layers = params_d['n_layers'] if 'n_layers' in params_d else 2
    dropout = params_d['dropout'] if 'dropout' in params_d else 0.25
    epochs = params_d['epochs'] if 'epochs' in params_d else 10
    batch_size = params_d['batch_size'] if 'batch_size' in params_d else 16
    input_rep = params_d['input_rep'] if 'input_rep' in params_d else 1
    bidirectional = True

    # set padding indices
    vocab_pad_idx = vocab[pad_token]
    tag_pad_idx = tags_vocab[pad_token]

    if input_rep == 0:
        lstm_model = VanillaBiLSTM(vocab_size,
                                   embedding_dim,
                                   hidden_dim,
                                   n_layers,
                                   dropout,
                                   bidirectional,
                                   tagset_size,
                                   vocab_pad_idx)
    elif input_rep == 1:
        case_in_dim = 3  # 3 dimensional binary vector for each word
        lstm_model = CaseBasedBLSTM(vocab_size,
                                    embedding_dim,
                                    hidden_dim,
                                    n_layers,
                                    dropout,
                                    bidirectional,
                                    tagset_size,
                                    vocab_pad_idx,
                                    case_in_dim
                                    )
    else:
        lstm_model = None  # not implemented

    # load embedding vectors
    if 'pretrained_embeddings_fn' in params_d:
        vectors = load_pretrained_embeddings(params_d['pretrained_embeddings_fn'], vocab.get_itos())
    else:
        vectors = load_pretrained_embeddings('../glove.6B.100d.txt', vocab.vocab.get_itos())

    # set the weights of the embedding layer as the pre-trained weights
    if vectors is not None:

        # get the first vocab_size vectors (only if vocab didn't used when calling to load_pretrained_embeddings)
        if len(vectors) > vocab_size:
            vectors = vectors[:vocab_size]

        lstm_model.embedding.weight.data.copy_(vectors)
        lstm_model.embedding.weight.data[vocab_pad_idx] = torch.zeros(embedding_dim)

    # return dictionary with all the required params to train the model.
    model = {
        'lstm': lstm_model,
        'input_rep': input_rep,
        'vocab': vocab,
        'tags_vocab': tags_vocab,
        'vocab_pad_idx': vocab_pad_idx,
        'tag_pad_idx': tag_pad_idx,
        'epochs': epochs,
        'batch_size': batch_size
    }

    return model


def load_pretrained_embeddings(path, vocab=None):
    """ Returns an object with the pre-trained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        You can also access the vectors at:
         https://www.dropbox.com/s/qxak38ybjom696y/glove.6B.100d.txt?dl=0
         (for efficiency (time and memory) - load only the vectors you need)
        The format of the vectors object is not specified as it will be used
        internally in your code, so you can use the data structure of your choice.

    Args:
        path (str): full path to the embeddings file
        vocab (list): a list of words to have embeddings for. Defaults to None.

    """
    # TODO
    # default parameters
    max_vectors = 100000
    name = '6B'
    dims = 100
    file = 'glove.{}.{}d.txt'.format(name, str(dims))
    vectors = None
    # init unknown vector with normal distribution values
    unk_func = torch.Tensor.normal_

    try:
        # get only the path without the filename
        if path.endswith(file):
            path = path.replace(file, '')
            if path == '':
                path = './'

        # get vocabulary words only
        if vocab is not None:
            embedding = torchtext.vocab.GloVe(name, cache=path, dim=dims, unk_init=unk_func)
            vectors = embedding.get_vecs_by_tokens(vocab)

        # if vocab is None return max_vectors
        else:
            embedding = torchtext.vocab.GloVe(name, cache=path, dim=dims, unk_init=unk_func, max_vectors=max_vectors)
            vectors = embedding.vectors

    except Exception as e:
        print("Error while loading embedding file.")

    return vectors


def train_rnn(model, train_data, val_data=None):
    """Trains the BiLSTM model on the specified data.

    Args:
        model (dict): the model dict as returned by initialize_rnn_model()
        train_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus()
        val_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus() to be used for validation.
                            Defaults to None
        input_rep (int): sets the input representation. Defaults to 0 (vanilla),
                         1: case-base; <other int>: other models, if you are playful
    """
    # Tips:
    # 1. you have to specify an optimizer
    # 2. you have to specify the loss function and the stopping criteria
    # 3. consider using batching
    # 4. some of the above could be implemented in helper functions (not part of
    #    the required API)

    # TODO complete the code
    # get the model, index of the padding in the tags set and the type of the lstm model.
    lstm_model = model['lstm']
    TAG_PAD_IDX = model['tag_pad_idx']
    input_rep = model['input_rep']

    # set criterion and optimizer - I am ignoring the pad index because it shouldn't affect the training
    criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX)
    optimizer = optim.Adam(lstm_model.parameters())

    lstm_model = lstm_model.to(device)
    criterion = criterion.to(device)

    # set number of epochs, batch_size, and early stopping
    N_EPOCHS = model['epochs'] if 'epochs' in model else 20
    BATCH_SIZE = model['batch_size'] if 'batch_size' in model else 16
    early_stop = model['early_stop'] if 'early_stop' in model else False

    # get batches of the training data (and validation data if val_data is not None)
    train_iterator = __get_batches(train_data, model['vocab'], model['tags_vocab'], model['vocab_pad_idx'],
                                   model['tag_pad_idx'], model['input_rep'], BATCH_SIZE)

    if val_data is not None:
        valid_iterator = __get_batches(val_data, model['vocab'], model['tags_vocab'], model['vocab_pad_idx'],
                                       model['tag_pad_idx'], model['input_rep'], BATCH_SIZE)

    # best validation loss
    best_valid_loss = float('inf')
    # set to true if early stopping is active and the model have to stop the training
    stop = False

    # run epochs
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        if input_rep == 0:
            train_loss, train_acc = __run_epoch(lstm_model, train_iterator, optimizer, criterion, TAG_PAD_IDX)
        elif input_rep == 1:
            train_loss, train_acc = __run_epoch_cases(lstm_model, train_iterator, optimizer, criterion, TAG_PAD_IDX)
        else:
            pass  # not implemented

        if val_data is not None:
            if input_rep == 0:
                valid_loss, valid_acc = __evaluate(lstm_model, valid_iterator, criterion, TAG_PAD_IDX)
            elif input_rep == 1:
                valid_loss, valid_acc = __evaluate_cases(lstm_model, valid_iterator, criterion, TAG_PAD_IDX)
            else:
                pass  # not implemented

            # update the best_valid_loss and, and if the current loss is greater than
            # the best one  stop the training (if early stop is active)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
            elif early_stop:
                stop = True

        end_time = time.time()

        epoch_mins, epoch_secs = __epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        if val_data is not None:
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

        if stop:
            print("Early stopping...")
            break


def rnn_tag_sentence(sentence, model):
    """ Returns a list of pairs (w,t) where each w corresponds to a word
        (same index) in the input sentence and t is the predicted tag.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict):  a dictionary with the trained BiLSTM model and all that is needed
                        to tag a sentence.

    Return:
        list: list of pairs
    """

    # TODO complete the code
    # set model, model type and evaluation phase.
    lstm_model = model['lstm']
    input_rep = model['input_rep']
    lstm_model.eval()

    # get indices of tokens and convert to tensor
    numericalized_tokens = [model['vocab'][word.lower()] for word in sentence]
    token_tensor = torch.LongTensor(numericalized_tokens).reshape(1, -1)
    token_tensor = token_tensor.to(device)

    if input_rep == 1:
        # get 3dimensional binary vectors for each word, convert to tensor and train the model
        cases = __get_case_vectors([word for word in sentence])
        cases_tensor = torch.LongTensor(cases).reshape(1, -1)
        cases_tensor = cases_tensor.to(device)
        predictions = lstm_model(token_tensor, cases_tensor)

    else:
        predictions = lstm_model(token_tensor)

    # get top predictions
    top_predictions = predictions.argmax(dim=-1).reshape(-1)

    # convert tags indices to real tags
    index_to_tag = model['tags_vocab'].get_itos()
    predicted_tags = [index_to_tag[t.item()] for t in top_predictions]

    # zip each word with its predicted tag
    tagged_sentence = list(zip(sentence, predicted_tags))

    return tagged_sentence


def get_best_performing_model_params():
    """Returns a disctionary specifying the parameters of your best performing
        BiLSTM model.
        IMPORTANT: this is a *hard coded* dictionary that will be used to create
        a model and train a model by calling
               initialize_rnn_model() and train_lstm()

    Return:
        model_params (dict): disctionary specifying the parameters of my best performing
        BiLSTM model.
    """
    # TODO complete the code
    model_params = {'max_vocab_size': -1,
                    'min_frequency': 2,
                    'input_rep': 1,
                    'embedding_dimension': 100,
                    'n_layers': 1,
                    'dropout': 0.25,
                    'hidden_dim': 10,
                    'epochs': 20,
                    'pretrained_embeddings_fn': '../glove.6B.100d.txt',
                    'data_fn': "../en-ud-train.upos.tsv"
                    }
    return model_params


def __get_vocab(corpus, max_size, min_freq, pad_token, unk_token):
    """Returns the vocabulary of the given corpus.

    Args:
        corpus (list): list of lists of pairs (w,t)
        max_size (int): max vocabulary size
        min_freq (int): the occurrence threshold to consider
        pad_token (str): token for padding
        unk_token (str): token for unknown words

    Return:
         vocabulary (torchtext.vocab.vocab): vocabulary of the given corpus
    """
    # get all words
    words = Counter([word.lower() for sentence in corpus for (word, tag) in sentence])

    # if max size set, get the max_size most common words
    if max_size != -1 and max_size < len(words):
        words = words.most_common(max_size)

    # sort words by frequency (if words = most_common its instance is of type list, else it is a Counter)
    if isinstance(words, Counter):
        sorted_by_freq_tuples = sorted(words.items(), key=lambda x: x[1], reverse=True)
    else:
        sorted_by_freq_tuples = sorted(words, key=lambda x: x[1], reverse=True)

    # cast into OrderedDict to use torchtext.vocab.vocab and get only words with min_freq
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    vocabulary = torchtext.vocab.vocab(ordered_dict, min_freq=min_freq)

    # append unk and padding token
    if unk_token not in vocabulary:
        vocabulary.insert_token(unk_token, 0)
    if pad_token not in vocabulary:
        vocabulary.insert_token(pad_token, 1)

    # make default index same as index of unk_token
    vocabulary.set_default_index(vocabulary[unk_token])
    return vocabulary


def __get_tags(corpus, pad_token):
    """Returns all the tags of the given corpus.

    Args:
        corpus (list): list of lists of pairs (w,t)
        pad_token (str): token for padding

    Return:
         tags_set (torchtext.vocab.vocab): tags vocab of the given corpus
    """
    # get all tags, sort by frequency and create vocabulary of tags
    all_tags = Counter([tag for sentence in corpus for (word, tag) in sentence])
    sorted_by_freq_tuples = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    tags_set = torchtext.vocab.vocab(ordered_dict)

    # append padding token
    if pad_token not in tags_set:
        tags_set.insert_token(pad_token, 1)

    return tags_set


def __get_batches(_data, vocab, tags_set, vocab_pad_idx, tag_pad_idx, input_rep, batch_size):
    """Return batches of size batch_size.

    Args:
        _data (list): list of sentences
        vocab (torchtext.vocab.vocab): words vocabulary of the given training data
        tags_set (torchtext.vocab.vocab): tags vocabulary of the given training data
        vocab_pad_idx (int): word padding index
        tag_pad_idx (int): tag padding index
        input_rep (int): 1 for case-based BLSTM, 0 for VanillaBLSTM
        batch_size (int): the size of the batches

    Return:
        batches (dict): dictionary of batches of data (contains text, tags and cases keys)

    """
    tensors = []
    labels = []
    cases = []
    batches = {}
    max_length = 0

    # convert each sentence to word indices and tags indices (if input_rep equals 1 append binary vectors)
    for sentence in _data:
        word_idx = [vocab[word.lower()] for (word, tag) in sentence]
        if len(word_idx) > max_length:
            max_length = len(word_idx)
        tag_idx = [tags_set[tag] for (word, tag) in sentence]
        tensors.append(torch.tensor(word_idx, dtype=torch.long))
        labels.append(torch.tensor(tag_idx, dtype=torch.long))
        if input_rep == 1:
            tokens = [word for (word, tag) in sentence]
            vector = __get_case_vectors(tokens)
            cases.append(vector)

    # pad each tensor with padding token/zeros according to the longest sentence (max_length)
    for index, tensor in enumerate(tensors):
        tensors[index] = torch.nn.functional.pad(tensors[index], (max_length - len(tensors[index]), 0), "constant",
                                                 vocab_pad_idx).reshape((1, max_length))
        labels[index] = torch.nn.functional.pad(labels[index],
                                                (max_length - len(labels[index]), 0), "constant", tag_pad_idx).reshape(
            (1, max_length))
        if input_rep == 1:
            cases[index] = torch.nn.functional.pad(cases[index],
                                                   (int((max_length - len(cases[index]) / 3) * 3), 0), "constant",
                                                   0).reshape(
                (1, -1))

    # concatenate all tensors into 1 tensor
    tensors = torch.cat(tensors)
    labels = torch.cat(labels)
    if input_rep == 1:
        cases = torch.cat(cases)

    # split the tensors into batches
    batches['text'] = torch.split(tensors, batch_size)
    batches['tags'] = torch.split(labels, batch_size)
    if input_rep == 1:
        batches['cases'] = torch.split(cases, batch_size)

    return batches


def __get_case_vectors(sentence):
    """Return vector of 3d binary vectors. For each word, create a 3d vector as follows:
    1. word is full lowercase: [1,0,0]
    2. word is full uppercase: [0,1,0]
    3. word is leading with capital letter: [0,0,1
    4. non of the above: [0,0,0]

    Args:
        sentence (list): list of tokens

    Return:
        vectors (tensor): flat vector with all words vectors
    """
    vector = []
    for word in sentence:
        if word == word.lower():
            vector.extend([1, 0, 0])
        elif word == word.upper():
            vector.extend([0, 1, 0])
        elif word == word.capitalize():
            vector.extend([0, 0, 1])
        else:
            vector.extend([0, 0, 0])

    vectors = torch.tensor(vector)
    return vectors


def __run_epoch(model, iterator, optimizer, criterion, tag_pad_idx):
    """Run one epoch and train the vanilla model.

    Args:
        model (object): torch.nn model.
        iterator (dict): batches as returned by __get_batches() function
        optimizer (optim object): optimizer object
        criterion (nn object): torch.nn criterion
        tag_pad_idx (int): tagging padd index

    Return:
         tuple of (epoch_loss, epoch_acc)
    """
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    # get text and tags batches
    text = iterator["text"]
    tags = iterator["tags"]

    total_batches = len(text)

    # iterate over each batch and train the model
    for text_batch, tag_batch in zip(text, tags):
        text_batch = text_batch.to(device)
        tag_batch = tag_batch.to(device)

        optimizer.zero_grad()

        predictions = model(text_batch)

        predictions = predictions.view(-1, predictions.shape[-1])
        tag_batch = tag_batch.view(-1)

        loss = criterion(predictions, tag_batch)

        acc = __categorical_accuracy(predictions, tag_batch, tag_pad_idx)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / total_batches, epoch_acc / total_batches


def __run_epoch_cases(model, iterator, optimizer, criterion, tag_pad_idx):
    """Run one epoch and train the case-based model.

    Args:
        model (object): torch.nn model.
        iterator (dict): batches as returned by __get_batches() function
        optimizer (optim object): optimizer object
        criterion (nn object): torch.nn criterion
        tag_pad_idx (int): tagging padd index

    Return:
         tuple of (epoch_loss, epoch_acc)
    """
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    # get text, tags and cases batches
    text = iterator["text"]
    tags = iterator["tags"]
    cases = iterator["cases"]

    total_batches = len(text)

    # iterate over each batch and train the model
    for text_batch, tag_batch, case_batch in zip(text, tags, cases):
        text_batch = text_batch.to(device)
        tag_batch = tag_batch.to(device)
        case_batch = case_batch.to(device)

        optimizer.zero_grad()

        predictions = model(text_batch, case_batch)

        predictions = predictions.view(-1, predictions.shape[-1])
        tag_batch = tag_batch.view(-1)

        loss = criterion(predictions, tag_batch)

        acc = __categorical_accuracy(predictions, tag_batch, tag_pad_idx)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / total_batches, epoch_acc / total_batches


def __evaluate(model, iterator, criterion, tag_pad_idx):
    """Evaluate the vanilla model on the given iterator.

    Args:
        model (object): torch.nn model.
        iterator (dict): batches as returned by __get_batches() function
        criterion (nn object): torch.nn criterion
        tag_pad_idx (int): tagging pad index

    Return:
         tuple of (epoch_loss, epoch_acc)
    """
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    text = iterator["text"]
    tags = iterator["tags"]
    total_batches = len(text)

    with torch.no_grad():
        for text_batch, tag_batch in zip(text, tags):
            text_batch = text_batch.to(device)
            tag_batch = tag_batch.to(device)

            predictions = model(text_batch)

            predictions = predictions.view(-1, predictions.shape[-1])
            tag_batch = tag_batch.view(-1)

            loss = criterion(predictions, tag_batch)

            acc = __categorical_accuracy(predictions, tag_batch, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / total_batches, epoch_acc / total_batches


def __evaluate_cases(model, iterator, criterion, tag_pad_idx):
    """Evaluate the case-based model on the given iterator.

    Args:
        model (object): torch.nn model.
        iterator (dict): batches as returned by __get_batches() function
        criterion (nn object): torch.nn criterion
        tag_pad_idx (int): tagging pad index

    Return:
         tuple of (epoch_loss, epoch_acc)
    """
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    text = iterator["text"]
    tags = iterator["tags"]
    cases = iterator["cases"]

    total_batches = len(text)

    with torch.no_grad():
        for text_batch, tag_batch, case_batch in zip(text, tags, cases):
            text_batch = text_batch.to(device)
            tag_batch = tag_batch.to(device)
            case_batch = case_batch.to(device)

            predictions = model(text_batch, case_batch)

            predictions = predictions.view(-1, predictions.shape[-1])
            tag_batch = tag_batch.view(-1)

            loss = criterion(predictions, tag_batch)

            acc = __categorical_accuracy(predictions, tag_batch, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / total_batches, epoch_acc / total_batches


def __categorical_accuracy(preds, y, tag_pad_idx):
    """Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8

    Args:
        preds (list): the model predictions
        y (list): the true tags
        tag_pad_idx (int): index of the tag padding

    Return:
        accuracy (float): the fraction of the corrected predictions
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / y[non_pad_elements].shape[0]


def __epoch_time(start_time, end_time):
    """Returns the total time elapsed between start_time and end_time

    Args:
        start_time (float): start time
        end_time (float): end time

    Return:
         tuple of (min, sec) time elapsed
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# ===========================================================
#       Wrapper function (tagging with a specified model)
# ===========================================================

def tag_sentence(sentence, model):
    """Returns a list of pairs (w,t) where pair corresponds to a word (same index) in
    the input sentence. Tagging is done with the specified model.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict): a dictionary where key is the model name and the value is
           an ordered list of the parameters of the trained model (baseline, HMM)
           or the model isteld and the input_rep flag (LSTMs).

        Models that must be supported (you can add more):
        1. baseline: {'baseline': [perWordTagCounts, allTagCounts]}
        2. HMM: {'hmm': [A,B]}
        3. Vanilla BiLSTM: {'blstm':[model_dict]}
        4. BiLSTM+case: {'cblstm': [model_dict]}
        5. (NOT REQUIRED: you can add other variations, agumenting the input
            with further subword information, with character-level word embedding etc.)

        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the HMM are:
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

        Parameters for an LSTM: the model dictionary (allows tagging the given sentence)


    Return:
        list: list of pairs
    """
    if list(model.keys())[0] == 'baseline':
        return baseline_tag_sentence(sentence, list(model.values())[0][0], list(model.values())[0][1])
    if list(model.keys())[0] == 'hmm':
        return hmm_tag_sentence(sentence, list(model.values())[0][0], list(model.values())[0][1])
    if list(model.keys())[0] == 'blstm':
        return rnn_tag_sentence(sentence, list(model.values())[0][0])
    if list(model.keys())[0] == 'cblstm':
        return rnn_tag_sentence(sentence, list(model.values())[0][0])


def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correcttly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    Return:
        tuple of three values: (correct, correctOOV, OOV)
    """
    assert len(gold_sentence) == len(pred_sentence)

    # TODO complete the code
    # get all training words
    words_set = [word for word in perWordTagCounts.keys()]
    # get all oov words from the given sentence - if they do not exist in words_set
    oov_words = [word for (word, tag) in gold_sentence if word.lower() not in words_set and word not in words_set]

    OOV = len(oov_words)
    correct = 0
    correctOOV = 0
    for index, (word, tag) in enumerate(gold_sentence):
        if word in oov_words:
            if tag == pred_sentence[index][1]:
                correctOOV += 1
                correct += 1
        elif tag == pred_sentence[index][1]:
            correct += 1

    return correct, correctOOV, OOV
