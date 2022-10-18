from tagger import load_annotated_corpus, get_best_performing_model_params, initialize_rnn_model, train_rnn, \
    learn_params, tag_sentence, use_seed

# NOTE: you can delete the imports, and copy & paster this driver to the end of the tagger.py file
# NOTE: this driver trains the models on all the train data without validation set, and test it on all the test data
# NOTE: config N_EPOCH & BATCH_SIZE as you like
# NOTE: you can use diffrent seed
# IMPORTANT NOTE:
# READ THE NOTE WRITTEN ABOVE THE test_lstm FUNCTION( later in this file), ABOUT THE "get_best_performing_model_params"

# Parameters for the lstm model training
BATCH_SIZE = 128
N_EPOCHS = 1
use_seed(seed=1234)


def get_acc_on_sentences(tagged_sentences, model):
    """This function receives a list of tagged sentences(list of pairs),
     and a model dict we should send to "tag_sentence" function
    The function iterates over the tagged sentence and count  return the accuracy of the model received
    Args:
        tagged_sentences(list): list of pairs (word, tag)

    Returns:
        accuracy(float): the accuracy of the model on the test set
    """
    num_of_correct = 0
    all_words = 0

    for i, original_tagged_sentence in enumerate(tagged_sentences):
        if i % 500 == 0:
            print(i)
        pred_tagged_sentence = tag_sentence(sentence=[x[0] for x in original_tagged_sentence], model=model)
        all_words += len(original_tagged_sentence)
        for original, predict in zip([x[1] for x in original_tagged_sentence], [x[1] for x in pred_tagged_sentence]):
            if original == predict:
                num_of_correct += 1

    return num_of_correct / all_words


# Note: Copy this function to "tagger.py" to make it work
def test_base(allTagCounts, perWordTagCounts):
    baseline_dict = {'baseline': [perWordTagCounts, allTagCounts]}
    print('baseline:', get_acc_on_sentences(test_tagged_sentences, baseline_dict))


# Note: Copy this function to "tagger.py" to make it work
def test_hmm(A, B):
    hmm_dict = {'hmm': [A, B]}
    print('hmm:', get_acc_on_sentences(test_tagged_sentences, hmm_dict))


# Note, "get_best_performing_model_params" gets the input_rep, only for the driver convenience,
# when submitting the project, "get_best_performing_model_params" shouldn't receive any arguments
def test_lstm(input_rep):
    import pickle

    model_params = get_best_performing_model_params()
    model_params['input_rep'] = input_rep
    model_dict = initialize_rnn_model(model_params)
    print('Training blstm')
    train_rnn(model_dict, train_tagged_sentences)
    if input_rep == 0:
        vanilia_blstm_dict = {'blstm': [model_dict]}

        with open(f'blstm_model_dict_trained_{N_EPOCHS}.pickle', 'wb') as handle:
            pickle.dump(vanilia_blstm_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("blstm accuracy:", get_acc_on_sentences(test_tagged_sentences, vanilia_blstm_dict))
    else:
        cblstm_dict = {'cblstm': [model_dict]}

        with open(f'cblstm_model_dict_trained_{N_EPOCHS}.pickle', 'wb') as handle:
            pickle.dump(cblstm_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("cblstm accuracy:", get_acc_on_sentences(test_tagged_sentences, cblstm_dict))


train_tagged_sentences = load_annotated_corpus('en-ud-train.upos.tsv')
test_tagged_sentences = load_annotated_corpus('en-ud-dev.upos.tsv')
allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B = learn_params(train_tagged_sentences)
test_base(allTagCounts, perWordTagCounts)
test_hmm(A, B)
test_lstm(input_rep=0)
test_lstm(input_rep=1)
