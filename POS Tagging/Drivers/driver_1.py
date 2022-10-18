import tagger

tagger.use_seed()
train_data = tagger.load_annotated_corpus('en-ud-train.upos.tsv')
dev_data = tagger.load_annotated_corpus('en-ud-dev.upos.tsv')

allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B = tagger.learn_params(train_data)
params = tagger.get_best_performing_model_params()
params['input_rep'] = 0
model_blstm = tagger.initialize_rnn_model(params)
tagger.train_rnn(model_blstm, train_data)

params = tagger.get_best_performing_model_params()
model_cblstm = tagger.initialize_rnn_model(params)
tagger.train_rnn(model_cblstm, train_data)

for model in ['baseline', 'hmm', 'blstm', 'cblstm']:
    sum_correct, sum_examples, sum_correctOOV, sum_OVV = 0, 0, 0, 0
    for test_sentence in dev_data:
        words_to_tag = [word for word, tag in test_sentence]
        if model == 'baseline':
            tagged_sentence = tagger.tag_sentence(words_to_tag, {'baseline': [perWordTagCounts, allTagCounts]})
        elif model == 'hmm':
            tagged_sentence = tagger.tag_sentence(words_to_tag, {'hmm': [A, B]})
        elif model == 'blstm':
            tagged_sentence = tagger.tag_sentence(words_to_tag, {'blstm': [model_blstm]})
        elif model == 'cblstm':
            tagged_sentence = tagger.tag_sentence(words_to_tag, {'cblstm': [model_cblstm]})

        correct, correctOOV, OOV = tagger.count_correct(test_sentence, tagged_sentence)
        sum_correct += correct
        sum_OVV += OOV
        sum_correctOOV += correctOOV
        sum_examples += len(words_to_tag)

    print(f"{model} - {sum_correct / sum_examples} \t sum OOV: {sum_OVV}, correctOOV {sum_correctOOV}")
