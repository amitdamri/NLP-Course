import tagger


def main():
    train_path = "../en-ud-train.upos.tsv"
    test_path = "../en-ud-dev.upos.tsv"

    print("Start Training...")
    tagger.use_seed()
    train_sentences = tagger.load_annotated_corpus(train_path)
    test_sentences = tagger.load_annotated_corpus(test_path)

    all_params = tagger.learn_params(train_sentences)
    allTagCounts = all_params[0]
    perWordTagCounts = all_params[1]
    transitionCounts = all_params[2]
    emissionCounts = all_params[3]
    A = all_params[4]
    B = all_params[5]
    print("Finish training...")

    params_d1 = {'max_vocab_size': -1,
                'min_frequency': 2,
                'input_rep': 0,
                'embedding_dimension': 100,
                'n_layers': 1,
                'dropout': 0.25,
                'hidden_dim': 10,
                'epochs': 20,
                'pretrained_embeddings_fn': '../glove.6B.100d.txt',
                'data_fn': "../en-ud-train.upos.tsv"
                }

    params_d2 = {'max_vocab_size': -1,
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

    model_dict_1 = tagger.initialize_rnn_model(params_d1)
    tagger.train_rnn(model_dict_1, train_sentences, test_sentences)

    model_dict_2 = tagger.initialize_rnn_model(params_d2)
    tagger.train_rnn(model_dict_2, train_sentences, test_sentences)

    model_base = {'baseline': [perWordTagCounts, allTagCounts]}
    model_hmm = {'hmm': [A, B]}
    model_blstm = {'blstm': [model_dict_1]}
    model_cblstm = {'cblstm': [model_dict_2]}
    models = [model_base, model_hmm, model_blstm, model_cblstm]
    # models = [model_base, model_hmm, model_blstm]
    # models = [model_blstm, model_cblstm]
    results = {}

    print("Start Testing...")
    for model in models:
        print("Model: ", list(model.keys())[0])

        total_words = 0
        total_oov_words = 0
        words_correct = 0
        oov_correct = 0

        for sentence in test_sentences:
            tokens = [word for (word, tag) in sentence]
            tagged = tagger.tag_sentence(tokens, model)
            correct, correct_oov, oov = tagger.count_correct(sentence, tagged)
            total_words += len(sentence)
            total_oov_words += oov
            words_correct += correct
            oov_correct += correct_oov

        results[list(model.keys())[0]] = {'Total Words': total_words, 'Accuracy': (words_correct / total_words),
                                          'OOV words': total_oov_words, 'OOV Accuracy': (oov_correct / total_oov_words)}

    for item in results.items():
        print(item)

    for model in models:
        print("Model: ", list(model.keys())[0])

        tokens = [word for (word, tag) in test_sentences[0]]
        tagged = tagger.tag_sentence(tokens, model)
        print(tagged)
        print(test_sentences[0])


if __name__ == '__main__':
    main()
