import unittest
import tagger


class TestTagger(unittest.TestCase):
    sentences = None
    allTagCounts = None
    perWordTagCounts = None
    transitionCounts = None
    emissionCounts = None
    A = None
    B = None

    @classmethod
    def setUpClass(cls):
        # path = "../en-ud-train.upos.tsv"
        path = "../tests/test_dataset.tsv"
        cls.sentences = tagger.load_annotated_corpus(path)
        all_params = tagger.learn_params(cls.sentences)
        cls.allTagCounts = all_params[0]
        cls.perWordTagCounts = all_params[1]
        cls.transitionCounts = all_params[2]
        cls.emissionCounts = all_params[3]
        cls.A = all_params[4]
        cls.B = all_params[5]

    def test_1_read(self):
        print("Test sentences: ", self.sentences)

    def test_2_learn_params(self):
        # test_3_all_tag_counts:
        self.assertEqual(self.allTagCounts["PUNCT"], 11)
        self.assertEqual(self.allTagCounts["NOUN"], 11)
        self.assertEqual(self.allTagCounts["VERB"], 6)

        # test_4_word_tag_counts:
        self.assertEqual(self.perWordTagCounts["of"]["ADP"], 2)
        self.assertEqual(self.perWordTagCounts["."]["PUNCT"], 3)
        self.assertEqual(self.perWordTagCounts["the"]["DET"], 4)

        # def test_transition_counts:
        self.assertEqual(self.transitionCounts["NOUN"]["ADP"], 5)
        self.assertEqual(self.transitionCounts["NOUN"]["VERB"], 3)
        self.assertEqual(self.transitionCounts["PUNCT"]["DET"], 2)

        # test_emission_counts:
        self.assertEqual(self.emissionCounts["ADP"]["of"], 2)
        self.assertEqual(self.emissionCounts["PUNCT"]["."], 3)
        self.assertEqual(self.emissionCounts["DET"]["the"], 4)

    def test_3_baseline(self):
        tokens = ["near", "the", "beautiful", "Syrian", "border"]
        pos_tags = tagger.baseline_tag_sentence(tokens, self.perWordTagCounts, self.allTagCounts)
        print(pos_tags)

    def test_4_hmm(self):
        tokens = ["near", "the", "beautiful", "Syrian", "border"]
        pos_tags = tagger.hmm_tag_sentence(tokens, self.A, self.B)
        print(pos_tags)

    def test_5_embedding(self):
        path = '../glove.6B.100d.txt'
        a = tagger.load_pretrained_embeddings(path, None)
        b = tagger.load_pretrained_embeddings(path, vocab=['the', 'hotel', 'was', 'here', 'amit'])
        print("hi")

    def test_6_init(self):
        params_d = {'max_vocab_size': 300,
                    'min_frequency': 5,
                    'input_rep': 0,
                    'embedding_dimension': 100,
                    'n_layers': 4,
                    'dropout': 0.25,
                    'hidden_dim': 128,
                    'pretrained_embeddings_fn': '../glove.6B.100d.txt',
                    'data_fn': "../en-ud-train.upos.tsv"
                    }
        train_data = tagger.load_annotated_corpus(params_d['data_fn'])
        test_path = "../en-ud-dev.upos.tsv"
        test_sentences = tagger.load_annotated_corpus(test_path)
        model_dict = tagger.initialize_rnn_model(params_d)
        tagger.train_rnn(model_dict, train_data)
        model_blstm = {'blstm': [model_dict]}
        models = [model_blstm]
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
                                              'OOV words': total_oov_words,
                                              'OOV Accuracy': (oov_correct / total_oov_words)}

        for item in results.items():
            print(item)

if __name__ == "__main__":
    unittest.main()
