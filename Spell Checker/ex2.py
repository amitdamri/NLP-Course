from collections import defaultdict, Counter
import re
import math


class Spell_Checker:
    """The class implements a context sensitive spell checker. The corrections
            are done in the Noisy Channel framework, based on a language model and
            an error distribution model.
    """

    def __init__(self, lm=None):
        """Initializing a spell checker object with a language model as an
        instance  variable. The language model should support the evaluate()
        and the get_model() functions as defined in assignment #1.

        Args:
            lm: a language model object. Defaults to None
        """
        self.lm                 = lm
        self.error_tables       = dict()
        self.dictionary         = Counter()
        self.char_unigram_count = Counter()
        self.char_bigram_count  = Counter()

    def build_model(self, text, n=3):
        """Returns a language model object built on the specified text. The language
            model should support evaluate() and the get_model() functions as defined
            in assignment #1.

            Args:
                text (str): the text to construct the model from.
                n (int): the order of the n-gram model (defaults to 3).

            Returns:
                A language model object
        """
        if n is None or n < 1:
            n = 1

        if text is None:
            return self.lm

        norm_text   = self.__normalize_text(text, n)
        n_grams     = self.__get_n_grams(norm_text, n)

        if self.lm:
            model_dict  = self.lm.get_model()
            count_ngram = defaultdict(int, Counter(n_grams))

            for key, val in count_ngram.items():
                model_dict[key] = val

            self.dictionary         = self.__create_dictionary(self.lm)
            self.char_unigram_count = self.__create_char_unigram_counter(self.lm)
            self.char_bigram_count  = self.__create_char_bigram_counter(self.lm)

        return self.lm

    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM dictionary and unigrams + bigrams dictionaries if set)

            Args:
                lm: a language model object
        """
        if lm is not None:
            self.lm                 = lm
            self.dictionary         = self.__create_dictionary(self.lm)
            self.char_unigram_count = self.__create_char_unigram_counter(self.lm)
            self.char_bigram_count  = self.__create_char_bigram_counter(self.lm)

    def learn_error_tables(self, errors_file):
        """Returns a nested dictionary {str:dict} where str is in:
            <'deletion', 'insertion', 'transposition', 'substitution'> and the
            inner dict {str: int} represents the confusion matrix of the
            specific errors, where str is a string of two characters matching the
            row and column "indices" in the relevant confusion matrix and the int is the
            observed count of such an error (computed from the specified errors file).
            Examples of such string are 'xy', for deletion of a 'y'
            after an 'x', insertion of a 'y' after an 'x'  and substitution
            of 'x' (incorrect) by a 'y'; and example of a transposition is 'xy'
            indicates the characters that are transposed.


            Notes:
                1. Ultimately, one can use only 'deletion' and 'insertion' and have
                    'substitution' and 'transposition' derived. Again,  we use all
                    four types explicitly in order to keep things simple.
            Args:
                errors_file (str): full path to the errors file. File format, TSV:
                                    <error>    <correct>


            Returns:
                A dictionary of confusion "matrices" by error type (dict).
        """
        confusion_dict = {'deletion': defaultdict(int), 'insertion': defaultdict(int),
                          'transposition': defaultdict(int), 'substitution': defaultdict(int)}

        if errors_file is not None:

            try:
                with open(errors_file, 'r') as file:
                    for line in file:
                        if line != '':
                            words = line.strip('\n').split('\t')
                            typo, _type = self.__get_difference_btw_strings(words[0], words[1])
                            confusion_dict[_type][typo] += 1

                self.add_error_tables(confusion_dict)

            except OSError as e:
                pass

        return confusion_dict

    def add_error_tables(self, error_tables):
        """ Adds the speficied dictionary of error tables as an instance variable.
            (Replaces an older value disctionary if set)

            Args:
                error_tables (dict): a dictionary of error tables in the format
                returned by  learn_error_tables()
        """
        if error_tables is not None:
            self.error_tables = error_tables

    def evaluate(self, text):
        """Returns the log-likelihod of the specified text given the language
            model in use. Smoothing is applied on texts containing OOV words

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        if self.lm is not None:
            return self.lm.evaluate(text)
        else:
            return 0

    def spell_check(self, text, alpha):
        """ Returns the most probable fix for the specified text. Use a simple
            noisy channel model if the number of tokens in the specified text is
            smaller than the length (n) of the language model.

            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.

            Return:
                A modified string (or a copy of the original if no corrections are made.)
        """
        fix_candid = self.__generate_spelling_candidates(text, alpha)
        return max(fix_candid.keys(), key=lambda k: fix_candid.get(k, 0))

    """Private Methods"""

    def __create_dictionary(self, lm):
        """Initiates the vocabulary dictionary of the spell checker model based on the language model n-grams.
        The function gets all the n-grams from the lm, and finds only valid words with length greater than two
        (except specific words with only one letter: 'a', 'I'). Finally, returns the final dictionary with all
        relevant words and their counts.

            Args:
                lm (object): a language model object which implements the evaluate() and get_model() functions.

            Return:
                 counter dict. counter dictionary of the model vocabulary based on the language model's n-gram.
        """
        dictionary = Counter()

        if lm is not None:
            model_dict = lm.get_model()
            all_lm_dict = " ".join([k for k in model_dict.keys()])

            # choose valid words ('a', 'i', and two and more letter words includes ' and - words.
            only_words = re.findall(r"\W+[ai]\W+|\w+['-]?\w+", all_lm_dict)

            # remove whitspaces
            only_words = [word.strip() for word in only_words]
            dictionary.update(only_words)

        return dictionary

    def __create_char_unigram_counter(self, lm):
        """Initiates the unigram char dictionary of the spell checker model based on the language model n-grams.
        The function gets all the n-grams from the lm, separates them by whitespaces, adds '#' at the beginning
        of each word (in order to normalize errors like thello -> hello, error: '#t'). Finally, returns the dict which
        contains the counter of each char.

            Args:
                lm (object): a language model object which implements the evaluate() and get_model() functions.

            Return:
                 counter dict. counter with unigram chars as keys and their counts as value.
        """
        if lm is not None:
            model_dict  = lm.get_model()
            agg_dict    = defaultdict(int)
            for k, v in model_dict.items():
                agg_dict['#' + k.split(' ')[0]] += v

            # Multiple each word by the counter of the n-gram, to count all appearances of these chars
            # and join all words to one sentence to easily count the chars.
            long_str = "".join([k * v for k, v in agg_dict.items()])
            uni_dict = Counter(long_str)
            return uni_dict

        return Counter()

    def __create_char_bigram_counter(self, lm):
        """Initiates the bigram chars dictionary of the spell checker model based on the language model n-grams.
        1. The function gets all the n-grams from the lm
        2. separates them by whitespaces
        3. gets the first word of each n-gram - counting without duplications (I assume that the model could miss
           the last n-1 words but it doesn't have an affect on the results because the corpus is large enough).
        4. adds '#' at the beginning of each word (in order to normalize errors like ello -> hello, error: '#h')
        5. adds whitespace at the end (to normalize errors like allyears -> all years, possible error: 'l ').

        Finally, returns the dict which contains the counter of each two chars.

            Args:
                lm (object): a language model object which implements the evaluate() and get_model() functions.

            Return:
                 counter dict. counter with bigram chars as keys and their counts as value.
        """
        if lm is not None:
            model_dict  = lm.get_model()
            agg_dict    = defaultdict(int)
            for k, v in model_dict.items():
                agg_dict['#' + k.split(' ')[0] + " "] += v

            # get every two adjacent chars and adds them to the list according to the number of the n-gram counter.
            bigram_word_chars = [k[j] + k[j + 1] for k, v in agg_dict.items() for i in range(v) for j in
                                 range(len(k) - 1)]

            bi_dict = Counter(bigram_word_chars)
            return bi_dict

        return Counter()

    def __get_n_grams(self, text, n):
        """Return the n-gram of word tokens of the given text.

            Args:
                text (str): the text to construct the n-grams of word tokens from.

            Returns:
                list:   the n-grams list.
        """
        tokens  = re.split(' ', text)
        n_grams = [' '.join(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]
        return n_grams

    def __generate_spelling_candidates(self, text, alpha):
        """Returns dictionary of new fixed sentences with their probabilities to fix the given text.
        1. first the text is normalized to find word in dictionary: Hello -> hello
        2. find all valid words
        3. check if all words are valid
            3.1 if yes, calculate for each word its candidates and the noisy chanel probabilities for each.
            3.2 if not, find the erroneous word (I assume there is only one erroneous word) and
            generate candidates for this word and calculate their noisy chanel probs.
        4. Finally switch between the candidate and the word (keep punctuations if exists).
        5. calculate the language model probs and add those probs with the noisy channel probs.
        6. return candidate sentences and thier probs.

            Args:
                text    (str):  the text to spell check.
                alpha (float):  the probability of keeping a lexical word as is.

            Return:
                dict. dictionary of optional sentences and their probabilities.
        """
        norm_text   = self.__normalize_text(text)
        all_words   = norm_text.split(' ')
        sent_probs  = defaultdict(int)
        only_words  = re.findall(r"\w+['-]?\w+", norm_text)

        # get the n of the model from the n-grams
        model_n = len(list(self.lm.get_model().keys())[0].split(' '))
        is_real = [self.__is_dictionary_word(word) for word in only_words]


        if all(is_real):

            for index, word in enumerate(all_words):

                if word in only_words:

                    candid_words = defaultdict(int)
                    word_candid  = self.__calculate_candidates(word)

                    for candidate in word_candid + [(word, 'None')]:
                        noisy_prob = self.__get_noisy_channel_real_word_prob(word, candidate[0], word_candid, alpha,
                                                                             candidate[1])
                        candid_words[candidate[0]] += noisy_prob

                    for cand in candid_words.keys():
                        sent_candid = ' '.join(all_words[:index] + [cand] + all_words[index + 1:])

                        if len(all_words) >= model_n:
                            lm_prob     = self.evaluate(sent_candid)
                        else:
                            smooth_num  = self.dictionary[cand] + 1
                            smooth_den  = sum(self.dictionary.values()) + len(self.dictionary.keys())
                            lm_prob     = smooth_num / smooth_den

                        sent_probs[sent_candid] = math.log(candid_words[cand]) + lm_prob

        else:

            wrong_word = only_words[is_real.index(False)]
            word_index = all_words.index(wrong_word)

            candid_words = defaultdict(int)
            word_candid  = self.__calculate_candidates(wrong_word)

            for candidate in word_candid:
                noisy_prob = self.__get_non_word_noisy_channel_prob(wrong_word, candidate[0], candidate[1])
                candid_words[candidate[0]] += noisy_prob

            for cand in candid_words.keys():
                sent_candid = ' '.join(all_words[:word_index] + [cand] + all_words[word_index + 1:])

                if len(all_words) >= model_n:
                    lm_prob = self.evaluate(sent_candid)
                else:
                    smooth_num = self.dictionary[cand] + 1
                    smooth_den = sum(self.dictionary.values()) + len(self.dictionary.keys())
                    lm_prob = smooth_num / smooth_den

                sent_probs[sent_candid] = math.log(candid_words[cand]) + lm_prob

        # if no candidates found
        if len(sent_probs) == 0:
            sent_probs[text] = 0

        return sent_probs

    def __calculate_candidates(self, word):
        """All edits that are one edit away from `word`.

            Args:
                word (str): calculate candidates for this word

            Return:
                list. list of all real word candidates of the given word.
        """
        # contains also words with ' and two words which missing whitespace
        letters     = "abcdefghijklmnopqrstuvwxyz' -"
        splits      = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes     = [(L + c + R, 'deletion') for L, R in splits for c in letters]
        inserts     = [(L + R[1:], 'insertion') for L, R in splits if R]
        sub         = [(L + c + R[1:], 'substitution') for L, R in splits if R for c in letters if c != R[0]]
        trans       = [(L + R[1] + R[0] + R[2:], 'transposition') for L, R in splits if len(R) > 1 and R[0] != R[1]]
        candidates  = deletes + inserts + sub + trans
        real_w      = [candidate for candidate in candidates if self.__is_dictionary_word(candidate[0])]
        return real_w

    def __is_dictionary_word(self, word):
        """Check if the input word is a real word (if the word exists in the model dictionary)

            Args:
                word (str): check if valid word.

            Return:
                Boolean. true if the word is a real word.
        """
        if word is not None:

            if ' ' in word:
                split = word.split(' ')
                return split[0] in self.dictionary and split[1] in self.dictionary
            else:
                return word in self.dictionary

        else:
            return False

    def __get_noisy_channel_real_word_prob(self, word, candidate, candidates, alpha, table_type=None):
        """Return the noisy channel probability.

            Args:
                word        (str): real word from the given sentence.
                candidate   (str)  candidate fot the given word.
                candidates  (list)  list of all candidates of the given word.
                alpha       (float) the probability to keep the original word.
                table_type  (str)   the type of the error.

            Return:
                float. the probability of the noisy channel.
        """
        if word == candidate:
            return alpha
        else:
            uni_dist = (1 - alpha) / len(candidates)

            # multiply the unified dist with the error probability
            prob = self.__get_non_word_noisy_channel_prob(word, candidate, table_type)
            return uni_dist * prob

    def __get_non_word_noisy_channel_prob(self, error, correct, table_type=None):
        """Return noisy channel probability of non words - the original word does not exists in the dictionary.

            Args:
                error   (str):  original word - non word.
                correct (str):  candidate word from the dictionary.
                table_type  (str): the type of the error

            Return:
                float.  noisy channel probability of a candidate (valid) for a non valid word (error).

        """
        if table_type:
            typo_chars, table_type = self.__get_difference_btw_strings(error, correct, table_type)
        else:
            typo_chars, table_type = self.__get_difference_btw_strings(error, correct)

        # smooth the numerator by +1 and the denominator by + total_of_errors in order to
        # eliminate from unknown errors.
        numerator   = self.__get_confusion_matrix_value(typo_chars, table_type) + 1
        total_errors_smoothing = sum([sum(self.error_tables[table].values()) for table in self.error_tables])
        denominator = self.__get_typo_count(typo_chars, table_type) + total_errors_smoothing
        return numerator / denominator

    def __get_confusion_matrix_value(self, typo_chars, table_type):
        """Return the number of typo_chars errors given the table_type.

            Args:
                typo_chars  (str): the error chars x and y columns.
                table_type  (str): the type of the error

            Return:
                int.    the value of the typo_chars error from the table_type confusion matrix.
        """
        if typo_chars in self.error_tables[table_type]:
            return self.error_tables[table_type][typo_chars]
        else:
            return 0

    def __get_typo_count(self, typo_chars, table_type):
        """Return the count (the denominator of the noisy channel) of the given error
        (typo chars) based on the error type (table_type).

            Args:
                typo_chars  (str): the error chars x and y columns.
                table_type  (str): the type of the error

            Return:
                int. counter of the given chars in the corpus.

        """
        unigram_table   = ['insertion', 'substitution']
        bigram_table    = ['deletion', 'transposition']

        if table_type in unigram_table:
            typo_char = typo_chars[0] if table_type == 'insertion' else typo_chars[1]

            if typo_char in self.char_unigram_count:
                return self.char_unigram_count[typo_char]
            else:
                return 0

        elif table_type in bigram_table and typo_chars in self.char_bigram_count:
            return self.char_bigram_count[typo_chars]
        else:
            return 0

    def __get_difference_btw_strings(self, error, correct, table_type=None):
        """Returns tuple of the difference and the difference type between
        the two given strings (error, correct). The type of the difference can be
        one of 4 types of errors: insertion, deletion, substitution and transposition.

            Args:
                error       (str):  string of the error word.
                correct     (str):  string of the correct word.
                table_type  (str):  the type of the difference (error). default value None,
                                    given only when the error type is known (when generating candidates).

            Return:
                tuple. tuple of (error, error_type). the error is of two chars and the error_type
                has 4 options: <insertion, deletion, transposition, substitution>
        """
        functions = {'deletion': self.__get_difference_deletion, 'insertion': self.__get_difference_insertion,
                     'transposition': self.__get_difference_transposition,
                     'substitution': self.__get_difference_substitution}

        if table_type:
            typo = functions[table_type](error, correct)
            return typo, table_type

        if len(error) > len(correct):  # insertion
            typo = self.__get_difference_insertion(error, correct)
            return typo, 'insertion'

        elif len(correct) > len(error):  # deletion
            typo = self.__get_difference_deletion(error, correct)
            return typo, 'deletion'

        elif sorted(error) == sorted(correct):  # transposition
            typo = self.__get_difference_transposition(error, correct)
            return typo, 'transposition'

        else:
            typo = self.__get_difference_substitution(error, correct)
            return typo, 'substitution'

    def __get_difference_deletion(self, error, correct):
        """Returns the the deletion difference between the two given strings (error, correct).

            Args:
                error       (str):  string of the error word.
                correct     (str):  string of the correct word.

            Return:
                str. string of two chars - the char which was deleted and its adjacent next char.
        """
        if len(error) < len(correct):
            for i in range(len(correct)):
                if i < len(error) and correct[i] != error[i]:
                    x = '#' if i == 0 else correct[i - 1]
                    y = correct[i]
                    return x + y

                elif i >= len(error):
                    x = correct[i - 1]
                    y = correct[i]
                    return x + y

    def __get_difference_insertion(self, error, correct):
        """Returns the the insertion difference between the two given strings (error, correct).

            Args:
                error       (str):  string of the error word.
                correct     (str):  string of the correct word.

            Return:
                str. string of two chars - the previous adjacent correct char and the error char which was inserted.
        """
        if len(error) > len(correct):

            for i in range(len(error)):
                if i < len(correct) and error[i] != correct[i]:
                    x = '#' if i == 0 else correct[i - 1]
                    y = error[i]
                    return x + y
                elif i >= len(correct):
                    x = correct[i - 1]
                    y = error[i]
                    return x + y

    def __get_difference_transposition(self, error, correct):
        """Returns the the transposition difference between the two given strings (error, correct).

            Args:
                error       (str):  string of the error word.
                correct     (str):  string of the correct word.

            Return:
                str. string of two chars - the correct order of the two chars (based on the correct string).
        """
        if len(error) == len(correct):
            for i in range(len(correct)):
                if correct[i] != error[i] and correct[i] == error[i + 1]:
                    x = correct[i]
                    y = correct[i + 1]
                    return x + y

    def __get_difference_substitution(self, error, correct):
        """Returns the the substitution difference between the two given strings (error, correct).

            Args:
                error       (str):  string of the error word.
                correct     (str):  string of the correct word.

            Return:
                str. string of two chars - the wrong i char and the right i char.
        """
        if len(error) == len(correct):
            for i in range(len(correct)):
                if correct[i] != error[i]:
                    x = error[i]
                    y = correct[i]
                    return x + y

    def __normalize_text(self, text, n=1):
        """Returns a normalized version of the specified string.
        The process of normalization is as follows:
        1. lower case -  transform all words into their lower case to map same words (The, the).
        2. pad punctuations with whitespaces - separate between words and punctuations

              Args:
                text (str): the text to normalize

              Returns:
                string: the normalized text.
        """

        if text is not None:
            # transform text to lower case
            norm_text = text.lower()

            # convert tab and newline to . I guess it describes an end of a sentence.
            norm_text = re.sub(r"[\n\t]", '.', norm_text)

            # pad all punctuations with whitespaces
            norm_text = re.sub(
                r'([_/<>%\"`~.,!?(){\}:;&\[\]$^*\\\n\t])', r' \1 ', norm_text)

            # padd ' if it appears in the beginning or in the end without continuation
            norm_text = re.sub(r"(^\')|( \')|(\' )", r" ' ", norm_text)

            # change -- to  - and padd if appears in the beginning or end without second word
            norm_text = re.sub(r"--", r" - ", norm_text)
            norm_text = re.sub(r"(^-)|( \-)|(\- )", r" - ", norm_text)

            # remove double whitespaces
            norm_text = re.sub(r'\s{2,}', ' ', norm_text)
            norm_text = norm_text.strip()

            return norm_text

        else:
            return text


def who_am_i():
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Amit Damri', 'id': '312199698', 'email': 'amitdamr@post.bgu.ac.il'}
