import sys
import importlib
import collections
import ex1


def test1():
    t = 'A cat sat on the mat. A fat cat sat on the mat. A rat sat on the mat. The rat sat on the cat. A bat spat on the rat that sat on the cat on the mat.'
    return t


def test2():
    with open("../corpora/big.txt", 'r', errors='ignore') as file:
        t = file.read()
        return t


if __name__ == '__main__':
    text = test1()
    nt = ex1.normalize_text(text)  # lower casing, padding punctuation with white spaces
    print(nt)

    lm = ex1.Ngram_Language_Model(n=3, chars=True)
    lm.build_model(nt)  # *
    print(lm.get_model_dictionary())  # *
    t = lm.generate(context='a cat', n=30)
    for e in [t, 'a cat sat on the mat . a fat cat sat on the mat . a bat spat on the mat . a rat sat on the mat .', 'a cat sat on the mat', 'the rat sat on the cat']:  # *
        print('%s | %.3f' % (e, lm.evaluate(e)))