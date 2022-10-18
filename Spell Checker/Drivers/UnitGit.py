# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:53:31 2019

@author: mor
"""

from TestGit import *

corpus_path = 'Corpora/big.txt'
errors_file = 'Corpora/common_errors.txt'

with open(corpus_path, 'r') as f:
    text = f.read()

norm_text = normalize_text(text)

sc = Spell_Checker()
model = sc.build_model(norm_text)
d = sc.learn_error_distribution(errors_file)

sent = 'sir arthur conan doyle copyright law are changing all over the world'
print('before: ', sent)
print('after:  ', sc.fix_sentence(sent, 0.95))

sent = 'information about speciffic rights and restrictions'
print('before: ', sent)
print('after:  ', sc.fix_sentence(sent, 0.95))

sent = 'the winter is comming'
print('before: ', sent)
print('after:  ', sc.fix_sentence(sent, 0.05))

sent = 'botle'
print('before: ', sent)
print('after:  ', sc.fix_sentence(sent, 0.05))

sentences = ['watch out there is a cat in the end of the stret',
             'watch out there i a cat in the end of the street',
             'the winter is comming',
             'botle']
for sent in sentences:
    print('before: ', sent)
    print('after:  ', sc.fix_sentence(sent, 0.05))

alpha = 0.95
text = "you are ther very man"
print(sc.spell_check(text, alpha))
text = "you are the vary man"
print(sc.spell_check(text, alpha))
text = "yo are the very man"
print(sc.spell_check(text, alpha))
text = "you aer the very man"
print(sc.spell_check(text, alpha))
text = "you are the very man"
print(sc.spell_check(text, alpha))
text = "yuo are the very man"
print(sc.spell_check(text, alpha))

print(sc.spell_check("the movie acress is famous. my name is john and i went to schol.", 0.01))

# =============================================================================
# need to check the model format
# =============================================================================
ng3 = Ngram_Language_Model()
ng3.build_model(norm_text, 3)

sc = Spell_Checker(ng3)
sc.learn_error_distribution(errors_file)
print(sc.spell_check("the movie acress is famous. my name is john and i went to schol.", 0.01))

ng5 = Ngram_Language_Model()
ng5.build_model(norm_text, 5)

sc.add_language_model(ng5)
print(sc.spell_check("the movie acress is famous. my name is john and i went to schol.", 0.01))

ng7 = Ngram_Language_Model()
ng7.build_model(norm_text, 7)

sc.add_language_model(ng7)
print(sc.spell_check("the movie acress is famous. my name is john and i went to schol.", 0.01))

ng9 = Ngram_Language_Model()
ng9.build_model(norm_text, 9)

sc.add_language_model(ng9)
print(sc.spell_check("the movie acress is famous. my name is john and i went to schol.", 0.01))

del d['deletion']
sc.add_error_tables(d)
sc.add_language_model(ng5)
print(sc.spell_check("the movie acress is famous. my name is john and i went to schol.", 0.01))
