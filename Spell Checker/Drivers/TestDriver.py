import ex2
import spelling_confusion_matrices
import ex1


def get_lm(path, n=3, chars=False):
    with open(path, 'r', errors='ignore') as file:
        text = file.read()

    lm = ex1.Ngram_Language_Model(chars=chars)
    return lm, text



# # test build model
# lan_m, text_n = get_lm('./Corpora/big.txt', 3, False)
# s = ex2.Spell_Checker(lan_m)
# s.build_model(None, 3)
# s.build_model(None, -1)
# s.build_model('', 3)
# s.build_model('', None)
#
# # learn error tables
# s.learn_error_tables(None)
# s.learn_error_tables('corpors')
# t = s.learn_error_tables('./Corpora/common_errors.txt')
# s.add_error_tables(None)
# error_tables = spelling_confusion_matrices.error_tables
# s.add_error_tables(error_tables)
#
#
# s = ex2.Spell_Checker(lan_m)
# s.build_model(text_n, 3)
# s.learn_error_tables('./Corpora/common_errors.txt')

# # spell check
# error_test=["I went to th beach", "I went ta the beach", "dfsjkdfs", ""]
# #print error text
# print("errors tested:")
# for e in error_test:
#     print(s._Spell_Checker__normalize_text(e))
# print("")
# print("error correction with given tables")
# s.add_error_tables(error_tables)
# alpha=0.9
# for e in error_test:
#     print(s.spell_check(s._Spell_Checker__normalize_text(e),alpha=alpha))
#
# print("")
# print("error correction learned tables")
# #test on learned error tables
# errors_tables=sc.learn_error_tables(os.path.join(os.getcwd(),"common_errors.txt"))
# for e in error_test:
#     print(sc.spell_check(ex2.normalize_text(e),alpha=alpha))



if __name__ == '__main__':
    lan_m, text_n = get_lm('./Corpora/big.txt', 3, False)
    error_tables = spelling_confusion_matrices.error_tables
    spell_model = ex2.Spell_Checker(lan_m)
    spell_model.build_model(text_n, 3)
    spell_model.add_error_tables(error_tables)
    print(spell_model.spell_check("dsdfsefeasf word like that", 0.95))
    a = spell_model.learn_error_tables('./Corpora/common_errors.txt')
    print(spell_model.spell_check("My nmae is sherlock", 0.95))