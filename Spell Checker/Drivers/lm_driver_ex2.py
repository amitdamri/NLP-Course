import spelling_confusion_matrices
import os
import ex2

f = open("big.txt", "r")
text=f.read().rstrip()
size=10000

#Make sure things work as in ex1:
nt = ex2.normalize_text(text)[:size] #lower casing, padding punctuation with white spaces
print(nt)
sc=ex2.Spell_Checker()
lm=sc.build_model(nt, n=3)
for e in [ex2.normalize_text("OOV it's not summer, it's not winter, it's Beer Sheva"), 'a cat sat on the mat','the rat sat on the cat' ]:
    print('%s | %.3f' %(e, lm.evaluate(e)))



error_test=["I went to th beach", "I went ta the beach", "dfsjkdfs", ""]
#print error text
print("errors tested:")
for e in error_test:
    print(ex2.normalize_text(e))
print("")
print("error correction with given tables")
sc.add_error_tables(spelling_confusion_matrices.error_tables)
alpha=0.9
for e in error_test:
    print(sc.spell_check(ex2.normalize_text(e),alpha=alpha))

print("")
print("error correction learned tables")
#test on learned error tables
errors_tables=sc.learn_error_tables(os.path.join(os.getcwd(),"common_errors.txt"))
for e in error_test:
    print(sc.spell_check(ex2.normalize_text(e),alpha=alpha))
