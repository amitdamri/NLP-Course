# The Noisy Channel and a Probabilistic Spell Checker

<p align="center">
	<img height=400 width=400 src="https://user-images.githubusercontent.com/49988048/196615615-15e29465-075c-4c0b-b4d8-b2d3af1afcc8.png">
</p>
![image]()


Implementation of a spell checker that handles both non-word and real-word errors given in a sentential context.
In order to do that I learned a language model as well as reconstructed the error distribution tables (according to error type) 
from lists of common errors. Finally, I combined it all into a context-sensitive noisy channel model.

The purpose of this assignment is to give hands-on experience with probabilistic models of language by implementing the full algorithmic pipeline. 

-	Used the noisy channel model to correct the errors, that is - a correction of a misspelled word depends on both the most probable 
correction on the error type-character level and words priors; A correction of a word in a sentence depends on the error 
type-character level and on the language model -- the correction should maximize the likelihood of getting the full corrected sentence. 
-	Used the language model when correcting words in a sentential context. 
-	Assumed a word has at most two errors (that is, ‘character’ will be considered a valid correction for ‘karacter’ [sub+del], 
while it will not be considered for ‘karakter’).
-	Assumed a sentence has at most one erroneous word.  

### Corpora Folder
-	Norvig’s big.txt file 
-	Trump’s historical tweets (~14K tweets and retweets by Trump, each tweet in a new line)
-	An even bigger corpus corpus.data (preprocessed, sentences are separated by &lt;s&gt;)

### Error lists Folder
-	Created the error types matrices from common_errors.txt list of common errors
(containing only the single error pairs in <a href="https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/For_machines">Wikipedia common misspellings</a>). 
File format: each line is a tab separated tuple: <error>    <correct>	
-	A file containing the confusion matrices  used in <a href="https://aclanthology.org/C90-2036/v">A Spelling Correction Program Based on a Noisy Channel Model</a>, Kernighan, Church and Gale, 1990:  
  spelling_confusion_matrics.py (this file is in the format returned by learn_error_tables function)


### API
1. def __init__(self, lm=None)
  ```
  Initializing a spell checker object with a language model as an
  instance  variable. The language model should support the evaluate()
  and the get_model() functions as defined in assignment #1.

  Args:
    lm: a language model object. Defaults to None
  ```
  
2. def add_language_model(self, lm)
  ```
    Adds the specified language model as an instance variable.
    (Replaces an older LM dictionary and unigrams + bigrams dictionaries if set)

    Args:
        lm: a language model object
  ```
  
3. def build_model(self, text, n)
  ```
  Returns a language model object built on the specified text. The language
  model should support evaluate() and the get_model() functions as defined
  in assignment #1.

  Args:
      text (str): the text to construct the model from.
      n (int): the order of the n-gram model (defaults to 3).

  Returns:
      A language model object
  ```
  
4. def learn_error_tables(self, errors_file)
  ```
  Returns a nested dictionary {str:dict} where str is in:
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
    ```
   ```
                                
5. def add_error_tables(self, error_tables) 
  ```
  Adds the speficied dictionary of error tables as an instance variable.
    (Replaces an older value disctionary if set)

    Args:
        error_tables (dict): a dictionary of error tables in the format
        returned by  learn_error_tables()
  ```

6. def spell_check(self, text, alpha)
  ```
   Returns the most probable fix for the specified text. Use a simple
    noisy channel model if the number of tokens in the specified text is
    smaller than the length (n) of the language model.

    Args:
        text (str): the text to spell check.
        alpha (float): the probability of keeping a lexical word as is.

    Return:
        A modified string (or a copy of the original if no corrections are made.)                             
  ```                                

7. def evaluate(self,text)
```
  Returns the log-likelihod of the specified text given the language
    model in use. Smoothing is applied on texts containing OOV words

   Args:
       text (str): Text to evaluate.

   Returns:
       Float. The float should reflect the (log) probability.                                
 ```


Python 3.9 | Python PEP 257

