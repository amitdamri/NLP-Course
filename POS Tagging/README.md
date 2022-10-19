# Part of Speech (POS) Tagging

<p align="center">
	<img height=300 width=300 src="https://user-images.githubusercontent.com/49988048/196628382-8a43ce69-9905-4e7f-b066-04a60a274095.png">
</p>

Implementation of a Hidden Markov Model and a BiLSTM model for Part of Speech tagging.

### Data
- Extracted tokens and POS tags are taken from the English Web Treebank via the Universal Dependencies Project (http://universaldependencies.org/).
Tagset overviews:
  * Universal: http://people.cs.georgetown.edu/nschneid/p/UPOS-English.pdf
  * Penn (full new-style tagset): https://spacy.io/docs/usage/pos-tagging#pos-tagging-english
  * Penn (examples): http://surdeanu.info/mihai/teaching/ista555-fall13/readings/PennTreebankTagset.html
- en-ud-train.upos.tsv - train set
- en-ud-dev.upos.tsv - development set
- Pretrained GloVe embedding vectors are available here: https://www.dropbox.com/s/qxak38ybjom696y/glove.6B.100d.txt?dl=0

### Pipeline for the RNN model
1.	Initializing model with all that is needed, including dimensions, training data and pretrained embeddings. It is assumed that preprocessing functions will be called from the initialize_rnn_model function. This stage returns a dictionary object model_d. 
2.	Training the RNN model: this is done given the output of the initialization model_d and a list of annotated sentences.
3.	Use the trained model (again, using model_d) to tag a new sentence (the sentence is given as a list of words). This is done via the tag_sentence(sentence, model) function. 
4.	Evaluation with count_correct() can be called. Note that this is a general function. 

### Code
1. VanillaBiLSTM.py - vanilla biLSTM in which the input layer is based on simple word embeddings
2. CaseBasedBLSTM.py - a case-based BiLSTM in which input vectors combine a 3-dim binary vector encoding case information, see https://arxiv.org/pdf/1510.06168.pdf
3. tagger.py - implementation of Hidden Markov Model and a POS tagger with HMM and BiLSTM models.
4. Tests.py - Unit tests
5. Main.py - testing the pipeline
