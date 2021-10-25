# ***LegalNLP*** - Natural Language Processing Methods for the Brazilian Legal Language :balance_scale:

### The library of Natural Language Processing for Brazilian legal language, *LegalNLP*, was born in a partnership between Brazilian researchers and the legal tech Tikal Tech based in São Paulo, Brazil. Besides containing pre-trained language models for the Brazilian legal language, ***LegalNLP*** provides functions that can facilitate the manipulation of legal texts in Portuguese and demonstration/tutorials to help people in their own work.

You can access our paper by clicking **here**. 

If you use our library in your academic work, please cite us in the following way

  

--------------

## Summary

0. [Accessing the Language Models](#0)
1. [ Introduction / Installing package](#1)
2. [Fuctions ](#2)
    1.  [ Text Cleaning Functions](#2.1)
    2.  [Other Functions](#2.2)
3. [ Language Models (Details / How to use)](#3)
    1.  [ Phraser ](#3.1)
    2.  [ Word2Vec/Doc2Vec ](#3.2)
    3.  [ FastText ](#3.3)
    4.  [ BERTikal ](#3.4)
4. [ Demonstrations/Tutorials](#4)
5. [ References](#5)

--------------

<a name="0"></a>
## 0\. Address for Language Models


All our models can be found [here](https://drive.google.com/drive/folders/1tCccOXPLSEAEUQtcWXvED3YaNJi3p7la?usp=sharing).

Some models can be download directly using our function `get_premodel`.


Please contact *felipemaiapolo@gmail.com* if you have some problem accessing the language models. 

--------------

<a name="1"></a>
## 1\. Introduction / Installing package
*LegalNLP* is promising given the scarcity of Natural Language Processing resources focused on the Brazilian legal language. It is worth mentioning that our library was made for Python, one of the most well-known programming languages for machine learning.


You can install our package running the following command on terminal
``` :sh
$ pip install git+https://github.com/felipemaiapolo/legalnlp
```

You can load all our functions running the following command

```python
from legalnlp import *
```


--------------

<a name="2"></a>
## 2\. Functions
<a name="2.1"></a>
### 2.1\.  Text Cleaning Functions


<a name="2.1.1"></a>
#### 2.1.1\. `clean(text, lower=True, return_masked=False)`
Function for cleaning texts to be used (optional) in conjunction with Doc2Vec, Word2Vec, and FastText models. We use RegEx to mask/extract information such as email addresses, URLs, dates, numbers, monetary values, etc.

**input**:  

- *text*, **str**;
 
- *lower*, **bool**, default=**True**. If lower==True, function lower cases the whole text. Note that all the models (except BERT) were trained with lower cased texts;

- *return_masked*, **bool**, default=**True**.  If return_masked == False, the function outputs a clean text. Otherwise, it returns a dictionary containing the clean text and the information extracted by RegEx;

**output**:

-  Clean text or dictionary, depending on the *return_masked* parameter;


<a name="2.1.2"></a>
#### 2.1.2\.`clean_bert(text)`

Function for cleaning the texts to be used (optional) in conjunction with the BERT model.

**input:**  

- *text*, **str**.

**output:** 

-  **str** with clean text.

<a name="2.2"></a>
### 2.2\.  Other functions

#### 2.2.2\. `get_premodel(model)` 

Function to download a pre-trained model in the same folder as the file that is being executed.

**input:**  

- *model*, **str**. Must contain the name of the pre-trained model that one wants to use. There are these options:  
    - **model = "bert"**: Download a .zip file containing BERTikal model and unzip it.
    - **model = "wdoc"**: Download Word2Vec and Do2vec pre-trained models in a.zip file and unzip it. It has 2 two files, one with an size 100 Doc2Vec Distributed Memory/ Word2Vec Continuous Bag-of-Words (CBOW) embeddings model and other with an size 100 Doc2Vec Distributed Bag-of-Words (DBOW)/ Word2Vec Skip-Gram (SG)  embeddings model.
    - **model = "fasttext"**: Download a .zip file containing 100 sized FastText CBOW/SG models and unzip it.
    - **model = "phraser"**: Download Phraser pre-trained model in a .zip file and unzip it. It has 2 two files with phraser1 and phreaser2. We explain how to use them in Section [ Phraser ](#3.1). 
    - **model = "w2vnilc"**: Download size 100 Word2Vec CBOW model trained by "Núcleo Interinstitucional de Linguística Computacional" embeddings model in a .zip file and unzip it. [Click here for more details](http://nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc).
    - **model = "neuralmindbase"**: Download a .zip file containing base BERT model (PyTorch), trained by NeuralMind and unzip it. For more informations about BERT models made by NeuralMind go to [their GitHub repo](https://github.com/neuralmind-ai/portuguese-bert).
    - **model = "neuralmindlarge"**: Download a .zip file containing large BERT model (PyTorch), trained by NeuralMind and unzip it. For more informations about BERT models made by NeuralMind go to [their GitHub repo](https://github.com/neuralmind-ai/portuguese-bert).


**output:** 

- True if download of some model was made and False otherwise.


#### 2.2.1\. `extract_features_bert(path_model, path_tokenizer, data, gpu=True)`

Function for extracting features with the BERT model (This function is not accessed through the package installation, but you can find it [here](https://github.com/legalnlp21/legalnlp/blob/main/demo/BERT/extract_features_bert.ipynb)).


**Input:**  

- *path_model*, **str**. Must contain the path of the pre-trained model;

- *path_tokenizer*, **str**. Must contain the path of tokenizer;

- *data*, **list**. Must contain a list of texts that will be extracted features;

- *gpu*, **bool**, default=**True**. If gpu==False, the GPU will not be used in the model application (we recommend feature extraction to be done using Google Colab).


**Output:** 

- **DataFrame** with features extracted by BERT model.


<a name="3"></a>
## 3\. Model Languages

<a name="3.1"></a>
### 3.1\. Phraser

Phraser is a statistical method proposed in the natural language processing
literature [1] for identifying which words when they appear
together, can be considered as unique tokens. This method application is able to
identify the relevance of the occurrence of a bigram against the occurrence of the
words that make it up separately. Thus, we can identify that a bigram like "São
Paulo" should be treated as a single token, for example. If the method is applied
a second time in sequence, we can check which are the relevant trigrams and
quadrigrams. Since the two applications should be done with different Phraser
models, it can be the case that the second application identifies bigrams that were
not identified by the first model.

This model is compatible with the `clean` function, but it is not necessary to use it before. Remember to at least make all letters lowercase. Please check our paper or [Gensim page](https://radimrehurek.com/gensim_3.8.3/models/phrases.html) for more details. Preferably use Gensim version 3.8.3.

#### Using *Phraser*
Installing Gensim


```python
!pip install gensim=='3.8.3' 
```

Importing package and loading our two Phraser models.


```python
#Importing packages
from gensim.models.phrases import Phraser 

#Loading two Phraser models
phraser1=Phraser.load('models_phraser/phraser1')
phraser2=Phraser.load('models_phraser/phraser2')
```


Applying Phraser once and twice to check output


```python
txt='direito do consumidor origem : bangu regional xxix juizado especial civel ação : [processo] - - recte : fundo de investimento em direitos creditórios'
tokens=txt.split()

print('Clean Text: "'+' '.join(tokens)+'"')
print('\nApplying Phraser 1x: "'+' '.join(phraser1[tokens])+'"')
print('\nApplying Phraser 2x: "'+' '.join(phraser2[phraser1[tokens]])+'"')
```

    Clean Text: "direito do consumidor origem : bangu regional xxix juizado especial civel ação : [processo] - - recte : fundo de investimento em direitos creditórios"
    
    Applying Phraser 1x: "direito do consumidor origem : bangu regional xxix juizado_especial civel_ação : [processo] - - recte : fundo de investimento em direitos_creditórios"
    
    Applying Phraser 2x: "direito do consumidor origem : bangu_regional xxix juizado_especial_civel_ação : [processo] - - recte : fundo de investimento em direitos_creditórios"

<a name="3.2"></a>
### 3.2\. Word2Vec/Doc2Vec

Our first models for generating vector representation for tokens and
texts (embeddings) are variations of the Word2Vec [1,
2] and Doc2Vec [3] methods. In short, the
Word2Vec methods generate embeddings for tokens5 and that somehow capture
the meaning of the various textual elements, based on the contexts in which these
elements appear. Doc2Vec methods are extensions/modifications of Word2Vec
for generating whole text representations.

The Word2Vec and Doc2Vec methods are presented together in this section because they were trained together using the Gensim package. Both models are compatible with the `clean` function, but it is not necessary to use it before. Remember to at least make all letters lowercase. Please check our paper or [Gensim page](https://radimrehurek.com/gensim_3.8.3/models/doc2vec.html) for more details. Preferably use Gensim version 3.8.3.


Below we have a summary table with some important information about the trained models:



| Filenames       |  Doc2Vec | Word2Vec   | Size | Windows
|:-------------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| ```w2v_d2v_dm*```     | Distributed Memory       (DM)             | Continuous Bag-of-Words (CBOW)          | 100, 200, 300 | 15 
| ```w2v_d2v_dbow*``` | Distributed Bag-of-Words (DBOW)               | Skip-Gram (SG)                   | 100, 200, 300      | 15 





#### Using *Word2Vec*

Installing Gensim


```python
!pip install gensim=='3.8.3' 
```

Loading W2V (all the files for the specific model should be in the same folder)


```python
from gensim.models import KeyedVectors

#Loading a W2V model
w2v=KeyedVectors.load('models_w2v_d2v/w2v_d2v_dm_size_100_window_15_epochs_20')
w2v=w2v.wv
```
Viewing the first 10 entries of 'juiz' vector


```python
w2v['juiz'][:10]
```




    array([ 6.570131  , -1.262787  ,  5.156106  , -8.943866  , -5.884408  ,
           -7.717058  ,  1.8819941 , -8.02803   , -0.66901577,  6.7223144 ],
          dtype=float32)




Viewing closest tokens to 'juiz'

```python
w2v.most_similar('juiz')
```




    [('juíza', 0.8210258483886719),
     ('juiza', 0.7306275367736816),
     ('juíz', 0.691645085811615),
     ('juízo', 0.6605231165885925),
     ('magistrado', 0.6213295459747314),
     ('mmª_juíza', 0.5510469675064087),
     ('juizo', 0.5494943261146545),
     ('desembargador', 0.5313084721565247),
     ('mmjuiz', 0.5277603268623352),
     ('fabíola_melo_feijão_juíza', 0.5043971538543701)]


#### Using *Doc2Vec*
Installing Gensim


```python
!pip install gensim=='3.8.3' 
```

Loading D2V (all the files for the specific model should be in the same folder)


```python
from gensim.models import Doc2Vec

#Loading a D2V model
d2v=Doc2Vec.load('models_w2v_d2v/w2v_d2v_dm_size_100_window_15_epochs_20')
```

Inferring vector for a text


```python
txt='direito do consumidor origem : bangu regional xxix juizado especial civel ação : [processo] - - recte : fundo de investimento em direitos creditórios'
tokens=txt.split()

txt_vec=d2v.infer_vector(tokens, epochs=20)
txt_vec[:10]
```




    array([ 0.02626514, -0.3876521 , -0.24873355, -0.0318402 ,  0.3343679 ,
           -0.21307918,  0.07193747,  0.02030687,  0.407305  ,  0.20065512],
          dtype=float32)




<a name="3.3"></a>
### 3.3\. FastText

The FastText [4] methods, like Word2Vec, form a class of
models for creating vector representations (embeddings) for tokens. Unlike
Word2Vec, which disregards the morphology of the tokens and allocates a
different vector for each one of them, the FastText methods consider that each one
of the tokens is formed by n-grams of characters or substrings. In this way, the
representation of tokens which do not appear in the training set can be inferred
from the representation of substrings. Also, rare tokens can have more robust
representations than those returned by the Word2Vec methods.

Models are compatible with the `clean` function, but it is not necessary to use it. Remember to at least make all letters lowercase. Please check our paper or the [Gensim page](https://radimrehurek.com/gensim/models/fasttext.html) for more details. Preferably use Gensim version 4.0.1.

Below we have a summary table with some important information about the trained models:
| Filenames      | FastText   | Sizes | Windows
|:-------------------:|:--------------:|:--------------:|:--------------:|
| ```fasttext_cbow*```         | Continuous Bag-of-Words (CBOW)          | 100, 200, 300 | 15 
| ```fasttext_sg*```             | Skip-Gram (SG)                   | 100, 200, 300      | 15 


#### Using *FastText*

installing Gensim


```python
!pip install gensim=='4.0.1' 
```

Loading FastText (all the files for the specific model should be in the same folder)


```python
from gensim.models import FastText

#Loading a FastText model
fast=FastText.load('models_fasttext/fasttext_sg_size_100_window_15_epochs_20')
fast=fast.wv
```

Viewing the first 10 entries of 'juiz' vector



```python
fast['juiz'][:10]
```




    array([ 0.46769685,  0.62529474,  0.08549586,  0.09621219, -0.09998254,
           -0.07897531,  0.32838237, -0.33229044, -0.05959201, -0.5865443 ],
          dtype=float32)



Viewing the first 10 vector entries of a token that was not in our vocabulary


```python
fast['juizasjashdkjhaskda'][:10]
```




    array([ 0.02795791,  0.1361525 ,  0.1340836 , -0.36824217, -0.11549155,
           -0.11167661,  0.32045627, -0.33701468, -0.05198409, -0.05513595],
          dtype=float32)


<a name="3.4"></a>
### 3.4\. BERTikal


We call BERTikal our BERT-Base model   (cased) [5] for Brazilian legal language. BERT models are models based on neural network architectures called Transformers. BERT models are trained with large sets of texts using the self-supervised paradigm, which is basically solving unsupervised problems using supervised techniques. A pre-trained BERT model is capable of generating representations for entire texts and can be adapted for a supervised task, e.g., text classification or question answering, using the fine-tuning mechanism. 

BERTikal was trained using the Python package [Transformers](https://huggingface.co/transformers/})  in its 4.2.2 version and its checkpoint made available by us is compatible with [PyTorch](https://pytorch.org/) 1.9.0. Although we expose the versions of both packages, more current versions can be used in applications of the model, as long as there are no relevant version conflicts.

Our model was trained from the checkpoint made available in [Neuralmind’s Github repository](https://github.com/neuralmind-ai/portuguese-bert) by the authors of recent research [6].

#### Using *BERTikal*

Installing Torch e Transformers


```python
!pip install torch=='1.8.1' transformers=='4.2.2'
```

Loading BERT (all the files for the specific model should be in the same folder)


```python
from transformers import BertModel, BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('model_bertikal/', do_lower_case=False)
bert_model = BertModel.from_pretrained('model_bertikal/')
```

--------------

<a name="4"></a>
## 4\. Demonstrations

For a better understanding of the application of these models, below are the links to notebooks where we apply them to a legal dataset using various classification models such as Logistic Regression and CatBoost:

- **BERT notebook** : 
[https://github.com/legalnlp21/legalnlp/blob/main/demo/BERT/BERT_TUTORIAL.ipynb](https://github.com/legalnlp21/legalnlp/blob/main/demo/BERT/BERT_TUTORIAL.ipynb)
- **Word2Vec notebook** :
[https://github.com/legalnlp21/legalnlp/blob/main/demo/Word2Vec/Word2Vec_TUTORIAL.ipynb](https://github.com/legalnlp21/legalnlp/blob/main/demo/Word2Vec/Word2Vec_TUTORIAL.ipynb)
- **Doc2Vec notebook** :
[https://github.com/legalnlp21/legalnlp/tree/main/demo/Doc2Vec](https://github.com/legalnlp21/legalnlp/tree/main/demo/Doc2Vec)



--------------

<a name="5"></a>
## 5\. References

[1] Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., and Dean, J. (2013b).
Distributed representations of words and phrases and their compositionality.
In Advances in neural information processing systems, pages 3111–3119.

[2] Mikolov, T., Chen, K., Corrado, G., and Dean, J. (2013a). Efficient estimation of
word representations in vector space. arXiv preprint arXiv:1301.3781.

[3] Le, Q. and Mikolov, T. (2014). Distributed representations of sentences and
documents. In International conference on machine learning, pages 1188–1196.
PMLR.

[4] Bojanowski, P., Grave, E., Joulin, A., and Mikolov, T. (2017). Enriching
word vectors with subword information. Transactions of the Association for
Computational Linguistics, 5:135–146.

[5] Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2018). Bert: Pre-training
of deep bidirectional transformers for language understanding. arXiv preprint
arXiv:1810.04805.

[6] Souza, F., Nogueira, R., and Lotufo, R. (2020). BERTimbau: pretrained BERT
models for Brazilian Portuguese. In 9th Brazilian Conference on Intelligent
Systems, BRACIS, Rio Grande do Sul, Brazil, October 20-23

