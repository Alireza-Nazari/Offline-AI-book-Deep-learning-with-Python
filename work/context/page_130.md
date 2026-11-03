# Generation bundle for page_130  (PDF page 130, printed page 106)

Page kind: content
Chapter: 4 Classification and regression
Section: 4.1 Classifying movie reviews: A binary classification example
Subsection: 4.1.1 The IMDb dataset
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_130.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_130.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p130_e03] (heading_section) 4.1
[p130_e04] (heading_section) Classifying movie reviews: A binary classification example
[p130_e05] (body) Two-class classification, or binary classification, is one of the most common kinds of machine learning problem. In this example, you’ll learn to classify movie reviews as positive or negative, based on the text content of the reviews.
[p130_e06] (heading_subsection) 4.1.1
[p130_e07] (heading_subsection) The IMDb dataset
[p130_e08] (body) You’ll work with the IMDb dataset: a set of 50,000 highly polarized reviews from the Internet Movie Database. They’re split into 25,000 reviews for training and 25,000 reviews for testing, each set consisting of 50% negative and 50% positive reviews.
[p130_e09] (body) Just like the MNIST dataset, the IMDb dataset comes packaged with Keras. It has already been preprocessed: the reviews (sequences of words) have been turned into sequences of integers, where each integer stands for a specific word in a dictionary. This enables us to focus on model building, training, and evaluation. In chapter 14, you’ll learn how to process raw text input from scratch. The following code will load the dataset (when you run it the first time, about 80 MB of data will be downloaded to your machine).
[p130_e10] (listing_caption) Listing 4.1 Loading the IMDb dataset
[p130_e11] (code)
```
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000
)
```
[p130_e12] (body) The argument `num_words=10000` means you’ll only keep the top 10,000 most frequently occurring words in the training data. Rare words will be discarded. This allows you to work with vector data of manageable size. If we didn’t set this limit, we’d be working with 88,585 unique words in the training data, which is unnecessarily large. Many of these words only occur in a single sample, and thus can’t be meaningfully used for classification.
[p130_e13] (body) The variables `train_data` and `test_data` are NumPy arrays of reviews; each review is a list of word indices (encoding a sequence of words). `train_labels` and `test_labels` are NumPy arrays of 0s and 1s, where 0 stands for negative and 1 stands for positive:
[p130_e14] (code)
```
>>> train_data[0]
[1, 14, 22, 16, ... 178, 32]
>>> train_labels[0]
1
```
[p130_e15] (body) Because you’re restricting yourself to the top 10,000 most frequent words, no word index will exceed 10,000:

## Listings
- Listing 4.1: Loading the IMDb dataset (code elements p130_e11)

## Equation candidates (formulas written inline in the text)
- [p130_e12] The argument `num_words=10000` means you’ll only keep the top 10,000 most frequently occurring words in the training data.

## Chapter outline
Chapter: 4 Classification and regression (PDF pages 128-159)
  - 4.1 Classifying movie reviews: A binary classification example (p.130)
      - 4.1.1 The IMDb dataset (p.130)
      - 4.1.2 Preparing the data (p.131)
      - 4.1.3 Building your model (p.132)
      - 4.1.4 Validating your approach (p.135)
      - 4.1.5 Using a trained model to generate predictions on new data (p.139)
      - 4.1.6 Further experiments (p.139)
      - 4.1.7 Wrapping up (p.140)
  - 4.2 Classifying newswires: A multiclass classification example (p.140)
      - 4.2.1 The Reuters dataset (p.140)
      - 4.2.2 Preparing the data (p.142)
      - 4.2.3 Building your model (p.142)
      - 4.2.4 Validating your approach (p.144)
      - 4.2.5 Generating predictions on new data (p.148)
      - 4.2.6 A different way to handle the labels and the loss (p.148)
      - 4.2.7 The importance of having sufficiently large intermediate layers (p.149)
      - 4.2.8 Further experiments (p.149)
      - 4.2.9 Wrapping up (p.150)
  - 4.3 Predicting house prices: A regression example (p.150)
      - 4.3.1 The California Housing Price dataset (p.150)
      - 4.3.2 Preparing the data (p.152)
      - 4.3.3 Building your model (p.152)
      - 4.3.4 Validating your approach using K-fold validation (p.153)
      - 4.3.5 Generating predictions on new data (p.158)
      - 4.3.6 Wrapping up (p.158)

## Previous page (PDF 129) text, for continuity
By the end of this chapter, you’ll be able to use neural networks to handle simple classification and regression tasks over vector data. You’ll then be ready to start building a more principled, theory-driven understanding of machine learning in chapter 5.

Classification and regression glossary

Classification and regression involve many specialized terms. You’ve come across some of them in earlier examples, and you’ll see more of them in future chapters. They have precise, machine learning–specific definitions, and you should be familiar with them:

Sample or input—One data point that goes into your model.

Prediction or output—What comes out of your model.

Target—The truth. What your model should ideally have predicted, according to

an external source of data.

Prediction error or loss value—A measure of the distance between your model’s

prediction and the target.

Classes—A set of possible labels to choose from in a classification problem.

For example, when classifying cat and dog pictures, “dog” and “cat” are the two classes.

Label—A specific instance of a class annotation in a classification problem.

For instance, if picture #1234 is annotated as containing the class “dog,” then “dog” is a label of picture #1234.

Ground-truth or annotations—All targets for a dataset, typically collected by

humans.

Binary classification—A classification task where each input sample should be

categorized into two exclusive categories.

Categorical classification or multiclass classification—A classification task

where each input sample should be categorized into more than two categories: for instance, classifying handwritten digits.

Multilabel classification—A classification task where each input sample can

be assigned multiple labels. For instance, a given image may co …

## Next page (PDF 131) text, for continuity
For kicks, let’s quickly decode one of these reviews back to English words.

Listing 4.2 Decoding reviews back to text

Let’s take a look at what we got:

Note that the leading `?` corresponds to a start token that has been prefixed to each review.

4.1.2

Preparing the data

You can’t directly feed lists of integers into a neural network. They have all different lengths, while a neural network expects to process contiguous batches of data. You have to turn your lists into tensors. There are two ways to do that:

Pad your lists so that they all have the same length, then turn them into an integer tensor of shape `(samples,` `max_length)`, and start your model with a layer capable of handling such integer tensors (the `Embedding` layer, which we’ll cover in detail later in the book).

Multi-hot encode your lists to turn them into vectors of 0s and 1s reflecting the presence or absence of all possible words. This would mean, for instance, turning the sequence `[8, 5]` into a 10,000-dimensional vector that would be all 0s except for indices 5 and 8, which would be 1s.

Let’s go with the latter solution to vectorize the data. When done manually, the process looks like the following.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.420 — 14 Text classification › 14.3 Sets vs. sequences
- PDF p.141 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.482 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer
- PDF p.421 — 14 Text classification › 14.3 Sets vs. sequences
- PDF p.13 — contents › 
- PDF p.423 — 14 Text classification › 14.4 Set models
- PDF p.439 — 14 Text classification › 14.5 Sequence models
- PDF p.422 — 14 Text classification › 14.4 Set models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.420 (printed 396) — 14 Text classification › 14.3 Sets vs. sequences › 14.3.1 Loading the IMDb classification dataset
We can see both a train and test set with positive and negative examples. Movie reviews with a low user rating on the IMDb site were sorted into the `neg/` directory and those with a high rating into the `pos/` directory. We can also see an `unsup/` directory, which is short for unsupervised. These are reviews deliberately left unlabeled by the dataset creator; they could be negative or positive reviews.

Let’s look at the content of a few of these text files. Whether you’re working with text or image data, remember to inspect what your data looks like before you dive into modeling. It will ground your intuition about what your model is actually doing.

### PDF p.141 (printed 117) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › 
As with the IMDb reviews, each example is a list of integers (word indices):

### PDF p.420 (printed 396) — 14 Text classification › 14.3 Sets vs. sequences › 
Historically, most early applications of machine learning to NLP just involved bag-of-words models that discarded sequence data. Interest in sequence models only started rising in 2015, with the rebirth of RNNs. Today, both approaches remain relevant. Let’s see how they work and when to use which. We will demonstrate each approach on a well-known text classification benchmark: the IMDb movie review sentiment-classification dataset. In chapters 4 and 5, you worked with a prevectorized version of the IMDb dataset; now let’s process the raw IMDb text data, just like you would do when approaching a new text-classification problem in the real world.

### PDF p.13 (printed xiii) — contents ›  › 
14.3

Sets vs. sequences 395

Loading the IMDb classification dataset 396

14.4

Set models 398

Training a bag-of-words model 399 ■ Training a bigram model 403

### PDF p.141 (printed 117) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › 
As with the IMDb dataset, the argument `num_words=10000` restricts the data to the 10,000 most frequently occurring words found in the data.

You have 8,982 training examples and 2,246 test examples:

### PDF p.482 (printed 458) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 15.4.3 Preprocessing IMDb movie reviews
We can reuse the IMDb loading code we used in chapter 14 unchanged. This will download the movie review data to a `train_dir` and `test_dir` and split a validation dataset into a `val_dir`:

### PDF p.429 (printed 405) — 14 Text classification › 14.5 Sequence models › 14.5 Sequence models
This will allow our batch computation to proceed much faster, although we will need to be careful with our padding tokens to ensure they do not affect the quality of our model predictions.

To keep a manageable input size, we can truncate our IMDb reviews after the first 600 words. This is a reasonable choice, since the average review length is 233 words, and only 5% of reviews are longer than 600 words. Once again, we can use the `Vectorization` layer, which has an option for padding or truncating inputs and includes a `"[PAD]"` at index zero of the learned vocabulary.

### PDF p.482 (printed 458) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 15.4.3 Preprocessing IMDb movie reviews
After loading, we once again have a training set of 20,000 movie reviews and a validation set of 5,000 movie reviews.

Before fine-tuning our classification model, we must tokenize our movie reviews with the RoBERTa tokenizer we loaded. During pretraining, RoBERTa used a specific form of “packing” tokens into a sequence, similar to what we did for our translation model. Each sequence would begin with an `<s>` token, end with an `</s>` token, and be followed by any number of `<pad>` tokens like this:
