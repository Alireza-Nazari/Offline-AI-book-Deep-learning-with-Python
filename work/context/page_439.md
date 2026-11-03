# Generation bundle for page_439  (PDF page 439, printed page 415)

Page kind: content
Chapter: 14 Text classification
Section: 14.5 Sequence models
Subsection: 14.5.4 Pretraining a word embedding
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_439.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_439.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p439_e03] (body) text in a dataset, where we continuously attempt to guess a missing word based on the words that appear to its direct right and left (figure 14.6). For example, if our “bag” of surrounding words contained the words “sail,” “wave,” and “mast,” we might guess that the middle word is “boat” or “ocean.”
[p439_e04] (figure_caption) Figure 14.6 The Continuous Bag of Words predicts a word based on its surrounding context with a shallow neural network.
[p439_e05] (body) In our particular IMDb classification problem, we are interested in “priming” the word embedding of the LSTM model we just trained. We can reuse the `TextVectorization` vocabulary we computed earlier. All we are trying to do here is to learn a good 64-dimensional vector for each word in this vocabulary.
[p439_e06] (body) We can create a new `TextVectorization` layer with the same vocabulary that does not truncate or pad input. We will preprocess the output tokens of this layer by sliding a context window across our text.
[p439_e07] (listing_caption) Listing 14.26 Removing padding from our `TextVectorization` preprocessing layer
[p439_e08] (code)
```
imdb_vocabulary = text_vectorization.get_vocabulary()
tokenize_no_padding = keras.layers.TextVectorization(
    vocabulary=imdb_vocabulary,
    split="whitespace",
    output_mode="int",
)
```
[p439_e09] (body) To preprocess our data, we will slide a window across our training data, creating “bags” of nine consecutive tokens. Then, we use the middle word as our label and the remaining eight words as an unordered context to predict our label.
[p439_e10] (body) To do this, we will again use `tf.data` to preprocess our inputs, although this choice does not limit the backend we use for actual model training.
[p439_e11] (listing_caption) Listing 14.27 Preprocessing our IMDb data for pretraining a CBOW model
[p439_e12] (code)
```
import tensorflow as tf
context_size = 4
```

## Code annotations printed next to listings (callouts, in order)
- Words to the left
- or right of label

## Listings
- Listing 14.26: Removing padding from our `TextVectorization` preprocessing layer (code elements p439_e08)
- Listing 14.27: Preprocessing our IMDb data for pretraining a CBOW model (code elements p439_e12)

## Figures
- Figure 14.6: The Continuous Bag of Words predicts a word based on its surrounding context with a shallow neural network.  | labels inside the figure: “quick”, “the”, “fox”, “jumped”, sum, “brown”, Target word

## Chapter outline
Chapter: 14 Text classification (PDF pages 405-444)
  - 14.1 A brief history of natural language processing (p.405)
  - 14.2 Preparing text data (p.408)
      - 14.2.1 Character and word tokenization (p.411)
      - 14.2.2 Subword tokenization (p.414)
  - 14.3 Sets vs. sequences (p.419)
      - 14.3.1 Loading the IMDb classification dataset (p.420)
  - 14.4 Set models (p.422)
      - 14.4.1 Training a bag-of-words model (p.423)
      - 14.4.2 Training a bigram model (p.427)
  - 14.5 Sequence models (p.429)
      - 14.5.1 Training a recurrent model (p.430)
      - 14.5.2 Understanding word embeddings (p.433)
      - 14.5.3 Using a word embedding (p.434)
      - 14.5.4 Pretraining a word embedding (p.438)
      - 14.5.5 Using the pretrained embedding for classification (p.442)

## Previous page (PDF 438) text, for continuity
(continued)

or swap the position of words in a sentence, so the sentence “The rain in Spain falls mainly on the plain” becomes “The rain Spain falls plain on the mainly.” Training your model on such edited inputs can make it robust to typos and grammar mistakes. However, this example also concisely shows the big pitfall with text augmentation—it is easy to alter the meaning of an input example unintentionally. Unlike image data, where you can crop, rotate, and adjust color levels on a picture of a cat and still have a recognizable cat on the other end, language is order-dependent and highly sensitive to tiny changes. A sentence with two words swapped might mean the opposite of the input sentence. Some augmentation techniques seek to solve this by replacing words from a table of known synonyms, but this, too, can be brittle if we choose the incorrect meaning of a word. These issues have kept text augmentation from being widespread in practice. It is usually a better idea to seek more text examples than to sink time into text augmentation techniques. Generative models, which we will see in the coming chapters, are beginning to offer a new form of text augmentation that can alleviate these pain points. By generating outputs from a model that has learned how to produce consistent and coherent text, we can create completely unseen inputs that plausibly resemble our input data. This poses its own challenges but opens up a new frontier for text augmentation on problems where data is particularly sparse and challenging to collect.

14.5.4 Pretraining a word embedding

The last decade of rapid advancement in NLP has coincided with the rise of pretraining as the dominant approach for text modeling problems. Once we move past simple set-based regression models to sequence models  …

## Next page (PDF 440) text, for continuity
After preprocessing, we can see that we have eight integer token IDs as context paired with a single token ID label.

The model we train with this data is exceedingly simple. We will use an `Embedding` layer to embed all context tokens and a `GlobalAveragePooling1D` to compute the average embedding of our “bag” of context tokens. Then, we use that average embedding to predict the value of our middle label token. That’s it! By repeatedly refining our embedding space so that we are good at predicting a word based on nearby word embeddings, we learn a rich embedding of tokens used in movie reviews.

Listing 14.28 Building a CBOW model

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.423 — 14 Text classification › 14.4 Set models
- PDF p.441 — 14 Text classification › 14.5 Sequence models
- PDF p.426 — 14 Text classification › 14.4 Set models
- PDF p.422 — 14 Text classification › 14.4 Set models
- PDF p.424 — 14 Text classification › 14.4 Set models
- PDF p.427 — 14 Text classification › 14.4 Set models
- PDF p.429 — 14 Text classification › 14.5 Sequence models
- PDF p.448 — 15 Language models and the Transformer › 15.1 The language model

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.423 (printed 399) — 14 Text classification › 14.4 Set models › 14.4.1 Training a bag-of-words model
To do this text processing in code, it would be easy enough to extend our `WordTokenizer` from earlier in the chapter. An even easier solution is to use the `TextVectorization` layer built into Keras. The `TextVectorization` handles word and character tokenization and comes with several additional features, including multi-hot encoding of the layer output.

The `TextVectorization` layer, like many preprocessing layers in Keras, has an `adapt()` method to learn a layer state from input data. In the case of `TextVectorization`, `adapt()` will learn a vocabulary for a dataset on the fly by iterating over an input dataset. Let’s use it to tokenize and encode our input data. We will build a vocabulary of 20,000 words, a good starting place for text classification problems.

Listing 14.12 Applying a bag-of-words encoding to the IMDb reviews

### PDF p.437 (printed 413) — 14 Text classification › 14.5 Sequence models › 
With the embedding, we have reduced both our training time and model size by an order of magnitude. A learned embedding is clearly far more efficient than one-hot encoding our input.

However, the LSTM’s overall performance did not change. Accuracy was stubbornly around 84%, still a far cry from the bag-of-words and bigram models. Does this mean that a “structured embedding space” for input tokens is not that practically useful? Or is it not useful for text classification tasks? Quite the contrary, a well-trained token embedding space can dramatically improve the practical performance ceiling of a model like this. The issue in this particular case is with our training setup. We lack enough data in our 20,000 review examples to effectively train a good word embedding. By the end of our 10 training epochs, our train set accuracy has cracked 99%. Our model has begun to overfit and memorize our input, and it turns out it is doing so well before we have learned an optimal set of word embeddings for the task at hand. For cases like this, we can turn to pretraining. Rather than training our word embedding jointly with the classification task, we can train it separately, on more data, with

### PDF p.442 (printed 418) — 14 Text classification › 14.5 Sequence models › 14.5.5 Using the pretrained embedding for classification
Now that we have trained a new word embedding, applying it to our LSTM model is simple. First, we create the model precisely as we did before.

Listing 14.30 Building another LSTM sequence model with an `Embedding` layer

### PDF p.427 (printed 403) — 14 Text classification › 14.4 Set models › 14.4.2 Training a bigram model
Of course, we can intuitively guess that discarding all word order is very reductive because even atomic concepts can be expressed via multiple words: the term “United States” conveys a concept that is quite distinct from the meaning of the words “states” and “united” taken separately. A movie that is “not bad” and a movie that is “bad” should probably get different sentiment scores.

Therefore, it is usually a good idea to inject some knowledge of local word ordering into a model, even for these simple set-based models we are currently building. One easy way to do that is to consider bigrams—a term for two tokens that appear consecutively in the input text. Given our example “this movie made me cry,” `{"this",` `"movie",` `"made",` `"me",` `"cry"}` is the set of all word unigrams in the input, and `{"this` `movie",` `"movie` `made",` `"made me",` `"me cry"}` is the set of all bigrams. The bag-of-words model we just trained could equivalently be called a unigram model, and the term n-gram refers to an ordered sequence of n tokens for any n. To add bigrams to our model, we want to consider the frequency of all bigrams while building our vocabulary. We could do this in two ways: by c

### PDF p.14 (printed xiv) — contents ›  › 
14.5

Sequence models 405

Training a recurrent model 406 ■ Understanding word embeddings 409 ■ Using a word embedding 410 ■ Pretraining a word embedding 414 ■ Using the pretrained embedding for classification 418

15

Language models and the Transformer 421 15.1

The language model 421

Training a Shakespeare language model 422 ■ Generating Shakespeare 426

15.2

Sequence-to-sequence learning 428

English-to-Spanish translation 430 ■ Sequence-to-sequence learning with RNNs 432

15.3

The Transformer architecture 437

Dot-product attention 439 ■ Transformer encoder block 444 Transformer decoder block 446 ■ Sequence-to-sequence learning with a Transformer 448 ■ Embedding positional information 451

15.4

Classification with a pretrained Transformer 454

Pretraining a Transformer encoder 454 ■ Loading a pretrained Transformer 455 ■ Preprocessing IMDb movie reviews 458 Fine-tuning a pretrained Transformer 460

15.5

What makes the Transformer effective? 461 16 Text generation 466 16.1 A brief history of sequence generation 468 16.2 Training a mini-GPT 470

Building the model 473 ■ Pretraining the model 476 Generative decoding 478 ■ Sampling strategies 480

16.3

Using a pretrained LLM

### PDF p.441 (printed 417) — 14 Text classification › 14.5 Sequence models › 
At the end of training, we are able to guess the middle word around 12% of the time based solely on the neighboring eight words. This may not sound like a great result, but given that we have 30,000 words to guess from each time, this is actually a reasonable accuracy score.

Let’s use this word embedding to improve the performance of our LSTM model.

### PDF p.429 (printed 405) — 14 Text classification › 14.5 Sequence models › 14.5 Sequence models
This will allow our batch computation to proceed much faster, although we will need to be careful with our padding tokens to ensure they do not affect the quality of our model predictions.

To keep a manageable input size, we can truncate our IMDb reviews after the first 600 words. This is a reasonable choice, since the average review length is 233 words, and only 5% of reviews are longer than 600 words. Once again, we can use the `Vectorization` layer, which has an option for padding or truncating inputs and includes a `"[PAD]"` at index zero of the learned vocabulary.

### PDF p.485 (printed 461) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › 15.5 What makes the Transformer effective?
In 2013, at Google, Tomas Mikolov and his colleagues noticed something remarkable. They were building a pretrained embedding called “Word2Vec,” similar to the Continuous Bag of Words (CBOW) embedding we built in the last chapter. Much like our CBOW model, their training objective sought to turn correlation relationships between words into distance relationships in the embedding space: a vector was associated with each word in a vocabulary, and the vectors were optimized so that the dot-product (cosine proximity) between vectors representing frequently co-occurring words would
