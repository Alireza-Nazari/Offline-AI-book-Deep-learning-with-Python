# Generation bundle for page_440  (PDF page 440, printed page 416)

Page kind: content
Chapter: 14 Text classification
Section: 14.5 Sequence models
Subsection: 14.5.4 Pretraining a word embedding
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_440.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_440.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p440_e03] (code)
```
window_size = 9
def window_data(token_ids):
    num_windows = tf.maximum(tf.size(token_ids) - context_size * 2, 0)
    windows = tf.range(window_size)[None, :]
    windows = windows + tf.range(num_windows)[:, None]
    windowed_tokens = tf.gather(token_ids, windows)
    return tf.data.Dataset.from_tensor_slices(windowed_tokens)
def split_label(window):
    left = window[:context_size]
    right = window[context_size + 1 :]
    bag = tf.concat((left, right), axis=0)
    label = window[4]
    return bag, label
dataset = keras.utils.text_dataset_from_directory(
    imdb_extract_dir / "train", batch_size=None
)
dataset = dataset.map(lambda x, y: x, num_parallel_calls=8)
dataset = dataset.map(tokenize_no_padding, num_parallel_calls=8)
dataset = dataset.interleave(window_data, cycle_length=8, num_parallel_calls=8)
dataset = dataset.map(split_label, num_parallel_calls=8)
```
[p440_e04] (body) After preprocessing, we can see that we have eight integer token IDs as context paired with a single token ID label.
[p440_e05] (body) The model we train with this data is exceedingly simple. We will use an `Embedding` layer to embed all context tokens and a `GlobalAveragePooling1D` to compute the average embedding of our “bag” of context tokens. Then, we use that average embedding to predict the value of our middle label token. That’s it! By repeatedly refining our embedding space so that we are good at predicting a word based on nearby word embeddings, we learn a rich embedding of tokens used in movie reviews.
[p440_e06] (listing_caption) Listing 14.28 Building a CBOW model
[p440_e07] (code)
```
hidden_dim = 64
inputs = keras.Input(shape=(2 * context_size,))
cbow_embedding = layers.Embedding(
    max_tokens,
    hidden_dim,
)
x = cbow_embedding(inputs)
x = layers.GlobalAveragePooling1D()(x)
```

## Code annotations printed next to listings (callouts, in order)
- Total window size
- Tokenizes
- Drops label
- Uses all training data,
- including the unsup/ directory
- Splits middle
- wonder into a label
- Creates context windows

## Listings
- Listing 14.28: Building a CBOW model (code elements p440_e07)

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

## Previous page (PDF 439) text, for continuity
text in a dataset, where we continuously attempt to guess a missing word based on the words that appear to its direct right and left (figure 14.6). For example, if our “bag” of surrounding words contained the words “sail,” “wave,” and “mast,” we might guess that the middle word is “boat” or “ocean.”

Figure 14.6 The Continuous Bag of Words predicts a word based on its surrounding context with a shallow neural network.

In our particular IMDb classification problem, we are interested in “priming” the word embedding of the LSTM model we just trained. We can reuse the `TextVectorization` vocabulary we computed earlier. All we are trying to do here is to learn a good 64-dimensional vector for each word in this vocabulary.

We can create a new `TextVectorization` layer with the same vocabulary that does not truncate or pad input. We will preprocess the output tokens of this layer by sliding a context window across our text.

Listing 14.26 Removing padding from our `TextVectorization` preprocessing layer

To preprocess our data, we will slide a window across our training data, creating “bags” of nine consecutive tokens. Then, we use the middle word as our label and the remaining eight words as an unordered context to predict our label.

To do this, we will again use `tf.data` to preprocess our inputs, although this choice does not limit the backend we use for actual model training.

Listing 14.27 Preprocessing our IMDb data for pretraining a CBOW model

## Next page (PDF 441) text, for continuity
Because our model is so simple, we can use a large batch size to speed up training without worrying about memory constraints.

We will also call `cache()` on this batched dataset so that we store the entire preprocessed dataset in memory rather than recomputing it each epoch. This is because for this very simple model, we are bottlenecked on preprocessing rather than training. That is, it is slower to tokenize our text and compute sliding windows on the CPU than to update our model parameters on the GPU. In such cases, saving your preprocessed outputs in memory or on disk is usually a good idea. You will notice how our later epochs are more than three times faster than the first. This is thanks to the cache of preprocessed training data.

Listing 14.29 Training the CBOW model

At the end of training, we are able to guess the middle word around 12% of the time based solely on the neighboring eight words. This may not sound like a great result, but given that we have 30,000 words to guess from each time, this is actually a reasonable accuracy score.

Let’s use this word embedding to improve the performance of our LSTM model.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.499 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.423 — 14 Text classification › 14.4 Set models
- PDF p.448 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.437 — 14 Text classification › 14.5 Sequence models
- PDF p.442 — 14 Text classification › 14.5 Sequence models
- PDF p.514 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.483 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer
- PDF p.436 — 14 Text classification › 14.5 Sequence models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.437 (printed 413) — 14 Text classification › 14.5 Sequence models › 
With the embedding, we have reduced both our training time and model size by an order of magnitude. A learned embedding is clearly far more efficient than one-hot encoding our input.

However, the LSTM’s overall performance did not change. Accuracy was stubbornly around 84%, still a far cry from the bag-of-words and bigram models. Does this mean that a “structured embedding space” for input tokens is not that practically useful? Or is it not useful for text classification tasks? Quite the contrary, a well-trained token embedding space can dramatically improve the practical performance ceiling of a model like this. The issue in this particular case is with our training setup. We lack enough data in our 20,000 review examples to effectively train a good word embedding. By the end of our 10 training epochs, our train set accuracy has cracked 99%. Our model has begun to overfit and memorize our input, and it turns out it is doing so well before we have learned an optimal set of word embeddings for the task at hand. For cases like this, we can turn to pretraining. Rather than training our word embedding jointly with the classification task, we can train it separately, on more data, with

### PDF p.438 (printed 414) — 14 Text classification › 14.5 Sequence models › 14.5.4 Pretraining a word embedding
The last decade of rapid advancement in NLP has coincided with the rise of pretraining as the dominant approach for text modeling problems. Once we move past simple set-based regression models to sequence models with millions or even billions of parameters, text models become incredibly data-hungry. We are usually limited by our ability to find labeled examples for a particular problem in the text domain.

The idea is to devise an unsupervised task to train model parameters that do not need labeled data. Pretraining data can be text in a similar domain to our final task, or even just arbitrary text in the languages we are interested in working with. Pretraining allows us to learn general patterns in language, effectively priming our model before we specialize it to the final task we are interested in. Word embeddings were one of the first big successes with text pretraining, and we will show how to pretrain a word embedding in this section. Remember the `unsup/` directory we ignored in our IMDb dataset preparation? It contains another 25,000 reviews— the same size as our training data. We will combine all our training data together and show how to pretrain the parameters of an `Emb

### PDF p.435 (printed 411) — 14 Text classification › 14.5 Sequence models › 
Figure 14.5 An `Embedding` layer acts as a dictionary mapping ints to floating point vectors.

When you instantiate an `Embedding` layer, its weights (its internal dictionary of token vectors) are initially random, just as with any other layer. During training, these word vectors are gradually adjusted via backpropagation, structuring the space into something the downstream model can exploit. Once fully trained, the embedding space will show a lot of structure—a kind of structure specialized for the specific problem for which you’re training your model.

Let’s build a model with an `Embedding` layer and benchmark it on our task.

Listing 14.24 Building an LSTM sequence model with an `Embedding` layer

### PDF p.444 (printed 420) — 14 Text classification › 14.5 Sequence models › 
An embedding is an efficient way to transform token IDs into a learned latent space. Embeddings can be trained normally with gradient descent.

Pretraining is critical for sequence models as a way to get around the data-hungry nature of these models. During pretraining, an unsupervised task allows models to learn from large amounts of unlabeled text data. The learned parameters can then be transferred to a downstream task.

### PDF p.442 (printed 418) — 14 Text classification › 14.5 Sequence models › 14.5.5 Using the pretrained embedding for classification
Now that we have trained a new word embedding, applying it to our LSTM model is simple. First, we create the model precisely as we did before.

Listing 14.30 Building another LSTM sequence model with an `Embedding` layer

### PDF p.434 (printed 410) — 14 Text classification › 14.5 Sequence models › 14.5.3 Using a word embedding
Is there an ideal word-embedding space that perfectly maps human language and can be used for any NPL task? Possibly, but we have yet to compute anything of the sort. Also, there is no single human language we could attempt to map—there are many different languages, and they aren’t isomorphic to one another because a language is the reflection of a specific culture and a particular context. More pragmatically, what makes a good word-embedding space depends heavily on your task: the perfect word-embedding space for an English-language movie review sentiment-analysis model may look different from the ideal embedding space for an English-language legal document classification model because the importance of certain semantic relationships varies from task to task.

It’s thus reasonable to learn a new embedding space with every new task. Fortunately, backpropagation makes this easy, and Keras makes it even easier. It’s about learning the weights of the Keras `Embedding` layer. The `Embedding` layer is best understood as a dictionary that maps integer indices (which stand for specific words) to dense vectors. It takes integers as input, looks them up in an internal dictionary, and return

### PDF p.486 (printed 462) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › 
The distance function used to compare tokens is cosine distance in both cases. Even the dimensionality of the embedding space is similar: a vector with somewhere between 1,000 and 10,000 dimensions to represent each word. At this point, you might interject: A Transformer is trained to predict missing words in a sequence, not to group tokens in an embedding space. How does the language model loss function relate to Word2Vec’s objective of maximizing the dot-product between co-occurring tokens? The answer is the attention mechanism. Attention is, by far, the most critical component in the Transformer architecture. It’s a mechanism for learning a new token embedding space by linearly recombining token embeddings from some prior space, in weighted combinations that give greater importance to tokens that are already “closer” to each other (i.e., that have a higher dot-product). It will tend to pull together the vectors of already close tokens, resulting over time in a space where token correlation relationships turn into embedding proximity relationships (in terms of cosine distance). Transformers work by learning a series of incrementally refined embedding spaces, each based on recombi

### PDF p.499 (printed 475) — 16 Text generation › 16.2 Training a mini-GPT › 
prediction layer because they deal with our vocabulary space. The token embedding weight has shape `(vocab_size, hidden_dim)` to embed every possible token. Our output projection has shape `(hidden_dim, vocab_size)` to make a floating-point prediction for every possible token.

We can actually tie these two weight matrices together. To compute our model’s final predictions, we will multiply our hidden states by the transpose of our token embedding matrix. You can very much think of our final projection as a “reverse embedding.” It maps from hidden space to token space, whereas an embedding maps from token space to hidden space. It turns out that using the same weights for this input and output projection is a good idea. Adding this to our `PositionalEmbedding` is simple; we will just add a `reverse` argument to the `call` method, which computes the projection by the transpose of the token embedding.

Listing 16.5 A positional embedding layer that can reverse a text embedding
