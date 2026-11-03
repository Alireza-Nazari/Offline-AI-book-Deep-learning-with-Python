# Generation bundle for page_435  (PDF page 435, printed page 411)

Page kind: content
Chapter: 14 Text classification
Section: 14.5 Sequence models
Subsection: 14.5.3 Using a word embedding
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_435.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_435.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p435_e03] (figure_caption) Figure 14.5 An `Embedding` layer acts as a dictionary mapping ints to floating point vectors.
[p435_e04] (body) When you instantiate an `Embedding` layer, its weights (its internal dictionary of token vectors) are initially random, just as with any other layer. During training, these word vectors are gradually adjusted via backpropagation, structuring the space into something the downstream model can exploit. Once fully trained, the embedding space will show a lot of structure—a kind of structure specialized for the specific problem for which you’re training your model.
[p435_e05] (body) Let’s build a model with an `Embedding` layer and benchmark it on our task.
[p435_e06] (listing_caption) Listing 14.24 Building an LSTM sequence model with an `Embedding` layer
[p435_e07] (code)
```
hidden_dim = 64
inputs = keras.Input(shape=(max_length,), dtype="int32")
x = keras.layers.Embedding(
    input_dim=max_tokens,
    output_dim=hidden_dim,
    mask_zero=True,
)(inputs)
x = keras.layers.Bidirectional(keras.layers.LSTM(hidden_dim))(x)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs, name="lstm_with_embedding")
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
```
[p435_e08] (body) The first two arguments to the `Embedding` layer are fairly straightforward. `input_dim` sets the total range of possible values for the integer inputs to the layer—that is, how many possible keys are there in our dictionary lookup. `output_dim` sets the dimensionality of the output vector we look up—that is, the dimensionality of our structured vector space for words.

## Listings
- Listing 14.24: Building an LSTM sequence model with an `Embedding` layer (code elements p435_e07)

## Figures
- Figure 14.5: An `Embedding` layer acts as a dictionary mapping ints to floating point vectors.  | labels inside the figure: Tokenizer, 0, 1, 2, 3, 4, 5, 6, 7, Tokens, Integers, Embedding, “arrow”, “rabbit”, “lodge”, “who”

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

## Previous page (PDF 434) text, for continuity
Besides being dense representations, word embeddings are also structured representations, and their structure is learned from data. Similar words get embedded in close locations, and further, specific directions in the embedding space are meaningful. To make this clearer, let’s look at a concrete example. In figure 14.4, four words are embedded on a 2D plane: cat, dog, wolf, and tiger. With the vector representations we chose here, some semantic relationships

Figure 14.4 A toy

between these words can be encoded as geometric transfor

example of a word-

mations. For instance, the same vector allows us to go from

embedding space

cat to tiger and from dog to wolf: this vector could be interpreted as the “from pet to wild animal” vector. Similarly, another vector lets us go from dog to cat and from wolf to tiger, which could be interpreted as a “from canine to feline” vector.

In real-world word-embedding spaces, typical examples of meaningful geometric transformations are “gender” vectors and “plural” vectors. For instance, by adding a “female” vector to the vector “king,” we obtain the vector “queen.” By adding a “plural” vector, we obtain “kings.” Word-embedding spaces typically feature thousands of such interpretable and potentially useful vectors. Let’s look at how to use such an embedding space in practice.

14.5.3 Using a word embedding

Is there an ideal word-embedding space that perfectly maps human language and can be used for any NPL task? Possibly, but we have yet to compute anything of the sort. Also, there is no single human language we could attempt to map—there are many different languages, and they aren’t isomorphic to one another because a language is the reflection of a specific culture and a particular context. More pragmatically, what makes a good  …

## Next page (PDF 436) text, for continuity
The third argument, `mask_zero=True`, is a little more subtle. This argument tells Keras which inputs in our sequence are `"[PAD]"` tokens, so we can mask these entries later in the model. Remember that when preprocessing our sequence input, we might add a lot of padding tokens to our original input so that a token sequence might look like:

All of those padding tokens will be first embedded and then fed into the `LSTM` layer. This means the last representation we receive from the `LSTM` cell might contain the results of processing the `"[PAD]"` token representation over and over recurrently. We are not very interested in the learned `LSTM` representation for the last `"[PAD]"` token in the previous sequence. Instead, we are interested in the representation of `"awful"`, the last non-padding token. Or, put equivalently, we want to mask all of the `"[PAD]"` tokens so that they do not affect our final output prediction.

`mask_zero=True` is simply a shorthand to easily do such masking in Keras with the `Embedding` layer. Keras will mark all elements in our sequence that initially contained a zero value, where zero is assumed to be the token ID for the `"[PAD]"` token. This mask will be used internally by the `LSTM` layer. Instead of outputting the last learned representation for the whole sequence, it will output the last non-masked representation. This form of masking is implicit and easy to use, but you can always be explicit about which items in a sequence you would like to mask if the need arises. The `LSTM` layer takes an optional `mask` call argument, for explicit or custom masking. Before we train this new model, let’s take a look at the model summary:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.442 — 14 Text classification › 14.5 Sequence models
- PDF p.431 — 14 Text classification › 14.5 Sequence models
- PDF p.499 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.475 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.440 — 14 Text classification › 14.5 Sequence models
- PDF p.437 — 14 Text classification › 14.5 Sequence models
- PDF p.424 — 14 Text classification › 14.4 Set models
- PDF p.441 — 14 Text classification › 14.5 Sequence models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.442 (printed 418) — 14 Text classification › 14.5 Sequence models › 14.5.5 Using the pretrained embedding for classification
Now that we have trained a new word embedding, applying it to our LSTM model is simple. First, we create the model precisely as we did before.

Listing 14.30 Building another LSTM sequence model with an `Embedding` layer

### PDF p.475 (printed 451) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 15.3.5 Embedding positional information
The idea behind a positional embedding is very simple: to give the model access to word order information, we will add the word’s position in the sentence to each word embedding. Our input word embeddings will have two components: the usual word vector, which represents the word independently of any specific context, and a position vector, which represents the position of the word in the current sentence. Hopefully, the model will then figure out how to best use this additional information.

The most straightforward scheme to add position information would be concatenating each word’s position to its embedding vector. You’d add a “position” axis to the vector and fill it with `0` for the first word in the sequence, `1` for the second, and so on. However, that may not be ideal because the positions can potentially be very large integers, which will disrupt the range of values in the embedding vector. As you know, neural networks don’t like very large input values or discrete input distributions. The “Attention is all you need” authors used an interesting trick to encode word positions: they added to the word embeddings a vector containing values in the range `[-1, 1]` that varied cy

### PDF p.499 (printed 475) — 16 Text generation › 16.2 Training a mini-GPT › 
prediction layer because they deal with our vocabulary space. The token embedding weight has shape `(vocab_size, hidden_dim)` to embed every possible token. Our output projection has shape `(hidden_dim, vocab_size)` to make a floating-point prediction for every possible token.

We can actually tie these two weight matrices together. To compute our model’s final predictions, we will multiply our hidden states by the transpose of our token embedding matrix. You can very much think of our final projection as a “reverse embedding.” It maps from hidden space to token space, whereas an embedding maps from token space to hidden space. It turns out that using the same weights for this input and output projection is a good idea. Adding this to our `PositionalEmbedding` is simple; we will just add a `reverse` argument to the `call` method, which computes the projection by the transpose of the token embedding.

Listing 16.5 A positional embedding layer that can reverse a text embedding

### PDF p.437 (printed 413) — 14 Text classification › 14.5 Sequence models › 
With the embedding, we have reduced both our training time and model size by an order of magnitude. A learned embedding is clearly far more efficient than one-hot encoding our input.

However, the LSTM’s overall performance did not change. Accuracy was stubbornly around 84%, still a far cry from the bag-of-words and bigram models. Does this mean that a “structured embedding space” for input tokens is not that practically useful? Or is it not useful for text classification tasks? Quite the contrary, a well-trained token embedding space can dramatically improve the practical performance ceiling of a model like this. The issue in this particular case is with our training setup. We lack enough data in our 20,000 review examples to effectively train a good word embedding. By the end of our 10 training epochs, our train set accuracy has cracked 99%. Our model has begun to overfit and memorize our input, and it turns out it is doing so well before we have learned an optimal set of word embeddings for the task at hand. For cases like this, we can turn to pretraining. Rather than training our word embedding jointly with the classification task, we can train it separately, on more data, with

### PDF p.442 (printed 418) — 14 Text classification › 14.5 Sequence models › 14.5.5 Using the pretrained embedding for classification
Listing 14.30 Building another LSTM sequence model with an `Embedding` layer
inputs = keras.Input(shape=(max_length,))
lstm_embedding = layers.Embedding(
    input_dim=max_tokens,
    output_dim=hidden_dim,
    mask_zero=True,
)
x = lstm_embedding(inputs)
x = layers.Bidirectional(layers.LSTM(hidden_dim))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs, name="lstm_with_cbow")

### PDF p.440 (printed 416) — 14 Text classification › 14.5 Sequence models › 
After preprocessing, we can see that we have eight integer token IDs as context paired with a single token ID label.

The model we train with this data is exceedingly simple. We will use an `Embedding` layer to embed all context tokens and a `GlobalAveragePooling1D` to compute the average embedding of our “bag” of context tokens. Then, we use that average embedding to predict the value of our middle label token. That’s it! By repeatedly refining our embedding space so that we are good at predicting a word based on nearby word embeddings, we learn a rich embedding of tokens used in movie reviews.

Listing 14.28 Building a CBOW model

### PDF p.438 (printed 414) — 14 Text classification › 14.5 Sequence models › 14.5.4 Pretraining a word embedding
The last decade of rapid advancement in NLP has coincided with the rise of pretraining as the dominant approach for text modeling problems. Once we move past simple set-based regression models to sequence models with millions or even billions of parameters, text models become incredibly data-hungry. We are usually limited by our ability to find labeled examples for a particular problem in the text domain.

The idea is to devise an unsupervised task to train model parameters that do not need labeled data. Pretraining data can be text in a similar domain to our final task, or even just arbitrary text in the languages we are interested in working with. Pretraining allows us to learn general patterns in language, effectively priming our model before we specialize it to the final task we are interested in. Word embeddings were one of the first big successes with text pretraining, and we will show how to pretrain a word embedding in this section. Remember the `unsup/` directory we ignored in our IMDb dataset preparation? It contains another 25,000 reviews— the same size as our training data. We will combine all our training data together and show how to pretrain the parameters of an `Emb

### PDF p.486 (printed 462) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › 
The distance function used to compare tokens is cosine distance in both cases. Even the dimensionality of the embedding space is similar: a vector with somewhere between 1,000 and 10,000 dimensions to represent each word. At this point, you might interject: A Transformer is trained to predict missing words in a sequence, not to group tokens in an embedding space. How does the language model loss function relate to Word2Vec’s objective of maximizing the dot-product between co-occurring tokens? The answer is the attention mechanism. Attention is, by far, the most critical component in the Transformer architecture. It’s a mechanism for learning a new token embedding space by linearly recombining token embeddings from some prior space, in weighted combinations that give greater importance to tokens that are already “closer” to each other (i.e., that have a higher dot-product). It will tend to pull together the vectors of already close tokens, resulting over time in a space where token correlation relationships turn into embedding proximity relationships (in terms of cosine distance). Transformers work by learning a series of incrementally refined embedding spaces, each based on recombi
