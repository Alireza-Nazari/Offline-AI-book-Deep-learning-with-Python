# Generation bundle for page_427  (PDF page 427, printed page 403)

Page kind: content
Chapter: 14 Text classification
Section: 14.4 Set models
Subsection: 14.4.2 Training a bigram model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_427.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_427.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p427_e03] (heading_subsection) 14.4.2 Training a bigram model
[p427_e04] (body) Of course, we can intuitively guess that discarding all word order is very reductive because even atomic concepts can be expressed via multiple words: the term “United States” conveys a concept that is quite distinct from the meaning of the words “states” and “united” taken separately. A movie that is “not bad” and a movie that is “bad” should probably get different sentiment scores.
[p427_e05] (body) Therefore, it is usually a good idea to inject some knowledge of local word ordering into a model, even for these simple set-based models we are currently building. One easy way to do that is to consider bigrams—a term for two tokens that appear consecutively in the input text. Given our example “this movie made me cry,” `{"this",` `"movie",` `"made",` `"me",` `"cry"}` is the set of all word unigrams in the input, and `{"this` `movie",` `"movie` `made",` `"made me",` `"me cry"}` is the set of all bigrams. The bag-of-words model we just trained could equivalently be called a unigram model, and the term n-gram refers to an ordered sequence of n tokens for any n. To add bigrams to our model, we want to consider the frequency of all bigrams while building our vocabulary. We could do this in two ways: by creating a vocabulary of only bigrams or by allowing both bigrams and unigrams to compete for space in the same vocabulary. For the latter case, the term `"United States"` will be included in our vocabulary before `"ventriloquism"` if it occurs more frequently in the input text. Again, we could build this by extending our `WordTokenizer` from earlier in the chapter, but there is no need. `TextVectorization` provides this out of the box. We will train a slightly larger vocabulary to account for the presence of bigrams, `adapt()` a new vocabulary, and multi-hot encode output vectors including bigrams.
[p427_e06] (listing_caption) Listing 14.16 Applying a bigram encoding to the IMDb reviews
[p427_e07] (code)
```
max_tokens = 30_000
text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    split="whitespace",
    output_mode="multi_hot",
    ngrams=2,
)
text_vectorization.adapt(train_ds_no_labels)
bigram_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y), num_parallel_calls=8
)
bigram_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y), num_parallel_calls=8
)
bigram_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y), num_parallel_calls=8
)
```
[p427_e08] (body) Let’s examine a batch of our preprocessed input again:

## Code annotations printed next to listings (callouts, in order)
- Learns a word-level
- vocabulary
- Considers all
- unigrams and bigrams

## Listings
- Listing 14.16: Applying a bigram encoding to the IMDb reviews (code elements p427_e07)

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

## Previous page (PDF 426) text, for continuity
Listing 14.15 Evaluating the bag-of-words regression model

We can correctly predict the sentiment of a review 88% of the time with a training job light enough that it could run efficiently on a single CPU.

It is worth noting our choice of word tokenization in this example. The reason to avoid character-level tokenization here is pretty obvious—a “bag” of all characters in a movie review will tell you very little about its content. Subword tokenization with a large enough vocabulary would be a good choice, but there is little need for it here. Since the model we are training is so small, it’s convenient to use a vocabulary that is quick to train and have our weights correspond to actual English words.

Preprocessing text efficiently

In all applied machine learning, the speed and efficiency of preprocessing are important concerns. A faster program is always desirable, but this becomes more urgent when the cost of accelerators (GPUs and TPUs) is so high. You want to avoid letting expensive GPUs idle while you preprocess your input! Text preprocessing is unique because it must always run on a CPU. GPUs strictly handle numeric inputs, so all tokenization must happen before your GPU’s train step. One option is to precompute your tokenized input—tokenization does not depend on model weights, so you could tokenize all input text files and resave them as integer sequences before starting training. However, this is not always practical. Tokenizing text on the fly allows for more rapid experimentation. If you are running inference on an unseen example, there is no way to precompute the tokenized input; you need to tokenize and run a forward pass in rapid succession. The name of the game when preprocessing text input on the fly is to be “fast enough.” You want to ensure your exp …

## Next page (PDF 428) text, for continuity
If we look at a small subsection of our vocabulary, we can see both unigram and bigram terms:

With our new encoding for our input data, we can train a linear model unaltered from before.

Listing 14.17 Training the bigram regression model

This model will be slightly larger than our bag-of words models (30,001 parameters instead of 20,001 parameters), but it still trains in about the same amount of time. How did it do?

Listing 14.18 Evaluating the bigram regression model

We’re now getting 90% test accuracy, a noticeable improvement!

We could improve this number even further by considering trigrams (triplets of words), although beyond trigrams, the problem quickly becomes intractable. The space of possible 4-grams of words in the English language is immense, and the problem grows exponentially as sequences get longer and longer. You would need an immense vocabulary to provide decent coverage of 4-grams, and your model would lose its ability to generalize, simply memorizing entire snippets of sentences with weights attached. To robustly consider longer-ordered text sequences, we will need more advanced modeling techniques.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.430 — 14 Text classification › 14.5 Sequence models
- PDF p.422 — 14 Text classification › 14.4 Set models
- PDF p.423 — 14 Text classification › 14.4 Set models
- PDF p.439 — 14 Text classification › 14.5 Sequence models
- PDF p.437 — 14 Text classification › 14.5 Sequence models
- PDF p.424 — 14 Text classification › 14.4 Set models
- PDF p.443 — 14 Text classification › 14.5 Sequence models
- PDF p.429 — 14 Text classification › 14.5 Sequence models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.422 (printed 398) — 14 Text classification › 14.4 Set models › 14.4 Set models
The simplest approach we can take regarding the ordering of tokens in text is to discard it. We still tokenize our input reviews normally as a sequence of token IDs, but immediately after tokenization, we convert the entire training example to a set—a simple unordered “bag” of tokens that are either present or absent in a movie review.

The idea here is to use these sets to build a very simple model that assigns a weight to every individual word in a review. The presence of the word `"terrible"` would probably (though not always) indicate a bad review, and `"riveting"` might indicate a good review. We can build a small model that can learn these weights—called a bag-of-words model. For example, let’s say you had a simple input sentence and vocabulary:

### PDF p.422 (printed 398) — 14 Text classification › 14.4 Set models › 14.4 Set models
"this movie made me cry"
{"[UNK]": 0, "movie": 1, "film": 2, "made": 3, "laugh": 4, "cry": 5}

### PDF p.443 (printed 419) — 14 Text classification › 14.5 Sequence models › 
With the pretrained embedding weights, we have boosted our LSTM performance to roughly the same as our set-based models. We do slightly better than the unigram model and slightly worse than the bigram model.

This may seem like a bit of a letdown after all the work we put in. Is learning on the entire sequence, with order information, just a bad idea? The problem is that we are still data-constrained for our final LSTM model. The model is expressive and powerful enough that with enough movie reviews, we would easily outperform set-based approaches, but we need a lot more training on ordered data before our model’s performance ceiling is reached. This is an easily solvable problem with enough compute resources. In the next chapter, we will cover the transformer model. The model is slightly better at learning dependencies across longer token sequences, but most critically, these models are often trained on large amounts of English text, including all word order information. This allows the model to learn, roughly speaking, a statistical form of the grammatical patterns that govern language. These types of statistical patterns around word order are precisely why our current LSTM model

### PDF p.439 (printed 415) — 14 Text classification › 14.5 Sequence models › 
text in a dataset, where we continuously attempt to guess a missing word based on the words that appear to its direct right and left (figure 14.6). For example, if our “bag” of surrounding words contained the words “sail,” “wave,” and “mast,” we might guess that the middle word is “boat” or “ocean.”

Figure 14.6 The Continuous Bag of Words predicts a word based on its surrounding context with a shallow neural network.

In our particular IMDb classification problem, we are interested in “priming” the word embedding of the LSTM model we just trained. We can reuse the `TextVectorization` vocabulary we computed earlier. All we are trying to do here is to learn a good 64-dimensional vector for each word in this vocabulary.

We can create a new `TextVectorization` layer with the same vocabulary that does not truncate or pad input. We will preprocess the output tokens of this layer by sliding a context window across our text.

Listing 14.26 Removing padding from our `TextVectorization` preprocessing layer

### PDF p.429 (printed 405) — 14 Text classification › 14.5 Sequence models › 14.5 Sequence models
Our last two models indicated that sequence information is important. We improved a basic linear model by adding features with some info on local word order.

However, this was done by manually engineering input features, and we can see how the approach will only scale up to a local ordering of just a few words. As is often the case in deep learning, rather than attempting to build these features ourselves, we should expose the model to the raw word sequence and let it directly learn positional dependencies between tokens. Models that ingest a complete token sequence are called, simply enough, sequence models. We have a few choices for architecture here. We could build an RNN model as we just did for timeseries modeling. We could build a 1D ConvNet, similar to our image processing models, but convolving filters over a single sequence dimension. And as we will dig into in the next chapter, we can build a Transformer. Before taking on any of these approaches, we must preprocess our inputs into ordered sequences. We want an integer sequence of token IDs, as we saw in the tokenization portion of this chapter, but with one additional wrinkle to handle. When we run computations on a batc

### PDF p.437 (printed 413) — 14 Text classification › 14.5 Sequence models › 
With the embedding, we have reduced both our training time and model size by an order of magnitude. A learned embedding is clearly far more efficient than one-hot encoding our input.

However, the LSTM’s overall performance did not change. Accuracy was stubbornly around 84%, still a far cry from the bag-of-words and bigram models. Does this mean that a “structured embedding space” for input tokens is not that practically useful? Or is it not useful for text classification tasks? Quite the contrary, a well-trained token embedding space can dramatically improve the practical performance ceiling of a model like this. The issue in this particular case is with our training setup. We lack enough data in our 20,000 review examples to effectively train a good word embedding. By the end of our 10 training epochs, our train set accuracy has cracked 99%. Our model has begun to overfit and memorize our input, and it turns out it is doing so well before we have learned an optimal set of word embeddings for the task at hand. For cases like this, we can turn to pretraining. Rather than training our word embedding jointly with the classification task, we can train it separately, on more data, with

### PDF p.423 (printed 399) — 14 Text classification › 14.4 Set models › 14.4.1 Training a bag-of-words model
To do this text processing in code, it would be easy enough to extend our `WordTokenizer` from earlier in the chapter. An even easier solution is to use the `TextVectorization` layer built into Keras. The `TextVectorization` handles word and character tokenization and comes with several additional features, including multi-hot encoding of the layer output.

The `TextVectorization` layer, like many preprocessing layers in Keras, has an `adapt()` method to learn a layer state from input data. In the case of `TextVectorization`, `adapt()` will learn a vocabulary for a dataset on the fly by iterating over an input dataset. Let’s use it to tokenize and encode our input data. We will build a vocabulary of 20,000 words, a good starting place for text classification problems.

Listing 14.12 Applying a bag-of-words encoding to the IMDb reviews

### PDF p.13 (printed xiii) — contents ›  › 
14.3

Sets vs. sequences 395

Loading the IMDb classification dataset 396

14.4

Set models 398

Training a bag-of-words model 399 ■ Training a bigram model 403
