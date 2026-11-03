# Generation bundle for page_428  (PDF page 428, printed page 404)

Page kind: content
Chapter: 14 Text classification
Section: 14.4 Set models
Subsection: 14.4.2 Training a bigram model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_428.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_428.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p428_e03] (code)
```
>>> x, y = next(bigram_train_ds.as_numpy_iterator())
>>> x.shape
(32, 30000)
```
[p428_e04] (body) If we look at a small subsection of our vocabulary, we can see both unigram and bigram terms:
[p428_e05] (code)
```
>>> text_vectorization.get_vocabulary()[100:108]
["in a", "most", "him", "dont", "it was", "one of", "for the", "made"]
```
[p428_e06] (body) With our new encoding for our input data, we can train a linear model unaltered from before.
[p428_e07] (listing_caption) Listing 14.17 Training the bigram regression model
[p428_e08] (code)
```
model = build_linear_classifier(max_tokens, "bigram_classifier")
model.fit(
    bigram_train_ds,
    validation_data=bigram_val_ds,
    epochs=10,
    callbacks=[early_stopping],
)
```
[p428_e09] (body) This model will be slightly larger than our bag-of words models (30,001 parameters instead of 20,001 parameters), but it still trains in about the same amount of time. How did it do?
[p428_e10] (listing_caption) Listing 14.18 Evaluating the bigram regression model
[p428_e11] (code)
```
>>> test_loss, test_acc = model.evaluate(bigram_test_ds)
>>> test_acc
0.90116
```
[p428_e12] (body) We’re now getting 90% test accuracy, a noticeable improvement!
[p428_e13] (body) We could improve this number even further by considering trigrams (triplets of words), although beyond trigrams, the problem quickly becomes intractable. The space of possible 4-grams of words in the English language is immense, and the problem grows exponentially as sequences get longer and longer. You would need an immense vocabulary to provide decent coverage of 4-grams, and your model would lose its ability to generalize, simply memorizing entire snippets of sentences with weights attached. To robustly consider longer-ordered text sequences, we will need more advanced modeling techniques.

## Listings
- Listing 14.17: Training the bigram regression model (code elements p428_e08)
- Listing 14.18: Evaluating the bigram regression model (code elements p428_e11)

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

## Previous page (PDF 427) text, for continuity
14.4.2 Training a bigram model

Of course, we can intuitively guess that discarding all word order is very reductive because even atomic concepts can be expressed via multiple words: the term “United States” conveys a concept that is quite distinct from the meaning of the words “states” and “united” taken separately. A movie that is “not bad” and a movie that is “bad” should probably get different sentiment scores.

Therefore, it is usually a good idea to inject some knowledge of local word ordering into a model, even for these simple set-based models we are currently building. One easy way to do that is to consider bigrams—a term for two tokens that appear consecutively in the input text. Given our example “this movie made me cry,” `{"this",` `"movie",` `"made",` `"me",` `"cry"}` is the set of all word unigrams in the input, and `{"this` `movie",` `"movie` `made",` `"made me",` `"me cry"}` is the set of all bigrams. The bag-of-words model we just trained could equivalently be called a unigram model, and the term n-gram refers to an ordered sequence of n tokens for any n. To add bigrams to our model, we want to consider the frequency of all bigrams while building our vocabulary. We could do this in two ways: by creating a vocabulary of only bigrams or by allowing both bigrams and unigrams to compete for space in the same vocabulary. For the latter case, the term `"United States"` will be included in our vocabulary before `"ventriloquism"` if it occurs more frequently in the input text. Again, we could build this by extending our `WordTokenizer` from earlier in the chapter, but there is no need. `TextVectorization` provides this out of the box. We will train a slightly larger vocabulary to account for the presence of bigrams, `adapt()` a new vocabulary, and multi-hot enc …

## Next page (PDF 429) text, for continuity
14.5 Sequence models

Our last two models indicated that sequence information is important. We improved a basic linear model by adding features with some info on local word order.

However, this was done by manually engineering input features, and we can see how the approach will only scale up to a local ordering of just a few words. As is often the case in deep learning, rather than attempting to build these features ourselves, we should expose the model to the raw word sequence and let it directly learn positional dependencies between tokens. Models that ingest a complete token sequence are called, simply enough, sequence models. We have a few choices for architecture here. We could build an RNN model as we just did for timeseries modeling. We could build a 1D ConvNet, similar to our image processing models, but convolving filters over a single sequence dimension. And as we will dig into in the next chapter, we can build a Transformer. Before taking on any of these approaches, we must preprocess our inputs into ordered sequences. We want an integer sequence of token IDs, as we saw in the tokenization portion of this chapter, but with one additional wrinkle to handle. When we run computations on a batch of inputs, we want all inputs to be rectangular so all calculations can be effectively parallelized across the batch on a GPU. However, tokenized inputs will almost always have varying lengths. IMDb movie reviews range from just a few sentences to multiple paragraphs, with varying word counts. To accommodate this fact, we can truncate our input sequences or “pad” them with another special token `"[PAD]"`, similar to the `"[UNK]"` token we used earlier. For example, given two input sentences and a desired length of eight

we would tokenize to the integer IDs for the foll …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.437 — 14 Text classification › 14.5 Sequence models
- PDF p.426 — 14 Text classification › 14.4 Set models
- PDF p.424 — 14 Text classification › 14.4 Set models
- PDF p.432 — 14 Text classification › 14.5 Sequence models
- PDF p.443 — 14 Text classification › 14.5 Sequence models
- PDF p.423 — 14 Text classification › 14.4 Set models
- PDF p.414 — 14 Text classification › 14.2 Preparing text data
- PDF p.425 — 14 Text classification › 14.4 Set models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.432 (printed 408) — 14 Text classification › 14.5 Sequence models › 
This is quite the step up in size from the unigram and bigram models. At about 15 million parameters, this is one of the larger models we have trained in the book so far, with only a single LSTM layer. Let’s trying training the model.

Listing 14.22 Training the LSTM sequence model

### PDF p.443 (printed 419) — 14 Text classification › 14.5 Sequence models › 
With the pretrained embedding weights, we have boosted our LSTM performance to roughly the same as our set-based models. We do slightly better than the unigram model and slightly worse than the bigram model.

This may seem like a bit of a letdown after all the work we put in. Is learning on the entire sequence, with order information, just a bad idea? The problem is that we are still data-constrained for our final LSTM model. The model is expressive and powerful enough that with enough movie reviews, we would easily outperform set-based approaches, but we need a lot more training on ordered data before our model’s performance ceiling is reached. This is an easily solvable problem with enough compute resources. In the next chapter, we will cover the transformer model. The model is slightly better at learning dependencies across longer token sequences, but most critically, these models are often trained on large amounts of English text, including all word order information. This allows the model to learn, roughly speaking, a statistical form of the grammatical patterns that govern language. These types of statistical patterns around word order are precisely why our current LSTM model

### PDF p.13 (printed xiii) — contents ›  › 
14.3

Sets vs. sequences 395

Loading the IMDb classification dataset 396

14.4

Set models 398

Training a bag-of-words model 399 ■ Training a bigram model 403

### PDF p.424 (printed 400) — 14 Text classification › 14.4 Set models › 
This model is dead simple. We have only 20,001 parameters, one for each word in our vocabulary and one for a bias term. Let’s train it. We’ll add on the `EarlyStopping` callback first covered in chapter 7, which will automatically stop when training when the validation loss stops improving and restore weights from the best epoch.

Listing 14.14 Training the bag-of-words regression model

### PDF p.437 (printed 413) — 14 Text classification › 14.5 Sequence models › 
With the embedding, we have reduced both our training time and model size by an order of magnitude. A learned embedding is clearly far more efficient than one-hot encoding our input.

However, the LSTM’s overall performance did not change. Accuracy was stubbornly around 84%, still a far cry from the bag-of-words and bigram models. Does this mean that a “structured embedding space” for input tokens is not that practically useful? Or is it not useful for text classification tasks? Quite the contrary, a well-trained token embedding space can dramatically improve the practical performance ceiling of a model like this. The issue in this particular case is with our training setup. We lack enough data in our 20,000 review examples to effectively train a good word embedding. By the end of our 10 training epochs, our train set accuracy has cracked 99%. Our model has begun to overfit and memorize our input, and it turns out it is doing so well before we have learned an optimal set of word embeddings for the task at hand. For cases like this, we can turn to pretraining. Rather than training our word embedding jointly with the classification task, we can train it separately, on more data, with

### PDF p.424 (printed 400) — 14 Text classification › 14.4 Set models › 
You can see that after preprocessing, each sample in our batch is converted into a vector of 20,000 numbers, each tracking the presence or absence of a vocabulary term.

Next, we can train a very simple linear model. We will save our model-building code as a function so we can use it again later.

Listing 14.13 Building a bag-of-words regression model

### PDF p.485 (printed 461) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › 
In just a single epoch of training, our model reached 93%, a noticeable improvement from the 90% ceiling we hit in our last chapter. Of course, this is a far more expensive model to use than the simple bigram classifier we built previously, but there are clear benefits to using such a large model. And this is all with the smaller size of the RoBERTa model. Using the larger 300 million parameter model, we could achieve an accuracy of over 95%.

### PDF p.438 (printed 414) — 14 Text classification › 14.5 Sequence models › 14.5.4 Pretraining a word embedding
The last decade of rapid advancement in NLP has coincided with the rise of pretraining as the dominant approach for text modeling problems. Once we move past simple set-based regression models to sequence models with millions or even billions of parameters, text models become incredibly data-hungry. We are usually limited by our ability to find labeled examples for a particular problem in the text domain.

The idea is to devise an unsupervised task to train model parameters that do not need labeled data. Pretraining data can be text in a similar domain to our final task, or even just arbitrary text in the languages we are interested in working with. Pretraining allows us to learn general patterns in language, effectively priming our model before we specialize it to the final task we are interested in. Word embeddings were one of the first big successes with text pretraining, and we will show how to pretrain a word embedding in this section. Remember the `unsup/` directory we ignored in our IMDb dataset preparation? It contains another 25,000 reviews— the same size as our training data. We will combine all our training data together and show how to pretrain the parameters of an `Emb
