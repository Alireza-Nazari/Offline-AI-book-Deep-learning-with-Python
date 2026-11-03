# Generation bundle for page_437  (PDF page 437, printed page 413)

Page kind: content
Chapter: 14 Text classification
Section: 14.5 Sequence models
Subsection: 14.5.3 Using a word embedding
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_437.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_437.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p437_e03] (code)
```
 Total params: 1,986,177 (7.58 MB)
 Trainable params: 1,986,177 (7.58 MB)
 Non-trainable params: 0 (0.00 B)
```
[p437_e04] (body) We have reduced the number of parameters for our one-hot-encoded LSTM model from 15 million to 2 million. Let’s train and evaluate the model.
[p437_e05] (listing_caption) Listing 14.25 Training and evaluating the LSTM with an `Embedding` layer
[p437_e06] (code)
```
>>> model.fit(
...     sequence_train_ds,
...     validation_data=sequence_val_ds,
...     epochs=10,
...     callbacks=[early_stopping],
... )
>>> test_loss, test_acc = model.evaluate(sequence_test_ds)
>>> test_acc
0.8443599939346313
```
[p437_e07] (body) With the embedding, we have reduced both our training time and model size by an order of magnitude. A learned embedding is clearly far more efficient than one-hot encoding our input.
[p437_e08] (body) However, the LSTM’s overall performance did not change. Accuracy was stubbornly around 84%, still a far cry from the bag-of-words and bigram models. Does this mean that a “structured embedding space” for input tokens is not that practically useful? Or is it not useful for text classification tasks? Quite the contrary, a well-trained token embedding space can dramatically improve the practical performance ceiling of a model like this. The issue in this particular case is with our training setup. We lack enough data in our 20,000 review examples to effectively train a good word embedding. By the end of our 10 training epochs, our train set accuracy has cracked 99%. Our model has begun to overfit and memorize our input, and it turns out it is doing so well before we have learned an optimal set of word embeddings for the task at hand. For cases like this, we can turn to pretraining. Rather than training our word embedding jointly with the classification task, we can train it separately, on more data, without the need for positive and negative review labels. Let’s take a look.
[p437_e09] (heading_subsection) Augmenting text data
[p437_e10] (sidebar) After seeing the importance of data augmentation for computer vision problems, you might wonder if we can do the same for text. The short answer is yes, you can, although it is not nearly as effective in the text domain. Basic text augmentation techniques look for basic edits we can make to input text that might help make our model more robust. For example, we might randomly delete

## Listings
- Listing 14.25: Training and evaluating the LSTM with an `Embedding` layer (code elements p437_e06)

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

## Previous page (PDF 436) text, for continuity
The third argument, `mask_zero=True`, is a little more subtle. This argument tells Keras which inputs in our sequence are `"[PAD]"` tokens, so we can mask these entries later in the model. Remember that when preprocessing our sequence input, we might add a lot of padding tokens to our original input so that a token sequence might look like:

All of those padding tokens will be first embedded and then fed into the `LSTM` layer. This means the last representation we receive from the `LSTM` cell might contain the results of processing the `"[PAD]"` token representation over and over recurrently. We are not very interested in the learned `LSTM` representation for the last `"[PAD]"` token in the previous sequence. Instead, we are interested in the representation of `"awful"`, the last non-padding token. Or, put equivalently, we want to mask all of the `"[PAD]"` tokens so that they do not affect our final output prediction.

`mask_zero=True` is simply a shorthand to easily do such masking in Keras with the `Embedding` layer. Keras will mark all elements in our sequence that initially contained a zero value, where zero is assumed to be the token ID for the `"[PAD]"` token. This mask will be used internally by the `LSTM` layer. Instead of outputting the last learned representation for the whole sequence, it will output the last non-masked representation. This form of masking is implicit and easy to use, but you can always be explicit about which items in a sequence you would like to mask if the need arises. The `LSTM` layer takes an optional `mask` call argument, for explicit or custom masking. Before we train this new model, let’s take a look at the model summary:

## Next page (PDF 438) text, for continuity
(continued)

or swap the position of words in a sentence, so the sentence “The rain in Spain falls mainly on the plain” becomes “The rain Spain falls plain on the mainly.” Training your model on such edited inputs can make it robust to typos and grammar mistakes. However, this example also concisely shows the big pitfall with text augmentation—it is easy to alter the meaning of an input example unintentionally. Unlike image data, where you can crop, rotate, and adjust color levels on a picture of a cat and still have a recognizable cat on the other end, language is order-dependent and highly sensitive to tiny changes. A sentence with two words swapped might mean the opposite of the input sentence. Some augmentation techniques seek to solve this by replacing words from a table of known synonyms, but this, too, can be brittle if we choose the incorrect meaning of a word. These issues have kept text augmentation from being widespread in practice. It is usually a better idea to seek more text examples than to sink time into text augmentation techniques. Generative models, which we will see in the coming chapters, are beginning to offer a new form of text augmentation that can alleviate these pain points. By generating outputs from a model that has learned how to produce consistent and coherent text, we can create completely unseen inputs that plausibly resemble our input data. This poses its own challenges but opens up a new frontier for text augmentation on problems where data is particularly sparse and challenging to collect.

14.5.4 Pretraining a word embedding

The last decade of rapid advancement in NLP has coincided with the rise of pretraining as the dominant approach for text modeling problems. Once we move past simple set-based regression models to sequence models  …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.442 — 14 Text classification › 14.5 Sequence models
- PDF p.432 — 14 Text classification › 14.5 Sequence models
- PDF p.441 — 14 Text classification › 14.5 Sequence models
- PDF p.435 — 14 Text classification › 14.5 Sequence models
- PDF p.443 — 14 Text classification › 14.5 Sequence models
- PDF p.440 — 14 Text classification › 14.5 Sequence models
- PDF p.430 — 14 Text classification › 14.5 Sequence models
- PDF p.439 — 14 Text classification › 14.5 Sequence models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.443 (printed 419) — 14 Text classification › 14.5 Sequence models › 
With the pretrained embedding weights, we have boosted our LSTM performance to roughly the same as our set-based models. We do slightly better than the unigram model and slightly worse than the bigram model.

This may seem like a bit of a letdown after all the work we put in. Is learning on the entire sequence, with order information, just a bad idea? The problem is that we are still data-constrained for our final LSTM model. The model is expressive and powerful enough that with enough movie reviews, we would easily outperform set-based approaches, but we need a lot more training on ordered data before our model’s performance ceiling is reached. This is an easily solvable problem with enough compute resources. In the next chapter, we will cover the transformer model. The model is slightly better at learning dependencies across longer token sequences, but most critically, these models are often trained on large amounts of English text, including all word order information. This allows the model to learn, roughly speaking, a statistical form of the grammatical patterns that govern language. These types of statistical patterns around word order are precisely why our current LSTM model

### PDF p.442 (printed 418) — 14 Text classification › 14.5 Sequence models › 14.5.5 Using the pretrained embedding for classification
Now that we have trained a new word embedding, applying it to our LSTM model is simple. First, we create the model precisely as we did before.

Listing 14.30 Building another LSTM sequence model with an `Embedding` layer

### PDF p.442 (printed 418) — 14 Text classification › 14.5 Sequence models › 14.5.5 Using the pretrained embedding for classification
Then, we apply our embedding weights from the CBOW embedding layer to the LSTM embedding layer. This effectively acts as a new and better initializer for the roughly 2 million embedding parameters in the LSTM model.

Listing 14.31 Reusing the CBOW embedding to prime the LSTM model

### PDF p.444 (printed 420) — 14 Text classification › 14.5 Sequence models › 
An embedding is an efficient way to transform token IDs into a learned latent space. Embeddings can be trained normally with gradient descent.

Pretraining is critical for sequence models as a way to get around the data-hungry nature of these models. During pretraining, an unsupervised task allows models to learn from large amounts of unlabeled text data. The learned parameters can then be transferred to a downstream task.

### PDF p.440 (printed 416) — 14 Text classification › 14.5 Sequence models › 
After preprocessing, we can see that we have eight integer token IDs as context paired with a single token ID label.

The model we train with this data is exceedingly simple. We will use an `Embedding` layer to embed all context tokens and a `GlobalAveragePooling1D` to compute the average embedding of our “bag” of context tokens. Then, we use that average embedding to predict the value of our middle label token. That’s it! By repeatedly refining our embedding space so that we are good at predicting a word based on nearby word embeddings, we learn a rich embedding of tokens used in movie reviews.

Listing 14.28 Building a CBOW model

### PDF p.435 (printed 411) — 14 Text classification › 14.5 Sequence models › 
Figure 14.5 An `Embedding` layer acts as a dictionary mapping ints to floating point vectors.

When you instantiate an `Embedding` layer, its weights (its internal dictionary of token vectors) are initially random, just as with any other layer. During training, these word vectors are gradually adjusted via backpropagation, structuring the space into something the downstream model can exploit. Once fully trained, the embedding space will show a lot of structure—a kind of structure specialized for the specific problem for which you’re training your model.

Let’s build a model with an `Embedding` layer and benchmark it on our task.

Listing 14.24 Building an LSTM sequence model with an `Embedding` layer

### PDF p.434 (printed 410) — 14 Text classification › 14.5 Sequence models › 14.5.3 Using a word embedding
Is there an ideal word-embedding space that perfectly maps human language and can be used for any NPL task? Possibly, but we have yet to compute anything of the sort. Also, there is no single human language we could attempt to map—there are many different languages, and they aren’t isomorphic to one another because a language is the reflection of a specific culture and a particular context. More pragmatically, what makes a good word-embedding space depends heavily on your task: the perfect word-embedding space for an English-language movie review sentiment-analysis model may look different from the ideal embedding space for an English-language legal document classification model because the importance of certain semantic relationships varies from task to task.

It’s thus reasonable to learn a new embedding space with every new task. Fortunately, backpropagation makes this easy, and Keras makes it even easier. It’s about learning the weights of the Keras `Embedding` layer. The `Embedding` layer is best understood as a dictionary that maps integer indices (which stand for specific words) to dense vectors. It takes integers as input, looks them up in an internal dictionary, and return

### PDF p.479 (printed 455) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
model were released a year after “Attention Is All You Need.” The model structure was exactly the same as the encoder part of the translation Transformer we just built. This encoder model is bidirectional in that every position in the sequence can attend to positions in front of and behind it. This means it’s a good model for computing a rich representation of input text, but not a model meant to run generation in a loop.

BERT was trained in sizes between 100 million and 300 million parameters, much bigger than the 14 million parameter Transformer we just trained. This meant the model needed a lot of training data to perform well. To achieve this, the authors used a riff on the classic language modeling setup called masked language modeling. To pretrain the model, we take a sequence of text and replace about 15% of the tokens with a special `[MASK]` token. The model will attempt to predict the original masked token values during training. While the classic language model, sometimes called a causal language model, attempts to predict `p(token|past` `tokens)`, the masked language model attempts to predict `p(token|surrounding` `tokens)`. This training setup is unsupervised. You don’
