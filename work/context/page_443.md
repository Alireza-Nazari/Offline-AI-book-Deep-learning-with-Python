# Generation bundle for page_443  (PDF page 443, printed page 419)

Page kind: content
Chapter: 14 Text classification
Section: 14.5 Sequence models
Subsection: 14.5.5 Using the pretrained embedding for classification
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_443.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_443.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p443_e03] (body) With the pretrained embedding weights, we have boosted our LSTM performance to roughly the same as our set-based models. We do slightly better than the unigram model and slightly worse than the bigram model.
[p443_e04] (body) This may seem like a bit of a letdown after all the work we put in. Is learning on the entire sequence, with order information, just a bad idea? The problem is that we are still data-constrained for our final LSTM model. The model is expressive and powerful enough that with enough movie reviews, we would easily outperform set-based approaches, but we need a lot more training on ordered data before our model’s performance ceiling is reached. This is an easily solvable problem with enough compute resources. In the next chapter, we will cover the transformer model. The model is slightly better at learning dependencies across longer token sequences, but most critically, these models are often trained on large amounts of English text, including all word order information. This allows the model to learn, roughly speaking, a statistical form of the grammatical patterns that govern language. These types of statistical patterns around word order are precisely why our current LSTM model is too data-constrained to learn effectively. However, as we move on to large, more advanced models that will push the limits of text-classification performance, it is worth pointing out that simple set-based regression approaches like our bigram model give you a lot of “bang for your buck.” Set-based models are lightning-fast and can contain just a few thousand parameters, a far cry from the billion-parameter large language models that dominate the news today. If you are working in an environment where compute is limited and you can sacrifice some accuracy, set-based models can often be the most cost-effective approach.
[p443_e05] (heading_section) Summary
[p443_e06] (bullet) All text modeling problems involve a preprocessing step where text is broken up and transformed to integer data, called tokenization.
[p443_e07] (bullet) Tokenization can be divided into three steps: standardization, splitting, and indexing. Standardization normalizes text, splitting breaks text up into tokens, and indexing assigns each token a unique integer ID.
[p443_e08] (bullet) There are three main types of tokenization: character, word, and subword tokenization. With an expressive-enough model and sufficient training data, the subword tokenization is usually the most effective.
[p443_e09] (bullet) NLP models differ primarily in handling the order of input tokens: – Set models discard most order information and learn simple and fast models based solely on the presence or absence of tokens in the input. Bigram or trigram models consider the presence or absence of two or three consecutive tokens. Set models are incredibly fast to train and deploy. – Sequence models attempt to learn with the ordered sequence of tokens in the input data. Sequence models need large amounts of data to learn effectively.

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

## Previous page (PDF 442) text, for continuity
14.5.5 Using the pretrained embedding for classification

Now that we have trained a new word embedding, applying it to our LSTM model is simple. First, we create the model precisely as we did before.

Listing 14.30 Building another LSTM sequence model with an `Embedding` layer

Then, we apply our embedding weights from the CBOW embedding layer to the LSTM embedding layer. This effectively acts as a new and better initializer for the roughly 2 million embedding parameters in the LSTM model.

Listing 14.31 Reusing the CBOW embedding to prime the LSTM model

With that, we can compile and train our LSTM model as normal.

Listing 14.32 Training the LSTM model with a pretrained embedding.

Let’s evaluate our LSTM model.

Listing 14.33 Evaluating the LSTM model with a pretrained embedding

## Next page (PDF 444) text, for continuity
An embedding is an efficient way to transform token IDs into a learned latent space. Embeddings can be trained normally with gradient descent.

Pretraining is critical for sequence models as a way to get around the data-hungry nature of these models. During pretraining, an unsupervised task allows models to learn from large amounts of unlabeled text data. The learned parameters can then be transferred to a downstream task.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.419 — 14 Text classification › 14.3 Sets vs. sequences
- PDF p.429 — 14 Text classification › 14.5 Sequence models
- PDF p.422 — 14 Text classification › 14.4 Set models
- PDF p.427 — 14 Text classification › 14.4 Set models
- PDF p.437 — 14 Text classification › 14.5 Sequence models
- PDF p.438 — 14 Text classification › 14.5 Sequence models
- PDF p.488 — 15 Language models and the Transformer › 15.5 What makes the Transformer effective?
- PDF p.411 — 14 Text classification › 14.2 Preparing text data

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.437 (printed 413) — 14 Text classification › 14.5 Sequence models › 
With the embedding, we have reduced both our training time and model size by an order of magnitude. A learned embedding is clearly far more efficient than one-hot encoding our input.

However, the LSTM’s overall performance did not change. Accuracy was stubbornly around 84%, still a far cry from the bag-of-words and bigram models. Does this mean that a “structured embedding space” for input tokens is not that practically useful? Or is it not useful for text classification tasks? Quite the contrary, a well-trained token embedding space can dramatically improve the practical performance ceiling of a model like this. The issue in this particular case is with our training setup. We lack enough data in our 20,000 review examples to effectively train a good word embedding. By the end of our 10 training epochs, our train set accuracy has cracked 99%. Our model has begun to overfit and memorize our input, and it turns out it is doing so well before we have learned an optimal set of word embeddings for the task at hand. For cases like this, we can turn to pretraining. Rather than training our word embedding jointly with the classification task, we can train it separately, on more data, with

### PDF p.438 (printed 414) — 14 Text classification › 14.5 Sequence models › 14.5.4 Pretraining a word embedding
The last decade of rapid advancement in NLP has coincided with the rise of pretraining as the dominant approach for text modeling problems. Once we move past simple set-based regression models to sequence models with millions or even billions of parameters, text models become incredibly data-hungry. We are usually limited by our ability to find labeled examples for a particular problem in the text domain.

The idea is to devise an unsupervised task to train model parameters that do not need labeled data. Pretraining data can be text in a similar domain to our final task, or even just arbitrary text in the languages we are interested in working with. Pretraining allows us to learn general patterns in language, effectively priming our model before we specialize it to the final task we are interested in. Word embeddings were one of the first big successes with text pretraining, and we will show how to pretrain a word embedding in this section. Remember the `unsup/` directory we ignored in our IMDb dataset preparation? It contains another 25,000 reviews— the same size as our training data. We will combine all our training data together and show how to pretrain the parameters of an `Emb

### PDF p.429 (printed 405) — 14 Text classification › 14.5 Sequence models › 14.5 Sequence models
Our last two models indicated that sequence information is important. We improved a basic linear model by adding features with some info on local word order.

However, this was done by manually engineering input features, and we can see how the approach will only scale up to a local ordering of just a few words. As is often the case in deep learning, rather than attempting to build these features ourselves, we should expose the model to the raw word sequence and let it directly learn positional dependencies between tokens. Models that ingest a complete token sequence are called, simply enough, sequence models. We have a few choices for architecture here. We could build an RNN model as we just did for timeseries modeling. We could build a 1D ConvNet, similar to our image processing models, but convolving filters over a single sequence dimension. And as we will dig into in the next chapter, we can build a Transformer. Before taking on any of these approaches, we must preprocess our inputs into ordered sequences. We want an integer sequence of token IDs, as we saw in the tokenization portion of this chapter, but with one additional wrinkle to handle. When we run computations on a batc

### PDF p.479 (printed 455) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
model were released a year after “Attention Is All You Need.” The model structure was exactly the same as the encoder part of the translation Transformer we just built. This encoder model is bidirectional in that every position in the sequence can attend to positions in front of and behind it. This means it’s a good model for computing a rich representation of input text, but not a model meant to run generation in a loop.

BERT was trained in sizes between 100 million and 300 million parameters, much bigger than the 14 million parameter Transformer we just trained. This meant the model needed a lot of training data to perform well. To achieve this, the authors used a riff on the classic language modeling setup called masked language modeling. To pretrain the model, we take a sequence of text and replace about 15% of the tokens with a special `[MASK]` token. The model will attempt to predict the original masked token values during training. While the classic language model, sometimes called a causal language model, attempts to predict `p(token|past` `tokens)`, the masked language model attempts to predict `p(token|surrounding` `tokens)`. This training setup is unsupervised. You don’

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

### PDF p.488 (printed 464) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › Summary
A language model is a model that learns a specific probability distribution— `p(token|past tokens)`: – Language models have broad applications, but the most important is that you can generate text by calling them in a loop, where the output token at one time step becomes the input token in the next. – A masked language model learns a related probability distribution `p(tokens|surrounding` `tokens)` and can be helpful for classifying text and individual tokens. – A sequence-to-sequence language model learns to predict the next token given both past tokens in a target sequence and an entirely separate, fixed source sequence. Sequence-to-sequence models are useful for problems like translation and question answering. – A sequence-to-sequence model usually has two separate components. An encoder computes a representation of the source sequence, and a decoder takes this representation as input and predicts the next token in a target sequence based on past tokens.

Attention is a mechanism that allows a model to pull information from anywhere in a sequence selectively based on the context of the token currently being processed: – Attention avoids the problems RNNs have with long-range de

### PDF p.422 (printed 398) — 14 Text classification › 14.4 Set models › 14.4 Set models
The simplest approach we can take regarding the ordering of tokens in text is to discard it. We still tokenize our input reviews normally as a sequence of token IDs, but immediately after tokenization, we convert the entire training example to a set—a simple unordered “bag” of tokens that are either present or absent in a movie review.

The idea here is to use these sets to build a very simple model that assigns a weight to every individual word in a review. The presence of the word `"terrible"` would probably (though not always) indicate a bad review, and `"riveting"` might indicate a good review. We can build a small model that can learn these weights—called a bag-of-words model. For example, let’s say you had a simple input sentence and vocabulary:
