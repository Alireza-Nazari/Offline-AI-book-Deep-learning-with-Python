# Generation bundle for page_449  (PDF page 449, printed page 425)

Page kind: content
Chapter: 15 Language models and the Transformer
Section: 15.1 The language model
Subsection: 15.1.1 Training a Shakespeare language model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_449.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_449.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p449_e03] (listing_caption) Listing 15.4 Building a miniature language model
[p449_e04] (code)
```
embedding_dim = 256
hidden_dim = 1024
inputs = layers.Input(shape=(sequence_length,), dtype="int", name="token_ids")
x = layers.Embedding(vocabulary_size, embedding_dim)(inputs)
x = layers.GRU(hidden_dim, return_sequences=True)(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(vocabulary_size, activation="softmax")(x)
model = keras.Model(inputs, outputs)
```
[p449_e05] (body) Let’s take a look at our model summary:
[p449_e06] (code)
```
>>> model.summary()
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ token_ids (InputLayer)            │ (None, 100)              │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ embedding (Embedding)             │ (None, 100, 256)         │        17,152 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ gru (GRU)                         │ (None, 100, 1024)        │     3,938,304 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ dropout (Dropout)                 │ (None, 100, 1024)        │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ dense (Dense)                     │ (None, 100, 67)          │        68,675 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
 Total params: 4,024,131 (15.35 MB)
 Trainable params: 4,024,131 (15.35 MB)
 Non-trainable params: 0 (0.00 B)
```
[p449_e07] (body) This model outputs a softmax probability for every possible character in our vocabulary, and we will `compile()` it with a crossentropy loss. Note that our model is still training on a classification problem, it’s just that we will make one classification prediction for every token in our sequence. For our batch of 64 samples with 100 characters each, we will predict 6,400 individual labels. Loss and accuracy metrics reported by Keras during training will be averaged first across each sequence and, second, across each batch.
[p449_e08] (body) Let’s go ahead and train our language model.
[p449_e09] (listing_caption) Listing 15.5 Training a miniature language model
[p449_e10] (code)
```
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
```

## Code annotations printed next to listings (callouts, in order)
- Outputs a probability
- distribution over all potential
- tokens in our vocabulary

## Listings
- Listing 15.4: Building a miniature language model (code elements p449_e04)
- Listing 15.5: Training a miniature language model (code elements p449_e10)

## Chapter outline
Chapter: 15 Language models and the Transformer (PDF pages 445-489)
  - 15.1 The language model (p.445)
      - 15.1.1 Training a Shakespeare language model (p.446)
      - 15.1.2 Generating Shakespeare (p.450)
  - 15.2 Sequence-to-sequence learning (p.452)
      - 15.2.1 English-to-Spanish translation (p.454)
      - 15.2.2 Sequence-to-sequence learning with RNNs (p.456)
  - 15.3 The Transformer architecture (p.461)
      - 15.3.1 Dot-product attention (p.463)
      - 15.3.2 Transformer encoder block (p.468)
      - 15.3.3 Transformer decoder block (p.470)
      - 15.3.4 Sequence-to-sequence learning with a Transformer (p.472)
      - 15.3.5 Embedding positional information (p.475)
  - 15.4 Classification with a pretrained Transformer (p.478)
      - 15.4.1 Pretraining a Transformer encoder (p.478)
      - 15.4.2 Loading a pretrained Transformer (p.479)
      - 15.4.3 Preprocessing IMDb movie reviews (p.482)
      - 15.4.4 Fine-tuning a pretrained Transformer (p.484)
  - 15.5 What makes the Transformer effective? (p.485)

## Previous page (PDF 448) text, for continuity
To map this input to a sequence of integers, we can again use the `TextVectorization` layer we saw in the last chapter. To learn a character-level vocabulary instead of a word-level vocabulary, we can change our `split` argument. Rather than the default `"whitespace"` splitting, we instead split by `"character"`. We will do no standardization here—to keep things simple, we will preserve case and pass punctuation through unaltered.

Listing 15.3 Learning a character-level vocabulary with the `TextVectorization` layer

Let’s inspect the vocabulary:

We need only 67 characters to handle the full source text.

Next, we can apply our tokenization layer to our input text. And finally, we can shuffle, batch, and cache our dataset so we don’t need to recompute it every epoch:

With that, we are ready to start modeling.

To build our simple language model, we want to predict the probability of a character given all past characters. Of all the modeling possibilities we have seen so far in this book, an RNN is the most natural fit, as the recurrent state of each cell allows the model to propagate information about past characters when predicting the label of the current character. We can also use an `Embedding`, as we saw in the previous chapter, to embed each input character as a unique 256-dimensional vector. We will use only a single recurrent layer to keep this model small and easy to train. Any recurrent layer would do here, but to keep things simple, we will use a `GRU`, which is fast and has a simpler internal state than an `LSTM`.

## Next page (PDF 450) text, for continuity
After 20 epochs, our model can eventually predict the next character in our input sequences around 70% of the time.

15.1.2 Generating Shakespeare

Now that we have trained a model that can predict the next individual tokens with some accuracy, we would like to use it to extrapolate an entire predicted sequence. We can do this by calling the model in a loop, where the model’s predicted output at one time step becomes the model’s input at the next time step. A model built for this kind of feedback loop is sometimes called an autoregressive model.

To run such a loop, we need to perform a slight surgery on the model we just trained. During training, our model handled only a fixed sequence length of 100 tokens, and the `GRU` cell’s state was handled implicitly when calling the layer. During generation, we would like to predict a single output token at a time and explicitly output the state of the `GRU`’s cell. We need to propagate that state, which contains all information the model has encoded about past input characters, the next time we call the model. Let’s make a model that handles a single input character at a time and allows explicitly passing the RNN state. Because this model will have the same computational structure, with slightly modified inputs and outputs, we can assign weights from one model to another.

Listing 15.6 Modifying the language model for autoregressive inference

With this, we can call the model to predict an output sequence in a loop. Before we do, we will make explicit lookup tables so we switch from characters to integers and choose a prompt—a snippet of text we will feed as input to the model before we begin predicting new tokens:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.459 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.474 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.481 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer
- PDF p.473 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.458 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.476 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.517 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.432 — 14 Text classification › 14.5 Sequence models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.446 (printed 422) — 15 Language models and the Transformer › 15.1 The language model › 
What about other text-based tasks like question answering or translation? For many real-world problems, we are interested in a model that can generate a text output for a given input. Just like we needed tokenizers and embeddings to help us handle text on the way in to a model, we must build up some techniques before we can produce text on the way out. We don’t need to start from scratch here; we can continue to use the idea of an integer sequence as a natural numeric representation for text. In the previous chapter, we covered tokenizing a string, where we split inputs into tokens and map each token to an int. We can detokenize a sequence by proceeding in reverse—map ints back to string tokens and join them together. With this approach, our problem becomes building a model that can predict an integer sequence of tokens. The simplest option to consider might be to train a direct classifier over the space of all possible output integer sequences, but some back-of-the-envelope math will quickly show this is intractable. With a vocabulary of 20,000 words, there are 20,000 ^ 4, or 160 quadrillion possible 4-word sequences, and fewer atoms in the universe than possible 20-word sequences

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

### PDF p.452 (printed 428) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 
We have yet to produce the next great tragedy, but this is not terrible for two minutes of training on a minimal dataset. The goal of this toy example is to show the power of the language model setup. We trained the model on the narrow problem of guessing a single character at a time but still use it for a much broader problem, generating an open-ended, Shakespearean-like text response.

It’s important to notice that this training setup only works because a recurrent neural network only passes information forward in the sequence. If you’d like, try replacing the `GRU` layer with a `Bidirectional(GRU(...))`. The training accuracy will zoom to above 99% immediately, and generation will stop working entirely. During training, our model sees the entire sequence each train step. If we “cheat” by letting information from the next token in the sequence affect the current token’s prediction, we’ve made our problem trivially easy. This language modeling setup is fundamental to countless problems in the text domain. It is also somewhat unique compared to other modeling problems we have seen so far in this book. We cannot simply call `model.predict()` to get the desired output. There is an en

### PDF p.446 (printed 422) — 15 Language models and the Transformer › 15.1 The language model › 15.1.1 Training a Shakespeare language model
Listing 15.1 Downloading an abbreviated collection of Shakespeare’s work
import keras
filename = keras.utils.get_file(
    origin=(
        "https://storage.googleapis.com/download.tensorflow.org/"
        "data/shakespeare.txt"
    ),
)
shakespeare = open(filename, "r").read()

### PDF p.447 (printed 423) — 15 Language models and the Transformer › 15.1 The language model › 
To build a language model from this input, we will need to massage our source text. First, we will split our data into equal-length chunks that we can batch and use for model training, much as we did for weather measurements in the timeseries chapters. Because we will be using a character-level tokenizer here, we can do this chunking directly on the string input. A 100-character string will map to a 100-integer sequence.

We will also split each input into two separate feature and label sequences, with each label sequence simply being the input sequence offset by a single character.

Listing 15.2 Splitting text into chunks for language model training

### PDF p.488 (printed 464) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › Summary
A language model is a model that learns a specific probability distribution— `p(token|past tokens)`: – Language models have broad applications, but the most important is that you can generate text by calling them in a loop, where the output token at one time step becomes the input token in the next. – A masked language model learns a related probability distribution `p(tokens|surrounding` `tokens)` and can be helpful for classifying text and individual tokens. – A sequence-to-sequence language model learns to predict the next token given both past tokens in a target sequence and an entirely separate, fixed source sequence. Sequence-to-sequence models are useful for problems like translation and question answering. – A sequence-to-sequence model usually has two separate components. An encoder computes a representation of the source sequence, and a decoder takes this representation as input and predicts the next token in a target sequence based on past tokens.

Attention is a mechanism that allows a model to pull information from anywhere in a sequence selectively based on the context of the token currently being processed: – Attention avoids the problems RNNs have with long-range de

### PDF p.639 (printed 615) — index ›  › 
learning, seq2seq (sequence-to-sequence)

manifold hypothesis 144 428–437 .map(function, num_parallel_calls) method 249 English-to-Spanish translation 430–432 mask argument 412 with RNNs 433–437 masked language model 455, 464 LearningRateSchedule 476 mask_zero argument 412 lemmatization, defined 386–387 matmul operation 550 lifelong learning 607 Matplotlib axis object 322 linear transformations 110 MaxPooling1D layer 362 LLMs (large language models) 468, 495–504, 565 MaxPooling2D layer 232, 233, 234, 239, 245, 315 multimodal LLM 498–501 max-pooling layers 602 RAG (Retrieval Augmented Generation) 501 max-pooling operation 239–241 reasoning models 502–504 max tensor operation 240 RLHF 495–498 metrics using pretrained 484–494

handling in custom train_step() 226–228 local generalization 573, 574 low-level usage of 221 local minimum 45 writing 207 logit 478 MHR (Modularity-Hierarchy-Reuse) formula 269 logs argument 211 middle stage 522 LoRA (Low-Rank Adaptation) 490–494 mini-batch or just batch 105 loss.backward() method 217 mini-batch SGD (mini-batch stochastic gradient loss function 9, 181, 596

descent) 44 loss scaling with mixed precision 560 mini-GPT (Generative Pretrained Transfo

### PDF p.479 (printed 455) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
model were released a year after “Attention Is All You Need.” The model structure was exactly the same as the encoder part of the translation Transformer we just built. This encoder model is bidirectional in that every position in the sequence can attend to positions in front of and behind it. This means it’s a good model for computing a rich representation of input text, but not a model meant to run generation in a loop.

BERT was trained in sizes between 100 million and 300 million parameters, much bigger than the 14 million parameter Transformer we just trained. This meant the model needed a lot of training data to perform well. To achieve this, the authors used a riff on the classic language modeling setup called masked language modeling. To pretrain the model, we take a sequence of text and replace about 15% of the tokens with a special `[MASK]` token. The model will attempt to predict the original masked token values during training. While the classic language model, sometimes called a causal language model, attempts to predict `p(token|past` `tokens)`, the masked language model attempts to predict `p(token|surrounding` `tokens)`. This training setup is unsupervised. You don’
