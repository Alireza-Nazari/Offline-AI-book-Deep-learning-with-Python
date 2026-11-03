# Generation bundle for page_475  (PDF page 475, printed page 451)

Page kind: content
Chapter: 15 Language models and the Transformer
Section: 15.3 The Transformer architecture
Subsection: 15.3.5 Embedding positional information
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_475.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_475.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p475_e03] (body) For RNNs, we relied on the layer’s computation to be order aware. In the case of the Transformer, we instead inject positional information directly into our embedded sequence itself. This is called a positional embedding. Let’s take a look.
[p475_e04] (heading_subsection) 15.3.5 Embedding positional information
[p475_e05] (body) The idea behind a positional embedding is very simple: to give the model access to word order information, we will add the word’s position in the sentence to each word embedding. Our input word embeddings will have two components: the usual word vector, which represents the word independently of any specific context, and a position vector, which represents the position of the word in the current sentence. Hopefully, the model will then figure out how to best use this additional information.
[p475_e06] (body) The most straightforward scheme to add position information would be concatenating each word’s position to its embedding vector. You’d add a “position” axis to the vector and fill it with `0` for the first word in the sequence, `1` for the second, and so on. However, that may not be ideal because the positions can potentially be very large integers, which will disrupt the range of values in the embedding vector. As you know, neural networks don’t like very large input values or discrete input distributions. The “Attention is all you need” authors used an interesting trick to encode word positions: they added to the word embeddings a vector containing values in the range `[-1, 1]` that varied cyclically depending on the position (they used cosine functions to achieve this). This trick offers a way to uniquely characterize any integer in a large range via a vector of small values. It’s clever, but it turns out we can do something simpler and more effective: we’ll learn positional embedding vectors the same way we learn to embed word indices. We’ll then add our positional embeddings to the corresponding word embeddings to obtain a position-aware word embedding. This is called a positional embedding. Let’s implement it.
[p475_e07] (listing_caption) Listing 15.17 A learned position embedding layer
[p475_e08] (code)
```
from keras import ops
class PositionalEmbedding(keras.Layer):
    def __init__(self, sequence_length, input_dim, output_dim):
        super().__init__()
        self.token_embeddings = layers.Embedding(input_dim, output_dim)
        self.position_embeddings = layers.Embedding(sequence_length, output_dim)
    def call(self, inputs):
        positions = ops.cumsum(ops.ones_like(inputs), axis=-1) - 1
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
```

## Code annotations printed next to listings (callouts, in order)
- Computes incrementing positions
- [0, 1, 2…] for each sequence in the batch

## Listings
- Listing 15.17: A learned position embedding layer (code elements p475_e08)

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

## Previous page (PDF 474) text, for continuity
Our model has almost exactly the same structure as the `GRU` translation model we trained earlier, with attention now substituting for recurrent layers as the mechanism to pass information across the sequence. Let’s try training the model:

After training, we get to about 58% accuracy: on average, the model correctly predicts the next word in the Spanish sentence 58% of the time. Something is off here. Training is worse than the RNN model by 7 percentage points. Either this Transformer architecture is not what it was hyped up to be, or we missed something in our implementation. Can you spot what it is?

This section is ostensibly about sequence models. In the previous chapter, we saw how vital word order could be to meaning. And yet, the Transformer we just built wasn’t a sequence model at all. Did you notice? It’s composed of dense layers that process sequence tokens independently of each other and an attention layer that looks at the tokens as a set. You could change the order of the tokens in a sequence, and you’d get identical pairwise attention scores and the same context-aware representations. If you were to rearrange every word in every English source sentence completely, the model wouldn’t notice, and you’d still get the same accuracy. Attention is a set-processing mechanism, focused on the relationships between pairs of sequence elements—it’s blind to whether these elements occur at the beginning, at the end, or in the middle of a sequence. So why do we say that Transformer is a sequence model? And how could it possibly be suitable for machine translation if it doesn’t look at word order?

## Next page (PDF 476) text, for continuity
We would use this `PositionalEmbedding` layer just like a regular `Embedding` layer. Let’s see it in action as we try training our Transformer for a second time.

Listing 15.18 Building a Transformer model with positional embeddings

With the positional embedding now added to our model, let’s try training again:

With positional information back in the model, things went much better. We achieved a 67% accuracy when guessing the next word. It’s a noticeable improvement from the `GRU` model, and that’s all the more impressive when you consider that this model has half the parameters of the GRU counterpart.

There’s one other important thing to notice about this training run. Training is noticeably faster than the RNN—each epoch takes about a third of the time. This would be true even if we matched parameter count with the RNN model, and it is a side effect of getting rid of the looped state passing of our `GRU` layers. With attention, there is no looping computation to handle during training, meaning that on a GPU or TPU, we can handle the entire attention computation in one go. This makes the `Transformer` quicker to train on accelerators.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.498 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.499 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.435 — 14 Text classification › 14.5 Sequence models
- PDF p.434 — 14 Text classification › 14.5 Sequence models
- PDF p.627 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.438 — 14 Text classification › 14.5 Sequence models
- PDF p.457 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.470 — 15 Language models and the Transformer › 15.3 The Transformer architecture

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.498 (printed 474) — 16 Text generation › 16.2 Training a mini-GPT › 
Next, we can copy the `PositionalEmbedding` layer from chapter 15. Recall that this layer gives us a simple way to learn an embedding for each position in a sequence and combine that with our token embeddings.

There’s a neat trick we can employ here to save some GPU memory. The biggest weights in a Transformer model are the input token embeddings and output dense

### PDF p.434 (printed 410) — 14 Text classification › 14.5 Sequence models › 
Besides being dense representations, word embeddings are also structured representations, and their structure is learned from data. Similar words get embedded in close locations, and further, specific directions in the embedding space are meaningful. To make this clearer, let’s look at a concrete example. In figure 14.4, four words are embedded on a 2D plane: cat, dog, wolf, and tiger. With the vector representations we chose here, some semantic relationships

Figure 14.4 A toy

between these words can be encoded as geometric transfor

example of a word-

mations. For instance, the same vector allows us to go from

embedding space

cat to tiger and from dog to wolf: this vector could be interpreted as the “from pet to wild animal” vector. Similarly, another vector lets us go from dog to cat and from wolf to tiger, which could be interpreted as a “from canine to feline” vector.

In real-world word-embedding spaces, typical examples of meaningful geometric transformations are “gender” vectors and “plural” vectors. For instance, by adding a “female” vector to the vector “king,” we obtain the vector “queen.” By adding a “plural” vector, we obtain “kings.” Word-embedding spaces typicall

### PDF p.627 (printed 603) — 20 Conclusions › 20.1 Key concepts in review › Transformers
A Transformer looks at a set of vectors (such as word vectors) and uses neural attention to transform each vector into a representation that is aware of the context provided by the other vectors in the set. When the set in question is an ordered sequence, you can also use positional encoding to create Transformers that can take into account both global context and word order, capable of processing long text paragraphs much more effectively than RNNs or 1D ConvNets.

Transformers can be used for any set-processing or sequence-processing task, including text classification, but they excel especially at sequence-to-sequence learning, such as translating paragraphs in a source language into a target language. A sequence-to-sequence Transformer is made of two parts:

A `TransformerEncoder` that turns an input vector sequence into a context-aware, order-aware output vector sequence

A `TransformerDecoder` that takes the output of the `TransformerEncoder`, as well as a target sequence, and predicts what should come next in the target sequence

If you’re only processing a single sequence (or set) of vectors, you’d only use the `TransformerEncoder`.

Following is a sequence-to-sequence Tran

### PDF p.434 (printed 410) — 14 Text classification › 14.5 Sequence models › 14.5.3 Using a word embedding
Is there an ideal word-embedding space that perfectly maps human language and can be used for any NPL task? Possibly, but we have yet to compute anything of the sort. Also, there is no single human language we could attempt to map—there are many different languages, and they aren’t isomorphic to one another because a language is the reflection of a specific culture and a particular context. More pragmatically, what makes a good word-embedding space depends heavily on your task: the perfect word-embedding space for an English-language movie review sentiment-analysis model may look different from the ideal embedding space for an English-language legal document classification model because the importance of certain semantic relationships varies from task to task.

It’s thus reasonable to learn a new embedding space with every new task. Fortunately, backpropagation makes this easy, and Keras makes it even easier. It’s about learning the weights of the Keras `Embedding` layer. The `Embedding` layer is best understood as a dictionary that maps integer indices (which stand for specific words) to dense vectors. It takes integers as input, looks them up in an internal dictionary, and return

### PDF p.435 (printed 411) — 14 Text classification › 14.5 Sequence models › 
Figure 14.5 An `Embedding` layer acts as a dictionary mapping ints to floating point vectors.

When you instantiate an `Embedding` layer, its weights (its internal dictionary of token vectors) are initially random, just as with any other layer. During training, these word vectors are gradually adjusted via backpropagation, structuring the space into something the downstream model can exploit. Once fully trained, the embedding space will show a lot of structure—a kind of structure specialized for the specific problem for which you’re training your model.

Let’s build a model with an `Embedding` layer and benchmark it on our task.

Listing 14.24 Building an LSTM sequence model with an `Embedding` layer

### PDF p.486 (printed 462) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › 
The distance function used to compare tokens is cosine distance in both cases. Even the dimensionality of the embedding space is similar: a vector with somewhere between 1,000 and 10,000 dimensions to represent each word. At this point, you might interject: A Transformer is trained to predict missing words in a sequence, not to group tokens in an embedding space. How does the language model loss function relate to Word2Vec’s objective of maximizing the dot-product between co-occurring tokens? The answer is the attention mechanism. Attention is, by far, the most critical component in the Transformer architecture. It’s a mechanism for learning a new token embedding space by linearly recombining token embeddings from some prior space, in weighted combinations that give greater importance to tokens that are already “closer” to each other (i.e., that have a higher dot-product). It will tend to pull together the vectors of already close tokens, resulting over time in a space where token correlation relationships turn into embedding proximity relationships (in terms of cosine distance). Transformers work by learning a series of incrementally refined embedding spaces, each based on recombi

### PDF p.442 (printed 418) — 14 Text classification › 14.5 Sequence models › 14.5.5 Using the pretrained embedding for classification
Now that we have trained a new word embedding, applying it to our LSTM model is simple. First, we create the model precisely as we did before.

Listing 14.30 Building another LSTM sequence model with an `Embedding` layer

### PDF p.485 (printed 461) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › 15.5 What makes the Transformer effective?
In 2013, at Google, Tomas Mikolov and his colleagues noticed something remarkable. They were building a pretrained embedding called “Word2Vec,” similar to the Continuous Bag of Words (CBOW) embedding we built in the last chapter. Much like our CBOW model, their training objective sought to turn correlation relationships between words into distance relationships in the embedding space: a vector was associated with each word in a vocabulary, and the vectors were optimized so that the dot-product (cosine proximity) between vectors representing frequently co-occurring words would
