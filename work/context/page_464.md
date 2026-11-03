# Generation bundle for page_464  (PDF page 464, printed page 440)

Page kind: content
Chapter: 15 Language models and the Transformer
Section: 15.3 The Transformer architecture
Subsection: 15.3.1 Dot-product attention
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_464.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_464.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p464_e03] (figure_caption) Figure 15.4 Attention assigns a relevance score to each vector in a source for each vector in a target sequence.
[p464_e04] (body) For attention to work well, we want to avoid passing information about important tokens through a loop potentially as long as our combined source and target sequence length—this is where RNNs start to fail. A simple way to do this is to take a weighted sum of all the source vectors based on this score we will compute. It would also be convenient if the sum of all attention scores for a given target were 1, as this would give our weighted sum a predictable magnitude. We can achieve this by running the scores through a `softmax` function—something like this, in NumPy pseudocode:
[p464_e05] (code)
```
scores = [score(target, source) for source in sources]
scores = softmax(scores)
combined = np.sum(scores * sources)
```
[p464_e06] (body) But how should we compute this relevance score? When researchers first worked with attention, this question was a big topic of inquiry. It turns out that one of the most straightforward approaches is best. We can use a dot-product as a simple measure of the distance between target and source vectors. If the source and target vectors are close together, we assume that means the source token is relevant to our prediction. At the end of this chapter, we will examine why this assumption makes intuitive sense.
[p464_e07] (body) Let’s update our pseudocode. We can make our snippet more complete by handling the entire target sequence at once—it will be equivalent to running our previous snippet in a loop for each token in the target sequence. When both `target` and `source` are sequences, the attention scores will be a matrix. Each row represents how much a target word will value a source word in the weighted sum (see figure 15.5). We will use the Einsum notation as a convenient way to write the dot-product and weighted sum:
[p464_e08] (code)
```
def dot_product_attention(target, source):
    scores = np.einsum("btd,bsd->bts", target, source)
    scores = softmax(scores, axis=-1)
```

## Code annotations printed next to listings (callouts, in order)
- Takes the dot-product between all target and source
- vectors, where b = batch size, t = target length,
- s = source length, and d = vector size

## Figures
- Figure 15.4: Attention assigns a relevance score to each vector in a source for each vector in a target sequence.  | labels inside the figure: (none - image)

## Equation candidates (formulas written inline in the text)
- [p464_e05] scores = [score(target, source) for source in sources]
scores = softmax(scores)
combined = np.sum(scores * sources)

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

## Previous page (PDF 463) text, for continuity
What is Einsum notation?

In machine learning codebases, you will frequently see little snippets that look like this: `np.einsum('ij,jk->ik', a, b)`. This is called Einsum notation, short for Einstein summation notation. Once you learn to read them, they can be a clear way to write complex array operations. You will see them frequently in Transformer code for this reason. The idea with an Einsum equation is to represent each axis of an input with a unique letter. For example, you could represent a rank-3 input as `ijk`. Then, you write an equation with any number of inputs and a single output `input1,input2->output`. The rules of this equation are as follows:

For any repeated letters across inputs, multiply values along these axes togeth

er. The size of these dimensions should match.

For any letters in the input but not the output, sum over these axes so they do

not appear in the returned array.

Output axes can be returned in any order.

This gets much clearer if we take a look at some examples:

In Keras, you can use `einsum`s in two ways. `keras.ops.einsum` is a drop-in replace

ment for `np.einsum`, and `keras.layers.EinsumDense` is a `Dense` layer where the

`matmul` is replaced by an `einsum` operation.

15.3.1 Dot-product attention

Let’s revisit our translation RNN and try to add the notion of selective attention. Consider predicting just a single token. After passing the `source` and `target` sequences through our `GRU` layers, we will have a vector representing the target token we are about to predict and a sequence of vectors representing each word in the source text.

With attention, our goal is to give the model a way to score every single vector in our source sequence based on its relevance to the current word we are trying to predict (figure 15.4). If …

## Next page (PDF 465) text, for continuity
We can make the hypothesis space of this attention mechanism much richer if we give the model parameters to control the attention score. If we project both source and target vectors with `Dense` layers, the model can find a good shared space where source vectors are close to target vectors if they help the overall prediction quality. Similarly, we should allow the model to project the source vectors into an entirely separate space before they are combined and once again after the summation.

Figure 15.5 When both target and source are

We can also adopt a slightly dif

sequences, attention scores are a 2D matrix. Each

ferent naming for inputs that has

row shows the attention scores for the word we are

become standard in the field. What

trying to predict (in green).

we just wrote is roughly summarized as `sum(score(target,` `source)` `* source)`. We will write this equivalently with different input names as `sum(score(query, key) * value)`. This three-argument version is more general—in rare cases, you might not want to use the same vector to score your source inputs as you use to sum your source inputs.

The terminology comes from search engines and recommender systems. Imagine a search tool to look up photos in a database—the “query” is your search term, the “keys” are photo tags you use to match with the query, and finally, the “values” are the photos themselves (figure 15.6). The attention mechanism we are building is roughly analogous to this sort of lookup. Let’s update our pseudocode, so we have a parameterized attention using our new terminology:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.466 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.467 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.468 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.471 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.470 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.627 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.476 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.458 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.466 (printed 442) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
Figure 15.6 Retrieving images from a database: the query is compared to a set of keys, and the match scores are used to rank values (images).

This block is a perfectly functional attention mechanism! We just wrote a function that will allow the model to pull information from anywhere in the source sequence, contextually, depending on the target word we are decoding.

The “Attention is all you need” authors made two more changes to our mechanism through trial and error. The first is a simple scaling factor. When input vectors get long, the dot-product scores can get quite large, which can affect the stability of our softmax gradients. The fix is simple: we can scale down our softmax scores slightly. Scaling by the square root of the vector length works well for any vector size. The other has to do with the expressivity of the attention mechanism. The softmax sum we are doing is powerful—it allows a direct connection across distant parts of a sequence. But the summation is also blunt: if the model tries to attend to too many

### PDF p.467 (printed 443) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
By projecting the query and key differently, one head might learn to match the subject of the source sentence, while another head might attend to punctuation. This multi-headed attention avoids the limitation of needing to combine the entire source sequence with a single softmax sum (figure 15.7).

Figure 15.7 Multi-headed attention allows each target word to attend to different parts of the source sequence in separate partitions of the eventual output vector.

Of course, in practice, you would want to write this code as a reusable layer. Here, Keras has you covered. We can recreate our previous code with the `MultiHeadAttention` layer as follows:

### PDF p.468 (printed 444) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 15.3.2 Transformer encoder block
This is called self-attention, and it is quite powerful. With self-attention, each token can attend to every token in its own sequence, including itself, allowing the model to learn a representation of the word in context.

Consider an example sentence: “The train left the station on time.” Now, consider one word in the sentence: “station.” What kind of station are we talking about? Could it be a radio station? Maybe the International Space Station? With self-attention, the model could learn to give a high attention score to the pair of “station” and “train,” summing the vector used to represent “train” into the representation of the word “station.” Self-attention gives the model an effective way to go from representing a word in a vacuum to representing a word conditioned on all other tokens that appear in the sequence. This sounds a lot like what an RNN is supposed to do. Can we just go ahead and replace our RNN layers with `MultiHeadAttention`? Almost! But not quite; we still need an essential ingredient for any deep neural network—a nonlinear activation function. The `MultiHeadAttention` layer combines linear projections of every element in a source sequence, but that’s it. In 

### PDF p.471 (printed 447) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
Our decoder layer takes in both a `target` and `source`. Like with the `Transformer` `Encoder`, we take in a `source_mask` that marks the location of all padding in the source input (`True` for non-padding, `False` for padding) and use it as an `attention_mask` for the cross-attention layer.

For the decoder’s self-attention layer, we need a different type of attention mask. Recall that when we built our RNN decoder, we avoided using a `Bidirectional` RNN. If we had used one, the model would be able to cheat by seeing the label it was trying to predict as a feature! Attention is inherently bidirectional; in self-attention, any token position in the target sequence can attend to any other position. Without special care,

### PDF p.468 (printed 444) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 15.3.2 Transformer encoder block
One way to use the `MultiHeadAttention` layer would be to add it to our existing RNN translation model. We could pass the sequence output from our encoder and decoder into an attention layer and use its output to update our target sequence before prediction. Attention would allow the model to handle long-range dependencies in text that the `GRU` layer will struggle with. This does, in fact, improve an RNN model’s capabilities and is how attention was first used in the mid-2010s.

However, the authors of “Attention is all you need” realized you could go further and use attention as a general mechanism for handling all sequence data in a model. Although so far we have only looked at attention as a way to handle information passing between two sequences, you could also use attention as a way to let a sequence attend to itself:

### PDF p.486 (printed 462) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › 
The distance function used to compare tokens is cosine distance in both cases. Even the dimensionality of the embedding space is similar: a vector with somewhere between 1,000 and 10,000 dimensions to represent each word. At this point, you might interject: A Transformer is trained to predict missing words in a sequence, not to group tokens in an embedding space. How does the language model loss function relate to Word2Vec’s objective of maximizing the dot-product between co-occurring tokens? The answer is the attention mechanism. Attention is, by far, the most critical component in the Transformer architecture. It’s a mechanism for learning a new token embedding space by linearly recombining token embeddings from some prior space, in weighted combinations that give greater importance to tokens that are already “closer” to each other (i.e., that have a higher dot-product). It will tend to pull together the vectors of already close tokens, resulting over time in a space where token correlation relationships turn into embedding proximity relationships (in terms of cosine distance). Transformers work by learning a series of incrementally refined embedding spaces, each based on recombi

### PDF p.471 (printed 447) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
source and target sequence as input. We will call this attention layer cross-attention as it brings information across the encoder and decoder.

Listing 15.15 A Transformer decoder block

### PDF p.627 (printed 603) — 20 Conclusions › 20.1 Key concepts in review › Transformers
A Transformer looks at a set of vectors (such as word vectors) and uses neural attention to transform each vector into a representation that is aware of the context provided by the other vectors in the set. When the set in question is an ordered sequence, you can also use positional encoding to create Transformers that can take into account both global context and word order, capable of processing long text paragraphs much more effectively than RNNs or 1D ConvNets.

Transformers can be used for any set-processing or sequence-processing task, including text classification, but they excel especially at sequence-to-sequence learning, such as translating paragraphs in a source language into a target language. A sequence-to-sequence Transformer is made of two parts:

A `TransformerEncoder` that turns an input vector sequence into a context-aware, order-aware output vector sequence

A `TransformerDecoder` that takes the output of the `TransformerEncoder`, as well as a target sequence, and predicts what should come next in the target sequence

If you’re only processing a single sequence (or set) of vectors, you’d only use the `TransformerEncoder`.

Following is a sequence-to-sequence Tran
