# Generation bundle for page_467  (PDF page 467, printed page 443)

Page kind: content
Chapter: 15 Language models and the Transformer
Section: 15.3 The Transformer architecture
Subsection: 15.3.1 Dot-product attention
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_467.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_467.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p467_e03] (body) tokens at once, the interesting features of individual source tokens will get “washed out” in the combined representation. A simple trick that works well is to do this attention operation several times for the same sequence, with several different attention heads running the same computation with different parameters:
[p467_e04] (code)
```
query_dense = [layers.Dense(head_dim) for i in range(num_heads)]
key_dense = [layers.Dense(head_dim) for i in range(num_heads)]
value_dense = [layers.Dense(head_dim) for i in range(num_heads)]
output_dense = layers.Dense(head_dim * num_heads)
def multi_head_attention(query, key, value):
    head_outputs = []
    for i in range(num_heads):
        query = query_dense[i](query)
        key = key_dense[i](key)
        value = value_dense[i](value)
        scores = np.einsum("btd,bsd->bts", target, source)
        scores = softmax(scores / math.sqrt(head_dim), axis=-1)
        head_output = np.einsum("bts,bsd->btd", scores, source)
        head_outputs.append(head_output)
    outputs = ops.concatenate(head_outputs, axis=-1)
    return output_dense(outputs)
multi_head_attention(query=target, key=source, value=source)
```
[p467_e05] (body) By projecting the query and key differently, one head might learn to match the subject of the source sentence, while another head might attend to punctuation. This multi-headed attention avoids the limitation of needing to combine the entire source sequence with a single softmax sum (figure 15.7).
[p467_e06] (figure_caption) Figure 15.7 Multi-headed attention allows each target word to attend to different parts of the source sequence in separate partitions of the eventual output vector.
[p467_e07] (body) Of course, in practice, you would want to write this code as a reusable layer. Here, Keras has you covered. We can recreate our previous code with the `MultiHeadAttention` layer as follows:

## Figures
- Figure 15.7: Multi-headed attention allows each target word to attend to different parts of the source sequence in separate partitions of the eventual output vector.  | labels inside the figure: (none - image)

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

## Previous page (PDF 466) text, for continuity
Figure 15.6 Retrieving images from a database: the query is compared to a set of keys, and the match scores are used to rank values (images).

This block is a perfectly functional attention mechanism! We just wrote a function that will allow the model to pull information from anywhere in the source sequence, contextually, depending on the target word we are decoding.

The “Attention is all you need” authors made two more changes to our mechanism through trial and error. The first is a simple scaling factor. When input vectors get long, the dot-product scores can get quite large, which can affect the stability of our softmax gradients. The fix is simple: we can scale down our softmax scores slightly. Scaling by the square root of the vector length works well for any vector size. The other has to do with the expressivity of the attention mechanism. The softmax sum we are doing is powerful—it allows a direct connection across distant parts of a sequence. But the summation is also blunt: if the model tries to attend to too many

## Next page (PDF 468) text, for continuity
15.3.2 Transformer encoder block

One way to use the `MultiHeadAttention` layer would be to add it to our existing RNN translation model. We could pass the sequence output from our encoder and decoder into an attention layer and use its output to update our target sequence before prediction. Attention would allow the model to handle long-range dependencies in text that the `GRU` layer will struggle with. This does, in fact, improve an RNN model’s capabilities and is how attention was first used in the mid-2010s.

However, the authors of “Attention is all you need” realized you could go further and use attention as a general mechanism for handling all sequence data in a model. Although so far we have only looked at attention as a way to handle information passing between two sequences, you could also use attention as a way to let a sequence attend to itself:

This is called self-attention, and it is quite powerful. With self-attention, each token can attend to every token in its own sequence, including itself, allowing the model to learn a representation of the word in context.

Consider an example sentence: “The train left the station on time.” Now, consider one word in the sentence: “station.” What kind of station are we talking about? Could it be a radio station? Maybe the International Space Station? With self-attention, the model could learn to give a high attention score to the pair of “station” and “train,” summing the vector used to represent “train” into the representation of the word “station.” Self-attention gives the model an effective way to go from representing a word in a vacuum to representing a word conditioned on all other tokens that appear in the sequence. This sounds a lot like what an RNN is supposed to do. Can we just go ahead and replace our RNN l …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.465 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.464 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.471 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.469 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.498 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.470 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.627 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.472 — 15 Language models and the Transformer › 15.3 The Transformer architecture

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.463 (printed 439) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 15.3.1 Dot-product attention
Let’s revisit our translation RNN and try to add the notion of selective attention. Consider predicting just a single token. After passing the `source` and `target` sequences through our `GRU` layers, we will have a vector representing the target token we are about to predict and a sequence of vectors representing each word in the source text.

With attention, our goal is to give the model a way to score every single vector in our source sequence based on its relevance to the current word we are trying to predict (figure 15.4). If the vector representation of a source token has a high score, we consider it particularly important; if not, we care less about it. For now, let’s assume we have this function `score(target_vector, source_vector)`.

### PDF p.464 (printed 440) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
Figure 15.4 Attention assigns a relevance score to each vector in a source for each vector in a target sequence.

For attention to work well, we want to avoid passing information about important tokens through a loop potentially as long as our combined source and target sequence length—this is where RNNs start to fail. A simple way to do this is to take a weighted sum of all the source vectors based on this score we will compute. It would also be convenient if the sum of all attention scores for a given target were 1, as this would give our weighted sum a predictable magnitude. We can achieve this by running the scores through a `softmax` function—something like this, in NumPy pseudocode:

### PDF p.464 (printed 440) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
But how should we compute this relevance score? When researchers first worked with attention, this question was a big topic of inquiry. It turns out that one of the most straightforward approaches is best. We can use a dot-product as a simple measure of the distance between target and source vectors. If the source and target vectors are close together, we assume that means the source token is relevant to our prediction. At the end of this chapter, we will examine why this assumption makes intuitive sense.

Let’s update our pseudocode. We can make our snippet more complete by handling the entire target sequence at once—it will be equivalent to running our previous snippet in a loop for each token in the target sequence. When both `target` and `source` are sequences, the attention scores will be a matrix. Each row represents how much a target word will value a source word in the weighted sum (see figure 15.5). We will use the Einsum notation as a convenient way to write the dot-product and weighted sum:

### PDF p.465 (printed 441) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
We can make the hypothesis space of this attention mechanism much richer if we give the model parameters to control the attention score. If we project both source and target vectors with `Dense` layers, the model can find a good shared space where source vectors are close to target vectors if they help the overall prediction quality. Similarly, we should allow the model to project the source vectors into an entirely separate space before they are combined and once again after the summation.

Figure 15.5 When both target and source are

We can also adopt a slightly dif

sequences, attention scores are a 2D matrix. Each

ferent naming for inputs that has

row shows the attention scores for the word we are

become standard in the field. What

trying to predict (in green).

we just wrote is roughly summarized as `sum(score(target,` `source)` `* source)`. We will write this equivalently with different input names as `sum(score(query, key) * value)`. This three-argument version is more general—in rare cases, you might not want to use the same vector to score your source inputs as you use to sum your source inputs.

The terminology comes from search engines and recommender systems. Imagin

### PDF p.470 (printed 446) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
While `BatchNormalization` collects information from many samples to obtain accurate statistics for the feature means and variances, `LayerNormalization` pools data within each sequence separately, which is more appropriate for sequence data.

We also pass a new input to the `MultiHeadAttention` layer called `attention_mask`. This Boolean tensor input will be broadcast to the same shape as our attention scores `(batch_size, target_length, source_length)`. When set, it will zero the attention score in specific locations, stopping the source tokens at these locations from being used in the attention calculation. We will use this to prevent any token in the sequence from attending to padding tokens, which contain no information. Our encoder layer takes a `source_mask` input that will mark all the non-padding tokens in our inputs and upranks it to shape `(batch_size, 1, source_length)` to use as an `attention_mask`. Note that the input and outputs of this layer have the same shape, so encoder blocks can be stacked on top of each other, building a progressively more expressive representation of the input English sentence.

### PDF p.471 (printed 447) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
source and target sequence as input. We will call this attention layer cross-attention as it brings information across the encoder and decoder.

Listing 15.15 A Transformer decoder block

### PDF p.472 (printed 448) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
Each row `i` can be read as a mask for attention for the target token at position `i`. In the first row, the first token can only attend to itself. In the second row, the second token can attend to both the first and second tokens, and so forth. This gives us the same effect as our RNN layer, where information can only propagate forward in the sequence, not backward. In Keras, you can specify this lower-triangular mask simply by passing `use_casual_mask` to the `MultiHeadAttention` layer when calling it. Figure 15.8 shows a visual representation of the layers in both the encoder and decoder layers, when stacked into a Transformer model.

Figure 15.8 A visual representation of the computations for both

`TransformerEncoder` and `TransformerDecoder` blocks

### PDF p.471 (printed 447) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
Our decoder layer takes in both a `target` and `source`. Like with the `Transformer` `Encoder`, we take in a `source_mask` that marks the location of all padding in the source input (`True` for non-padding, `False` for padding) and use it as an `attention_mask` for the cross-attention layer.

For the decoder’s self-attention layer, we need a different type of attention mask. Recall that when we built our RNN decoder, we avoided using a `Bidirectional` RNN. If we had used one, the model would be able to cheat by seeing the label it was trying to predict as a feature! Attention is inherently bidirectional; in self-attention, any token position in the target sequence can attend to any other position. Without special care,
