# Generation bundle for page_469  (PDF page 469, printed page 445)

Page kind: content
Chapter: 15 Language models and the Transformer
Section: 15.3 The Transformer architecture
Subsection: 15.3.2 Transformer encoder block
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_469.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_469.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p469_e03] (body) to a single matrix multiplication! That’s a real problem with the expressiveness of our model.
[p469_e04] (body) At some point, all recurrent cells pass the input vector for each token through a dense projection and apply an activation function; we need a plan for something similar. The authors of “Attention is all you need” decided to add this back in the simplest way possible—stacking a feedforward network of two dense layers with an activation in the middle. Attention passes information across the sequence, and the feedforward network updates the representation of individual sequence items. We are ready to start building a Transformer model. Let’s start by replacing the encoder of our translation model. We will use self-attention to pass information along the source sequence of English words. We will also add in two things we learned to be particularly important when building ConvNets back in chapter 9, normalization and residual connections.
[p469_e05] (listing_caption) Listing 15.14 A Transformer encoder block
[p469_e06] (code)
```
class TransformerEncoder(keras.Layer):
    def __init__(self, hidden_dim, intermediate_dim, num_heads):
        super().__init__()
        key_dim = hidden_dim // num_heads
        self.self_attention = layers.MultiHeadAttention(num_heads, key_dim)
        self.self_attention_layernorm = layers.LayerNormalization()
        self.feed_forward_1 = layers.Dense(intermediate_dim, activation="relu")
        self.feed_forward_2 = layers.Dense(hidden_dim)
        self.feed_forward_layernorm = layers.LayerNormalization()
    def call(self, source, source_mask):
        residual = x = source
        mask = source_mask[:, None, :]
        x = self.self_attention(query=x, key=x, value=x, attention_mask=mask)
        x = x + residual
        x = self.self_attention_layernorm(x)
        residual = x
        x = self.feed_forward_1(x)
        x = self.feed_forward_2(x)
        x = x + residual
        x = self.feed_forward_layernorm(x)
        return x
```
[p469_e07] (body) You’ll note that the normalization layers we’re using here aren’t `BatchNormalization` layers like those we’ve used in image models. That’s because `BatchNormalization` doesn’t work well for sequence data. Instead, we’re using the `LayerNormalization` layer, which normalizes each sequence independently from other sequences in the batch— like this, in NumPy-like pseudocode:

## Code annotations printed next to listings (callouts, in order)
- Feedforward layers
- Self-attention layers
- Self-attention computation
- Feedforward
- computation

## Listings
- Listing 15.14: A Transformer encoder block (code elements p469_e06)

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

## Previous page (PDF 468) text, for continuity
15.3.2 Transformer encoder block

One way to use the `MultiHeadAttention` layer would be to add it to our existing RNN translation model. We could pass the sequence output from our encoder and decoder into an attention layer and use its output to update our target sequence before prediction. Attention would allow the model to handle long-range dependencies in text that the `GRU` layer will struggle with. This does, in fact, improve an RNN model’s capabilities and is how attention was first used in the mid-2010s.

However, the authors of “Attention is all you need” realized you could go further and use attention as a general mechanism for handling all sequence data in a model. Although so far we have only looked at attention as a way to handle information passing between two sequences, you could also use attention as a way to let a sequence attend to itself:

This is called self-attention, and it is quite powerful. With self-attention, each token can attend to every token in its own sequence, including itself, allowing the model to learn a representation of the word in context.

Consider an example sentence: “The train left the station on time.” Now, consider one word in the sentence: “station.” What kind of station are we talking about? Could it be a radio station? Maybe the International Space Station? With self-attention, the model could learn to give a high attention score to the pair of “station” and “train,” summing the vector used to represent “train” into the representation of the word “station.” Self-attention gives the model an effective way to go from representing a word in a vacuum to representing a word conditioned on all other tokens that appear in the sequence. This sounds a lot like what an RNN is supposed to do. Can we just go ahead and replace our RNN l …

## Next page (PDF 470) text, for continuity
Compare to `BatchNormalization` (during training):

While `BatchNormalization` collects information from many samples to obtain accurate statistics for the feature means and variances, `LayerNormalization` pools data within each sequence separately, which is more appropriate for sequence data.

We also pass a new input to the `MultiHeadAttention` layer called `attention_mask`. This Boolean tensor input will be broadcast to the same shape as our attention scores `(batch_size, target_length, source_length)`. When set, it will zero the attention score in specific locations, stopping the source tokens at these locations from being used in the attention calculation. We will use this to prevent any token in the sequence from attending to padding tokens, which contain no information. Our encoder layer takes a `source_mask` input that will mark all the non-padding tokens in our inputs and upranks it to shape `(batch_size, 1, source_length)` to use as an `attention_mask`. Note that the input and outputs of this layer have the same shape, so encoder blocks can be stacked on top of each other, building a progressively more expressive representation of the input English sentence.

15.3.3 Transformer decoder block

Next up is the decoder block. This layer will be almost identical to the encoder block, except we want the decoder to use the encoder output sequence as an input. To do this, we can use attention twice. We first apply a self-attention layer like our encoder, which allows each position in the target sequence to use information from other target positions. We then add another `MultiHeadAttention` layer, which receives both the

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.471 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.498 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.627 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.476 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.473 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.458 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.472 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.474 — 15 Language models and the Transformer › 15.3 The Transformer architecture

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.471 (printed 447) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
source and target sequence as input. We will call this attention layer cross-attention as it brings information across the encoder and decoder.

Listing 15.15 A Transformer decoder block

### PDF p.489 (printed 465) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › 
The Transformer is a sequence modeling architecture that uses attention as the only mechanism to pass information across a sequence: – The Transformer works by stacking blocks of alternating attention and two-layer feedforward networks. – The Transformer can scale to many parameters and lots of training data while still improving accuracy in the language modeling problem. – Unlike RNNs, the Transformer involves no sequence-length loops at training time, making the model much easier to train in parallel across many machines. – A Transformer encoder uses bidirectional attention to build a rich representation of a sequence. – A Transformer decoder uses causal attention to predict the next word in a language model setup.

### PDF p.461 (printed 437) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 15.3 The Transformer architecture
In 2017, Vaswani et al. introduced the Transformer architecture in the seminal paper “Attention Is All You Need.”1 The authors were working on translation systems like the one we just built, and the critical discovery is right in the title. As it turned out, a simple mechanism called attention can be used to construct powerful sequence models that don’t feature recurrent layers at all. The idea of attention was not new and had been used in NLP systems for a couple of years when they published. But the idea that attention was so useful it could be the only mechanism used to pass information across a sequence was quite surprising at the time.

This finding unleashed nothing short of a revolution in natural language processing—and beyond. Attention has fast become one of the most influential ideas in deep learning. In this section, you’ll get an in-depth explanation of how it works and why it has proven so effective for sequence modeling. We’ll then use attention to rebuild our English-to-Spanish translation model. So, with all that as build-up, what exactly is attention? And how does it offer a replacement for the recurrent neural networks we have used so far? Attention was actually 

### PDF p.498 (printed 474) — 16 Text generation › 16.2 Training a mini-GPT › 
France?”, the learned representation of the word “Where” cannot attend to the words “capital” and “France” in the attention layer. This limits the expressivity of the model but has a massive advantage in terms of simplicity of pretraining. We don’t need to curate datasets with pairs of inputs and outputs; everything can be a single sequence. We can train on any text we can find on the internet at a massive scale.

Let’s copy the `TransformerDecoder` from chapter 15 but remove the cross-attention layer, which allowed the decoder to attend to the encoder sequence. We will also make one minor change, adding dropout after the attention and feedforward blocks. In chapter 15, we only used a single Transformer layer in our encoder and decoder, so we could get away with only using a single dropout layer at the end of our entire model. For our GPT model, we will stack quite a few layers, so adding dropout within each decoder layer is important to prevent overfitting.

Listing 16.4 A Transformer decoder block without cross-attention

### PDF p.474 (printed 450) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
After training, we get to about 58% accuracy: on average, the model correctly predicts the next word in the Spanish sentence 58% of the time. Something is off here. Training is worse than the RNN model by 7 percentage points. Either this Transformer architecture is not what it was hyped up to be, or we missed something in our implementation. Can you spot what it is?

This section is ostensibly about sequence models. In the previous chapter, we saw how vital word order could be to meaning. And yet, the Transformer we just built wasn’t a sequence model at all. Did you notice? It’s composed of dense layers that process sequence tokens independently of each other and an attention layer that looks at the tokens as a set. You could change the order of the tokens in a sequence, and you’d get identical pairwise attention scores and the same context-aware representations. If you were to rearrange every word in every English source sentence completely, the model wouldn’t notice, and you’d still get the same accuracy. Attention is a set-processing mechanism, focused on the relationships between pairs of sequence elements—it’s blind to whether these elements occur at the beginning, at the end,

### PDF p.471 (printed 447) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
Our decoder layer takes in both a `target` and `source`. Like with the `Transformer` `Encoder`, we take in a `source_mask` that marks the location of all padding in the source input (`True` for non-padding, `False` for padding) and use it as an `attention_mask` for the cross-attention layer.

For the decoder’s self-attention layer, we need a different type of attention mask. Recall that when we built our RNN decoder, we avoided using a `Bidirectional` RNN. If we had used one, the model would be able to cheat by seeing the label it was trying to predict as a feature! Attention is inherently bidirectional; in self-attention, any token position in the target sequence can attend to any other position. Without special care,

### PDF p.472 (printed 448) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
Each row `i` can be read as a mask for attention for the target token at position `i`. In the first row, the first token can only attend to itself. In the second row, the second token can attend to both the first and second tokens, and so forth. This gives us the same effect as our RNN layer, where information can only propagate forward in the sequence, not backward. In Keras, you can specify this lower-triangular mask simply by passing `use_casual_mask` to the `MultiHeadAttention` layer when calling it. Figure 15.8 shows a visual representation of the layers in both the encoder and decoder layers, when stacked into a Transformer model.

Figure 15.8 A visual representation of the computations for both

`TransformerEncoder` and `TransformerDecoder` blocks

### PDF p.484 (printed 460) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 15.4.4 Fine-tuning a pretrained Transformer
Before we fine-tune our backbone to predict movie reviews, we need to update it so it outputs a binary classification label. The backbone outputs an entire sequence with shape `(batch_size, sequence_length, 768)`, where each 768-dimensional vector represents an input word in the context of its surrounding words. Before predicting a label, we must condense this sequence to a single vector per sample.

One option would be to do mean pooling or max pooling across the whole sequence, computing an average of all token vectors. What works slightly better is simply using the first token’s representation as the pooled value. This is due to the nature of the attention in our model—the first position in the final encoder layer will be able to attend to all other positions in the sequence and pull information from them. So rather than pooling information with something coarse, like taking an average, attention allows us to pool information contextually across the sequence. Let’s now add a classification head to our backbone. We will also add one final `Dense` projection with a nonlinearity before generating an output prediction.
