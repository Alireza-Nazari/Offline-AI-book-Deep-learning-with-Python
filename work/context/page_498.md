# Generation bundle for page_498  (PDF page 498, printed page 474)

Page kind: content
Chapter: 16 Text generation
Section: 16.2 Training a mini-GPT
Subsection: 16.2.1 Building the model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_498.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_498.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p498_e03] (body) France?”, the learned representation of the word “Where” cannot attend to the words “capital” and “France” in the attention layer. This limits the expressivity of the model but has a massive advantage in terms of simplicity of pretraining. We don’t need to curate datasets with pairs of inputs and outputs; everything can be a single sequence. We can train on any text we can find on the internet at a massive scale.
[p498_e04] (body) Let’s copy the `TransformerDecoder` from chapter 15 but remove the cross-attention layer, which allowed the decoder to attend to the encoder sequence. We will also make one minor change, adding dropout after the attention and feedforward blocks. In chapter 15, we only used a single Transformer layer in our encoder and decoder, so we could get away with only using a single dropout layer at the end of our entire model. For our GPT model, we will stack quite a few layers, so adding dropout within each decoder layer is important to prevent overfitting.
[p498_e05] (listing_caption) Listing 16.4 A Transformer decoder block without cross-attention
[p498_e06] (code)
```
from keras import layers
class TransformerDecoder(keras.Layer):
    def __init__(self, hidden_dim, intermediate_dim, num_heads):
        super().__init__()
        key_dim = hidden_dim // num_heads
        self.self_attention = layers.MultiHeadAttention(
            num_heads, key_dim, dropout=0.1
        )
        self.self_attention_layernorm = layers.LayerNormalization()
        self.feed_forward_1 = layers.Dense(intermediate_dim, activation="relu")
        self.feed_forward_2 = layers.Dense(hidden_dim)
        self.feed_forward_layernorm = layers.LayerNormalization()
        self.dropout = layers.Dropout(0.1)
    def call(self, inputs):
        residual = x = inputs
        x = self.self_attention(query=x, key=x, value=x, use_causal_mask=True)
        x = self.dropout(x)
        x = x + residual
        x = self.self_attention_layernorm(x)
        residual = x
        x = self.feed_forward_1(x)
        x = self.feed_forward_2(x)
        x = self.dropout(x)
        x = x + residual
        x = self.feed_forward_layernorm(x)
        return x
```
[p498_e07] (body) Next, we can copy the `PositionalEmbedding` layer from chapter 15. Recall that this layer gives us a simple way to learn an embedding for each position in a sequence and combine that with our token embeddings.
[p498_e08] (body) There’s a neat trick we can employ here to save some GPU memory. The biggest weights in a Transformer model are the input token embeddings and output dense

## Code annotations printed next to listings (callouts, in order)
- Self-attention layers
- Feedforward layers
- Self-attention
- computation
- Feedforward
- computation

## Listings
- Listing 16.4: A Transformer decoder block without cross-attention (code elements p498_e06)

## Chapter outline
Chapter: 16 Text generation (PDF pages 490-531)
  - 16.1 A brief history of sequence generation (p.492)
  - 16.2 Training a mini-GPT (p.494)
      - 16.2.1 Building the model (p.497)
      - 16.2.2 Pretraining the model (p.500)
      - 16.2.3 Generative decoding (p.502)
      - 16.2.4 Sampling strategies (p.504)
  - 16.3 Using a pretrained LLM (p.508)
      - 16.3.1 Text generation with the Gemma model (p.509)
      - 16.3.2 Instruction fine-tuning (p.512)
      - 16.3.3 Low-Rank Adaptation (LoRA) (p.514)
  - 16.4 Going further with LLMs (p.519)
      - 16.4.1 Reinforcement Learning with Human Feedback (RLHF) (p.519)
      - 16.4.2 Multimodal LLMs (p.522)
      - 16.4.3 Retrieval Augmented Generation (RAG) (p.525)
      - 16.4.4 “Reasoning” models (p.526)
  - 16.5 Where are LLMs heading next? (p.528)

## Previous page (PDF 497) text, for continuity
As we first did in chapter 8, we will end our `tf.data` pipeline with a call to `prefetch()`. This will make sure we always have some batches loaded onto our GPU and ready for the model.

We have 29,373 batches. You could count this yourself if you would like—the line `ds.reduce(0, lambda c, _: c + 1)` will iterate over the entire dataset and increment a counter. But simply tokenizing a dataset of this size will take a few minutes on a decently fast CPU. At 128 samples per batch and 256 tokens per sample, this is just under a billion tokens of data. Let’s split off 500 batches as a quick validation set, and we are ready to start pretraining:

16.2.1 Building the model

The original GPT model simplifies the sequence-to-sequence Transformer we saw in the last chapter. Rather than take in a source and target sequence with an encoder and decoder, as we did for our translation model, the GPT approach does away with the encoder entirely and only uses the decoder. This means that information can only travel from left to right in a sequence.

This was an interesting bet on the part of the GPT developers. A decoder-only model can still handle sequence-to-sequence problems like question-answering. However, rather than feeding in the question and answer as separate inputs, we must combine both into a single sequence to feed it to our model. So, unlike the original Transformer, the question tokens would not be handled any differently than answer tokens. All tokens are embedded into the same latent space with the same set of parameters. The other consequence of this approach is that the information flow is no longer bidirectional, even for input sequences. Given an input, such as “Where is the capital of

## Next page (PDF 499) text, for continuity
prediction layer because they deal with our vocabulary space. The token embedding weight has shape `(vocab_size, hidden_dim)` to embed every possible token. Our output projection has shape `(hidden_dim, vocab_size)` to make a floating-point prediction for every possible token.

We can actually tie these two weight matrices together. To compute our model’s final predictions, we will multiply our hidden states by the transpose of our token embedding matrix. You can very much think of our final projection as a “reverse embedding.” It maps from hidden space to token space, whereas an embedding maps from token space to hidden space. It turns out that using the same weights for this input and output projection is a good idea. Adding this to our `PositionalEmbedding` is simple; we will just add a `reverse` argument to the `call` method, which computes the projection by the transpose of the token embedding.

Listing 16.5 A positional embedding layer that can reverse a text embedding

Let’s build our model. We will stack eight decoder layers into a single “mini” GPT model.

We will also turn on a Keras setting called mixed precision to speed up training. This will allow Keras to run some of the model’s computations much faster by sacrificing some numerical fidelity. For now, this will remain a little mysterious, but a full explanation is waiting in chapter 18.

Listing 16.6 Creating a mini-GPT functional model

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.471 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.469 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.472 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.476 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.468 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.470 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.473 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.474 — 15 Language models and the Transformer › 15.3 The Transformer architecture

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.469 (printed 445) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
to a single matrix multiplication! That’s a real problem with the expressiveness of our model.

At some point, all recurrent cells pass the input vector for each token through a dense projection and apply an activation function; we need a plan for something similar. The authors of “Attention is all you need” decided to add this back in the simplest way possible—stacking a feedforward network of two dense layers with an activation in the middle. Attention passes information across the sequence, and the feedforward network updates the representation of individual sequence items. We are ready to start building a Transformer model. Let’s start by replacing the encoder of our translation model. We will use self-attention to pass information along the source sequence of English words. We will also add in two things we learned to be particularly important when building ConvNets back in chapter 9, normalization and residual connections.

Listing 15.14 A Transformer encoder block

### PDF p.468 (printed 444) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 15.3.2 Transformer encoder block
One way to use the `MultiHeadAttention` layer would be to add it to our existing RNN translation model. We could pass the sequence output from our encoder and decoder into an attention layer and use its output to update our target sequence before prediction. Attention would allow the model to handle long-range dependencies in text that the `GRU` layer will struggle with. This does, in fact, improve an RNN model’s capabilities and is how attention was first used in the mid-2010s.

However, the authors of “Attention is all you need” realized you could go further and use attention as a general mechanism for handling all sequence data in a model. Although so far we have only looked at attention as a way to handle information passing between two sequences, you could also use attention as a way to let a sequence attend to itself:

### PDF p.471 (printed 447) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
source and target sequence as input. We will call this attention layer cross-attention as it brings information across the encoder and decoder.

Listing 15.15 A Transformer decoder block

### PDF p.470 (printed 446) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
While `BatchNormalization` collects information from many samples to obtain accurate statistics for the feature means and variances, `LayerNormalization` pools data within each sequence separately, which is more appropriate for sequence data.

We also pass a new input to the `MultiHeadAttention` layer called `attention_mask`. This Boolean tensor input will be broadcast to the same shape as our attention scores `(batch_size, target_length, source_length)`. When set, it will zero the attention score in specific locations, stopping the source tokens at these locations from being used in the attention calculation. We will use this to prevent any token in the sequence from attending to padding tokens, which contain no information. Our encoder layer takes a `source_mask` input that will mark all the non-padding tokens in our inputs and upranks it to shape `(batch_size, 1, source_length)` to use as an `attention_mask`. Note that the input and outputs of this layer have the same shape, so encoder blocks can be stacked on top of each other, building a progressively more expressive representation of the input English sentence.

### PDF p.458 (printed 434) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 
Next, let’s add the decoder—a simple `GRU` layer that takes as its initial state the encoded source sentence. On top of it, we add a `Dense` layer that produces a probability distribution over the Spanish vocabulary for each output step. Here, we want to predict the next tokens based only on what came before, so a `Bidirectional` RNN would break training by making the loss function trivially easy.

Listing 15.12 Building a sequence-to-sequence decoder

### PDF p.479 (printed 455) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
model were released a year after “Attention Is All You Need.” The model structure was exactly the same as the encoder part of the translation Transformer we just built. This encoder model is bidirectional in that every position in the sequence can attend to positions in front of and behind it. This means it’s a good model for computing a rich representation of input text, but not a model meant to run generation in a loop.

BERT was trained in sizes between 100 million and 300 million parameters, much bigger than the 14 million parameter Transformer we just trained. This meant the model needed a lot of training data to perform well. To achieve this, the authors used a riff on the classic language modeling setup called masked language modeling. To pretrain the model, we take a sequence of text and replace about 15% of the tokens with a special `[MASK]` token. The model will attempt to predict the original masked token values during training. While the classic language model, sometimes called a causal language model, attempts to predict `p(token|past` `tokens)`, the masked language model attempts to predict `p(token|surrounding` `tokens)`. This training setup is unsupervised. You don’

### PDF p.468 (printed 444) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 15.3.2 Transformer encoder block
This is called self-attention, and it is quite powerful. With self-attention, each token can attend to every token in its own sequence, including itself, allowing the model to learn a representation of the word in context.

Consider an example sentence: “The train left the station on time.” Now, consider one word in the sentence: “station.” What kind of station are we talking about? Could it be a radio station? Maybe the International Space Station? With self-attention, the model could learn to give a high attention score to the pair of “station” and “train,” summing the vector used to represent “train” into the representation of the word “station.” Self-attention gives the model an effective way to go from representing a word in a vacuum to representing a word conditioned on all other tokens that appear in the sequence. This sounds a lot like what an RNN is supposed to do. Can we just go ahead and replace our RNN layers with `MultiHeadAttention`? Almost! But not quite; we still need an essential ingredient for any deep neural network—a nonlinear activation function. The `MultiHeadAttention` layer combines linear projections of every element in a source sequence, but that’s it. In 

### PDF p.476 (printed 452) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
We would use this `PositionalEmbedding` layer just like a regular `Embedding` layer. Let’s see it in action as we try training our Transformer for a second time.

Listing 15.18 Building a Transformer model with positional embeddings
