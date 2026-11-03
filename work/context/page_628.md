# Generation bundle for page_628  (PDF page 628, printed page 604)

Page kind: content
Chapter: 20 Conclusions
Section: 20.1 Key concepts in review
Subsection: 20.1.6 Key network architectures
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_628.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_628.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p628_e03] (code)
```
decoder_inputs = keras.Input(shape=(dst_seq_length,), dtype="int64")
x = TokenAndPositionEmbedding(vocab_size, dst_seq_length, embed_dim)(
    decoder_inputs
)
x = TransformerDecoder(intermediate_dim=256, num_heads=8)(x, encoder_outputs)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
transformer.compile(optimizer="adamw", loss="categorical_crossentropy")
```
[p628_e04] (body) And this is a lone `TransformerEncoder` for binary classification of integer sequences:
[p628_e05] (code)
```
inputs = keras.Input(shape=(seq_length,), dtype="int64")
x = TokenAndPositionEmbedding(vocab_size, seq_length, embed_dim)(inputs)
x = TransformerEncoder(intermediate_dim=256, num_heads=8)(x)
x = layers.GlobalMaxPooling1D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="adamw", loss="binary_crossentropy")
```
[p628_e06] (heading_subsubsection) Recurrent neural networks
[p628_e07] (body) Recurrent neural networks (RNNs) work by processing sequences of inputs one timestep at a time and maintaining a state throughout (a state is typically a vector or set of vectors). They should be used preferentially over 1D ConvNets in the case of sequences where patterns of interest aren’t invariant by temporal translation (for instance, timeseries data where the recent past is more important than the distant past).
[p628_e08] (body) Three RNN layers are available in Keras: `SimpleRNN`, `GRU`, and `LSTM`. For most practical purposes, you should use either `GRU` or `LSTM`. `LSTM` is the more powerful of the two but is also more expensive; you can think of `GRU` as a simpler, cheaper alternative to it. To stack multiple RNN layers on top of each other, each layer prior to the last layer in the stack should return the full sequence of its outputs (each input timestep will correspond to an output timestep); if you aren’t stacking any further RNN layers, then it’s common to return only the last output, which contains information about the entire sequence. Following is a single RNN layer for binary classification of vector sequences:
[p628_e09] (code)
```
inputs = keras.Input(shape=(num_timesteps, num_features))
x = layers.LSTM(32)(inputs)
outputs = layers.Dense(num_classes, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy")
```

## Code annotations printed next to listings (callouts, in order)
- Predictions for target sequence
- one step in the future
- Target sequence so far

## Chapter outline
Chapter: 20 Conclusions (PDF pages 619-632)
  - 20.1 Key concepts in review (p.619)
      - 20.1.1 Various approaches to artificial intelligence (p.620)
      - 20.1.2 What makes deep learning special within the field of machine learning (p.620)
      - 20.1.3 How to think about deep learning (p.621)
      - 20.1.4 Key enabling technologies (p.622)
      - 20.1.5 The universal machine learning workflow (p.623)
      - 20.1.6 Key network architectures (p.624)
  - 20.2 Limitations of deep learning (p.629)
  - 20.3 What might lie ahead (p.630)
  - 20.4 Staying up to date in a fast-moving field (p.631)
      - 20.4.1 Practice on real-world problems using Kaggle (p.631)
      - 20.4.2 Read about the latest developments on arXiv (p.631)
      - 20.4.3 Explore the Keras ecosystem (p.632)
  - 20.5 Final words (p.632)

## Previous page (PDF 627) text, for continuity
When building a very deep ConvNet, it’s common to add batch normalization layers as well as residual connections—two architecture patterns that help gradient information flow smoothly through the network.

Transformers

A Transformer looks at a set of vectors (such as word vectors) and uses neural attention to transform each vector into a representation that is aware of the context provided by the other vectors in the set. When the set in question is an ordered sequence, you can also use positional encoding to create Transformers that can take into account both global context and word order, capable of processing long text paragraphs much more effectively than RNNs or 1D ConvNets.

Transformers can be used for any set-processing or sequence-processing task, including text classification, but they excel especially at sequence-to-sequence learning, such as translating paragraphs in a source language into a target language. A sequence-to-sequence Transformer is made of two parts:

A `TransformerEncoder` that turns an input vector sequence into a context-aware, order-aware output vector sequence

A `TransformerDecoder` that takes the output of the `TransformerEncoder`, as well as a target sequence, and predicts what should come next in the target sequence

If you’re only processing a single sequence (or set) of vectors, you’d only use the `TransformerEncoder`.

Following is a sequence-to-sequence Transformer for mapping a source sequence to a target sequence (this setup could be used for machine translation or question-answering, for instance):

## Next page (PDF 629) text, for continuity
And this is a stacked RNN layer for binary classification of vector sequences:

20.2 Limitations of deep learning

Building deep learning models is like playing with LEGO bricks: layers can be plugged together to map essentially anything to anything, given that you have appropriate training data available and that the mapping is achievable via a continuous geometric transformation of reasonable complexity.

Here’s the catch, though—this mapping is often not learnable in a way that will generalize. Deep learning models operate like vast, interpolative databases of patterns. Their pattern-matching strength is also their core weakness:

They fundamentally struggle to adapt to novelty. Because their parameters are fixed after training, they can only retrieve or replicate patterns similar to their training data. Faced with inputs significantly outside this familiar distribution—no matter how simple the underlying task—their performance degrades drastically, as they lack mechanisms for fluid generalization beyond their memorized experience. This explains why even large models fail on novel tasks or simple variations of familiar problems—like ARC-AGI tasks.

They’re very sensitive to phrasing and other distractors. Deep learning models exhibit high sensitivity to superficial variations in input presentation, such as minor phrasing changes (consider prompt sensitivity in LLMs) or imperceptible perturbations (consider adversarial examples in vision), indicating a lack of robust, human-like understanding.

They often can’t learn generalizable algorithms. The continuous, geometric nature of deep learning models makes them fundamentally ill-suited for learning exact, discrete, step-by-step algorithms, such as those that are the bread and butter of classical computer science. Models …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.625 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.626 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.457 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.392 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.431 — 14 Text classification › 14.5 Sequence models
- PDF p.219 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.251 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.220 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.392 (printed 368) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3.2 A recurrent layer in Keras
This is especially useful if your model is meant to process sequences of variable length. However, if all of your sequences have the same length, I recommend specifying a complete input shape, since it enables `model.summary()` to display output length information, which is always nice, and it can unlock some performance optimizations (see the sidebar “On RNN runtime performance” in section 13.3.4).

All recurrent layers in Keras (`SimpleRNN`, `LSTM`, and `GRU`) can be run in two different modes: they can return either full sequences of successive outputs for each timestep (a rank-3 tensor of shape `(batch_size, timesteps, output_features)`) or only the last output for each input sequence (a rank-2 tensor of shape `(batch_size, output_features)`). These two modes are controlled by the `return_sequences` constructor argument. Let’s look at an example that uses `SimpleRNN` and returns only the output at the last timestep.

Listing 13.17 An RNN layer that returns only its last output step

### PDF p.396 (printed 372) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3.3 Getting the most out of recurrent neural networks
By this point, you’ve learned

What RNNs are and how they work

What an LSTM is and why it works better on long sequences than a naive RNN

How to use Keras RNN layers to process sequence data

Next, we’ll review a number of more advanced features of RNNs, which can help you get the most out of your deep learning sequence models. By the end of the section, you’ll know most of what there is to know about using recurrent networks with Keras.

We’ll cover the following:

Recurrent dropout—This is a variant of dropout, used to fight overfitting in recurrent layers.

Stacking recurrent layers—This increases the representational power of the model (at the cost of higher computational loads).

Bidirectional recurrent layers—These present the same information to a recurrent network in different ways, increasing accuracy and mitigating forgetting issues.

We’ll use these techniques to refine our temperature forecasting RNN.

### PDF p.392 (printed 368) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3.2 A recurrent layer in Keras
The process you just naively implemented in NumPy corresponds to an actual Keras layer—the `SimpleRNN` layer.

There is one minor difference: `SimpleRNN` processes batches of sequences, like all other Keras layers, not a single sequence as in the NumPy example. This means it takes inputs of shape `(batch_size,` `timesteps,` `input_features)` rather than `(timesteps, ` `input_features)`. When specifying the `shape` argument of your initial `Input()`, note that you can set the `timesteps` entry to `None`, which enables your network to process sequences of arbitrary length.

Listing 13.16 An RNN layer that can process sequences of any length

### PDF p.404 (printed 380) — 13 Timeseries forecasting › 13.4 Going even further › Summary
As you first learned in chapter 6, when approaching a new problem, it’s good to first establish commonsense baselines for your metric of choice. If you don’t have a baseline to beat, you can’t tell whether you’re making real progress.

Try simple models before expensive ones, to justify the additional expense. Sometimes a simple model will turn out to be your best option.

When you have data where ordering matters—in particular, for timeseries data— recurrent networks are a great fit and easily outperform models that first flatten the temporal data. The two essential RNN layers available in Keras are the `LSTM` layer and the `GRU` layer.

To use dropout with recurrent networks, you should use a time-constant dropout mask and recurrent dropout mask. These are built into Keras recurrent layers, so all you have to do is use the `recurrent_dropout` arguments of recurrent layers.

Stacked RNNs provide more representational power than a single RNN layer. They’re also much more expensive and thus not always worth it. Although they offer clear gains on complex problems (such as machine translation), they may not always be relevant to smaller, simpler problems.

### PDF p.389 (printed 365) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3.1 Understanding recurrent neural networks
A major characteristic of all neural networks you’ve seen so far, such as densely connected networks and ConvNets, is that they have no memory. Each input shown to them is processed independently, with no state kept between inputs. With such networks, to process a sequence or a temporal series of data points, you have to show the entire sequence to the network at once: turn it into a single data point. For instance, this is what we did in the densely connected network example: we flattened our five days of data into a single large vector and processed it in one go. Such networks are called feedforward networks.

In contrast, as you’re reading the present sentence, you’re processing it word by word—or rather, eye saccade by eye saccade—while keeping memories of what came before; this gives you a fluid representation of the meaning conveyed by this sentence. Biological intelligence processes information incrementally while maintaining an internal model of what it’s processing, built from past information and constantly updated as new information comes in. A recurrent neural network (RNN) adopts the same principle, albeit in an extremely simplified version: it processes sequences by i

### PDF p.399 (printed 375) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3.5 Stacking recurrent layers
Because you’re no longer overfitting, but seem to have hit a performance bottleneck, you should consider increasing the capacity and expressive power of the network. Recall the description of the universal machine learning workflow: it’s generally a good idea to increase the capacity of your model until overfitting becomes the primary obstacle (assuming you’re already taking basic steps to mitigate overfitting, such as using dropout). As long as you aren’t overfitting too badly, you’re likely under capacity.

Increasing network capacity is typically done by increasing the number of units in the layers or adding more layers. Recurrent layer stacking is a classic way to build more powerful recurrent networks: for instance, not too long ago the Google Translate algorithm was powered by a stack of seven large `LSTM` layers—that’s huge. To stack recurrent layers on top of each other in Keras, all intermediate layers should return their full sequence of outputs (a rank-3 tensor) rather than their output at the last timestep. As you’ve already learned, this is done by specifying `return_sequences=True`. In the following example, we’ll try a stack of two dropout-regularized recurrent layer

### PDF p.390 (printed 366) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
The state of the RNN is reset between processing two different, independent sequences (such as two samples in a batch), so you still consider one sequence to be a single data point: a single input to the network. What changes is that this data point is no longer processed in a single step; rather, the network internally loops over sequence elements. To make these notions of loop and state clear, let’s implement the forward pass of a toy RNN. This RNN takes as input

Figure 13.6 A recurrent

a sequence of vectors, which we’ll encode as a rank-2 ten

network: a network with a loop

sor of size `(timesteps, input_features)`. It loops over timesteps, and at each timestep, it considers its current state at `t` and the input at `t` (of shape `(input_features,)`), and combines them to obtain the output at `t`. We’ll then set the state for the next step to be this previous output. For the first timestep, the previous output isn’t defined; hence, there is no current state. So we’ll initialize the state as an all-zero vector called the initial state of the network.

In pseudocode, this is the RNN.

Listing 13.13 Pseudocode RNN

### PDF p.461 (printed 437) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
Note that this inference setup, while very simple, is inefficient, since we reprocess the entire source sentence and the entire generated target sentence every time we sample a new word. In a practical application, you’d want to be careful not to recompute any state that has not changed. All we really need to predict a new token in the decoder is the current token and the previous RNN state, which we could cache before each loop iteration. There are many ways this toy model could be improved. We could use a deep stack of recurrent layers for both the encoder and the decoder, we could try other RNN layers like `LSTM`, and so on. Beyond such tweaks, however, the RNN approach to sequence-to-sequence learning has a few fundamental limitations:

The source sequence representation has to be held entirely in the encoder state vector, which significantly limits the size and complexity of the sentences you can translate.

RNNs have trouble dealing with very long sequences since they tend to progressively forget about the past—by the time you’ve reached the 100th token in either sequence, little information remains about the start of the sequence.

Recurrent neural networks dominated sequenc
