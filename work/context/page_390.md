# Generation bundle for page_390  (PDF page 390, printed page 366)

Page kind: content
Chapter: 13 Timeseries forecasting
Section: 13.3 Recurrent neural networks
Subsection: 13.3.1 Understanding recurrent neural networks
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_390.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_390.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p390_e03] (body) The state of the RNN is reset between processing two different, independent sequences (such as two samples in a batch), so you still consider one sequence to be a single data point: a single input to the network. What changes is that this data point is no longer processed in a single step; rather, the network internally loops over sequence elements. To make these notions of loop and state clear, let’s implement the forward pass of a toy RNN. This RNN takes as input
[p390_e04] (figure_caption) Figure 13.6 A recurrent
[p390_e05] (body) a sequence of vectors, which we’ll encode as a rank-2 ten
[p390_e06] (figure_caption) network: a network with a loop
[p390_e07] (body) sor of size `(timesteps, input_features)`. It loops over timesteps, and at each timestep, it considers its current state at `t` and the input at `t` (of shape `(input_features,)`), and combines them to obtain the output at `t`. We’ll then set the state for the next step to be this previous output. For the first timestep, the previous output isn’t defined; hence, there is no current state. So we’ll initialize the state as an all-zero vector called the initial state of the network.
[p390_e08] (body) In pseudocode, this is the RNN.
[p390_e09] (listing_caption) Listing 13.13 Pseudocode RNN
[p390_e10] (code)
```
state_t = 0
for input_t in input_sequence:
    output_t = f(input_t, state_t)
    state_t = output_t
```
[p390_e11] (body) You can even flesh out the function `f`: the transformation of the input and state into an output will be parameterized by two matrices, `W` and `U`, and a bias vector. It’s similar to the transformation operated by a densely connected layer in a feedforward network.
[p390_e12] (listing_caption) Listing 13.14 More detailed pseudocode for the RNN
[p390_e13] (code)
```
state_t = 0
for input_t in input_sequence:
    output_t = activation(dot(W, input_t) + dot(U, state_t) + b)
    state_t = output_t
```
[p390_e14] (body) To make these notions absolutely unambiguous, let’s write a naive NumPy implementation of the forward pass of the simple RNN.
[p390_e15] (listing_caption) Listing 13.15 NumPy implementation of a simple RNN
[p390_e16] (code)
```
import numpy as np
timesteps = 100
input_features = 32
```

## Code annotations printed next to listings (callouts, in order)
- The state at t
- Iterates over sequence elements
- The previous output becomes
- the state for the next iteration.
- Number of timesteps in the input sequence
- Dimensionality of the input feature space

## Listings
- Listing 13.13: Pseudocode RNN (code elements p390_e10)
- Listing 13.14: More detailed pseudocode for the RNN (code elements p390_e13)
- Listing 13.15: NumPy implementation of a simple RNN (code elements p390_e16)

## Figures
- Figure 13.6: A recurrent  | labels inside the figure: RNN, Output, Recurrent, connection
- Figure None: network: a network with a loop  | labels inside the figure: (none - image)

## Chapter outline
Chapter: 13 Timeseries forecasting (PDF pages 375-404)
  - 13.1 Different kinds of timeseries tasks (p.375)
  - 13.2 A temperature forecasting example (p.376)
      - 13.2.1 Preparing the data (p.380)
      - 13.2.2 A commonsense, non-machine-learning baseline (p.383)
      - 13.2.3 Let’s try a basic machine learning model (p.384)
      - 13.2.4 Let’s try a 1D convolutional model (p.386)
  - 13.3 Recurrent neural networks (p.388)
      - 13.3.1 Understanding recurrent neural networks (p.389)
      - 13.3.2 A recurrent layer in Keras (p.392)
      - 13.3.3 Getting the most out of recurrent neural networks (p.396)
      - 13.3.4 Using recurrent dropout to fight overfitting (p.396)
      - 13.3.5 Stacking recurrent layers (p.399)
      - 13.3.6 Using bidirectional RNNs (p.401)
  - 13.4 Going even further (p.403)

## Previous page (PDF 389) text, for continuity
Figure 13.5 Training and validation MAE on the Jena temperature forecasting task with an LSTM-based model. (Note that we omit epoch 1 on this graph because the high training MAE (7.75) at epoch 1 would distort the scale.)

But why did the LSTM model perform markedly better than the densely connected one or the ConvNet? And how can we further refine the model? To answer this, let’s take a closer look at recurrent neural networks.

13.3.1 Understanding recurrent neural networks

A major characteristic of all neural networks you’ve seen so far, such as densely connected networks and ConvNets, is that they have no memory. Each input shown to them is processed independently, with no state kept between inputs. With such networks, to process a sequence or a temporal series of data points, you have to show the entire sequence to the network at once: turn it into a single data point. For instance, this is what we did in the densely connected network example: we flattened our five days of data into a single large vector and processed it in one go. Such networks are called feedforward networks.

In contrast, as you’re reading the present sentence, you’re processing it word by word—or rather, eye saccade by eye saccade—while keeping memories of what came before; this gives you a fluid representation of the meaning conveyed by this sentence. Biological intelligence processes information incrementally while maintaining an internal model of what it’s processing, built from past information and constantly updated as new information comes in. A recurrent neural network (RNN) adopts the same principle, albeit in an extremely simplified version: it processes sequences by iterating through the sequence elements and maintaining a state containing information relative to what it has seen so  …

## Next page (PDF 391) text, for continuity
Easy enough: in summary, an RNN is a `for` loop that reuses quantities computed during the previous iteration of the loop, nothing more. Of course, there are many different RNNs fitting this definition that you could build—this example is one of the simplest RNN formulations. RNNs are characterized by their step function, such as the following function in this case (see figure 13.7):

Figure 13.7 A simple RNN, unrolled over time

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.395 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.392 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.393 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.628 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.401 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.399 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.396 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.457 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.628 (printed 604) — 20 Conclusions › 20.1 Key concepts in review › Recurrent neural networks
Recurrent neural networks (RNNs) work by processing sequences of inputs one timestep at a time and maintaining a state throughout (a state is typically a vector or set of vectors). They should be used preferentially over 1D ConvNets in the case of sequences where patterns of interest aren’t invariant by temporal translation (for instance, timeseries data where the recent past is more important than the distant past).

Three RNN layers are available in Keras: `SimpleRNN`, `GRU`, and `LSTM`. For most practical purposes, you should use either `GRU` or `LSTM`. `LSTM` is the more powerful of the two but is also more expensive; you can think of `GRU` as a simpler, cheaper alternative to it. To stack multiple RNN layers on top of each other, each layer prior to the last layer in the stack should return the full sequence of its outputs (each input timestep will correspond to an output timestep); if you aren’t stacking any further RNN layers, then it’s common to return only the last output, which contains information about the entire sequence. Following is a single RNN layer for binary classification of vector sequences:

### PDF p.392 (printed 368) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3.2 A recurrent layer in Keras
The process you just naively implemented in NumPy corresponds to an actual Keras layer—the `SimpleRNN` layer.

There is one minor difference: `SimpleRNN` processes batches of sequences, like all other Keras layers, not a single sequence as in the NumPy example. This means it takes inputs of shape `(batch_size,` `timesteps,` `input_features)` rather than `(timesteps, ` `input_features)`. When specifying the `shape` argument of your initial `Input()`, note that you can set the `timesteps` entry to `None`, which enables your network to process sequences of arbitrary length.

Listing 13.16 An RNN layer that can process sequences of any length

### PDF p.392 (printed 368) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3.2 A recurrent layer in Keras
This is especially useful if your model is meant to process sequences of variable length. However, if all of your sequences have the same length, I recommend specifying a complete input shape, since it enables `model.summary()` to display output length information, which is always nice, and it can unlock some performance optimizations (see the sidebar “On RNN runtime performance” in section 13.3.4).

All recurrent layers in Keras (`SimpleRNN`, `LSTM`, and `GRU`) can be run in two different modes: they can return either full sequences of successive outputs for each timestep (a rank-3 tensor of shape `(batch_size, timesteps, output_features)`) or only the last output for each input sequence (a rank-2 tensor of shape `(batch_size, output_features)`). These two modes are controlled by the `return_sequences` constructor argument. Let’s look at an example that uses `SimpleRNN` and returns only the output at the last timestep.

Listing 13.17 An RNN layer that returns only its last output step

### PDF p.450 (printed 426) — 15 Language models and the Transformer › 15.1 The language model › 15.1.2 Generating Shakespeare
Now that we have trained a model that can predict the next individual tokens with some accuracy, we would like to use it to extrapolate an entire predicted sequence. We can do this by calling the model in a loop, where the model’s predicted output at one time step becomes the model’s input at the next time step. A model built for this kind of feedback loop is sometimes called an autoregressive model.

To run such a loop, we need to perform a slight surgery on the model we just trained. During training, our model handled only a fixed sequence length of 100 tokens, and the `GRU` cell’s state was handled implicitly when calling the layer. During generation, we would like to predict a single output token at a time and explicitly output the state of the `GRU`’s cell. We need to propagate that state, which contains all information the model has encoded about past input characters, the next time we call the model. Let’s make a model that handles a single input character at a time and allows explicitly passing the RNN state. Because this model will have the same computational structure, with slightly modified inputs and outputs, we can assign weights from one model to another.

Listing 15.

### PDF p.461 (printed 437) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
Note that this inference setup, while very simple, is inefficient, since we reprocess the entire source sentence and the entire generated target sentence every time we sample a new word. In a practical application, you’d want to be careful not to recompute any state that has not changed. All we really need to predict a new token in the decoder is the current token and the previous RNN state, which we could cache before each loop iteration. There are many ways this toy model could be improved. We could use a deep stack of recurrent layers for both the encoder and the decoder, we could try other RNN layers like `LSTM`, and so on. Beyond such tweaks, however, the RNN approach to sequence-to-sequence learning has a few fundamental limitations:

The source sequence representation has to be held entirely in the encoder state vector, which significantly limits the size and complexity of the sentences you can translate.

RNNs have trouble dealing with very long sequences since they tend to progressively forget about the past—by the time you’ve reached the 100th token in either sequence, little information remains about the start of the sequence.

Recurrent neural networks dominated sequenc

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

### PDF p.457 (printed 433) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 
However, there is a critical issue with this approach. Due to the step-by-step nature of RNNs, the model will only look at tokens `0...N` in the source sequence to predict token `N` in the target sequence. Consider translating the sentence, “I will bring the bag to you.” In Spanish, that would be “Te traeré la bolsa,” where “Te,” the first word of the translation, corresponds to “you” in the English source text. There’s simply no way to output the first word of the translation without seeing the last word of the source English text!

If you’re a human translator, you’d start by reading the entire source sentence before beginning to translate it. This is especially important if you’re dealing with languages with wildly different word ordering. And that’s precisely what standard sequence-to-sequence models do. In a proper sequence-to-sequence setup (see figure 15.2), you would first use an encoder RNN to turn the entire source sequence into a single representation of the source text. This could be the last output of the RNN or, alternatively, its final internal state vectors. We can use this representation as the initial state of a decoder RNN in the language model setup instead of a

### PDF p.504 (printed 480) — 16 Text generation › 16.2 Training a mini-GPT › Cached generation
There’s still one more major inefficiency in the generation function we just built. Can you spot it? Each time we call our model, we call it for an entire sequence and then throw away everything but the predictions for a single position. This is wasteful—our sequence only changes by a single token between generation steps. When we did generation with an RNN in chapter 15, we could keep around our RNN state and only compute outputs for a single token at each step. This state vector contained all the information the model needed about the past sequence. Transformers that use a causal attention, like GPT, actually have a similar notion of state. If we think through the entire model we just built, you’ll note that attention is the only place where the model passes information from position to position. The feedforward blocks of a transformer only modify the hidden representation of each token position in isolation. Inside attention, we incorporate information about past tokens through the `key` and `value` vectors. For a given `query` at a position, we compute attention scores by dotting the `query` with all previous `key` vectors and combining all previous `value` vectors. These `key`
