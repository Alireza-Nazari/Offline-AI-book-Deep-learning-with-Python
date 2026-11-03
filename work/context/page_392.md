# Generation bundle for page_392  (PDF page 392, printed page 368)

Page kind: content
Chapter: 13 Timeseries forecasting
Section: 13.3 Recurrent neural networks
Subsection: 13.3.2 A recurrent layer in Keras
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_392.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_392.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p392_e03] (body) NOTE In this example, the final output is a rank-2 tensor of shape `(timesteps, ` `output_features)`, where each timestep is the output of the loop at time `t`. Each timestep `t` in the output tensor contains information about timesteps `0` to `t` in the input sequence—about the entire past. For this reason, in many cases, you don’t need this full sequence of outputs; you just need the last output (`output_t` at the end of the loop), because it already contains information about the entire sequence.
[p392_e04] (heading_subsection) 13.3.2 A recurrent layer in Keras
[p392_e05] (body) The process you just naively implemented in NumPy corresponds to an actual Keras layer—the `SimpleRNN` layer.
[p392_e06] (body) There is one minor difference: `SimpleRNN` processes batches of sequences, like all other Keras layers, not a single sequence as in the NumPy example. This means it takes inputs of shape `(batch_size,` `timesteps,` `input_features)` rather than `(timesteps, ` `input_features)`. When specifying the `shape` argument of your initial `Input()`, note that you can set the `timesteps` entry to `None`, which enables your network to process sequences of arbitrary length.
[p392_e07] (listing_caption) Listing 13.16 An RNN layer that can process sequences of any length
[p392_e08] (code)
```
num_features = 14
inputs = keras.Input(shape=(None, num_features))
outputs = layers.SimpleRNN(16)(inputs)
```
[p392_e09] (body) This is especially useful if your model is meant to process sequences of variable length. However, if all of your sequences have the same length, I recommend specifying a complete input shape, since it enables `model.summary()` to display output length information, which is always nice, and it can unlock some performance optimizations (see the sidebar “On RNN runtime performance” in section 13.3.4).
[p392_e10] (body) All recurrent layers in Keras (`SimpleRNN`, `LSTM`, and `GRU`) can be run in two different modes: they can return either full sequences of successive outputs for each timestep (a rank-3 tensor of shape `(batch_size, timesteps, output_features)`) or only the last output for each input sequence (a rank-2 tensor of shape `(batch_size, output_features)`). These two modes are controlled by the `return_sequences` constructor argument. Let’s look at an example that uses `SimpleRNN` and returns only the output at the last timestep.
[p392_e11] (listing_caption) Listing 13.17 An RNN layer that returns only its last output step
[p392_e12] (code)
```
>>> num_features = 14
>>> steps = 120
>>> inputs = keras.Input(shape=(steps, num_features))
>>> outputs = layers.SimpleRNN(16, return_sequences=False)(inputs)
>>> print(outputs.shape)
(None, 16)
```

## Code annotations printed next to listings (callouts, in order)
- Note that return_sequences=False is the default.

## Listings
- Listing 13.16: An RNN layer that can process sequences of any length (code elements p392_e08)
- Listing 13.17: An RNN layer that returns only its last output step (code elements p392_e12)

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

## Previous page (PDF 391) text, for continuity
Easy enough: in summary, an RNN is a `for` loop that reuses quantities computed during the previous iteration of the loop, nothing more. Of course, there are many different RNNs fitting this definition that you could build—this example is one of the simplest RNN formulations. RNNs are characterized by their step function, such as the following function in this case (see figure 13.7):

Figure 13.7 A simple RNN, unrolled over time

## Next page (PDF 393) text, for continuity
The following example returns the full output sequence.

Listing 13.18 An RNN layer that returns its full output sequence

It’s sometimes useful to stack several recurrent layers one after the other to increase the representational power of a network. In such a setup, you have to get all of the intermediate layers to return the full sequence of outputs.

Listing 13.19 Stacking RNN layers

Now, in practice, you’ll rarely work with the `SimpleRNN` layer. It’s generally too simplistic to be of real use. In particular, `SimpleRNN` has a major issue: although it should theoretically be able to retain at time `t` information about inputs seen many timesteps before, in practice, such long-term dependencies prove impossible to learn. This is due to the vanishing gradients problem, an effect that is similar to what is observed with non-recurrent networks (feedforward networks) that are many layers deep: as you keep adding layers to a network, the network eventually becomes untrainable. The theoretical reasons for this effect were studied by Hochreiter, Schmidhuber, and Bengio in the early 1990s.3

Thankfully, `SimpleRNN` isn’t the only recurrent layer available in Keras. There are two others: `LSTM` and `GRU`, which were designed to address these issues. Let’s consider the `LSTM` layer. The underlying Long Short-Term Memory (LSTM) algorithm was developed by Hochreiter and Schmidhuber in 1997;4 it was the culmination of their research on the vanishing gradients problem. This layer is a variant of the `SimpleRNN` layer you already know about; it adds a way to carry information across many timesteps. Imagine a conveyor belt running parallel to the sequence you’re processing. Information from the sequence can jump onto the conveyor belt at any point, be transported to a later timest …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.628 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.390 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.399 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.402 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.457 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.396 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.389 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.401 — 13 Timeseries forecasting › 13.3 Recurrent neural networks

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.628 (printed 604) — 20 Conclusions › 20.1 Key concepts in review › Recurrent neural networks
Recurrent neural networks (RNNs) work by processing sequences of inputs one timestep at a time and maintaining a state throughout (a state is typically a vector or set of vectors). They should be used preferentially over 1D ConvNets in the case of sequences where patterns of interest aren’t invariant by temporal translation (for instance, timeseries data where the recent past is more important than the distant past).

Three RNN layers are available in Keras: `SimpleRNN`, `GRU`, and `LSTM`. For most practical purposes, you should use either `GRU` or `LSTM`. `LSTM` is the more powerful of the two but is also more expensive; you can think of `GRU` as a simpler, cheaper alternative to it. To stack multiple RNN layers on top of each other, each layer prior to the last layer in the stack should return the full sequence of its outputs (each input timestep will correspond to an output timestep); if you aren’t stacking any further RNN layers, then it’s common to return only the last output, which contains information about the entire sequence. Following is a single RNN layer for binary classification of vector sequences:

### PDF p.390 (printed 366) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
The state of the RNN is reset between processing two different, independent sequences (such as two samples in a batch), so you still consider one sequence to be a single data point: a single input to the network. What changes is that this data point is no longer processed in a single step; rather, the network internally loops over sequence elements. To make these notions of loop and state clear, let’s implement the forward pass of a toy RNN. This RNN takes as input

Figure 13.6 A recurrent

a sequence of vectors, which we’ll encode as a rank-2 ten

network: a network with a loop

sor of size `(timesteps, input_features)`. It loops over timesteps, and at each timestep, it considers its current state at `t` and the input at `t` (of shape `(input_features,)`), and combines them to obtain the output at `t`. We’ll then set the state for the next step to be this previous output. For the first timestep, the previous output isn’t defined; hence, there is no current state. So we’ll initialize the state as an all-zero vector called the initial state of the network.

In pseudocode, this is the RNN.

Listing 13.13 Pseudocode RNN

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

### PDF p.399 (printed 375) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3.5 Stacking recurrent layers
Because you’re no longer overfitting, but seem to have hit a performance bottleneck, you should consider increasing the capacity and expressive power of the network. Recall the description of the universal machine learning workflow: it’s generally a good idea to increase the capacity of your model until overfitting becomes the primary obstacle (assuming you’re already taking basic steps to mitigate overfitting, such as using dropout). As long as you aren’t overfitting too badly, you’re likely under capacity.

Increasing network capacity is typically done by increasing the number of units in the layers or adding more layers. Recurrent layer stacking is a classic way to build more powerful recurrent networks: for instance, not too long ago the Google Translate algorithm was powered by a stack of seven large `LSTM` layers—that’s huge. To stack recurrent layers on top of each other in Keras, all intermediate layers should return their full sequence of outputs (a rank-3 tensor) rather than their output at the last timestep. As you’ve already learned, this is done by specifying `return_sequences=True`. In the following example, we’ll try a stack of two dropout-regularized recurrent layer

### PDF p.389 (printed 365) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3.1 Understanding recurrent neural networks
A major characteristic of all neural networks you’ve seen so far, such as densely connected networks and ConvNets, is that they have no memory. Each input shown to them is processed independently, with no state kept between inputs. With such networks, to process a sequence or a temporal series of data points, you have to show the entire sequence to the network at once: turn it into a single data point. For instance, this is what we did in the densely connected network example: we flattened our five days of data into a single large vector and processed it in one go. Such networks are called feedforward networks.

In contrast, as you’re reading the present sentence, you’re processing it word by word—or rather, eye saccade by eye saccade—while keeping memories of what came before; this gives you a fluid representation of the meaning conveyed by this sentence. Biological intelligence processes information incrementally while maintaining an internal model of what it’s processing, built from past information and constantly updated as new information comes in. A recurrent neural network (RNN) adopts the same principle, albeit in an extremely simplified version: it processes sequences by i

### PDF p.397 (printed 373) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
curves, it’s evident that the model is quickly overfitting, despite only having very few units: the training and validation losses start to diverge considerably after a few epochs. You’re already familiar with a classic technique for fighting this phenomenon: dropout, which randomly zeros out input units of a layer to break happenstance correlations in the training data that the layer is exposed to. But how to correctly apply dropout in recurrent networks isn’t a trivial question.

It has long been known that applying dropout before a recurrent layer hinders learning rather than helping with regularization. In 2015, Yarin Gal, as part of his PhD thesis on Bayesian deep learning,5 determined the proper way to use dropout with a recurrent network: the same dropout mask (the same pattern of dropped units) should be applied at every timestep, instead of a dropout mask that varies randomly from timestep to timestep. What’s more, to regularize the representations formed by the recurrent gates of layers such as `GRU` and `LSTM`, a temporally constant dropout mask should be applied to the inner recurrent activations of the layer (a recurrent dropout mask). Using the same dropout mask at ev

### PDF p.402 (printed 378) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
the better: they offer a new angle from which to look at your data, capturing aspects of the data that were missed by other approaches, and thus they can help boost performance on a task. This is the intuition behind ensembling, a concept we’ll explore in chapter 18.

A bidirectional RNN exploits this idea to improve on the performance of chronological-order RNNs. It looks at its input sequence both ways (see figure 13.13), obtaining potentially richer representations and capturing patterns that may have been missed by the chronologi

Figure 13.13 How a bidirectional RNN

cal-order version alone.

layer works

To instantiate a bidirectional RNN in Keras, you use the `Bidirectional` layer, which takes as its first argument a recurrent layer instance. `Bidirectional` creates a second, separate instance of this recurrent layer and uses one instance for processing the input sequences in chronological order and the other instance for processing the input sequences in reversed order. You can try it on our temperature forecasting task.

Listing 13.24 Training and evaluating a bidirectional LSTM

### PDF p.117 (printed 93) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
Different types of layers are appropriate for different tensor formats and different types of data processing. For instance, simple vector data, stored in 2D tensors of shape `(samples,` `features)`, is often processed by densely connected layers, also called fully connected or dense layers (the `Dense` class in Keras). Sequence data, stored in 3D tensors of shape `(samples,` `timesteps,` `features)`, is typically processed by recurrent layers, such as an LSTM layer, or 1D convolution layers (`Conv1D`). Image data, stored in rank-4 tensors, is usually processed by 2D convolution layers (`Conv2D`). You can think of layers as the LEGO bricks of deep learning, a metaphor that is made explicit by Keras. Building deep learning models in Keras is done by clipping together compatible layers to form useful data transformation pipelines.
