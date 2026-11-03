# Generation bundle for page_399  (PDF page 399, printed page 375)

Page kind: content
Chapter: 13 Timeseries forecasting
Section: 13.3 Recurrent neural networks
Subsection: 13.3.5 Stacking recurrent layers
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_399.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_399.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p399_e03] (sidebar) slowdown, which more or less forces you stick to what NVIDIA happens to provide. For instance, recurrent dropout isn’t supported by the LSTM and GRU cuDNN kernels, so adding it to your layers forces the runtime to fall back to the regular TensorFlow implementation, which is generally two to five times slower on GPU (even though its computational cost is the same). As a way to speed up your RNN layer when you can’t use cuDNN, you can try unrolling it. Unrolling a `for` loop consists of removing the loop and simply in-lining its content N times. In the case of the `for` loop of an RNN, unrolling can help TensorFlow optimize the underlying computation graph. However, it will also considerably increase the memory consumption of your RNN—as such, it’s only viable for relatively small sequences (around 100 steps or fewer). Also, you can only do this if the number of timesteps in the data is known in advance by the model (that is, if you pass a `shape` without any `None` entries to your initial `Input()`). It works like this:
[p399_e04] (code)
```
inputs = keras.Input(shape=(sequence_length, num_features))
x = layers.LSTM(32, recurrent_dropout=0.2, unroll=True)(inputs)
```
[p399_e05] (heading_subsection) 13.3.5 Stacking recurrent layers
[p399_e06] (body) Because you’re no longer overfitting, but seem to have hit a performance bottleneck, you should consider increasing the capacity and expressive power of the network. Recall the description of the universal machine learning workflow: it’s generally a good idea to increase the capacity of your model until overfitting becomes the primary obstacle (assuming you’re already taking basic steps to mitigate overfitting, such as using dropout). As long as you aren’t overfitting too badly, you’re likely under capacity.
[p399_e07] (body) Increasing network capacity is typically done by increasing the number of units in the layers or adding more layers. Recurrent layer stacking is a classic way to build more powerful recurrent networks: for instance, not too long ago the Google Translate algorithm was powered by a stack of seven large `LSTM` layers—that’s huge. To stack recurrent layers on top of each other in Keras, all intermediate layers should return their full sequence of outputs (a rank-3 tensor) rather than their output at the last timestep. As you’ve already learned, this is done by specifying `return_sequences=True`. In the following example, we’ll try a stack of two dropout-regularized recurrent layers. For a change, we’ll use `GRU` layers instead of `LSTM`. A Gated Recurrent Unit (GRU) is very similar to an LSTM—you can think of it as a slightly simpler, streamlined version of the LSTM architecture. It was introduced in 2014 by Cho et al. just when recurrent networks were starting to gain interest anew in the then-tiny research community.6
[p399_e08] (footnote) 6 See Cho et al., “On the Properties of Neural Machine Translation: Encoder-Decoder Approaches,” 2014, https:// arxiv.org/abs/1409.1259.

## Code annotations printed next to listings (callouts, in order)
- sequence_length cannot be None.
- Passes unroll=True to
- enable unrolling

## Equation candidates (formulas written inline in the text)
- [p399_e04] inputs = keras.Input(shape=(sequence_length, num_features))
x = layers.LSTM(32, recurrent_dropout=0.2, unroll=True)(inputs)
- [p399_e07] As you’ve already learned, this is done by specifying `return_sequences=True`.

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

## Previous page (PDF 398) text, for continuity
Figure 13.11 shows the results. Success! We’re no longer overfitting during the first 20 epochs. We achieve a validation MAE as low as 2.27 degrees (7% improvement over the no-learning baseline) and a test MAE of 2.45 degrees (6.5% improvement over the baseline). Not too bad.

Figure 13.11 Training and validation loss on the Jena temperature forecasting task with a dropout-regularized LSTM

On RNN runtime performance

Recurrent models with very few parameters, like the ones in this chapter, tend to be significantly faster on a multicore CPU than on GPU because they only involve small matrix multiplications, and the chain of multiplications is not well parallelizable due to the presence of a `for` loop. But larger RNNs can greatly benefit from a GPU runtime. When using a Keras `LSTM` or `GRU` layer on GPU with default keyword arguments, your layer will be using a cuDNN kernel, a highly optimized, low-level, NVIDIA-provided implementation of the underlying algorithm (we’ve mentioned those in the previous chapter). As usual, cuDNN kernels are a mixed blessing: they’re fast, but inflexible—if you try doing anything not supported by the default kernel, you will suffer a dramatic

## Next page (PDF 400) text, for continuity
Listing 13.23 Training and evaluating a dropout-regularized, stacked GRU model

Figure 13.12 shows the results. We achieve a test MAE of 2.39 degrees (an 8.8% improvement over the baseline). You can see that the added layer does improve the results a bit, though not dramatically. You may be seeing diminishing returns from increasing network capacity at this point.

Figure 13.12 Training and validation loss on the Jena temperature forecasting task with a stacked GRU network

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.396 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.393 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.397 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.392 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.628 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.404 — 13 Timeseries forecasting › 13.4 Going even further
- PDF p.388 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.402 — 13 Timeseries forecasting › 13.3 Recurrent neural networks

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

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

### PDF p.628 (printed 604) — 20 Conclusions › 20.1 Key concepts in review › Recurrent neural networks
Recurrent neural networks (RNNs) work by processing sequences of inputs one timestep at a time and maintaining a state throughout (a state is typically a vector or set of vectors). They should be used preferentially over 1D ConvNets in the case of sequences where patterns of interest aren’t invariant by temporal translation (for instance, timeseries data where the recent past is more important than the distant past).

Three RNN layers are available in Keras: `SimpleRNN`, `GRU`, and `LSTM`. For most practical purposes, you should use either `GRU` or `LSTM`. `LSTM` is the more powerful of the two but is also more expensive; you can think of `GRU` as a simpler, cheaper alternative to it. To stack multiple RNN layers on top of each other, each layer prior to the last layer in the stack should return the full sequence of its outputs (each input timestep will correspond to an output timestep); if you aren’t stacking any further RNN layers, then it’s common to return only the last output, which contains information about the entire sequence. Following is a single RNN layer for binary classification of vector sequences:

### PDF p.393 (printed 369) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
It’s sometimes useful to stack several recurrent layers one after the other to increase the representational power of a network. In such a setup, you have to get all of the intermediate layers to return the full sequence of outputs.

Listing 13.19 Stacking RNN layers

### PDF p.397 (printed 373) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
curves, it’s evident that the model is quickly overfitting, despite only having very few units: the training and validation losses start to diverge considerably after a few epochs. You’re already familiar with a classic technique for fighting this phenomenon: dropout, which randomly zeros out input units of a layer to break happenstance correlations in the training data that the layer is exposed to. But how to correctly apply dropout in recurrent networks isn’t a trivial question.

It has long been known that applying dropout before a recurrent layer hinders learning rather than helping with regularization. In 2015, Yarin Gal, as part of his PhD thesis on Bayesian deep learning,5 determined the proper way to use dropout with a recurrent network: the same dropout mask (the same pattern of dropped units) should be applied at every timestep, instead of a dropout mask that varies randomly from timestep to timestep. What’s more, to regularize the representations formed by the recurrent gates of layers such as `GRU` and `LSTM`, a temporally constant dropout mask should be applied to the inner recurrent activations of the layer (a recurrent dropout mask). Using the same dropout mask at ev

### PDF p.404 (printed 380) — 13 Timeseries forecasting › 13.4 Going even further › Summary
As you first learned in chapter 6, when approaching a new problem, it’s good to first establish commonsense baselines for your metric of choice. If you don’t have a baseline to beat, you can’t tell whether you’re making real progress.

Try simple models before expensive ones, to justify the additional expense. Sometimes a simple model will turn out to be your best option.

When you have data where ordering matters—in particular, for timeseries data— recurrent networks are a great fit and easily outperform models that first flatten the temporal data. The two essential RNN layers available in Keras are the `LSTM` layer and the `GRU` layer.

To use dropout with recurrent networks, you should use a time-constant dropout mask and recurrent dropout mask. These are built into Keras recurrent layers, so all you have to do is use the `recurrent_dropout` arguments of recurrent layers.

Stacked RNNs provide more representational power than a single RNN layer. They’re also much more expensive and thus not always worth it. Although they offer clear gains on complex problems (such as machine translation), they may not always be relevant to smaller, simpler problems.

### PDF p.392 (printed 368) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3.2 A recurrent layer in Keras
This is especially useful if your model is meant to process sequences of variable length. However, if all of your sequences have the same length, I recommend specifying a complete input shape, since it enables `model.summary()` to display output length information, which is always nice, and it can unlock some performance optimizations (see the sidebar “On RNN runtime performance” in section 13.3.4).

All recurrent layers in Keras (`SimpleRNN`, `LSTM`, and `GRU`) can be run in two different modes: they can return either full sequences of successive outputs for each timestep (a rank-3 tensor of shape `(batch_size, timesteps, output_features)`) or only the last output for each input sequence (a rank-2 tensor of shape `(batch_size, output_features)`). These two modes are controlled by the `return_sequences` constructor argument. Let’s look at an example that uses `SimpleRNN` and returns only the output at the last timestep.

Listing 13.17 An RNN layer that returns only its last output step

### PDF p.393 (printed 369) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
Now, in practice, you’ll rarely work with the `SimpleRNN` layer. It’s generally too simplistic to be of real use. In particular, `SimpleRNN` has a major issue: although it should theoretically be able to retain at time `t` information about inputs seen many timesteps before, in practice, such long-term dependencies prove impossible to learn. This is due to the vanishing gradients problem, an effect that is similar to what is observed with non-recurrent networks (feedforward networks) that are many layers deep: as you keep adding layers to a network, the network eventually becomes untrainable. The theoretical reasons for this effect were studied by Hochreiter, Schmidhuber, and Bengio in the early 1990s.3

Thankfully, `SimpleRNN` isn’t the only recurrent layer available in Keras. There are two others: `LSTM` and `GRU`, which were designed to address these issues. Let’s consider the `LSTM` layer. The underlying Long Short-Term Memory (LSTM) algorithm was developed by Hochreiter and Schmidhuber in 1997;4 it was the culmination of their research on the vanishing gradients problem. This layer is a variant of the `SimpleRNN` layer you already know about; it adds a way to carry information

### PDF p.461 (printed 437) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
Note that this inference setup, while very simple, is inefficient, since we reprocess the entire source sentence and the entire generated target sentence every time we sample a new word. In a practical application, you’d want to be careful not to recompute any state that has not changed. All we really need to predict a new token in the decoder is the current token and the previous RNN state, which we could cache before each loop iteration. There are many ways this toy model could be improved. We could use a deep stack of recurrent layers for both the encoder and the decoder, we could try other RNN layers like `LSTM`, and so on. Beyond such tweaks, however, the RNN approach to sequence-to-sequence learning has a few fundamental limitations:

The source sequence representation has to be held entirely in the encoder state vector, which significantly limits the size and complexity of the sentences you can translate.

RNNs have trouble dealing with very long sequences since they tend to progressively forget about the past—by the time you’ve reached the 100th token in either sequence, little information remains about the start of the sequence.

Recurrent neural networks dominated sequenc
