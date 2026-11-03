# Generation bundle for page_401  (PDF page 401, printed page 377)

Page kind: content
Chapter: 13 Timeseries forecasting
Section: 13.3 Recurrent neural networks
Subsection: 13.3.6 Using bidirectional RNNs
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_401.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_401.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p401_e03] (heading_subsection) 13.3.6 Using bidirectional RNNs
[p401_e04] (body) The last technique introduced in this section is called bidirectional RNNs. A bidirectional RNN is a common RNN variant that can offer greater performance than a regular RNN on certain tasks. It’s frequently used in natural language processing—you could call it the Swiss Army knife of deep learning for natural language processing.
[p401_e05] (body) RNNs are notably order dependent: they process the timesteps of their input sequences in order, and shuffling or reversing the timesteps can completely change the representations the RNN extracts from the sequence. This is precisely the reason they perform well on problems where order is meaningful, such as the temperature forecasting problem. A bidirectional RNN exploits the order sensitivity of RNNs: it consists of using two regular RNNs, such as the `GRU` and `LSTM` layers you’re already familiar with, each of which processes the input sequence in one direction (chronologically and antichronologically), and then merging their representations. By processing a sequence both ways, a bidirectional RNN can catch patterns that may be overlooked by a unidirectional RNN. Remarkably, the fact that the RNN layers in this section have processed sequences in chronological order (older timesteps first) may have been an arbitrary decision. At least, it’s a decision we made no attempt to question so far. Could the RNNs have performed well enough if they processed input sequences in antichronological order, for instance, newer timesteps first? Let’s try this in practice and see what happens. All you need to do is write a variant of the data generator where the input sequences are reverted along the time dimension (replace the last line with `yield samples[:, ::-1, ` `:], targets`). When training the same LSTM-based model that you used in the first experiment in this section, you would find that such a reversed-order LSTM strongly underperforms even the commonsense baseline. This indicates that, in this case, chronological processing is important to the success of the approach. This makes perfect sense: the underlying `LSTM` layer will typically be better at remembering the recent past than the distant past, and, naturally, the more recent weather data points are more predictive than older data points for the problem (that’s what makes the commonsense baseline fairly strong). Thus the chronological version of the layer is bound to outperform the reversed-order version. However, this isn’t true for many other problems, including natural language: intuitively, the importance of a word in understanding a sentence isn’t strongly dependent on its position in the sentence. On text data, reversed-order processing works just as well as chronological processing—you can read text backwards just fine (try it!). Although word order does matter in understanding language, which order you use isn’t crucial. Importantly, an RNN trained on reversed sequences will learn different representations than one trained on the original sequences, much as you would have different mental models if time flowed backward in the real world—if you lived a life where you died on your first day and were born on your last day. In machine learning, representations that are different yet useful are always worth exploiting, and the more they differ,

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

## Previous page (PDF 400) text, for continuity
Listing 13.23 Training and evaluating a dropout-regularized, stacked GRU model

Figure 13.12 shows the results. We achieve a test MAE of 2.39 degrees (an 8.8% improvement over the baseline). You can see that the added layer does improve the results a bit, though not dramatically. You may be seeing diminishing returns from increasing network capacity at this point.

Figure 13.12 Training and validation loss on the Jena temperature forecasting task with a stacked GRU network

## Next page (PDF 402) text, for continuity
the better: they offer a new angle from which to look at your data, capturing aspects of the data that were missed by other approaches, and thus they can help boost performance on a task. This is the intuition behind ensembling, a concept we’ll explore in chapter 18.

A bidirectional RNN exploits this idea to improve on the performance of chronological-order RNNs. It looks at its input sequence both ways (see figure 13.13), obtaining potentially richer representations and capturing patterns that may have been missed by the chronologi

Figure 13.13 How a bidirectional RNN

cal-order version alone.

layer works

To instantiate a bidirectional RNN in Keras, you use the `Bidirectional` layer, which takes as its first argument a recurrent layer instance. `Bidirectional` creates a second, separate instance of this recurrent layer and uses one instance for processing the input sequences in chronological order and the other instance for processing the input sequences in reversed order. You can try it on our temperature forecasting task.

Listing 13.24 Training and evaluating a bidirectional LSTM

You’ll find that it doesn’t perform as well as the plain `LSTM` layer. It’s easy to understand why: all the predictive capacity must come from the chronological half of the network because the antichronological half is known to be severely underperforming on this task (again, because the recent past matters much more than the distant past in this case). At the same time, the presence of the antichronological half doubles the network’s capacity and causes it to start overfitting much earlier.

However, bidirectional RNNs are a great fit for text data—or any other kind of data where order matters, yet where which order you use doesn’t matter. In fact, for a while in 2016, bidirectional L …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.396 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.392 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.388 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.390 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.389 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.393 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.399 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.404 — 13 Timeseries forecasting › 13.4 Going even further

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.628 (printed 604) — 20 Conclusions › 20.1 Key concepts in review › Recurrent neural networks
Recurrent neural networks (RNNs) work by processing sequences of inputs one timestep at a time and maintaining a state throughout (a state is typically a vector or set of vectors). They should be used preferentially over 1D ConvNets in the case of sequences where patterns of interest aren’t invariant by temporal translation (for instance, timeseries data where the recent past is more important than the distant past).

Three RNN layers are available in Keras: `SimpleRNN`, `GRU`, and `LSTM`. For most practical purposes, you should use either `GRU` or `LSTM`. `LSTM` is the more powerful of the two but is also more expensive; you can think of `GRU` as a simpler, cheaper alternative to it. To stack multiple RNN layers on top of each other, each layer prior to the last layer in the stack should return the full sequence of its outputs (each input timestep will correspond to an output timestep); if you aren’t stacking any further RNN layers, then it’s common to return only the last output, which contains information about the entire sequence. Following is a single RNN layer for binary classification of vector sequences:

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

### PDF p.461 (printed 437) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
Note that this inference setup, while very simple, is inefficient, since we reprocess the entire source sentence and the entire generated target sentence every time we sample a new word. In a practical application, you’d want to be careful not to recompute any state that has not changed. All we really need to predict a new token in the decoder is the current token and the previous RNN state, which we could cache before each loop iteration. There are many ways this toy model could be improved. We could use a deep stack of recurrent layers for both the encoder and the decoder, we could try other RNN layers like `LSTM`, and so on. Beyond such tweaks, however, the RNN approach to sequence-to-sequence learning has a few fundamental limitations:

The source sequence representation has to be held entirely in the encoder state vector, which significantly limits the size and complexity of the sentences you can translate.

RNNs have trouble dealing with very long sequences since they tend to progressively forget about the past—by the time you’ve reached the 100th token in either sequence, little information remains about the start of the sequence.

Recurrent neural networks dominated sequenc

### PDF p.388 (printed 364) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3 Recurrent neural networks
Neither the fully connected approach nor the convolutional approach did well, but that doesn’t mean machine learning isn’t applicable to this problem. The densely connected approach first flattened the timeseries, which removed the notion of time from the input data. The convolutional approach treated every segment of the data in the same way, even applying pooling, which destroyed order information. Let’s instead look at the data as what it is: a sequence, where causality and order matter.

There’s a family of neural network architectures that were designed specifically for this use case: recurrent neural networks. Among them, the Long Short-Term Memory (LSTM) layer in particular has long been very popular. We’ll see in a minute how these models work—but let’s start by giving the LSTM layer a try.

Listing 13.12 A simple LSTM-based model

### PDF p.390 (printed 366) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
The state of the RNN is reset between processing two different, independent sequences (such as two samples in a batch), so you still consider one sequence to be a single data point: a single input to the network. What changes is that this data point is no longer processed in a single step; rather, the network internally loops over sequence elements. To make these notions of loop and state clear, let’s implement the forward pass of a toy RNN. This RNN takes as input

Figure 13.6 A recurrent

a sequence of vectors, which we’ll encode as a rank-2 ten

network: a network with a loop

sor of size `(timesteps, input_features)`. It loops over timesteps, and at each timestep, it considers its current state at `t` and the input at `t` (of shape `(input_features,)`), and combines them to obtain the output at `t`. We’ll then set the state for the next step to be this previous output. For the first timestep, the previous output isn’t defined; hence, there is no current state. So we’ll initialize the state as an all-zero vector called the initial state of the network.

In pseudocode, this is the RNN.

Listing 13.13 Pseudocode RNN

### PDF p.392 (printed 368) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3.2 A recurrent layer in Keras
The process you just naively implemented in NumPy corresponds to an actual Keras layer—the `SimpleRNN` layer.

There is one minor difference: `SimpleRNN` processes batches of sequences, like all other Keras layers, not a single sequence as in the NumPy example. This means it takes inputs of shape `(batch_size,` `timesteps,` `input_features)` rather than `(timesteps, ` `input_features)`. When specifying the `shape` argument of your initial `Input()`, note that you can set the `timesteps` entry to `None`, which enables your network to process sequences of arbitrary length.

Listing 13.16 An RNN layer that can process sequences of any length

### PDF p.389 (printed 365) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3.1 Understanding recurrent neural networks
A major characteristic of all neural networks you’ve seen so far, such as densely connected networks and ConvNets, is that they have no memory. Each input shown to them is processed independently, with no state kept between inputs. With such networks, to process a sequence or a temporal series of data points, you have to show the entire sequence to the network at once: turn it into a single data point. For instance, this is what we did in the densely connected network example: we flattened our five days of data into a single large vector and processed it in one go. Such networks are called feedforward networks.

In contrast, as you’re reading the present sentence, you’re processing it word by word—or rather, eye saccade by eye saccade—while keeping memories of what came before; this gives you a fluid representation of the meaning conveyed by this sentence. Biological intelligence processes information incrementally while maintaining an internal model of what it’s processing, built from past information and constantly updated as new information comes in. A recurrent neural network (RNN) adopts the same principle, albeit in an extremely simplified version: it processes sequences by i

### PDF p.474 (printed 450) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
After training, we get to about 58% accuracy: on average, the model correctly predicts the next word in the Spanish sentence 58% of the time. Something is off here. Training is worse than the RNN model by 7 percentage points. Either this Transformer architecture is not what it was hyped up to be, or we missed something in our implementation. Can you spot what it is?

This section is ostensibly about sequence models. In the previous chapter, we saw how vital word order could be to meaning. And yet, the Transformer we just built wasn’t a sequence model at all. Did you notice? It’s composed of dense layers that process sequence tokens independently of each other and an attention layer that looks at the tokens as a set. You could change the order of the tokens in a sequence, and you’d get identical pairwise attention scores and the same context-aware representations. If you were to rearrange every word in every English source sentence completely, the model wouldn’t notice, and you’d still get the same accuracy. Attention is a set-processing mechanism, focused on the relationships between pairs of sequence elements—it’s blind to whether these elements occur at the beginning, at the end,
