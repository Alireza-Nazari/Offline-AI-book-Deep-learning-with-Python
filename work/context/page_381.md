# Generation bundle for page_381  (PDF page 381, printed page 357)

Page kind: content
Chapter: 13 Timeseries forecasting
Section: 13.2 A temperature forecasting example
Subsection: 13.2.1 Preparing the data
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_381.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_381.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p381_e03] (body) Let’s say you’re using `data = [0 1 2 3 4 5 6]` and `sequence_length=3`; then `timeseries_dataset_from_array` will generate the following samples: `[0 1 2]`, `[1 2 ` `3]`, `[2 3 4]`, `[3 4 5]`, `[4 5 6]`.
[p381_e04] (sidebar) You can also pass a `target` array to `timeseries_dataset_from_array()`. The first entry of the `targets` array should match the desired target for the first sequence that will be generated from the `data` array. So if you’re doing timeseries forecasting, simply use as `targets` the same array as for `data`, offset by some amount.
[p381_e05] (body) For instance, with `data = [0 1 2 3 4 5 6 ...]` and `sequence_length=3`, you could
[p381_e06] (sidebar) create a dataset to predict the next step in the series by passing `targets = [3 4 5 6 ` `...]`. Let’s try it:
[p381_e07] (code)
```
import numpy as np
import keras
int_sequence = np.arange(10)
dummy_dataset = keras.utils.timeseries_dataset_from_array(
    data=int_sequence[:-3],
    targets=int_sequence[3:],
    sequence_length=3,
    batch_size=2,
)
for inputs, targets in dummy_dataset:
    for i in range(inputs.shape[0]):
        print([int(x) for x in inputs[i]], int(targets[i]))
```
[p381_e08] (sidebar) This bit of code prints the following results:
[p381_e09] (code)
```
[0, 1, 2] 3
[1, 2, 3] 4
[2, 3, 4] 5
[3, 4, 5] 6
[4, 5, 6] 7
```
[p381_e10] (body) We’ll use `timeseries_dataset_from_array` to instantiate three datasets: one for training, one for validation, and one for testing.
[p381_e11] (body) We’ll use the following parameter values:
[p381_e12] (bullet) `sampling_rate = 6` — Observations will be sampled at one data point per hour: we will only keep one data point out of six.
[p381_e13] (bullet) `sequence_length = 120` — Observations will go back five days (120 hours).
[p381_e14] (bullet) `delay = sampling_rate * (sequence_length` `+` `24 - 1)` — The target for a sequence will be the temperature 24 hours after the end of the sequence.

## Code annotations printed next to listings (callouts, in order)
- The sequences we generate will
- be sampled from [0 1 2 3 4 5 6].
- Generate an array of sorted
- integers from 0 to 9.
- The sequences will
- be 3 steps long.
- The sequences will be
- batched in batches of size 2.
- The target for the sequence that starts
- at data[N] will be data[N + 3].

## Equation candidates (formulas written inline in the text)
- [p381_e03] Let’s say you’re using `data = [0 1 2 3 4 5 6]` and `sequence_length=3`; then `timeseries_dataset_from_array` will generate the following samples: `[0 1 2]`, `[1 2 ` `3]`, `[2 3 4]`, `[3 4 5]`, `[4 5 6]`.
- [p381_e05] For instance, with `data = [0 1 2 3 4 5 6 ...]` and `sequence_length=3`, you could
- [p381_e12] `sampling_rate = 6` — Observations will be sampled at one data point per hour: we will only keep one data point out of six.
- [p381_e13] `sequence_length = 120` — Observations will go back five days (120 hours).
- [p381_e14] `delay = sampling_rate * (sequence_length` `+` `24 - 1)` — The target for a sequence will be the temperature 24 hours after the end of the sequence.

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

## Previous page (PDF 380) text, for continuity
13.2.1 Preparing the data

The exact formulation of the problem will be as follows: given data covering the previous five days and sampled once per hour, can we predict the temperature in 24 hours?

First, let’s preprocess the data to a format a neural network can ingest. This is easy: the data is already numerical, so you don’t need to do any vectorization. But each timeseries in the data is on a different scale (for example, atmospheric pressure, measured in mbar, is around 1,000, while H2OC, measured in millimoles per mole, is around 3). We’ll normalize each timeseries independently so that they all take small values on a similar scale. We’re going to use the first 210,225 timesteps as training data, so we’ll compute the mean and standard deviation only on this fraction of the data.

Listing 13.6 Normalizing the data

Next, let’s create a `Dataset` object that yields batches of data from the past five days along with a target temperature 24 hours in the future. Because the samples in the dataset are highly redundant (sample `N` and sample `N + 1` will have most of their timesteps in common), it would be wasteful to explicitly allocate memory for every sample. Instead, we’ll generate the samples on the fly while only keeping in memory the original `raw_data` and `temperature` arrays, and nothing more.

We could easily write a Python generator to do this, but there’s a built-in dataset utility in Keras that does just that (`timeseries_dataset_from_array()`), so we can save ourselves some work by using it. You can generally use it for any kind of timeseries forecasting task.

Understanding timeseries_dataset_from_array()

To understand what `timeseries_dataset_from_array()` does, let’s take a look at a simple example. The general idea is that you provide an array of tim …

## Next page (PDF 382) text, for continuity
`start_index = 0` and `end_index = num_train_samples`—For the training dataset, to only use the first 50% of the data.

`start_index = num_train_samples` and `end_index = num_train_samples + num_` `val_samples`—For the validation dataset, to only use the next 25% of the data.

`start_index = num_train_samples + num_val_samples`—For the test dataset, to use the remaining samples.

Listing 13.7 Instantiating datasets for training, validation, and testing

Each dataset yields a tuple `(samples, targets)`, where `samples` is a batch of 256 samples, each containing 120 consecutive hours of input data, and `targets` is the corresponding array of 256 target temperatures. Note that the samples are randomly shuffled, so two

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.377 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.379 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.383 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.152 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.378 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.94 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.384 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.376 — 13 Timeseries forecasting › 13.2 A temperature forecasting example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.376 (printed 352) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 13.2 A temperature forecasting example
Throughout this chapter, all of our code examples will target a single problem: predicting the temperature 24 hours in the future, given a timeseries of hourly measurements of quantities such as atmospheric pressure and humidity, recorded over the recent past by a set of sensors on the roof of a building. As you will see, it’s a fairly challenging problem!

We’ll use this temperature forecasting task to highlight what makes timeseries data fundamentally different from the kinds of datasets you’ve encountered so far, to show that densely connected networks and convolutional networks aren’t well equipped to deal with it, and to demonstrate a new kind of machine learning technique that really shines on this type of problem: recurrent neural networks (RNNs). We’ll work with a weather timeseries dataset recorded at the weather station at the Max Planck Institute for Biogeochemistry in Jena, Germany.1 In this dataset, 14 different quantities (such as temperature, atmospheric pressure, humidity, wind direction, and so on) were recorded every 10 minutes, over several years. The original data goes back to 2003, but the subset of the data we’ll download is limited to 2009–2016. Let’s start b

### PDF p.377 (printed 353) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 
Now, convert all 420,551 lines of data into NumPy arrays: one array for the temperature (in degrees Celsius), and another one for the rest of the data—the features we will use to predict future temperatures. Note that we discard the “Date Time” column.

Listing 13.2 Parsing the data

### PDF p.379 (printed 355) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 
Figure 13.2 Temperature over the first 10 days of the dataset (ºC)

On this plot, you can see daily periodicity, especially evident for the last four days. Also note that this 10-day period must be coming from a fairly cold winter month.

NOTE Periodicity over multiple timescales is an important and very common property of timeseries data. Whether you’re looking at the weather, mall parking occupancy, traffic to a website, sales of a grocery store, or steps logged in a fitness tracker, you’ll see daily cycles and yearly cycles (human-generated data also tends to feature weekly cycles). When exploring your data, make sure to look for these patterns.

With our dataset, if you were trying to predict average temperature for the next month given a few months of past data, the problem would be easy, due to the reliable year-scale periodicity of the data. But looking at the data over a scale of days, the temperature looks a lot more chaotic. Is this timeseries predictable at a daily scale? Let’s find out.

In all our experiments, we’ll use the first 50% of the data for training, the following 25% for validation, and the last 25% for testing. When working with timeseries data, it’s importa

### PDF p.378 (printed 354) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 
Figure 13.1 Temperature over the full temporal range of the dataset (ºC)

Figure 10.2 shows a more narrow plot of the first 10 days of temperature data. Because the data is recorded every 10 minutes, you get 24 × 6 = 144 data points per day.

Listing 13.4 Plotting the first 10 days of the temperature timeseries

### PDF p.152 (printed 128) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Preparing the data
Note that the quantities used for normalizing the test data are computed using the training data. You should never use in your workflow any quantity computed on the test data, even for something as simple as data normalization.

In addition, we should also scale the targets. Our normalized inputs have their value in a small range close to 0, and our model’s weights are initialized with small random values. This means that our model’s prediction will also be small values when we start training. If the targets are in the range 60,000–500,000, the model is going to need very large weight values to output those. With a small learning rate, it would take a very long time to get there. The simplest fix is to divide all target values by 100,000, so that the smallest target becomes 0.6, and the largest becomes 5. We can then convert the model’s predictions back to dollar values by multiplying them by 100,000 accordingly.

Listing 4.26 Scaling the targets

### PDF p.386 (printed 362) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 13.2.4 Let’s try a 1D convolutional model
Speaking of using the right architecture priors: since our input sequences feature daily cycles, perhaps a convolutional model could work? A temporal ConvNet could reuse the same representations across different days, much like a spatial ConvNet can reuse the same representations across different locations in an image.

You already know about the `Conv2D` and `SeparableConv2D` layers, which see their inputs through small windows that swipe across 2D grids. There are also 1D and even 3D versions of these layers: `Conv1D`, `SeparableConv1D`, and `Conv3D`.2 The `Conv1D` layer relies on 1D windows that slide across input sequences, and the `Conv3D` layer relies on cubic windows that slide across input volumes. You can thus build 1D ConvNets, strictly analogous to 2D ConvNets. They’re a great fit for any sequence data that follows the translation invariance assumption (meaning that if you slide a window over the sequence, the content of the window should follow the same properties independently of the location of the window). Let’s try one on our temperature forecasting problem. We’ll pick an initial window length of 24, so that we look at 24 hours of data at a time (one cycle). As we d

### PDF p.390 (printed 366) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
The state of the RNN is reset between processing two different, independent sequences (such as two samples in a batch), so you still consider one sequence to be a single data point: a single input to the network. What changes is that this data point is no longer processed in a single step; rather, the network internally loops over sequence elements. To make these notions of loop and state clear, let’s implement the forward pass of a toy RNN. This RNN takes as input

Figure 13.6 A recurrent

a sequence of vectors, which we’ll encode as a rank-2 ten

network: a network with a loop

sor of size `(timesteps, input_features)`. It loops over timesteps, and at each timestep, it considers its current state at `t` and the input at `t` (of shape `(input_features,)`), and combines them to obtain the output at `t`. We’ll then set the state for the next step to be this previous output. For the first timestep, the previous output isn’t defined; hence, there is no current state. So we’ll initialize the state as an all-zero vector called the initial state of the network.

In pseudocode, this is the RNN.

Listing 13.13 Pseudocode RNN

### PDF p.453 (printed 429) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 
The general template behind sequence-to-sequence models is described in figure 15.1. During training the following happen:

An encoder model turns the source sequence into an intermediate representation.

A decoder is trained using the language modeling setup we saw previously. It will recursively predict the next token in the target sequence by looking at all previous target tokens and our encoder’s representation of the source sequence.

During inference, we don’t have access to the target sequence—we’re trying to predict it from scratch. We will generate it one token at a time, just as we did with our Shakespeare generator:

We obtain the encoded source sequence from the encoder.

The decoder starts by looking at the encoded source sequence as well as an initial “seed” token (such as the string `"[start]"`) and uses them to predict the first real token in the sequence.

The predicted sequence so far is fed back into the decoder, in a loop, until it generates a stop token (such as the string `"[end]"`).

Figure 15.1 Sequence-to-sequence learning: the source sequence is processed by the encoder and is then sent to the decoder. The decoder looks at the target sequence so far and pr
