# Generation bundle for page_382  (PDF page 382, printed page 358)

Page kind: content
Chapter: 13 Timeseries forecasting
Section: 13.2 A temperature forecasting example
Subsection: 13.2.1 Preparing the data
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_382.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_382.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p382_e03] (bullet) `start_index = 0` and `end_index = num_train_samples`—For the training dataset, to only use the first 50% of the data.
[p382_e04] (bullet) `start_index = num_train_samples` and `end_index = num_train_samples + num_` `val_samples`—For the validation dataset, to only use the next 25% of the data.
[p382_e05] (bullet) `start_index = num_train_samples + num_val_samples`—For the test dataset, to use the remaining samples.
[p382_e06] (listing_caption) Listing 13.7 Instantiating datasets for training, validation, and testing
[p382_e07] (code)
```
sampling_rate = 6
sequence_length = 120
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 256
train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples,
)
val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples,
)
test_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples,
)
```
[p382_e08] (body) Each dataset yields a tuple `(samples, targets)`, where `samples` is a batch of 256 samples, each containing 120 consecutive hours of input data, and `targets` is the corresponding array of 256 target temperatures. Note that the samples are randomly shuffled, so two

## Listings
- Listing 13.7: Instantiating datasets for training, validation, and testing (code elements p382_e07)

## Equation candidates (formulas written inline in the text)
- [p382_e03] `start_index = 0` and `end_index = num_train_samples`—For the training dataset, to only use the first 50% of the data.
- [p382_e04] `start_index = num_train_samples` and `end_index = num_train_samples + num_` `val_samples`—For the validation dataset, to only use the next 25% of the data.
- [p382_e05] `start_index = num_train_samples + num_val_samples`—For the test dataset, to use the remaining samples.

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

## Previous page (PDF 381) text, for continuity
Let’s say you’re using `data = [0 1 2 3 4 5 6]` and `sequence_length=3`; then `timeseries_dataset_from_array` will generate the following samples: `[0 1 2]`, `[1 2 ` `3]`, `[2 3 4]`, `[3 4 5]`, `[4 5 6]`.

You can also pass a `target` array to `timeseries_dataset_from_array()`. The first entry of the `targets` array should match the desired target for the first sequence that will be generated from the `data` array. So if you’re doing timeseries forecasting, simply use as `targets` the same array as for `data`, offset by some amount.

For instance, with `data = [0 1 2 3 4 5 6 ...]` and `sequence_length=3`, you could

create a dataset to predict the next step in the series by passing `targets = [3 4 5 6 ` `...]`. Let’s try it:

This bit of code prints the following results:

We’ll use `timeseries_dataset_from_array` to instantiate three datasets: one for training, one for validation, and one for testing.

We’ll use the following parameter values:

`sampling_rate = 6` — Observations will be sampled at one data point per hour: we will only keep one data point out of six.

`sequence_length = 120` — Observations will go back five days (120 hours).

`delay = sampling_rate * (sequence_length` `+` `24 - 1)` — The target for a sequence will be the temperature 24 hours after the end of the sequence.

## Next page (PDF 383) text, for continuity
consecutive sequences in a batch (like `samples[0]` and `samples[1]`) aren’t necessarily temporally close.

Listing 13.8 Inspecting the dataset

13.2.2 A commonsense, non-machine-learning baseline

Before you start using black box, deep learning models to solve the temperature prediction problem, let’s try a simple, commonsense approach. It will serve as a sanity check, and it will establish a baseline that you’ll have to beat to demonstrate the usefulness of more advanced, machine learning models. Such commonsense baselines can be useful when you’re approaching a new problem for which there is no known solution (yet). A classic example is that of unbalanced classification tasks, where some classes are much more common than others. If your dataset contains 90% instances of class A and 10% instances of class B, then a commonsense approach to the classification task is to always predict “A” when presented with a new sample. Such a classifier is 90% accurate overall, and any learning-based approach should therefore beat this 90% score to demonstrate usefulness. Sometimes, such elementary baselines can prove surprisingly hard to beat.

In this case, the temperature timeseries can safely be assumed to be continuous (the temperatures tomorrow are likely to be close to the temperatures today) as well as periodic with a daily period. Thus, a commonsense approach is to always predict that the temperature 24 hours from now will be equal to the temperature right now. Let’s evaluate this approach, using the mean absolute error (MAE) metric, defined as follows:

Here’s the evaluation loop.

Listing 13.9 Computing the commonsense baseline MAE

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.380 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.379 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.497 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.384 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.272 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.399 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.388 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.430 — 14 Text classification › 14.5 Sequence models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.380 (printed 356) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 13.2.1 Preparing the data
Next, let’s create a `Dataset` object that yields batches of data from the past five days along with a target temperature 24 hours in the future. Because the samples in the dataset are highly redundant (sample `N` and sample `N + 1` will have most of their timesteps in common), it would be wasteful to explicitly allocate memory for every sample. Instead, we’ll generate the samples on the fly while only keeping in memory the original `raw_data` and `temperature` arrays, and nothing more.

We could easily write a Python generator to do this, but there’s a built-in dataset utility in Keras that does just that (`timeseries_dataset_from_array()`), so we can save ourselves some work by using it. You can generally use it for any kind of timeseries forecasting task.

### PDF p.379 (printed 355) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 
Listing 13.5 Computing the number of samples for each data split
>>> num_train_samples = int(0.5 * len(raw_data))
>>> num_val_samples = int(0.25 * len(raw_data))
>>> num_test_samples = len(raw_data) - num_train_samples - num_val_samples

### PDF p.380 (printed 356) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 
>>> print("num_train_samples:", num_train_samples)
>>> print("num_val_samples:", num_val_samples)
>>> print("num_test_samples:", num_test_samples)
num_train_samples: 210225
num_val_samples: 105112
num_test_samples: 105114

### PDF p.135 (printed 111) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › Validating your approach
You’ll now train the model for 20 epochs (20 iterations over all samples in the training data), in mini-batches of 512 samples. At the same time, you’ll monitor loss and accuracy on the 10,000 samples that you set apart. You do so by passing the validation data as the `validation_data` argument to `model.fit()`.

Listing 4.7 Training your model

### PDF p.377 (printed 353) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 
Now, convert all 420,551 lines of data into NumPy arrays: one array for the temperature (in degrees Celsius), and another one for the rest of the data—the features we will use to predict future temperatures. Note that we discard the “Date Time” column.

Listing 13.2 Parsing the data

### PDF p.144 (printed 120) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › Validating your approach
Let’s set apart 1,000 samples in the training data to use as a validation set.

Listing 4.17 Setting aside a validation set

### PDF p.379 (printed 355) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 
Figure 13.2 Temperature over the first 10 days of the dataset (ºC)

On this plot, you can see daily periodicity, especially evident for the last four days. Also note that this 10-day period must be coming from a fairly cold winter month.

NOTE Periodicity over multiple timescales is an important and very common property of timeseries data. Whether you’re looking at the weather, mall parking occupancy, traffic to a website, sales of a grocery store, or steps logged in a fitness tracker, you’ll see daily cycles and yearly cycles (human-generated data also tends to feature weekly cycles). When exploring your data, make sure to look for these patterns.

With our dataset, if you were trying to predict average temperature for the next month given a few months of past data, the problem would be easy, due to the reliable year-scale periodicity of the data. But looking at the data over a scale of days, the temperature looks a lot more chaotic. Is this timeseries predictable at a daily scale? Let’s find out.

In all our experiments, we’ll use the first 50% of the data for training, the following 25% for validation, and the last 25% for testing. When working with timeseries data, it’s importa

### PDF p.497 (printed 473) — 16 Text generation › 16.2 Training a mini-GPT › 
As we first did in chapter 8, we will end our `tf.data` pipeline with a call to `prefetch()`. This will make sure we always have some batches loaded onto our GPU and ready for the model.

We have 29,373 batches. You could count this yourself if you would like—the line `ds.reduce(0, lambda c, _: c + 1)` will iterate over the entire dataset and increment a counter. But simply tokenizing a dataset of this size will take a few minutes on a decently fast CPU. At 128 samples per batch and 256 tokens per sample, this is just under a billion tokens of data. Let’s split off 500 batches as a quick validation set, and we are ready to start pretraining:
