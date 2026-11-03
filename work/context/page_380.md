# Generation bundle for page_380  (PDF page 380, printed page 356)

Page kind: content
Chapter: 13 Timeseries forecasting
Section: 13.2 A temperature forecasting example
Subsection: 13.2.1 Preparing the data
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_380.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_380.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p380_e03] (code)
```
>>> print("num_train_samples:", num_train_samples)
>>> print("num_val_samples:", num_val_samples)
>>> print("num_test_samples:", num_test_samples)
num_train_samples: 210225
num_val_samples: 105112
num_test_samples: 105114
```
[p380_e04] (heading_subsection) 13.2.1 Preparing the data
[p380_e05] (body) The exact formulation of the problem will be as follows: given data covering the previous five days and sampled once per hour, can we predict the temperature in 24 hours?
[p380_e06] (body) First, let’s preprocess the data to a format a neural network can ingest. This is easy: the data is already numerical, so you don’t need to do any vectorization. But each timeseries in the data is on a different scale (for example, atmospheric pressure, measured in mbar, is around 1,000, while H2OC, measured in millimoles per mole, is around 3). We’ll normalize each timeseries independently so that they all take small values on a similar scale. We’re going to use the first 210,225 timesteps as training data, so we’ll compute the mean and standard deviation only on this fraction of the data.
[p380_e07] (listing_caption) Listing 13.6 Normalizing the data
[p380_e08] (code)
```
mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std
```
[p380_e09] (body) Next, let’s create a `Dataset` object that yields batches of data from the past five days along with a target temperature 24 hours in the future. Because the samples in the dataset are highly redundant (sample `N` and sample `N + 1` will have most of their timesteps in common), it would be wasteful to explicitly allocate memory for every sample. Instead, we’ll generate the samples on the fly while only keeping in memory the original `raw_data` and `temperature` arrays, and nothing more.
[p380_e10] (body) We could easily write a Python generator to do this, but there’s a built-in dataset utility in Keras that does just that (`timeseries_dataset_from_array()`), so we can save ourselves some work by using it. You can generally use it for any kind of timeseries forecasting task.
[p380_e11] (heading_subsection) Understanding timeseries_dataset_from_array()
[p380_e12] (sidebar) To understand what `timeseries_dataset_from_array()` does, let’s take a look at a simple example. The general idea is that you provide an array of timeseries data (the `data` argument), and `timeseries_dataset_from_array` gives you windows extracted from the original timeseries (we’ll call them “sequences”).

## Listings
- Listing 13.6: Normalizing the data (code elements p380_e08)

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

## Previous page (PDF 379) text, for continuity
Figure 13.2 Temperature over the first 10 days of the dataset (ºC)

On this plot, you can see daily periodicity, especially evident for the last four days. Also note that this 10-day period must be coming from a fairly cold winter month.

NOTE Periodicity over multiple timescales is an important and very common property of timeseries data. Whether you’re looking at the weather, mall parking occupancy, traffic to a website, sales of a grocery store, or steps logged in a fitness tracker, you’ll see daily cycles and yearly cycles (human-generated data also tends to feature weekly cycles). When exploring your data, make sure to look for these patterns.

With our dataset, if you were trying to predict average temperature for the next month given a few months of past data, the problem would be easy, due to the reliable year-scale periodicity of the data. But looking at the data over a scale of days, the temperature looks a lot more chaotic. Is this timeseries predictable at a daily scale? Let’s find out.

In all our experiments, we’ll use the first 50% of the data for training, the following 25% for validation, and the last 25% for testing. When working with timeseries data, it’s important to use validation and test data that is more recent than the training data because you’re trying to predict the future given the past, not the reverse, and your validation/test splits should reflect this temporal ordering. Some problems happen to be considerably simpler if you reverse the time axis!

Listing 13.5 Computing the number of samples for each data split

## Next page (PDF 381) text, for continuity
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

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.382 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.378 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.376 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.152 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.383 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.377 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.388 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.375 — 13 Timeseries forecasting › 13.1 Different kinds of timeseries tasks

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.376 (printed 352) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 13.2 A temperature forecasting example
Throughout this chapter, all of our code examples will target a single problem: predicting the temperature 24 hours in the future, given a timeseries of hourly measurements of quantities such as atmospheric pressure and humidity, recorded over the recent past by a set of sensors on the roof of a building. As you will see, it’s a fairly challenging problem!

We’ll use this temperature forecasting task to highlight what makes timeseries data fundamentally different from the kinds of datasets you’ve encountered so far, to show that densely connected networks and convolutional networks aren’t well equipped to deal with it, and to demonstrate a new kind of machine learning technique that really shines on this type of problem: recurrent neural networks (RNNs). We’ll work with a weather timeseries dataset recorded at the weather station at the Max Planck Institute for Biogeochemistry in Jena, Germany.1 In this dataset, 14 different quantities (such as temperature, atmospheric pressure, humidity, wind direction, and so on) were recorded every 10 minutes, over several years. The original data goes back to 2003, but the subset of the data we’ll download is limited to 2009–2016. Let’s start b

### PDF p.378 (printed 354) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 
Figure 13.1 Temperature over the full temporal range of the dataset (ºC)

Figure 10.2 shows a more narrow plot of the first 10 days of temperature data. Because the data is recorded every 10 minutes, you get 24 × 6 = 144 data points per day.

Listing 13.4 Plotting the first 10 days of the temperature timeseries

### PDF p.377 (printed 353) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 
Now, convert all 420,551 lines of data into NumPy arrays: one array for the temperature (in degrees Celsius), and another one for the rest of the data—the features we will use to predict future temperatures. Note that we discard the “Date Time” column.

Listing 13.2 Parsing the data

### PDF p.383 (printed 359) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 13.2.2 A commonsense, non-machine-learning baseline
Before you start using black box, deep learning models to solve the temperature prediction problem, let’s try a simple, commonsense approach. It will serve as a sanity check, and it will establish a baseline that you’ll have to beat to demonstrate the usefulness of more advanced, machine learning models. Such commonsense baselines can be useful when you’re approaching a new problem for which there is no known solution (yet). A classic example is that of unbalanced classification tasks, where some classes are much more common than others. If your dataset contains 90% instances of class A and 10% instances of class B, then a commonsense approach to the classification task is to always predict “A” when presented with a new sample. Such a classifier is 90% accurate overall, and any learning-based approach should therefore beat this 90% score to demonstrate usefulness. Sometimes, such elementary baselines can prove surprisingly hard to beat.

In this case, the temperature timeseries can safely be assumed to be continuous (the temperatures tomorrow are likely to be close to the temperatures today) as well as periodic with a daily period. Thus, a commonsense approach is to always predict 

### PDF p.378 (printed 354) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 
Listing 13.4 Plotting the first 10 days of the temperature timeseries
plt.plot(range(1440), temperature[:1440])

### PDF p.375 (printed 351) — 13 Timeseries forecasting › 13.1 Different kinds of timeseries tasks › This chapter covers
An overview of machine learning for timeseries

Understanding recurrent neural networks (RNNs)

Applying RNNs to a temperature forecasting

example

This chapter tackles timeseries, where temporal order is everything. We’ll focus on the most common and valuable timeseries task: forecasting. Using the recent past to predict the near future is a powerful capability, whether you’re trying to anticipate energy demand, manage inventory, or simply forecast the weather.

### PDF p.382 (printed 358) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 
Each dataset yields a tuple `(samples, targets)`, where `samples` is a batch of 256 samples, each containing 120 consecutive hours of input data, and `targets` is the corresponding array of 256 target temperatures. Note that the samples are randomly shuffled, so two

### PDF p.388 (printed 364) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
properties than data from an evening or from the middle of the night. Weather data is only translation-invariant for a very specific timescale.

Second, order in our data matters—a lot. The recent past is far more informative for predicting the next day’s temperature than data from five days ago. A 1D ConvNet is not able to make use of this fact. In particular, our max pooling and global average pooling layers are largely destroying order information.
