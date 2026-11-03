# Generation bundle for page_375  (PDF page 375, printed page 351)

Page kind: chapter_opener
Chapter: 13 Timeseries forecasting
Section: 13.1 Different kinds of timeseries tasks
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_375.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_375.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p375_e01] (chapter_number) 13
[p375_e02] (chapter_title) Timeseries forecasting
[p375_e03] (chapter_covers) This chapter covers
[p375_e04] (bullet) An overview of machine learning for timeseries
[p375_e05] (bullet) Understanding recurrent neural networks (RNNs)
[p375_e06] (bullet) Applying RNNs to a temperature forecasting
[p375_e07] (bullet) example
[p375_e08] (body) This chapter tackles timeseries, where temporal order is everything. We’ll focus on the most common and valuable timeseries task: forecasting. Using the recent past to predict the near future is a powerful capability, whether you’re trying to anticipate energy demand, manage inventory, or simply forecast the weather.
[p375_e09] (heading_section) 13.1 Different kinds of timeseries tasks
[p375_e10] (body) A timeseries can be any data obtained via measurements at regular intervals, like the daily price of a stock, the hourly electricity consumption of a city, or the weekly sales of a store. Timeseries are everywhere, whether we’re looking at natural phenomena (like seismic activity, the evolution of fish populations in a river, or the weather at a location) or human activity patterns (like visitors to a website, a country’s GDP, or credit card transactions). Unlike the types of data you’ve encountered so far,

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

## Previous page (PDF 374) text, for continuity
Summary

Object detection identifies and locates objects within an image using bounding boxes. It’s basically a weaker version of image segmentation, but one that can be run much more efficiently.

There are two primary approaches to object detection: – Region-based Convolutional Neural Networks (R-CNNs), which are two-stage models that first propose regions of interest and then classify them with a ConvNet. – Single-stage detectors (like RetinaNet and YOLO), which perform both tasks in a single step. Single-stage detectors are generally faster and more efficient, making them suitable for real-time applications (e.g., self-driving cars).

YOLO computes two separate outputs simultaneously during training—possible bounding boxes and a class probability map: – Each candidate bounding box is paired with a confidence score, which is trained to target the Intersection over Union of the predicted box and the ground truth box. – The class probability map classifies different regions of an image as belonging to different objects.

RetinaNet builds on this idea by using a feature pyramid network (FPN), which combines features from multiple ConvNet layers to create feature maps at different scales, allowing it to more accurately detect objects of different sizes.

## Next page (PDF 376) text, for continuity
working with timeseries involves understanding the dynamics of a system—its periodic cycles, how it trends over time, its regular regime, and its sudden spikes.

By far, the most common timeseries-related task is forecasting: predicting what happens next in the series. Forecast electricity consumption a few hours in advance so you can anticipate demand, forecast revenue a few months in advance so you can plan your budget, forecast the weather a few days in advance so you can plan your schedule. Forecasting is what this chapter focuses on. But there’s actually a wide range of other things you can do with timeseries, such as

Anomaly detection—Detect anything unusual happening within a continuous data stream. Unusual activity on your corporate network? Might be an attacker. Unusual readings on a manufacturing line? Time for a human to go take a look. Anomaly detection is typically done via unsupervised learning, because you often don’t know what kind of anomaly you’re looking for, and thus you can’t train on specific anomaly examples.

Classification—Assign one or more categorical labels to a timeseries. For instance, given the timeseries of activity of a visitor on a website, classify whether the visitor is a bot or a human.

Event detection—Identify the occurrence of a specific, expected event within a continuous data stream. A particularly useful application is “hotword detection,” where a model monitors an audio stream and detects utterances like “OK, Google” or “Hey, Alexa.”

In this chapter, you’ll learn about recurrent neural networks (RNNs) and how to apply them to timeseries forecasting.

13.2 A temperature forecasting example

Throughout this chapter, all of our code examples will target a single problem: predicting the temperature 24 hours in the future, given  …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.380 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.379 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.378 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.403 — 13 Timeseries forecasting › 13.4 Going even further
- PDF p.381 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.388 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.383 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.396 — 13 Timeseries forecasting › 13.3 Recurrent neural networks

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.379 (printed 355) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 
Figure 13.2 Temperature over the first 10 days of the dataset (ºC)

On this plot, you can see daily periodicity, especially evident for the last four days. Also note that this 10-day period must be coming from a fairly cold winter month.

NOTE Periodicity over multiple timescales is an important and very common property of timeseries data. Whether you’re looking at the weather, mall parking occupancy, traffic to a website, sales of a grocery store, or steps logged in a fitness tracker, you’ll see daily cycles and yearly cycles (human-generated data also tends to feature weekly cycles). When exploring your data, make sure to look for these patterns.

With our dataset, if you were trying to predict average temperature for the next month given a few months of past data, the problem would be easy, due to the reliable year-scale periodicity of the data. But looking at the data over a scale of days, the temperature looks a lot more chaotic. Is this timeseries predictable at a daily scale? Let’s find out.

In all our experiments, we’ll use the first 50% of the data for training, the following 25% for validation, and the last 25% for testing. When working with timeseries data, it’s importa

### PDF p.403 (printed 379) — 13 Timeseries forecasting › 13.4 Going even further › Markets and machine learning
Some readers are bound to want to take the techniques we’ve introduced here and try them on the problem of forecasting the future price of securities on the stock market (or currency exchange rates, and so on). However, markets have very different statistical characteristics than natural phenomena such as weather patterns. When it comes to markets, past performance is not a good predictor of future returns—looking in the rear-view mirror is a bad way to drive. Machine learning, on the other hand, is applicable to datasets where the past is a good predictor of the future, like weather, electricity consumption, or foot traffic at a store. Always remember that all trading is fundamentally information arbitrage: gaining an advantage by using data or insights that other market participants are missing. Trying to use well-known machine learning techniques and publicly available data to beat the markets is effectively a dead end, since you won’t have any information advantage compared to everyone else. You’re likely to waste your time and resources with nothing to show for it.

### PDF p.380 (printed 356) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 13.2.1 Preparing the data
Next, let’s create a `Dataset` object that yields batches of data from the past five days along with a target temperature 24 hours in the future. Because the samples in the dataset are highly redundant (sample `N` and sample `N + 1` will have most of their timesteps in common), it would be wasteful to explicitly allocate memory for every sample. Instead, we’ll generate the samples on the fly while only keeping in memory the original `raw_data` and `temperature` arrays, and nothing more.

We could easily write a Python generator to do this, but there’s a built-in dataset utility in Keras that does just that (`timeseries_dataset_from_array()`), so we can save ourselves some work by using it. You can generally use it for any kind of timeseries forecasting task.

### PDF p.388 (printed 364) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3 Recurrent neural networks
Neither the fully connected approach nor the convolutional approach did well, but that doesn’t mean machine learning isn’t applicable to this problem. The densely connected approach first flattened the timeseries, which removed the notion of time from the input data. The convolutional approach treated every segment of the data in the same way, even applying pooling, which destroyed order information. Let’s instead look at the data as what it is: a sequence, where causality and order matter.

There’s a family of neural network architectures that were designed specifically for this use case: recurrent neural networks. Among them, the Long Short-Term Memory (LSTM) layer in particular has long been very popular. We’ll see in a minute how these models work—but let’s start by giving the LSTM layer a try.

Listing 13.12 A simple LSTM-based model

### PDF p.380 (printed 356) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › Understanding timeseries_dataset_from_array()
To understand what `timeseries_dataset_from_array()` does, let’s take a look at a simple example. The general idea is that you provide an array of timeseries data (the `data` argument), and `timeseries_dataset_from_array` gives you windows extracted from the original timeseries (we’ll call them “sequences”).

### PDF p.381 (printed 357) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 
Let’s say you’re using `data = [0 1 2 3 4 5 6]` and `sequence_length=3`; then `timeseries_dataset_from_array` will generate the following samples: `[0 1 2]`, `[1 2 ` `3]`, `[2 3 4]`, `[3 4 5]`, `[4 5 6]`.

You can also pass a `target` array to `timeseries_dataset_from_array()`. The first entry of the `targets` array should match the desired target for the first sequence that will be generated from the `data` array. So if you’re doing timeseries forecasting, simply use as `targets` the same array as for `data`, offset by some amount.

For instance, with `data = [0 1 2 3 4 5 6 ...]` and `sequence_length=3`, you could

create a dataset to predict the next step in the series by passing `targets = [3 4 5 6 ` `...]`. Let’s try it:

### PDF p.388 (printed 364) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
properties than data from an evening or from the middle of the night. Weather data is only translation-invariant for a very specific timescale.

Second, order in our data matters—a lot. The recent past is far more informative for predicting the next day’s temperature than data from five days ago. A 1D ConvNet is not able to make use of this fact. In particular, our max pooling and global average pooling layers are largely destroying order information.

### PDF p.383 (printed 359) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 13.2.2 A commonsense, non-machine-learning baseline
Before you start using black box, deep learning models to solve the temperature prediction problem, let’s try a simple, commonsense approach. It will serve as a sanity check, and it will establish a baseline that you’ll have to beat to demonstrate the usefulness of more advanced, machine learning models. Such commonsense baselines can be useful when you’re approaching a new problem for which there is no known solution (yet). A classic example is that of unbalanced classification tasks, where some classes are much more common than others. If your dataset contains 90% instances of class A and 10% instances of class B, then a commonsense approach to the classification task is to always predict “A” when presented with a new sample. Such a classifier is 90% accurate overall, and any learning-based approach should therefore beat this 90% score to demonstrate usefulness. Sometimes, such elementary baselines can prove surprisingly hard to beat.

In this case, the temperature timeseries can safely be assumed to be continuous (the temperatures tomorrow are likely to be close to the temperatures today) as well as periodic with a daily period. Thus, a commonsense approach is to always predict 
