# Generation bundle for page_384  (PDF page 384, printed page 360)

Page kind: content
Chapter: 13 Timeseries forecasting
Section: 13.2 A temperature forecasting example
Subsection: 13.2.3 Let’s try a basic machine learning model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_384.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_384.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p384_e03] (code)
```
        preds = samples[:, -1, 1] * std[1] + mean[1]
        total_abs_err += np.sum(np.abs(preds - targets))
        samples_seen += samples.shape[0]
    return total_abs_err / samples_seen
print(f"Validation MAE: {evaluate_naive_method(val_dataset):.2f}")
print(f"Test MAE: {evaluate_naive_method(test_dataset):.2f}")
```
[p384_e04] (body) This commonsense baseline achieves a validation MAE of 2.44 degrees Celsius, and a test MAE of 2.62 degrees Celsius. So if you always assume that the temperature 24 hours in the future will be the same as it is now, you will be off by two and a half degrees on average. It’s not too bad, but you probably won’t launch a weather forecasting service based on this heuristic. Now, the game is to use your knowledge of deep learning to do better.
[p384_e05] (heading_subsection) 13.2.3 Let’s try a basic machine learning model
[p384_e06] (body) In the same way that it’s useful to establish a commonsense baseline before trying machine learning approaches, it’s useful to try simple, cheap, machine learning models (such as small, densely connected networks) before looking into complicated and computationally expensive models such as RNNs. This is the best way to make sure any further complexity you throw at the problem is legitimate and delivers real benefits.
[p384_e07] (body) Listing 13.10 shows a fully connected model that starts by flattening the data and then runs it through two `Dense` layers. Note the lack of activation function on the last `Dense` layer, which is typical for a regression problem. We use mean squared error (MSE) as the loss, rather than MAE, because unlike MAE, it’s smooth around zero, a useful property for gradient descent. We will monitor MAE by adding it as a metric in `compile()`.
[p384_e08] (listing_caption) Listing 13.10 Training and evaluating a densely connected model
[p384_e09] (code)
```
import keras
from keras import layers
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.Flatten()(inputs)
x = layers.Dense(16, activation="relu")(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
callbacks = [
    keras.callbacks.ModelCheckpoint("jena_dense.keras", save_best_only=True)
```

## Code annotations printed next to listings (callouts, in order)
- The temperature feature is at column 1, so samples[:, -1, 1] is the
- last temperature measurement in the input sequence. Recall that
- we normalized our features to retrieve a temperature in Celsius
- degrees, we need to un-normalize it, by multiplying it by the
- standard deviation and adding back the mean.
- We use a callback to save
- the best-performing model.

## Listings
- Listing 13.10: Training and evaluating a densely connected model (code elements p384_e09)

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

## Previous page (PDF 383) text, for continuity
consecutive sequences in a batch (like `samples[0]` and `samples[1]`) aren’t necessarily temporally close.

Listing 13.8 Inspecting the dataset

13.2.2 A commonsense, non-machine-learning baseline

Before you start using black box, deep learning models to solve the temperature prediction problem, let’s try a simple, commonsense approach. It will serve as a sanity check, and it will establish a baseline that you’ll have to beat to demonstrate the usefulness of more advanced, machine learning models. Such commonsense baselines can be useful when you’re approaching a new problem for which there is no known solution (yet). A classic example is that of unbalanced classification tasks, where some classes are much more common than others. If your dataset contains 90% instances of class A and 10% instances of class B, then a commonsense approach to the classification task is to always predict “A” when presented with a new sample. Such a classifier is 90% accurate overall, and any learning-based approach should therefore beat this 90% score to demonstrate usefulness. Sometimes, such elementary baselines can prove surprisingly hard to beat.

In this case, the temperature timeseries can safely be assumed to be continuous (the temperatures tomorrow are likely to be close to the temperatures today) as well as periodic with a daily period. Thus, a commonsense approach is to always predict that the temperature 24 hours from now will be equal to the temperature right now. Let’s evaluate this approach, using the mean absolute error (MAE) metric, defined as follows:

Here’s the evaluation loop.

Listing 13.9 Computing the commonsense baseline MAE

## Next page (PDF 385) text, for continuity
Let’s display the loss curves for validation and training (see figure 13.3).

Listing 13.11 Plotting results

Figure 13.3 Training and validation MAE on the Jena temperature forecasting task with a simple, densely connected network

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.388 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.398 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.387 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.400 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.397 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.402 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.153 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.386 — 13 Timeseries forecasting › 13.2 A temperature forecasting example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.388 (printed 364) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3 Recurrent neural networks
Figure 13.5 shows the results. Much better! We achieve a validation MAE as low as 2.39 degrees and a test MAE of 2.55 degrees. The LSTM-based model can finally beat the commonsense baseline (albeit just by a bit, for now), demonstrating the value of machine learning on this task.

### PDF p.387 (printed 363) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 
Figure 13.4 Training and validation MAE on the Jena temperature forecasting task with a 1D ConvNet

As it turns out, this model performs even worse than the densely connected one, only achieving a validation MAE of about 2.9 degrees, far from the commonsense baseline. What went wrong here? Two things:

First, weather data doesn’t quite respect the translation invariance assumption. While the data does feature daily cycles, data from a morning follows different

### PDF p.398 (printed 374) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
Figure 13.11 shows the results. Success! We’re no longer overfitting during the first 20 epochs. We achieve a validation MAE as low as 2.27 degrees (7% improvement over the no-learning baseline) and a test MAE of 2.45 degrees (6.5% improvement over the baseline). Not too bad.

Figure 13.11 Training and validation loss on the Jena temperature forecasting task with a dropout-regularized LSTM

### PDF p.389 (printed 365) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
Figure 13.5 Training and validation MAE on the Jena temperature forecasting task with an LSTM-based model. (Note that we omit epoch 1 on this graph because the high training MAE (7.75) at epoch 1 would distort the scale.)

But why did the LSTM model perform markedly better than the densely connected one or the ConvNet? And how can we further refine the model? To answer this, let’s take a closer look at recurrent neural networks.

### PDF p.400 (printed 376) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
Figure 13.12 shows the results. We achieve a test MAE of 2.39 degrees (an 8.8% improvement over the baseline). You can see that the added layer does improve the results a bit, though not dramatically. You may be seeing diminishing returns from increasing network capacity at this point.

Figure 13.12 Training and validation loss on the Jena temperature forecasting task with a stacked GRU network

### PDF p.376 (printed 352) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 13.2 A temperature forecasting example
Throughout this chapter, all of our code examples will target a single problem: predicting the temperature 24 hours in the future, given a timeseries of hourly measurements of quantities such as atmospheric pressure and humidity, recorded over the recent past by a set of sensors on the roof of a building. As you will see, it’s a fairly challenging problem!

We’ll use this temperature forecasting task to highlight what makes timeseries data fundamentally different from the kinds of datasets you’ve encountered so far, to show that densely connected networks and convolutional networks aren’t well equipped to deal with it, and to demonstrate a new kind of machine learning technique that really shines on this type of problem: recurrent neural networks (RNNs). We’ll work with a weather timeseries dataset recorded at the weather station at the Max Planck Institute for Biogeochemistry in Jena, Germany.1 In this dataset, 14 different quantities (such as temperature, atmospheric pressure, humidity, wind direction, and so on) were recorded every 10 minutes, over several years. The original data goes back to 2003, but the subset of the data we’ll download is limited to 2009–2016. Let’s start b

### PDF p.386 (printed 362) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 
Some of the validation losses are close to the no-learning baseline, but not reliably. This goes to show the merit of having this baseline in the first place: it turns out to be not easy to outperform. Your common sense contains a lot of valuable information to which a machine learning model doesn’t have access.

You may wonder, if a simple, well-performing model exists to go from the data to the targets (the commonsense baseline), why doesn’t the model you’re training find it and improve on it? Well, the space of models in which you’re searching for a solution—that is, your hypothesis space—is the space of all possible two-layer networks with the configuration you defined. The commonsense heuristic is just one model among millions that can be represented in this space. It’s like looking for a needle in a haystack. Just because a good solution technically exists in your hypothesis space doesn’t mean you’ll be able to find it via gradient descent. That’s a pretty significant limitation of machine learning in general: unless the learning algorithm is hardcoded to look for a specific kind of simple model, it can sometimes fail to find a simple solution to a simple problem. That’s why 

### PDF p.403 (printed 379) — 13 Timeseries forecasting › 13.4 Going even further › 13.4 Going even further
There are many other things you could try to improve performance on the temperature forecasting problem:

Adjust the number of units in each recurrent layer in the stacked setup, as well as the amount of dropout. The current choices are largely arbitrary and thus probably suboptimal.

Adjust the learning rate used by the `Adam` optimizer or try a different optimizer.

Try using a stack of `Dense` layers as the regressor on top of the recurrent layer, instead of a single `Dense` layer.

Improve the input to the model: try using longer or shorter sequences or a different sampling rate, or start doing feature engineering.

As always, deep learning is more an art than a science. We can provide guidelines that suggest what is likely to work or not work on a given problem, but, ultimately, every dataset is unique; you’ll have to evaluate different strategies empirically. There is currently no theory that will tell you in advance precisely what you should do to optimally solve a problem. You must iterate.

In our experience, improving on the no-learning baseline by about 10% is likely the best you can do with this dataset. This isn’t so great, but these results make sense: while near-futu
