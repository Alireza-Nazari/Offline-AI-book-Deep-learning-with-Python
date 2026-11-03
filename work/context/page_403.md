# Generation bundle for page_403  (PDF page 403, printed page 379)

Page kind: content
Chapter: 13 Timeseries forecasting
Section: 13.4 Going even further
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_403.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_403.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p403_e03] (heading_section) 13.4 Going even further
[p403_e04] (body) There are many other things you could try to improve performance on the temperature forecasting problem:
[p403_e05] (bullet) Adjust the number of units in each recurrent layer in the stacked setup, as well as the amount of dropout. The current choices are largely arbitrary and thus probably suboptimal.
[p403_e06] (bullet) Adjust the learning rate used by the `Adam` optimizer or try a different optimizer.
[p403_e07] (bullet) Try using a stack of `Dense` layers as the regressor on top of the recurrent layer, instead of a single `Dense` layer.
[p403_e08] (bullet) Improve the input to the model: try using longer or shorter sequences or a different sampling rate, or start doing feature engineering.
[p403_e09] (body) As always, deep learning is more an art than a science. We can provide guidelines that suggest what is likely to work or not work on a given problem, but, ultimately, every dataset is unique; you’ll have to evaluate different strategies empirically. There is currently no theory that will tell you in advance precisely what you should do to optimally solve a problem. You must iterate.
[p403_e10] (body) In our experience, improving on the no-learning baseline by about 10% is likely the best you can do with this dataset. This isn’t so great, but these results make sense: while near-future weather is highly predictable if you have access to data from a wide grid of different locations, it’s not very predictable if you only have measurements from a single location. The evolution of the weather where you are depends on current weather patterns in surrounding locations.
[p403_e11] (heading_subsection) Markets and machine learning
[p403_e12] (sidebar) Some readers are bound to want to take the techniques we’ve introduced here and try them on the problem of forecasting the future price of securities on the stock market (or currency exchange rates, and so on). However, markets have very different statistical characteristics than natural phenomena such as weather patterns. When it comes to markets, past performance is not a good predictor of future returns—looking in the rear-view mirror is a bad way to drive. Machine learning, on the other hand, is applicable to datasets where the past is a good predictor of the future, like weather, electricity consumption, or foot traffic at a store. Always remember that all trading is fundamentally information arbitrage: gaining an advantage by using data or insights that other market participants are missing. Trying to use well-known machine learning techniques and publicly available data to beat the markets is effectively a dead end, since you won’t have any information advantage compared to everyone else. You’re likely to waste your time and resources with nothing to show for it.

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

## Previous page (PDF 402) text, for continuity
the better: they offer a new angle from which to look at your data, capturing aspects of the data that were missed by other approaches, and thus they can help boost performance on a task. This is the intuition behind ensembling, a concept we’ll explore in chapter 18.

A bidirectional RNN exploits this idea to improve on the performance of chronological-order RNNs. It looks at its input sequence both ways (see figure 13.13), obtaining potentially richer representations and capturing patterns that may have been missed by the chronologi

Figure 13.13 How a bidirectional RNN

cal-order version alone.

layer works

To instantiate a bidirectional RNN in Keras, you use the `Bidirectional` layer, which takes as its first argument a recurrent layer instance. `Bidirectional` creates a second, separate instance of this recurrent layer and uses one instance for processing the input sequences in chronological order and the other instance for processing the input sequences in reversed order. You can try it on our temperature forecasting task.

Listing 13.24 Training and evaluating a bidirectional LSTM

You’ll find that it doesn’t perform as well as the plain `LSTM` layer. It’s easy to understand why: all the predictive capacity must come from the chronological half of the network because the antichronological half is known to be severely underperforming on this task (again, because the recent past matters much more than the distant past in this case). At the same time, the presence of the antichronological half doubles the network’s capacity and causes it to start overfitting much earlier.

However, bidirectional RNNs are a great fit for text data—or any other kind of data where order matters, yet where which order you use doesn’t matter. In fact, for a while in 2016, bidirectional L …

## Next page (PDF 404) text, for continuity
Summary

As you first learned in chapter 6, when approaching a new problem, it’s good to first establish commonsense baselines for your metric of choice. If you don’t have a baseline to beat, you can’t tell whether you’re making real progress.

Try simple models before expensive ones, to justify the additional expense. Sometimes a simple model will turn out to be your best option.

When you have data where ordering matters—in particular, for timeseries data— recurrent networks are a great fit and easily outperform models that first flatten the temporal data. The two essential RNN layers available in Keras are the `LSTM` layer and the `GRU` layer.

To use dropout with recurrent networks, you should use a time-constant dropout mask and recurrent dropout mask. These are built into Keras recurrent layers, so all you have to do is use the `recurrent_dropout` arguments of recurrent layers.

Stacked RNNs provide more representational power than a single RNN layer. They’re also much more expensive and thus not always worth it. Although they offer clear gains on complex problems (such as machine translation), they may not always be relevant to smaller, simpler problems.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.384 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.376 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.383 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.375 — 13 Timeseries forecasting › 13.1 Different kinds of timeseries tasks
- PDF p.386 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.388 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.396 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.397 — 13 Timeseries forecasting › 13.3 Recurrent neural networks

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.376 (printed 352) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 13.2 A temperature forecasting example
Throughout this chapter, all of our code examples will target a single problem: predicting the temperature 24 hours in the future, given a timeseries of hourly measurements of quantities such as atmospheric pressure and humidity, recorded over the recent past by a set of sensors on the roof of a building. As you will see, it’s a fairly challenging problem!

We’ll use this temperature forecasting task to highlight what makes timeseries data fundamentally different from the kinds of datasets you’ve encountered so far, to show that densely connected networks and convolutional networks aren’t well equipped to deal with it, and to demonstrate a new kind of machine learning technique that really shines on this type of problem: recurrent neural networks (RNNs). We’ll work with a weather timeseries dataset recorded at the weather station at the Max Planck Institute for Biogeochemistry in Jena, Germany.1 In this dataset, 14 different quantities (such as temperature, atmospheric pressure, humidity, wind direction, and so on) were recorded every 10 minutes, over several years. The original data goes back to 2003, but the subset of the data we’ll download is limited to 2009–2016. Let’s start b

### PDF p.383 (printed 359) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 13.2.2 A commonsense, non-machine-learning baseline
Before you start using black box, deep learning models to solve the temperature prediction problem, let’s try a simple, commonsense approach. It will serve as a sanity check, and it will establish a baseline that you’ll have to beat to demonstrate the usefulness of more advanced, machine learning models. Such commonsense baselines can be useful when you’re approaching a new problem for which there is no known solution (yet). A classic example is that of unbalanced classification tasks, where some classes are much more common than others. If your dataset contains 90% instances of class A and 10% instances of class B, then a commonsense approach to the classification task is to always predict “A” when presented with a new sample. Such a classifier is 90% accurate overall, and any learning-based approach should therefore beat this 90% score to demonstrate usefulness. Sometimes, such elementary baselines can prove surprisingly hard to beat.

In this case, the temperature timeseries can safely be assumed to be continuous (the temperatures tomorrow are likely to be close to the temperatures today) as well as periodic with a daily period. Thus, a commonsense approach is to always predict 

### PDF p.375 (printed 351) — 13 Timeseries forecasting › 13.1 Different kinds of timeseries tasks › This chapter covers
An overview of machine learning for timeseries

Understanding recurrent neural networks (RNNs)

Applying RNNs to a temperature forecasting

example

This chapter tackles timeseries, where temporal order is everything. We’ll focus on the most common and valuable timeseries task: forecasting. Using the recent past to predict the near future is a powerful capability, whether you’re trying to anticipate energy demand, manage inventory, or simply forecast the weather.

### PDF p.384 (printed 360) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 13.2.3 Let’s try a basic machine learning model
In the same way that it’s useful to establish a commonsense baseline before trying machine learning approaches, it’s useful to try simple, cheap, machine learning models (such as small, densely connected networks) before looking into complicated and computationally expensive models such as RNNs. This is the best way to make sure any further complexity you throw at the problem is legitimate and delivers real benefits.

Listing 13.10 shows a fully connected model that starts by flattening the data and then runs it through two `Dense` layers. Note the lack of activation function on the last `Dense` layer, which is typical for a regression problem. We use mean squared error (MSE) as the loss, rather than MAE, because unlike MAE, it’s smooth around zero, a useful property for gradient descent. We will monitor MAE by adding it as a metric in `compile()`.

Listing 13.10 Training and evaluating a densely connected model

### PDF p.386 (printed 362) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 
Some of the validation losses are close to the no-learning baseline, but not reliably. This goes to show the merit of having this baseline in the first place: it turns out to be not easy to outperform. Your common sense contains a lot of valuable information to which a machine learning model doesn’t have access.

You may wonder, if a simple, well-performing model exists to go from the data to the targets (the commonsense baseline), why doesn’t the model you’re training find it and improve on it? Well, the space of models in which you’re searching for a solution—that is, your hypothesis space—is the space of all possible two-layer networks with the configuration you defined. The commonsense heuristic is just one model among millions that can be represented in this space. It’s like looking for a needle in a haystack. Just because a good solution technically exists in your hypothesis space doesn’t mean you’ll be able to find it via gradient descent. That’s a pretty significant limitation of machine learning in general: unless the learning algorithm is hardcoded to look for a specific kind of simple model, it can sometimes fail to find a simple solution to a simple problem. That’s why 

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

### PDF p.397 (printed 373) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
curves, it’s evident that the model is quickly overfitting, despite only having very few units: the training and validation losses start to diverge considerably after a few epochs. You’re already familiar with a classic technique for fighting this phenomenon: dropout, which randomly zeros out input units of a layer to break happenstance correlations in the training data that the layer is exposed to. But how to correctly apply dropout in recurrent networks isn’t a trivial question.

It has long been known that applying dropout before a recurrent layer hinders learning rather than helping with regularization. In 2015, Yarin Gal, as part of his PhD thesis on Bayesian deep learning,5 determined the proper way to use dropout with a recurrent network: the same dropout mask (the same pattern of dropped units) should be applied at every timestep, instead of a dropout mask that varies randomly from timestep to timestep. What’s more, to regularize the representations formed by the recurrent gates of layers such as `GRU` and `LSTM`, a temporally constant dropout mask should be applied to the inner recurrent activations of the layer (a recurrent dropout mask). Using the same dropout mask at ev

### PDF p.386 (printed 362) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 13.2.4 Let’s try a 1D convolutional model
Speaking of using the right architecture priors: since our input sequences feature daily cycles, perhaps a convolutional model could work? A temporal ConvNet could reuse the same representations across different days, much like a spatial ConvNet can reuse the same representations across different locations in an image.

You already know about the `Conv2D` and `SeparableConv2D` layers, which see their inputs through small windows that swipe across 2D grids. There are also 1D and even 3D versions of these layers: `Conv1D`, `SeparableConv1D`, and `Conv3D`.2 The `Conv1D` layer relies on 1D windows that slide across input sequences, and the `Conv3D` layer relies on cubic windows that slide across input volumes. You can thus build 1D ConvNets, strictly analogous to 2D ConvNets. They’re a great fit for any sequence data that follows the translation invariance assumption (meaning that if you slide a window over the sequence, the content of the window should follow the same properties independently of the location of the window). Let’s try one on our temperature forecasting problem. We’ll pick an initial window length of 24, so that we look at 24 hours of data at a time (one cycle). As we d
