# Generation bundle for page_386  (PDF page 386, printed page 362)

Page kind: content
Chapter: 13 Timeseries forecasting
Section: 13.2 A temperature forecasting example
Subsection: 13.2.4 Let’s try a 1D convolutional model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_386.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_386.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p386_e03] (body) Some of the validation losses are close to the no-learning baseline, but not reliably. This goes to show the merit of having this baseline in the first place: it turns out to be not easy to outperform. Your common sense contains a lot of valuable information to which a machine learning model doesn’t have access.
[p386_e04] (body) You may wonder, if a simple, well-performing model exists to go from the data to the targets (the commonsense baseline), why doesn’t the model you’re training find it and improve on it? Well, the space of models in which you’re searching for a solution—that is, your hypothesis space—is the space of all possible two-layer networks with the configuration you defined. The commonsense heuristic is just one model among millions that can be represented in this space. It’s like looking for a needle in a haystack. Just because a good solution technically exists in your hypothesis space doesn’t mean you’ll be able to find it via gradient descent. That’s a pretty significant limitation of machine learning in general: unless the learning algorithm is hardcoded to look for a specific kind of simple model, it can sometimes fail to find a simple solution to a simple problem. That’s why using good feature engineering and relevant architecture priors is essential: you need to be precisely telling your model what it should be looking for.
[p386_e05] (heading_subsection) 13.2.4 Let’s try a 1D convolutional model
[p386_e06] (body) Speaking of using the right architecture priors: since our input sequences feature daily cycles, perhaps a convolutional model could work? A temporal ConvNet could reuse the same representations across different days, much like a spatial ConvNet can reuse the same representations across different locations in an image.
[p386_e07] (body) You already know about the `Conv2D` and `SeparableConv2D` layers, which see their inputs through small windows that swipe across 2D grids. There are also 1D and even 3D versions of these layers: `Conv1D`, `SeparableConv1D`, and `Conv3D`.2 The `Conv1D` layer relies on 1D windows that slide across input sequences, and the `Conv3D` layer relies on cubic windows that slide across input volumes. You can thus build 1D ConvNets, strictly analogous to 2D ConvNets. They’re a great fit for any sequence data that follows the translation invariance assumption (meaning that if you slide a window over the sequence, the content of the window should follow the same properties independently of the location of the window). Let’s try one on our temperature forecasting problem. We’ll pick an initial window length of 24, so that we look at 24 hours of data at a time (one cycle). As we downsample the sequences (via `MaxPooling1D` layers), we’ll reduce the window size accordingly (figure 13.4):
[p386_e08] (code)
```
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.Conv1D(8, 24, activation="relu")(inputs)
x = layers.MaxPooling1D(2)(x)
```
[p386_e09] (footnote) 2 There isn’t a `SeparableConv3D` layer, not for any theoretical reason, but simply because we haven’t implemented it.

## Equation candidates (formulas written inline in the text)
- [p386_e08] inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.Conv1D(8, 24, activation="relu")(inputs)
x = layers.MaxPooling1D(2)(x)

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

## Previous page (PDF 385) text, for continuity
Let’s display the loss curves for validation and training (see figure 13.3).

Listing 13.11 Plotting results

Figure 13.3 Training and validation MAE on the Jena temperature forecasting task with a simple, densely connected network

## Next page (PDF 387) text, for continuity
Figure 13.4 Training and validation MAE on the Jena temperature forecasting task with a 1D ConvNet

As it turns out, this model performs even worse than the densely connected one, only achieving a validation MAE of about 2.9 degrees, far from the commonsense baseline. What went wrong here? Two things:

First, weather data doesn’t quite respect the translation invariance assumption. While the data does feature daily cycles, data from a morning follows different

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.626 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.383 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.388 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.339 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.403 — 13 Timeseries forecasting › 13.4 Going even further
- PDF p.384 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.404 — 13 Timeseries forecasting › 13.4 Going even further
- PDF p.469 — 15 Language models and the Transformer › 15.3 The Transformer architecture

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.383 (printed 359) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 13.2.2 A commonsense, non-machine-learning baseline
Before you start using black box, deep learning models to solve the temperature prediction problem, let’s try a simple, commonsense approach. It will serve as a sanity check, and it will establish a baseline that you’ll have to beat to demonstrate the usefulness of more advanced, machine learning models. Such commonsense baselines can be useful when you’re approaching a new problem for which there is no known solution (yet). A classic example is that of unbalanced classification tasks, where some classes are much more common than others. If your dataset contains 90% instances of class A and 10% instances of class B, then a commonsense approach to the classification task is to always predict “A” when presented with a new sample. Such a classifier is 90% accurate overall, and any learning-based approach should therefore beat this 90% score to demonstrate usefulness. Sometimes, such elementary baselines can prove surprisingly hard to beat.

In this case, the temperature timeseries can safely be assumed to be continuous (the temperatures tomorrow are likely to be close to the temperatures today) as well as periodic with a daily period. Thus, a commonsense approach is to always predict 

### PDF p.292 (printed 268) — 9 ConvNet architecture patterns ›  › This chapter covers
The modularity-hierarchy-reuse formula for model

architecture

An overview of standard best practices for

building ConvNets: residual connections,

batch normalization, and depthwise separable

convolutions

Ongoing design trends for computer vision

models

A model’s “architecture” is the sum of the choices that went into creating it: which layers to use, how to configure them, in what arrangement to connect them. These choices define the hypothesis space of your model: the space of possible functions that gradient descent can search over, parameterized by the model’s weights. Like feature engineering, a good hypothesis space encodes prior knowledge that you have about the problem at hand and its solution. For instance, using convolution layers means that you know in advance that the relevant patterns present in your input

### PDF p.176 (printed 152) — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models › Beating a common-sense baseline
Besides the different evaluation protocols you have available, one last thing you should know about is the use of common-sense baselines.

Training a deep learning model is a bit like pressing a button that launches a rocket in a parallel world. You can’t hear it or see it. You can’t observe the manifold learning process—it’s happening in a space with thousands of dimensions, and even if you projected it to 3D, you couldn’t interpret it. The only feedback you have is your validation metrics—like an altitude meter on your invisible rocket. A particularly important point is to be able to tell whether you’re getting off the ground at all. What was the altitude you started at? Your model seems to have an accuracy of 15%, is that any good? Before you start working with a dataset, you should always pick a trivial baseline that you’ll try to beat. If you cross that threshold, you’ll know you’re doing something right: your model is actually using the information in the input data to make predictions that generalize—you can keep going. This baseline could be performance of a random classifier, or the performance of the simplest non-machine learning technique you can imagine. For instance, i

### PDF p.403 (printed 379) — 13 Timeseries forecasting › 13.4 Going even further › 13.4 Going even further
There are many other things you could try to improve performance on the temperature forecasting problem:

Adjust the number of units in each recurrent layer in the stacked setup, as well as the amount of dropout. The current choices are largely arbitrary and thus probably suboptimal.

Adjust the learning rate used by the `Adam` optimizer or try a different optimizer.

Try using a stack of `Dense` layers as the regressor on top of the recurrent layer, instead of a single `Dense` layer.

Improve the input to the model: try using longer or shorter sequences or a different sampling rate, or start doing feature engineering.

As always, deep learning is more an art than a science. We can provide guidelines that suggest what is likely to work or not work on a given problem, but, ultimately, every dataset is unique; you’ll have to evaluate different strategies empirically. There is currently no theory that will tell you in advance precisely what you should do to optimally solve a problem. You must iterate.

In our experience, improving on the no-learning baseline by about 10% is likely the best you can do with this dataset. This isn’t so great, but these results make sense: while near-futu

### PDF p.293 (printed 269) — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse › 
images are translation-invariant. To effectively learn from data, you need to make assumptions about what you’re looking for.

Model architecture is often the difference between success and failure. If you make inappropriate architecture choices, your model may be stuck with suboptimal metrics, and no amount of training data will save it. Inversely, a good model architecture will accelerate learning and will enable your model to make efficient use of the training data available, reducing the need for large datasets. A good model architecture is one that reduces the size of the search space or otherwise makes it easier to converge to a good point of the search space. Just like feature engineering and data curation, model architecture is all about making the problem simpler for gradient descent to solve—and remember that gradient descent is a pretty stupid search process, so it needs all the help it can get. Model architecture is more an art than a science. Experienced machine learning engineers are able to intuitively cobble together high-performing models on their first try, while beginners often struggle to create a model that trains at all. The keyword here is intuitively: no one

### PDF p.179 (printed 155) — 5 Fundamentals of machine learning › 5.3 Improving model fit › Using better architecture priors
You have a model that fits, but for some reason your validation metrics aren’t improving at all. They remain no better than what a random classifier would achieve: your model trains, but doesn’t generalize. What’s going on?

This is perhaps the worst machine learning situation you can find yourself in. It indicates that something is fundamentally wrong with your approach, and it may not be easy to tell what. Here are some tips. First, it may be that the input data you’re using simply doesn’t contain sufficient information to predict your targets: the problem as formulated is not solvable. This is what happened earlier when we tried to fit an MNIST model where the labels were shuffled: the model would train just fine, but validation accuracy would stay stuck at 10%, because it was plainly impossible to generalize with such a dataset. It may also be that the kind of model you’re using is not suited for the problem at hand. For instance, in chapter 13, you’ll see an example of a timeseries prediction problem where a densely connected architecture isn’t able to beat a trivial baseline, whereas a more appropriate recurrent architecture does manage to generalize well. Using a model that 

### PDF p.384 (printed 360) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 13.2.3 Let’s try a basic machine learning model
In the same way that it’s useful to establish a commonsense baseline before trying machine learning approaches, it’s useful to try simple, cheap, machine learning models (such as small, densely connected networks) before looking into complicated and computationally expensive models such as RNNs. This is the best way to make sure any further complexity you throw at the problem is legitimate and delivers real benefits.

Listing 13.10 shows a fully connected model that starts by flattening the data and then runs it through two `Dense` layers. Note the lack of activation function on the last `Dense` layer, which is typical for a regression problem. We use mean squared error (MSE) as the loss, rather than MAE, because unlike MAE, it’s smooth around zero, a useful property for gradient descent. We will monitor MAE by adding it as a metric in `compile()`.

Listing 13.10 Training and evaluating a densely connected model

### PDF p.404 (printed 380) — 13 Timeseries forecasting › 13.4 Going even further › Summary
As you first learned in chapter 6, when approaching a new problem, it’s good to first establish commonsense baselines for your metric of choice. If you don’t have a baseline to beat, you can’t tell whether you’re making real progress.

Try simple models before expensive ones, to justify the additional expense. Sometimes a simple model will turn out to be your best option.

When you have data where ordering matters—in particular, for timeseries data— recurrent networks are a great fit and easily outperform models that first flatten the temporal data. The two essential RNN layers available in Keras are the `LSTM` layer and the `GRU` layer.

To use dropout with recurrent networks, you should use a time-constant dropout mask and recurrent dropout mask. These are built into Keras recurrent layers, so all you have to do is use the `recurrent_dropout` arguments of recurrent layers.

Stacked RNNs provide more representational power than a single RNN layer. They’re also much more expensive and thus not always worth it. Although they offer clear gains on complex problems (such as machine translation), they may not always be relevant to smaller, simpler problems.
