# Generation bundle for page_204  (PDF page 204, printed page 180)

Page kind: content
Chapter: 6 The universal workflow of machine learning
Section: 6.2 Developing a model
Subsection: 6.2.2 Choosing an evaluation protocol
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_204.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_204.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p204_e03] (body) converging. To make learning easier for your network, your data should have the following characteristics:
[p204_e04] (bullet) Take small values—Typically, most values should be in the 0–1 range.
[p204_e05] (bullet) Be homogeneous—That is, all features should take values in roughly the same range.
[p204_e06] (body) Additionally, the following stricter normalization practice is common and can help, although it isn’t always necessary (for example, you didn’t do this in the digit-classification example):
[p204_e07] (bullet) Normalize each feature independently to have a mean of 0.
[p204_e08] (bullet) Normalize each feature independently to have a standard deviation of 1.
[p204_e09] (body) This is easy to do with NumPy arrays:
[p204_e10] (code)
```
x -= x.mean(axis=0)
x /= x.std(axis=0)
```
[p204_e11] (heading_subsubsection) Handling missing values
[p204_e12] (body) You may sometimes have missing values in your data. For instance, in the house price example, the second feature was the median age of houses in the district. What if this feature wasn’t available for all samples? You’d then have missing values in the training or test data.
[p204_e13] (body) You could just discard the feature entirely, but you don’t necessarily have to:
[p204_e14] (bullet) If the feature is categorical, it’s safe to create a new category that means “the value is missing.” The model will automatically learn what this implies with respect to the targets.
[p204_e15] (bullet) If the feature is numerical, avoid inputting an arbitrary value like 0 because it may create a discontinuity in the latent space formed by your features, making it harder for a model trained on it to generalize. Instead, consider replacing the missing value with the average or median value for the feature in the dataset. You could also train a model to predict the feature value given the values of other features.
[p204_e16] (body) Note that if you’re expecting missing categorical features in the test data, but the network was trained on data without any missing values, the network won’t have learned to ignore missing values! In this situation, you should artificially generate training samples with missing entries: copy some training samples several times, and drop some of the categorical features that you expect are likely to be missing in the test data.
[p204_e17] (heading_subsection) 6.2.2
[p204_e18] (heading_subsection) Choosing an evaluation protocol
[p204_e19] (body) As you’ve learned in the previous chapter, the purpose of a model is to achieve generalization, and every modeling decision you will make throughout the model development process will be guided by validation metrics that seek to measure generalization

## Code annotations printed next to listings (callouts, in order)
- Assuming x is a 2D data matrix
- of shape (samples, features)

## Equation candidates (formulas written inline in the text)
- [p204_e10] x -= x.mean(axis=0)
x /= x.std(axis=0)

## Chapter outline
Chapter: 6 The universal workflow of machine learning (PDF pages 195-213)
  - 6.1 Defining the task (p.196)
      - 6.1.1 Framing the problem (p.196)
      - 6.1.2 Collecting a dataset (p.198)
      - 6.1.3 Understanding your data (p.202)
      - 6.1.4 Choosing a measure of success (p.202)
  - 6.2 Developing a model (p.203)
      - 6.2.1 Preparing the data (p.203)
      - 6.2.2 Choosing an evaluation protocol (p.204)
      - 6.2.3 Beating a baseline (p.205)
      - 6.2.4 Scaling up: Developing a model that overfits (p.206)
      - 6.2.5 Regularizing and tuning your model (p.207)
  - 6.3 Deploying your model (p.207)
      - 6.3.1 Explaining your work to stakeholders and setting expectations (p.208)
      - 6.3.2 Shipping an inference model (p.208)
      - 6.3.3 Monitoring your model in the wild (p.212)
      - 6.3.4 Maintaining your model (p.212)

## Previous page (PDF 203) text, for continuity
6.2

Developing a model

Once you know how you will measure your progress, you can get started with model development. Most tutorials and research projects assume that this is the only step— skipping problem definition and dataset collection, which are assumed to be already done, and skipping model deployment and maintenance, which is assumed to be handled by someone else. In fact, model development is only one step in the machine learning workflow, and if you ask us, it’s not the most difficult one. The hardest things in machine learning are framing problems and collecting, annotating, and cleaning data. So cheer up, what comes next will be easy in comparison!

6.2.1

Preparing the data

As you’ve learned before, deep learning models typically don’t ingest raw data. Data preprocessing aims at making the raw data at hand more amenable to neural networks. This includes vectorization, normalization, or handling missing values. Many preprocessing techniques are domain specific (for example, specific to text data or image data); we’ll cover those in the following chapters as we encounter them in practical examples. For now, we’ll review the basics that are common to all data domains.

Vectorization

All inputs and targets in a neural network must be typically tensors of floating-point data (or, in specific cases, tensors of integers or strings). Whatever data you need to process—sound, images, text—you must first turn into tensors, a step called data vectorization. For instance, in the two previous text classification examples from chapter 4, we started from text represented as lists of integers (standing for sequences of words), and we used multi-hot encoding to turn them into a tensor of `float32` data. In the examples of classifying digits and predicting house prices, th …

## Next page (PDF 205) text, for continuity
performance. The goal of your validation protocol is to accurately estimate what your success metric of choice (such as accuracy) will be on actual production data. The reliability of that process is critical to building a useful model.

In chapter 5, we reviewed three common evaluation protocols:

Maintaining a hold-out validation set—The way to go when you have plenty of data

Doing K-fold cross-validation—The right choice when you have too few samples for hold-out validation to be reliable

Doing iterated K-fold validation—For performing highly accurate model evaluation when little data is available

Just pick one of these. In most cases, the first will work well enough. As you’ve learned before, always be mindful of the representativity of your validation set(s) and be careful not to have redundant samples between your training set and your validation set(s).

6.2.3

Beating a baseline

As you start working on the model itself, your initial goal is to achieve statistical power, as you saw in chapter 5—that is, to develop a small model that is capable of beating a simple baseline.

At this stage, these are the three most important things you should focus on:

Feature engineering—Filter out uninformative features (feature selection) and use your knowledge of the problem to develop new features that are likely to be useful.

Selecting the correct architecture priors—What type of model architecture will you use? A densely connected network, a ConvNet, a recurrent neural network, a Transformer? Is deep learning even a good approach for the task, or should you use something else?

Selecting a good enough training configuration—What loss function should you use? What batch size and learning rate?

Picking the right loss function

It’s often not possible to directly optimiz …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.152 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.202 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.183 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.153 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.207 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.191 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.193 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.158 — 4 Classification and regression › 4.3 Predicting house prices: A regression example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.152 (printed 128) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Preparing the data
It would be problematic to feed into a neural network values that all take wildly different ranges. The model might be able to automatically adapt to such heterogeneous data, but it would definitely make learning more difficult. A widespread best practice to deal with such data is to do feature-wise normalization: for each feature in the input data (a column in the input data matrix), you subtract the mean of the feature and divide by the standard deviation, so that the feature is centered around 0 and has a unit standard deviation. This is easily done in NumPy.

Listing 4.25 Normalizing the data

### PDF p.202 (printed 178) — 6 The universal workflow of machine learning › 6.1 Defining the task › Understanding your data
It’s bad practice to treat a dataset as a black box. Before you start training models, you should explore and visualize your data to gain insights about what makes it predictive— which will inform feature engineering—and screen for potential issues:

If your data includes images or natural language text, take a look at a few samples (and their labels) directly.

If your data contains numerical features, it’s a good idea to plot the histogram of feature values to get a feel for the range of values taken and the frequency of different values.

If your data includes location information, plot it on a map. Do any clear patterns emerge?

Are some samples missing values for some features? If so, you’ll need to deal with this when you prepare the data (we cover how to do this in the next section).

If your task is a classification problem, print the number of instances of each class in your data. Are the classes roughly equally represented? If not, you will need to account for this imbalance.

Check for target leaking—the presence of features in your data that provide information about the targets that may not be available in production. If you’re training a model on medical records to pr

### PDF p.183 (printed 159) — 5 Fundamentals of machine learning › 5.4 Improving generalization › Dataset curation
You’ve already learned that generalization in deep learning originates from the latent structure of your data. If your data makes it possible to smoothly interpolate between samples, then you will be able to train a deep learning model that generalizes. If your problem is overly noisy or fundamentally discrete, like, say, list sorting, deep learning will not help you. Deep learning is curve fitting, not magic.

As such, it is essential that you make sure that you’re working with an appropriate dataset. Spending more effort and money on data collection almost always yields a much greater return on investment than spending the same on developing a better model:

Make sure you have enough data. Remember that you need a dense sampling of the input-cross-output space. More data will yield a better model. Sometimes, problems that seem impossible at first become solvable with a larger dataset.

Minimize labeling errors—visualize your inputs to check for anomalies, and proofread your labels.

Clean your data and deal with missing values (we cover this in the next chapter).

If you have many features and you aren’t sure which ones are actually useful, do feature selection.

A particularly i

### PDF p.152 (printed 128) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Preparing the data
Note that the quantities used for normalizing the test data are computed using the training data. You should never use in your workflow any quantity computed on the test data, even for something as simple as data normalization.

In addition, we should also scale the targets. Our normalized inputs have their value in a small range close to 0, and our model’s weights are initialized with small random values. This means that our model’s prediction will also be small values when we start training. If the targets are in the range 60,000–500,000, the model is going to need very large weight values to output those. With a small learning rate, it would take a very long time to get there. The simplest fix is to divide all target values by 100,000, so that the smallest target becomes 0.6, and the largest becomes 5. We can then convert the model’s predictions back to dollar values by multiplying them by 100,000 accordingly.

Listing 4.26 Scaling the targets

### PDF p.159 (printed 135) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Summary
The three most common kinds of machine learning tasks on vector data are binary classification, multiclass classification, and scalar regression. Each task uses different loss functions: – `binary_crossentropy` for binary classification – `categorical_crossentropy` for multiclass classification – `mean_squared_error` for scalar regression

You’ll usually need to preprocess raw data before feeding it into a neural network.

When your data has features with different ranges, scale each feature independently as part of preprocessing.

As training progresses, neural networks eventually begin to overfit and obtain worse results on never-before-seen data.

If you don’t have much training data, use a small model with only one or two intermediate layers, to avoid severe overfitting.

If your data is divided into many categories, you may cause information bottlenecks if you make the intermediate layers too small.

When you’re working with little data, K-fold validation can help reliably evaluate your model.

### PDF p.207 (printed 183) — 6 The universal workflow of machine learning › 6.3 Deploying your model › Regularizing and tuning your model
Once you’ve achieved statistical power and you’re able to overfit, you know you’re on the right path. At this point, your goal becomes to maximize generalization performance.

This phase will take the most time: you’ll repeatedly modify your model, train it, evaluate on your validation data (not the test data, at this point), modify it again, and repeat, until the model is as good as it can get. Here are some things you should try:

Try different architectures; add or remove layers.

Add dropout.

If your model is small, add L1 or L2 regularization.

Try different hyperparameters (such as the number of units per layer or the learning rate of the optimizer) to find the optimal configuration.

Optionally, iterate on data curation or feature engineering: collect and annotate more data, develop better features, or remove features that don’t seem to be informative.

It’s possible to automate a large chunk of this work by using automated hyperparameter-tuning software, such as KerasTuner. We’ll cover this in chapter 18.

Be mindful of the following: every time you use feedback from your validation process to tune your model, you leak information about the validation process into the mode

### PDF p.300 (printed 276) — 9 ConvNet architecture patterns › 9.3 Batch normalization › Batch normalization
Normalization in machine learning is a broad category of methods that seek to make different samples seen by a machine learning model more similar to each other, which helps the model learn and generalize well to new data. The most common form of data normalization is one you’ve seen several times in this book already: centering the data on zero by subtracting the mean from the data and giving the data a unit standard deviation by dividing the data by its standard deviation. In effect, this makes the assumption that the data follows a normal (or Gaussian) distribution and makes sure this distribution is centered and scaled to unit variance:

### PDF p.172 (printed 148) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › 
Figure 5.11 A dense sampling of the input space is necessary to learn a model capable of accurate generalization.

past training inputs, without having to use common-sense, abstract reasoning, or external knowledge about the world—all things that machine learning models have no access to. As such, you should always keep in mind that the best way to improve a deep learning model is to train it on more data or better data (of course, adding overly noisy or inaccurate data will harm generalization). A denser coverage of the input data manifold will yield a model that generalizes better. You should never expect a deep learning model to perform anything more than crude interpolation between its training samples, and thus, you should do everything you can to make interpolation as easy as possible. The only thing you will find in a deep learning model is what you put into it: the priors encoded in its architecture and the data it was trained on.

When getting more data isn’t possible, the next-best solution is to modulate the quantity of information that your model is allowed to store, or to add constraints on the smoothness of the model curve. If a network can only afford to memorize a s
