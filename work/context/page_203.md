# Generation bundle for page_203  (PDF page 203, printed page 179)

Page kind: content
Chapter: 6 The universal workflow of machine learning
Section: 6.2 Developing a model
Subsection: 6.2.1 Preparing the data
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_203.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_203.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p203_e03] (heading_section) 6.2
[p203_e04] (heading_section) Developing a model
[p203_e05] (body) Once you know how you will measure your progress, you can get started with model development. Most tutorials and research projects assume that this is the only step— skipping problem definition and dataset collection, which are assumed to be already done, and skipping model deployment and maintenance, which is assumed to be handled by someone else. In fact, model development is only one step in the machine learning workflow, and if you ask us, it’s not the most difficult one. The hardest things in machine learning are framing problems and collecting, annotating, and cleaning data. So cheer up, what comes next will be easy in comparison!
[p203_e06] (heading_subsection) 6.2.1
[p203_e07] (heading_subsection) Preparing the data
[p203_e08] (body) As you’ve learned before, deep learning models typically don’t ingest raw data. Data preprocessing aims at making the raw data at hand more amenable to neural networks. This includes vectorization, normalization, or handling missing values. Many preprocessing techniques are domain specific (for example, specific to text data or image data); we’ll cover those in the following chapters as we encounter them in practical examples. For now, we’ll review the basics that are common to all data domains.
[p203_e09] (heading_subsubsection) Vectorization
[p203_e10] (body) All inputs and targets in a neural network must be typically tensors of floating-point data (or, in specific cases, tensors of integers or strings). Whatever data you need to process—sound, images, text—you must first turn into tensors, a step called data vectorization. For instance, in the two previous text classification examples from chapter 4, we started from text represented as lists of integers (standing for sequences of words), and we used multi-hot encoding to turn them into a tensor of `float32` data. In the examples of classifying digits and predicting house prices, the data already came in vectorized form, so you were able to skip this step.
[p203_e11] (heading_subsubsection) Value normalization
[p203_e12] (body) In the MNIST digit-classification example from chapter 2, you started from image data encoded as integers in the 0–255 range, encoding grayscale values. Before you fed this data into your network, you had to cast it to `float32` and divide by 255 so you’d end up with floating-point values in the 0–1 range. Similarly, when predicting house prices, you started from features that took a variety of ranges—some features had small floating-point values, others had fairly large integer values. Before you fed this data into your network, you had to normalize each feature independently so that it had a standard deviation of 1 and a mean of 0.
[p203_e13] (body) In general, it isn’t safe to feed into a neural network data that takes relatively large values (for example, multi-digit integers, which are much larger than the initial values taken by the weights of a network) or data that is heterogeneous (for example, data where one feature is in the range 0–1, and another is in the range 100–200). Doing so can trigger large gradient updates that will prevent the network from

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

## Previous page (PDF 202) text, for continuity
6.1.3

Understanding your data

It’s bad practice to treat a dataset as a black box. Before you start training models, you should explore and visualize your data to gain insights about what makes it predictive— which will inform feature engineering—and screen for potential issues:

If your data includes images or natural language text, take a look at a few samples (and their labels) directly.

If your data contains numerical features, it’s a good idea to plot the histogram of feature values to get a feel for the range of values taken and the frequency of different values.

If your data includes location information, plot it on a map. Do any clear patterns emerge?

Are some samples missing values for some features? If so, you’ll need to deal with this when you prepare the data (we cover how to do this in the next section).

If your task is a classification problem, print the number of instances of each class in your data. Are the classes roughly equally represented? If not, you will need to account for this imbalance.

Check for target leaking—the presence of features in your data that provide information about the targets that may not be available in production. If you’re training a model on medical records to predict whether someone will be treated for cancer in the future, and the records include the feature “This person has been diagnosed with cancer,” then your targets are being artificially leaked into your data. Always ask yourself, is every feature in your data something that will be available in the same form in production?

6.1.4

Choosing a measure of success

To control something, you need to be able to observe it. To achieve success on a project, you must first define what you mean by success. Accuracy? Precision and recall? Customer retention rate? Your met …

## Next page (PDF 204) text, for continuity
converging. To make learning easier for your network, your data should have the following characteristics:

Take small values—Typically, most values should be in the 0–1 range.

Be homogeneous—That is, all features should take values in roughly the same range.

Additionally, the following stricter normalization practice is common and can help, although it isn’t always necessary (for example, you didn’t do this in the digit-classification example):

Normalize each feature independently to have a mean of 0.

Normalize each feature independently to have a standard deviation of 1.

This is easy to do with NumPy arrays:

Handling missing values

You may sometimes have missing values in your data. For instance, in the house price example, the second feature was the median age of houses in the district. What if this feature wasn’t available for all samples? You’d then have missing values in the training or test data.

You could just discard the feature entirely, but you don’t necessarily have to:

If the feature is categorical, it’s safe to create a new category that means “the value is missing.” The model will automatically learn what this implies with respect to the targets.

If the feature is numerical, avoid inputting an arbitrary value like 0 because it may create a discontinuity in the latent space formed by your features, making it harder for a model trained on it to generalize. Instead, consider replacing the missing value with the average or median value for the feature in the dataset. You could also train a model to predict the feature value given the values of other features.

Note that if you’re expecting missing categorical features in the test data, but the network was trained on data without any missing values, the network won’t have learned to ignore missing va …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.152 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.183 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.159 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.623 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.142 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.132 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.75 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.44 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.152 (printed 128) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Preparing the data
It would be problematic to feed into a neural network values that all take wildly different ranges. The model might be able to automatically adapt to such heterogeneous data, but it would definitely make learning more difficult. A widespread best practice to deal with such data is to do feature-wise normalization: for each feature in the input data (a column in the input data matrix), you subtract the mean of the feature and divide by the standard deviation, so that the feature is centered around 0 and has a unit standard deviation. This is easily done in NumPy.

Listing 4.25 Normalizing the data

### PDF p.152 (printed 128) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Preparing the data
Note that the quantities used for normalizing the test data are computed using the training data. You should never use in your workflow any quantity computed on the test data, even for something as simple as data normalization.

In addition, we should also scale the targets. Our normalized inputs have their value in a small range close to 0, and our model’s weights are initialized with small random values. This means that our model’s prediction will also be small values when we start training. If the targets are in the range 60,000–500,000, the model is going to need very large weight values to output those. With a small learning rate, it would take a very long time to get there. The simplest fix is to divide all target values by 100,000, so that the smallest target becomes 0.6, and the largest becomes 5. We can then convert the model’s predictions back to dollar values by multiplying them by 100,000 accordingly.

Listing 4.26 Scaling the targets

### PDF p.131 (printed 107) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › Preparing the data
You can’t directly feed lists of integers into a neural network. They have all different lengths, while a neural network expects to process contiguous batches of data. You have to turn your lists into tensors. There are two ways to do that:

Pad your lists so that they all have the same length, then turn them into an integer tensor of shape `(samples,` `max_length)`, and start your model with a layer capable of handling such integer tensors (the `Embedding` layer, which we’ll cover in detail later in the book).

Multi-hot encode your lists to turn them into vectors of 0s and 1s reflecting the presence or absence of all possible words. This would mean, for instance, turning the sequence `[8, 5]` into a 10,000-dimensional vector that would be all 0s except for indices 5 and 8, which would be 1s.

Let’s go with the latter solution to vectorize the data. When done manually, the process looks like the following.

### PDF p.159 (printed 135) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Summary
The three most common kinds of machine learning tasks on vector data are binary classification, multiclass classification, and scalar regression. Each task uses different loss functions: – `binary_crossentropy` for binary classification – `categorical_crossentropy` for multiclass classification – `mean_squared_error` for scalar regression

You’ll usually need to preprocess raw data before feeding it into a neural network.

When your data has features with different ranges, scale each feature independently as part of preprocessing.

As training progresses, neural networks eventually begin to overfit and obtain worse results on never-before-seen data.

If you don’t have much training data, use a small model with only one or two intermediate layers, to avoid severe overfitting.

If your data is divided into many categories, you may cause information bottlenecks if you make the intermediate layers too small.

When you’re working with little data, K-fold validation can help reliably evaluate your model.

### PDF p.183 (printed 159) — 5 Fundamentals of machine learning › 5.4 Improving generalization › Dataset curation
You’ve already learned that generalization in deep learning originates from the latent structure of your data. If your data makes it possible to smoothly interpolate between samples, then you will be able to train a deep learning model that generalizes. If your problem is overly noisy or fundamentally discrete, like, say, list sorting, deep learning will not help you. Deep learning is curve fitting, not magic.

As such, it is essential that you make sure that you’re working with an appropriate dataset. Spending more effort and money on data collection almost always yields a much greater return on investment than spending the same on developing a better model:

Make sure you have enough data. Remember that you need a dense sampling of the input-cross-output space. More data will yield a better model. Sometimes, problems that seem impossible at first become solvable with a larger dataset.

Minimize labeling errors—visualize your inputs to check for anomalies, and proofread your labels.

Clean your data and deal with missing values (we cover this in the next chapter).

If you have many features and you aren’t sure which ones are actually useful, do feature selection.

A particularly i

### PDF p.10 (printed x) — contents ›  › 
3.6

Introduction to Keras 90

First steps with Keras 91 ■ Layers: The building blocks of deep learning 92 ■ From layers to models 96 ■ The “compile” step: Configuring the learning process 97 ■ Picking a loss function 99 Understanding the fit method 100 ■ Monitoring loss and metrics on validation data 101 ■ Inference: Using a model after training 102

4

Classification and regression 104 4.1

Classifying movie reviews: A binary classification example 106

The IMDb dataset 106 ■ Preparing the data 107 ■ Building your model 108 ■ Validating your approach 111 ■ Using a trained model to generate predictions on new data 115 Further experiments 115 ■ Wrapping up 116

4.2

Classifying newswires: A multiclass classification example 116

The Reuters dataset 116 ■ Preparing the data 118 ■ Building your model 118 ■ Validating your approach 120 ■ Generating predictions on new data 124 ■ A different way to handle the labels and the loss 124 ■ The importance of having sufficiently large intermediate layers 125 ■ Further experiments 125 Wrapping up 126

4.3

Predicting house prices: A regression example 126

The California Housing Price dataset 126 ■ Preparing the data 128 ■ Building your model 1

### PDF p.380 (printed 356) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 13.2.1 Preparing the data
The exact formulation of the problem will be as follows: given data covering the previous five days and sampled once per hour, can we predict the temperature in 24 hours?

First, let’s preprocess the data to a format a neural network can ingest. This is easy: the data is already numerical, so you don’t need to do any vectorization. But each timeseries in the data is on a different scale (for example, atmospheric pressure, measured in mbar, is around 1,000, while H2OC, measured in millimoles per mole, is around 3). We’ll normalize each timeseries independently so that they all take small values on a similar scale. We’re going to use the first 210,225 timesteps as training data, so we’ll compute the mean and standard deviation only on this fraction of the data.

Listing 13.6 Normalizing the data

### PDF p.623 (printed 599) — 20 Conclusions › 20.1 Key concepts in review › 20.1.5 The universal machine learning workflow
Having access to an extremely powerful tool for creating models that map any input space to any target space is great, but the difficult part of the machine learning workflow is often everything that comes before designing and training such models (and, for production models, what comes after, as well). Understanding the problem domain to be able to determine what to attempt to predict, given what data, and how to measure success is a prerequisite for any successful application of machine learning, and it isn’t something that advanced tools like Keras and TensorFlow can help you with. As a reminder, here’s a quick summary of the typical machine learning workflow as described in chapter 6:

Define the problem. What data is available, and what are you trying to predict? Will you need to collect more data or hire people to manually label a dataset?

Identify a way to reliably measure success on your goal. For simple tasks, this may be prediction accuracy, but in many cases, it will require sophisticated, domain-specific metrics.

Prepare the validation process that you’ll use to evaluate your models. In particular, you should define a training set, a validation set, and a test set. Th
