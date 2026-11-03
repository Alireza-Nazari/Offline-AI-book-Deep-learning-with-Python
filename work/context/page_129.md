# Generation bundle for page_129  (PDF page 129, printed page 105)

Page kind: content
Chapter: 4 Classification and regression
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_129.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_129.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p129_e04] (body) By the end of this chapter, you’ll be able to use neural networks to handle simple classification and regression tasks over vector data. You’ll then be ready to start building a more principled, theory-driven understanding of machine learning in chapter 5.
[p129_e05] (heading_subsection) Classification and regression glossary
[p129_e06] (sidebar) Classification and regression involve many specialized terms. You’ve come across some of them in earlier examples, and you’ll see more of them in future chapters. They have precise, machine learning–specific definitions, and you should be familiar with them:
[p129_e07] (bullet) Sample or input—One data point that goes into your model.
[p129_e08] (bullet) Prediction or output—What comes out of your model.
[p129_e09] (bullet) Target—The truth. What your model should ideally have predicted, according to
[p129_e10] (sidebar) an external source of data.
[p129_e11] (bullet) Prediction error or loss value—A measure of the distance between your model’s
[p129_e12] (sidebar) prediction and the target.
[p129_e13] (bullet) Classes—A set of possible labels to choose from in a classification problem.
[p129_e14] (sidebar) For example, when classifying cat and dog pictures, “dog” and “cat” are the two classes.
[p129_e15] (bullet) Label—A specific instance of a class annotation in a classification problem.
[p129_e16] (sidebar) For instance, if picture #1234 is annotated as containing the class “dog,” then “dog” is a label of picture #1234.
[p129_e17] (bullet) Ground-truth or annotations—All targets for a dataset, typically collected by
[p129_e18] (sidebar) humans.
[p129_e19] (bullet) Binary classification—A classification task where each input sample should be
[p129_e20] (sidebar) categorized into two exclusive categories.
[p129_e21] (bullet) Categorical classification or multiclass classification—A classification task
[p129_e22] (sidebar) where each input sample should be categorized into more than two categories: for instance, classifying handwritten digits.
[p129_e23] (bullet) Multilabel classification—A classification task where each input sample can
[p129_e24] (sidebar) be assigned multiple labels. For instance, a given image may contain both a cat and a dog and should be annotated with both the “cat” label and the “dog” label. The number of labels per image is usually variable.
[p129_e25] (bullet) Scalar regression—A task where the target is a continuous scalar value. Pre
[p129_e26] (sidebar) dicting house prices is a good example: the different target prices form a continuous space.
[p129_e27] (bullet) Vector regression—A task where the target is a set of continuous values: for
[p129_e28] (sidebar) example, a continuous vector. If you’re doing regression against multiple values (such as the coordinates of a bounding box in an image), then you’re doing vector regression.
[p129_e29] (bullet) Mini-batch or just batch—A small set of samples (typically between 8 and 128)
[p129_e30] (sidebar) that are processed simultaneously by the model. The number of samples is often a power of 2, to facilitate memory allocation on GPU. When training, a mini-batch is used to compute a single gradient-descent update applied to the weights of the model.

## Chapter outline
Chapter: 4 Classification and regression (PDF pages 128-159)
  - 4.1 Classifying movie reviews: A binary classification example (p.130)
      - 4.1.1 The IMDb dataset (p.130)
      - 4.1.2 Preparing the data (p.131)
      - 4.1.3 Building your model (p.132)
      - 4.1.4 Validating your approach (p.135)
      - 4.1.5 Using a trained model to generate predictions on new data (p.139)
      - 4.1.6 Further experiments (p.139)
      - 4.1.7 Wrapping up (p.140)
  - 4.2 Classifying newswires: A multiclass classification example (p.140)
      - 4.2.1 The Reuters dataset (p.140)
      - 4.2.2 Preparing the data (p.142)
      - 4.2.3 Building your model (p.142)
      - 4.2.4 Validating your approach (p.144)
      - 4.2.5 Generating predictions on new data (p.148)
      - 4.2.6 A different way to handle the labels and the loss (p.148)
      - 4.2.7 The importance of having sufficiently large intermediate layers (p.149)
      - 4.2.8 Further experiments (p.149)
      - 4.2.9 Wrapping up (p.150)
  - 4.3 Predicting house prices: A regression example (p.150)
      - 4.3.1 The California Housing Price dataset (p.150)
      - 4.3.2 Preparing the data (p.152)
      - 4.3.3 Building your model (p.152)
      - 4.3.4 Validating your approach using K-fold validation (p.153)
      - 4.3.5 Generating predictions on new data (p.158)
      - 4.3.6 Wrapping up (p.158)

## Previous page (PDF 128) text, for continuity
Classification

and regression

This chapter covers

Your first examples of real-world machine

learning workflows

Handling binary and categorical classification

problems

Handling continuous regression problems

This chapter is designed to get you started with using neural networks to solve real problems. You’ll consolidate the knowledge you gained from chapters 2 and 3, and you’ll apply what you’ve learned to three new tasks covering the three most common use cases of neural networks—binary classification, categorical classification, and scalar regression:

Classifying movie reviews as positive or negative (binary classification)

Classifying news wires by topic (categorical classification)

Estimating the price of a house, given real estate data (scalar regression)

These examples will be your first contact with end-to-end machine learning workflows: you’ll get introduced to data preprocessing, basic model architecture principles, and model evaluation.

## Next page (PDF 130) text, for continuity
4.1

Classifying movie reviews: A binary classification example

Two-class classification, or binary classification, is one of the most common kinds of machine learning problem. In this example, you’ll learn to classify movie reviews as positive or negative, based on the text content of the reviews.

4.1.1

The IMDb dataset

You’ll work with the IMDb dataset: a set of 50,000 highly polarized reviews from the Internet Movie Database. They’re split into 25,000 reviews for training and 25,000 reviews for testing, each set consisting of 50% negative and 50% positive reviews.

Just like the MNIST dataset, the IMDb dataset comes packaged with Keras. It has already been preprocessed: the reviews (sequences of words) have been turned into sequences of integers, where each integer stands for a specific word in a dictionary. This enables us to focus on model building, training, and evaluation. In chapter 14, you’ll learn how to process raw text input from scratch. The following code will load the dataset (when you run it the first time, about 80 MB of data will be downloaded to your machine).

Listing 4.1 Loading the IMDb dataset

The argument `num_words=10000` means you’ll only keep the top 10,000 most frequently occurring words in the training data. Rare words will be discarded. This allows you to work with vector data of manageable size. If we didn’t set this limit, we’d be working with 88,585 unique words in the training data, which is unnecessarily large. Many of these words only occur in a single sample, and thus can’t be meaningfully used for classification.

The variables `train_data` and `test_data` are NumPy arrays of reviews; each review is a list of word indices (encoding a sequence of words). `train_labels` and `test_labels` are NumPy arrays of 0s and 1s, where 0 sta …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.150 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.159 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.140 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.158 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.28 — 1 What is deep learning? › 1.4 Learning rules and representations from data
- PDF p.160 — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning
- PDF p.153 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.42 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.150 (printed 126) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Predicting house prices: A regression example
The two previous examples were considered classification problems, where the goal was to predict a single discrete label of an input data point. Another common type of machine learning problem is regression, which consists of predicting a continuous value instead of a discrete label: for instance, predicting the temperature tomorrow given meteorological data, or predicting the time that a software project will take to complete given its specifications.

NOTE Confusingly, logistic regression isn’t a regression algorithm—it’s a classification algorithm.

### PDF p.159 (printed 135) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Summary
The three most common kinds of machine learning tasks on vector data are binary classification, multiclass classification, and scalar regression. Each task uses different loss functions: – `binary_crossentropy` for binary classification – `categorical_crossentropy` for multiclass classification – `mean_squared_error` for scalar regression

You’ll usually need to preprocess raw data before feeding it into a neural network.

When your data has features with different ranges, scale each feature independently as part of preprocessing.

As training progresses, neural networks eventually begin to overfit and obtain worse results on never-before-seen data.

If you don’t have much training data, use a small model with only one or two intermediate layers, to avoid severe overfitting.

If your data is divided into many categories, you may cause information bottlenecks if you make the intermediate layers too small.

When you’re working with little data, K-fold validation can help reliably evaluate your model.

### PDF p.197 (printed 173) — 6 The universal workflow of machine learning › 6.1 Defining the task › 
What will your input data be? What are you trying to predict? You can only learn to predict something if you have available training data: for example, you can only learn to classify the sentiment of movie reviews if you have both movie reviews and sentiment annotations available. As such, data availability is usually the limiting factor at this stage. In many cases, you will have to resort to collecting and annotating new datasets yourself (which we cover in the next section).

What type of machine learning task are you facing? Is it binary classification? Multiclass classification? Scalar regression? Vector regression? Multiclass, multilabel classification? Image segmentation? Ranking? Something else, like clustering, generation, or reinforcement learning? In some cases, it may be that machine learning isn’t even the best way to make sense of your data, and you should use something else, such as plain old-school statistical analysis: – The photo search engine project is a multiclass, multilabel classification task. – The spam detection project is a binary classification task. If you set “offensive content” as a separate class, it’s a three-way classification task. – The music rec

### PDF p.140 (printed 116) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › Classifying newswires: A multiclass classification example
In the previous section, you saw how to classify vector inputs into two mutually exclusive classes using a densely connected neural network. But what happens when you have more than two classes?

In this section, you’ll build a model to classify Reuters newswires into 46 mutually exclusive topics. Because you have many classes, this problem is an instance of multiclass classification, and because each data point should be classified into only one category, the problem is more specifically an instance of single-label, multiclass classification. If each data point could belong to multiple categories (in this case, topics), you’d be facing a multilabel, multiclass classification problem.

### PDF p.158 (printed 134) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Generating predictions on new data
When calling `predict()` on our binary classification model, we retrieved a scalar score between 0 and 1 for each input sample. With our multiclass classification model, we retrieved a probability distribution over all classes for each sample. Now, with this scalar regression model, `predict()` returns the model’s guess for the sample’s price in hundreds of thousands of dollars:

### PDF p.150 (printed 126) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Wrapping up
Here’s what you should take away from this example:

If you’re trying to classify data points among N classes, your model should end with a `Dense` layer of size N.

In a single-label, multiclass classification problem, your model should end with a `softmax` activation so that it will output a probability distribution over the N output classes.

Categorical crossentropy is almost always the loss function you should use for such problems. It minimizes the distance between the probability distributions output by the model and the true distribution of the targets.

There are two ways to handle labels in multiclass classification: – Encoding the labels via categorical encoding (also known as one-hot encoding) and using `categorical_crossentropy` as a loss function – Encoding the labels as integers and using the `sparse_categorical_` `crossentropy` loss function

If you need to classify data into a large number of categories, you should avoid creating information bottlenecks in your model due to intermediate layers that are too small.

### PDF p.42 (printed 18) — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network › 
NOTE In machine learning, a category in a classification problem is called a class. Data points are called samples. The class associated with a specific sample is called

Figure 2.1 MNIST sample digits

a label.

The MNIST dataset comes preloaded in Keras, in the form of a set of four NumPy arrays.

Listing 2.1 Loading the MNIST dataset in Keras

### PDF p.28 (printed 4) — 1 What is deep learning? › 1.4 Learning rules and representations from data › Learning rules and representations from data
To define deep learning and understand the difference between deep learning and other machine learning approaches, first we need some idea of what machine learning algorithms do. We just stated that machine learning discovers rules to execute a data processing task, given examples of what’s expected. So, to do machine learning, we need three things:

Input data points—For instance, if the task is speech recognition, these data points could be sound files of people speaking. If the task is image tagging, they could be pictures.

Examples of the expected output—In a speech-recognition task, these could be human-generated transcripts of sound files. In an image task, expected outputs could be tags such as “dog,” “cat,” and so on.

A way to measure whether the algorithm is doing a good job—This is necessary to determine the distance between the algorithm’s current output and its expected
