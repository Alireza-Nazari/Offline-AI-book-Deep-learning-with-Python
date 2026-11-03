# Generation bundle for page_159  (PDF page 159, printed page 135)

Page kind: content
Chapter: 4 Classification and regression
Section: 4.3 Predicting house prices: A regression example
Subsection: 4.3.6 Wrapping up
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_159.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_159.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p159_e03] (heading_section) Summary
[p159_e04] (bullet) The three most common kinds of machine learning tasks on vector data are binary classification, multiclass classification, and scalar regression. Each task uses different loss functions: – `binary_crossentropy` for binary classification – `categorical_crossentropy` for multiclass classification – `mean_squared_error` for scalar regression
[p159_e05] (bullet) You’ll usually need to preprocess raw data before feeding it into a neural network.
[p159_e06] (bullet) When your data has features with different ranges, scale each feature independently as part of preprocessing.
[p159_e07] (bullet) As training progresses, neural networks eventually begin to overfit and obtain worse results on never-before-seen data.
[p159_e08] (bullet) If you don’t have much training data, use a small model with only one or two intermediate layers, to avoid severe overfitting.
[p159_e09] (bullet) If your data is divided into many categories, you may cause information bottlenecks if you make the intermediate layers too small.
[p159_e10] (bullet) When you’re working with little data, K-fold validation can help reliably evaluate your model.

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

## Previous page (PDF 158) text, for continuity
Here’s the final result:

We’re still off by about $31,000 on average.

4.3.5

Generating predictions on new data

When calling `predict()` on our binary classification model, we retrieved a scalar score between 0 and 1 for each input sample. With our multiclass classification model, we retrieved a probability distribution over all classes for each sample. Now, with this scalar regression model, `predict()` returns the model’s guess for the sample’s price in hundreds of thousands of dollars:

The first district in the test set is predicted to have an average home price of about $283,000.

4.3.6

Wrapping up

Here’s what you should take away from this scalar regression example:

Regression is done using a different loss function than what we used for classification. Mean squared error (MSE) is a loss function commonly used for regression.

Similarly, evaluation metrics to be used for regression differ from those used for classification; naturally, the concept of accuracy doesn’t apply for regression. A common regression metric is MAE.

When features in the input data have values in different ranges, each feature should be scaled independently as a preprocessing step.

When there is little data available, using K-fold validation is a great way to reliably evaluate a model.

When little training data is available, it’s preferable to use a small model with few intermediate layers (typically only one or two), in order to avoid severe overfitting.

## Next page (PDF 160) text, for continuity
Fundamentals of

machine learning

This chapter covers

Understanding the tension between

generalization and optimization, the fundamental

issue in machine learning

Evaluation methods for machine learning models

Best practices to improve model fitting

Best practices to achieve better generalization

After the three practical examples in chapter 4, you should be starting to feel familiar with how to approach classification and regression problems using neural networks, and you’ve witnessed the central problem of machine learning: overfitting. This chapter will formalize some of your new intuition about machine learning into a solid conceptual framework, highlighting the importance of accurate model evaluation and the balance between training and generalization.

5.1

Generalization: The goal of machine learning

In the three examples presented in chapter 4—predicting movie reviews, topic classification, and house-price regression—we split the data into a training set, a

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.140 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.129 — 4 Classification and regression › 
- PDF p.128 — 4 Classification and regression › 
- PDF p.152 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.150 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.203 — 6 The universal workflow of machine learning › 6.2 Developing a model
- PDF p.206 — 6 The universal workflow of machine learning › 6.2 Developing a model
- PDF p.10 — contents › 

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.140 (printed 116) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › Wrapping up
Here’s what you should take away from this example:

You usually need to do quite a bit of preprocessing on your raw data to be able to feed it—as tensors—into a neural network. Sequences of words can be encoded as binary vectors, but there are other encoding options, too.

Stacks of `Dense` layers with `relu` activations can solve a wide range of problems (including sentiment classification), and you’ll use them frequently.

In a binary classification problem (two output classes), your model should end with a `Dense` layer with one unit and a `sigmoid` activation: the output of your model should be a scalar between 0 and 1, encoding a probability.

With such a scalar sigmoid output on a binary classification problem, the loss function you should use is `binary_crossentropy`.

The `adam` optimizer is generally a good enough choice, whatever your problem. That’s one less thing for you to worry about.

As they get better on their training data, neural networks eventually start overfitting and end up obtaining increasingly worse results on data they’ve never seen before. Be sure to always monitor performance on data that is outside of the training set!

### PDF p.128 (printed 104) — 4 Classification and regression ›  › This chapter covers
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

### PDF p.197 (printed 173) — 6 The universal workflow of machine learning › 6.1 Defining the task › 
What will your input data be? What are you trying to predict? You can only learn to predict something if you have available training data: for example, you can only learn to classify the sentiment of movie reviews if you have both movie reviews and sentiment annotations available. As such, data availability is usually the limiting factor at this stage. In many cases, you will have to resort to collecting and annotating new datasets yourself (which we cover in the next section).

What type of machine learning task are you facing? Is it binary classification? Multiclass classification? Scalar regression? Vector regression? Multiclass, multilabel classification? Image segmentation? Ranking? Something else, like clustering, generation, or reinforcement learning? In some cases, it may be that machine learning isn’t even the best way to make sense of your data, and you should use something else, such as plain old-school statistical analysis: – The photo search engine project is a multiclass, multilabel classification task. – The spam detection project is a binary classification task. If you set “offensive content” as a separate class, it’s a three-way classification task. – The music rec

### PDF p.150 (printed 126) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Wrapping up
Here’s what you should take away from this example:

If you’re trying to classify data points among N classes, your model should end with a `Dense` layer of size N.

In a single-label, multiclass classification problem, your model should end with a `softmax` activation so that it will output a probability distribution over the N output classes.

Categorical crossentropy is almost always the loss function you should use for such problems. It minimizes the distance between the probability distributions output by the model and the true distribution of the targets.

There are two ways to handle labels in multiclass classification: – Encoding the labels via categorical encoding (also known as one-hot encoding) and using `categorical_crossentropy` as a loss function – Encoding the labels as integers and using the `sparse_categorical_` `crossentropy` loss function

If you need to classify data into a large number of categories, you should avoid creating information bottlenecks in your model due to intermediate layers that are too small.

### PDF p.129 (printed 105) — 4 Classification and regression ›  › 
By the end of this chapter, you’ll be able to use neural networks to handle simple classification and regression tasks over vector data. You’ll then be ready to start building a more principled, theory-driven understanding of machine learning in chapter 5.

### PDF p.129 (printed 105) — 4 Classification and regression ›  › Classification and regression glossary
Classification and regression involve many specialized terms. You’ve come across some of them in earlier examples, and you’ll see more of them in future chapters. They have precise, machine learning–specific definitions, and you should be familiar with them:

Sample or input—One data point that goes into your model.

Prediction or output—What comes out of your model.

Target—The truth. What your model should ideally have predicted, according to

an external source of data.

Prediction error or loss value—A measure of the distance between your model’s

prediction and the target.

Classes—A set of possible labels to choose from in a classification problem.

For example, when classifying cat and dog pictures, “dog” and “cat” are the two classes.

Label—A specific instance of a class annotation in a classification problem.

For instance, if picture #1234 is annotated as containing the class “dog,” then “dog” is a label of picture #1234.

Ground-truth or annotations—All targets for a dataset, typically collected by

humans.

Binary classification—A classification task where each input sample should be

categorized into two exclusive categories.

Categorical classification or multiclass 

### PDF p.150 (printed 126) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Predicting house prices: A regression example
The two previous examples were considered classification problems, where the goal was to predict a single discrete label of an input data point. Another common type of machine learning problem is regression, which consists of predicting a continuous value instead of a discrete label: for instance, predicting the temperature tomorrow given meteorological data, or predicting the time that a software project will take to complete given its specifications.

NOTE Confusingly, logistic regression isn’t a regression algorithm—it’s a classification algorithm.

### PDF p.124 (printed 100) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
correct loss. For instance, you’ll use binary crossentropy for a two-class classification problem, categorical crossentropy for a many-class classification problem, and so on. Only when you’re working on truly new research problems will you have to develop your own loss functions. In the next few chapters, we’ll detail explicitly which loss functions to choose for a wide range of common tasks.
