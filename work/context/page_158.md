# Generation bundle for page_158  (PDF page 158, printed page 134)

Page kind: content
Chapter: 4 Classification and regression
Section: 4.3 Predicting house prices: A regression example
Subsection: 4.3.6 Wrapping up
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_158.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_158.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p158_e03] (body) Here’s the final result:
[p158_e04] (code)
```
>>> round(test_mean_absolute_error, 3)
0.31
```
[p158_e05] (body) We’re still off by about $31,000 on average.
[p158_e06] (heading_subsection) 4.3.5
[p158_e07] (heading_subsection) Generating predictions on new data
[p158_e08] (body) When calling `predict()` on our binary classification model, we retrieved a scalar score between 0 and 1 for each input sample. With our multiclass classification model, we retrieved a probability distribution over all classes for each sample. Now, with this scalar regression model, `predict()` returns the model’s guess for the sample’s price in hundreds of thousands of dollars:
[p158_e09] (code)
```
>>> predictions = model.predict(x_test)
>>> predictions[0]
array([2.834494], dtype=float32)
```
[p158_e10] (body) The first district in the test set is predicted to have an average home price of about $283,000.
[p158_e11] (heading_subsection) 4.3.6
[p158_e12] (heading_subsection) Wrapping up
[p158_e13] (body) Here’s what you should take away from this scalar regression example:
[p158_e14] (bullet) Regression is done using a different loss function than what we used for classification. Mean squared error (MSE) is a loss function commonly used for regression.
[p158_e15] (bullet) Similarly, evaluation metrics to be used for regression differ from those used for classification; naturally, the concept of accuracy doesn’t apply for regression. A common regression metric is MAE.
[p158_e16] (bullet) When features in the input data have values in different ranges, each feature should be scaled independently as a preprocessing step.
[p158_e17] (bullet) When there is little data available, using K-fold validation is a great way to reliably evaluate a model.
[p158_e18] (bullet) When little training data is available, it’s preferable to use a small model with few intermediate layers (typically only one or two), in order to avoid severe overfitting.

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

## Previous page (PDF 157) text, for continuity
According to this plot (see figure 4.11), validation MAE stops improving significantly after 120–140 epochs (this number includes the 10 epochs we omitted). Past that point, you start overfitting.

Once you’re finished tuning other parameters of the model (in addition to the number of epochs, you could also adjust the size of the intermediate layers), you can train a final production model on all of the training data, with the best parameters, and then look at its performance on the test data.

Figure 4.11 Validation MAE by epoch, excluding the first 10 data points

Listing 4.33 Training the final model

## Next page (PDF 159) text, for continuity
Summary

The three most common kinds of machine learning tasks on vector data are binary classification, multiclass classification, and scalar regression. Each task uses different loss functions: – `binary_crossentropy` for binary classification – `categorical_crossentropy` for multiclass classification – `mean_squared_error` for scalar regression

You’ll usually need to preprocess raw data before feeding it into a neural network.

When your data has features with different ranges, scale each feature independently as part of preprocessing.

As training progresses, neural networks eventually begin to overfit and obtain worse results on never-before-seen data.

If you don’t have much training data, use a small model with only one or two intermediate layers, to avoid severe overfitting.

If your data is divided into many categories, you may cause information bottlenecks if you make the intermediate layers too small.

When you’re working with little data, K-fold validation can help reliably evaluate your model.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.153 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.148 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.129 — 4 Classification and regression › 
- PDF p.150 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.139 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.140 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.143 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.424 — 14 Text classification › 14.4 Set models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.153 (printed 129) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › 
The model ends with a single unit and no activation: it will be a linear layer. This is a typical setup for scalar regression—a regression where you’re trying to predict a single continuous value. Applying an activation function would constrain the range the output can take; for instance, if you applied a `sigmoid` activation function to the last layer, the model could only learn to predict values between 0 and 1. Here, because the last layer is purely linear, the model is free to learn to predict values in any range.

Note that you compile the model with the `mean_squared_error` loss function—mean squared error, the square of the difference between the predictions and the targets. This is a widely used loss function for regression problems. You’re also monitoring a new metric during training: mean absolute error (MAE). It’s the absolute value of the difference between the predictions and the targets. For instance, an MAE of 0.5 on this problem would mean your predictions are off by $50,000 on average (remember the target scaling of factor 100,000).

### PDF p.626 (printed 602) — 20 Conclusions › 20.1 Key concepts in review › 
To perform regression toward a vector of continuous values, end your stack of layers with a `Dense` layer with a number of units equal to the number of values you’re trying to predict (often a single one, such as the price of a house) and no activation. Various losses can be used for regression—most commonly `mean_squared_error` (MSE):

### PDF p.150 (printed 126) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Predicting house prices: A regression example
The two previous examples were considered classification problems, where the goal was to predict a single discrete label of an input data point. Another common type of machine learning problem is regression, which consists of predicting a continuous value instead of a discrete label: for instance, predicting the temperature tomorrow given meteorological data, or predicting the time that a software project will take to complete given its specifications.

NOTE Confusingly, logistic regression isn’t a regression algorithm—it’s a classification algorithm.

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

### PDF p.129 (printed 105) — 4 Classification and regression ›  › Classification and regression glossary
dicting house prices is a good example: the different target prices form a continuous space.

Vector regression—A task where the target is a set of continuous values: for

example, a continuous vector. If you’re doing regression against multiple values (such as the coordinates of a bounding box in an image), then you’re doing vector regression.

Mini-batch or just batch—A small set of samples (typically between 8 and 128)

that are processed simultaneously by the model. The number of samples is often a power of 2, to facilitate memory allocation on GPU. When training, a mini-batch is used to compute a single gradient-descent update applied to the weights of the model.

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

### PDF p.160 (printed 136) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › Generalization: The goal of machine learning
In the three examples presented in chapter 4—predicting movie reviews, topic classification, and house-price regression—we split the data into a training set, a
