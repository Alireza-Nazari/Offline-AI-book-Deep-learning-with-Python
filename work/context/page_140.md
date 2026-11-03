# Generation bundle for page_140  (PDF page 140, printed page 116)

Page kind: content
Chapter: 4 Classification and regression
Section: 4.2 Classifying newswires: A multiclass classification example
Subsection: 4.2.1 The Reuters dataset
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_140.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_140.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p140_e03] (bullet) Try using layers with more units or fewer units: 32 units, 64 units, and so on.
[p140_e04] (bullet) Try using the `mean_squared_error` loss function instead of `binary_crossentropy`.
[p140_e05] (bullet) Try using the `tanh` activation (an activation that was popular in the early days of neural networks) instead of `relu`.
[p140_e06] (heading_subsection) 4.1.7
[p140_e07] (heading_subsection) Wrapping up
[p140_e08] (body) Here’s what you should take away from this example:
[p140_e09] (bullet) You usually need to do quite a bit of preprocessing on your raw data to be able to feed it—as tensors—into a neural network. Sequences of words can be encoded as binary vectors, but there are other encoding options, too.
[p140_e10] (bullet) Stacks of `Dense` layers with `relu` activations can solve a wide range of problems (including sentiment classification), and you’ll use them frequently.
[p140_e11] (bullet) In a binary classification problem (two output classes), your model should end with a `Dense` layer with one unit and a `sigmoid` activation: the output of your model should be a scalar between 0 and 1, encoding a probability.
[p140_e12] (bullet) With such a scalar sigmoid output on a binary classification problem, the loss function you should use is `binary_crossentropy`.
[p140_e13] (bullet) The `adam` optimizer is generally a good enough choice, whatever your problem. That’s one less thing for you to worry about.
[p140_e14] (bullet) As they get better on their training data, neural networks eventually start overfitting and end up obtaining increasingly worse results on data they’ve never seen before. Be sure to always monitor performance on data that is outside of the training set!
[p140_e15] (heading_section) 4.2
[p140_e16] (heading_section) Classifying newswires: A multiclass classification example
[p140_e17] (body) In the previous section, you saw how to classify vector inputs into two mutually exclusive classes using a densely connected neural network. But what happens when you have more than two classes?
[p140_e18] (body) In this section, you’ll build a model to classify Reuters newswires into 46 mutually exclusive topics. Because you have many classes, this problem is an instance of multiclass classification, and because each data point should be classified into only one category, the problem is more specifically an instance of single-label, multiclass classification. If each data point could belong to multiple categories (in this case, topics), you’d be facing a multilabel, multiclass classification problem.
[p140_e19] (heading_subsection) 4.2.1
[p140_e20] (heading_subsection) The Reuters dataset
[p140_e21] (body) You’ll work with the Reuters dataset, a set of short newswires and their topics, published by Reuters in 1986. It’s a simple, widely used toy dataset for text classification. There are 46 different topics; some topics are more represented than others, but each topic has at least 10 examples in the training set.

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

## Previous page (PDF 139) text, for continuity
The final results are as follows:

This fairly naive approach achieves an accuracy of 88%. With state-of-the-art approaches, you should be able to get close to 95%.

4.1.5

Using a trained model to generate predictions on new data

After having trained a model, you’ll want to use it in a practical setting. You can generate the likelihood of reviews being positive by using the `predict` method, as you’ve learned in chapter 3:

As you can see, the model is confident for some samples (0.99 or more, or 0.01 or less) but less confident for others (0.6, 0.4).

4.1.6

Further experiments

The following experiments will help convince you that the architecture choices you’ve made are all fairly reasonable, although there’s still room for improvement:

You used two representation layers before the final classification layer. Try using one or three representation layers and see how doing so affects validation and test accuracy.

## Next page (PDF 141) text, for continuity
Like IMDb and MNIST, the Reuters dataset comes packaged as part of Keras. Let’s take a look.

Listing 4.11 Loading the Reuters dataset

As with the IMDb dataset, the argument `num_words=10000` restricts the data to the 10,000 most frequently occurring words found in the data.

You have 8,982 training examples and 2,246 test examples:

As with the IMDb reviews, each example is a list of integers (word indices):

Here’s how you can decode it back to words, in case you’re curious.

Listing 4.12 Decoding newswires back to text

The label associated with an example is an integer between 0 and 45—a topic index:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.150 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.132 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.134 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.142 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.159 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.129 — 4 Classification and regression › 
- PDF p.625 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.147 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.150 (printed 126) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Wrapping up
Here’s what you should take away from this example:

If you’re trying to classify data points among N classes, your model should end with a `Dense` layer of size N.

In a single-label, multiclass classification problem, your model should end with a `softmax` activation so that it will output a probability distribution over the N output classes.

Categorical crossentropy is almost always the loss function you should use for such problems. It minimizes the distance between the probability distributions output by the model and the true distribution of the targets.

There are two ways to handle labels in multiclass classification: – Encoding the labels via categorical encoding (also known as one-hot encoding) and using `categorical_crossentropy` as a loss function – Encoding the labels as integers and using the `sparse_categorical_` `crossentropy` loss function

If you need to classify data into a large number of categories, you should avoid creating information bottlenecks in your model due to intermediate layers that are too small.

### PDF p.159 (printed 135) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Summary
The three most common kinds of machine learning tasks on vector data are binary classification, multiclass classification, and scalar regression. Each task uses different loss functions: – `binary_crossentropy` for binary classification – `categorical_crossentropy` for multiclass classification – `mean_squared_error` for scalar regression

You’ll usually need to preprocess raw data before feeding it into a neural network.

When your data has features with different ranges, scale each feature independently as part of preprocessing.

As training progresses, neural networks eventually begin to overfit and obtain worse results on never-before-seen data.

If you don’t have much training data, use a small model with only one or two intermediate layers, to avoid severe overfitting.

If your data is divided into many categories, you may cause information bottlenecks if you make the intermediate layers too small.

When you’re working with little data, K-fold validation can help reliably evaluate your model.

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

### PDF p.142 (printed 118) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › Building your model
This topic classification problem looks similar to the previous movie review classification problem: in both cases, you’re trying to classify short snippets of text. But there is a new constraint here: the number of output classes has gone from 2 to 46. The dimensionality of the output space is much larger.

In a stack of `Dense` layers like those you’ve been using, each layer can only access information present in the output of the previous layer. If one layer drops some information relevant to the classification problem, this information can never be

### PDF p.134 (printed 110) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › 
So the layer could only learn linear transformations (affine transformations) of the input data: the hypothesis space of the layer would be the set of all possible linear transformations of the input data into a 16-dimensional space. Such a hypothesis space is too restricted and wouldn’t benefit from multiple layers of representations because a deep stack of linear layers would still implement a linear operation: adding more layers wouldn’t extend the hypothesis space (as you saw in chapter 2). To get access to a much richer hypothesis space that would benefit from deep representations, you need a non-linearity or activation function. `relu` is the most popular activation function in deep learning, but there are many other candidates, which all come with similarly strange names: `prelu`, `elu`, and so on.

Figure 4.2 The rectified linear unit function Figure 4.3 The sigmoid function

Finally, you need to choose a loss function and an optimizer. Because you’re facing a binary classification problem and the output of your model is a probability (you end your model with a single-unit layer with a sigmoid activation), it’s best to use the `binary_crossentropy` loss. It isn’t the only v

### PDF p.132 (printed 108) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › Building your model
The input data is vectors, and the labels are scalars (1s and 0s): this is one of the simplest problem setups you’ll ever encounter. A type of model that performs well on such a problem is a plain stack of densely connected (`Dense`) layers with `relu` activations.

There are two key architecture decisions to be made about such a stack of `Dense` layers:

How many layers to use

How many units to choose for each layer

In chapter 5, you’ll learn formal principles to guide you in making these choices. For the time being, you’ll have to trust us with the following architecture choice:

Two intermediate layers with 16 units each

A third layer that will output the scalar prediction regarding the sentiment of the current review

### PDF p.625 (printed 601) — 20 Conclusions › 20.1 Key concepts in review › 
To perform multilabel categorical classification (where each sample can have several classes), end your stack of layers with a `Dense` layer with a number of units equal to the number of classes and a `sigmoid` activation, and use `binary_crossentropy` as the loss. Your targets should be k-hot encoded:

### PDF p.197 (printed 173) — 6 The universal workflow of machine learning › 6.1 Defining the task › 
What will your input data be? What are you trying to predict? You can only learn to predict something if you have available training data: for example, you can only learn to classify the sentiment of movie reviews if you have both movie reviews and sentiment annotations available. As such, data availability is usually the limiting factor at this stage. In many cases, you will have to resort to collecting and annotating new datasets yourself (which we cover in the next section).

What type of machine learning task are you facing? Is it binary classification? Multiclass classification? Scalar regression? Vector regression? Multiclass, multilabel classification? Image segmentation? Ranking? Something else, like clustering, generation, or reinforcement learning? In some cases, it may be that machine learning isn’t even the best way to make sense of your data, and you should use something else, such as plain old-school statistical analysis: – The photo search engine project is a multiclass, multilabel classification task. – The spam detection project is a binary classification task. If you set “offensive content” as a separate class, it’s a three-way classification task. – The music rec
