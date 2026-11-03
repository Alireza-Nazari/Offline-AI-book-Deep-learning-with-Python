# Generation bundle for page_149  (PDF page 149, printed page 125)

Page kind: content
Chapter: 4 Classification and regression
Section: 4.2 Classifying newswires: A multiclass classification example
Subsection: 4.2.8 Further experiments
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_149.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_149.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p149_e03] (body) This new loss function is still mathematically the same as `categorical_crossentropy`; it just has a different interface.
[p149_e04] (heading_subsection) 4.2.7
[p149_e05] (heading_subsection) The importance of having sufficiently large intermediate layers
[p149_e06] (body) We mentioned earlier that because the final outputs are 46-dimensional, you should avoid intermediate layers with much fewer than 46 units. Now let’s see what happens when you introduce an information bottleneck by having intermediate layers that are significantly less than 46-dimensional: for example, 4-dimensional.
[p149_e07] (listing_caption) Listing 4.23 A model with an information bottleneck
[p149_e08] (code)
```
model = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(4, activation="relu"),
        layers.Dense(46, activation="softmax"),
    ]
)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=128,
    validation_data=(x_val, y_val),
)
```
[p149_e09] (body) The model now peaks at approximately 71% validation accuracy, an 8% absolute drop. This drop is mostly due to the fact that you’re trying to compress a lot of information (enough information to recover the separation hyperplanes of 46 classes) into an intermediate space that is too low-dimensional. The model is able to cram most of the necessary information into these 4-dimensional representations, but not all of it.
[p149_e10] (heading_subsection) 4.2.8
[p149_e11] (heading_subsection) Further experiments
[p149_e12] (body) Like in the previous example, we encourage you to try out the following experiments to train your intuition about the kind of configuration decisions you have to make with such models:
[p149_e13] (bullet) Try using larger or smaller layers: 32 units, 128 units, and so on.
[p149_e14] (bullet) You used two intermediate layers before the final softmax classification layer. Now try using a single intermediate layer, or three intermediate layers.

## Listings
- Listing 4.23: A model with an information bottleneck (code elements p149_e08)

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

## Previous page (PDF 148) text, for continuity
4.2.5

Generating predictions on new data

Calling the model’s `predict` method on new samples returns a class probability distribution over all 46 topics for each sample. Let’s generate topic predictions for all of the test data:

Each entry in “predictions” is a vector of length 46:

The coefficients in this vector sum to 1, as they form a probability distribution:

The largest entry is the predicted class—the class with the highest probability:

4.2.6

A different way to handle the labels and the loss

We mentioned earlier that another way to encode the labels would be to leave them untouched as integer tensors, like this:

The only thing this approach would change is the choice of the loss function. The loss function used in listing 4.22, `categorical_crossentropy`, expects the labels to follow a categorical encoding. With integer labels, you should use `sparse_categorical_crossentropy`:

## Next page (PDF 150) text, for continuity
4.2.9

Wrapping up

Here’s what you should take away from this example:

If you’re trying to classify data points among N classes, your model should end with a `Dense` layer of size N.

In a single-label, multiclass classification problem, your model should end with a `softmax` activation so that it will output a probability distribution over the N output classes.

Categorical crossentropy is almost always the loss function you should use for such problems. It minimizes the distance between the probability distributions output by the model and the true distribution of the targets.

There are two ways to handle labels in multiclass classification: – Encoding the labels via categorical encoding (also known as one-hot encoding) and using `categorical_crossentropy` as a loss function – Encoding the labels as integers and using the `sparse_categorical_` `crossentropy` loss function

If you need to classify data into a large number of categories, you should avoid creating information bottlenecks in your model due to intermediate layers that are too small.

4.3

Predicting house prices: A regression example

The two previous examples were considered classification problems, where the goal was to predict a single discrete label of an input data point. Another common type of machine learning problem is regression, which consists of predicting a continuous value instead of a discrete label: for instance, predicting the temperature tomorrow given meteorological data, or predicting the time that a software project will take to complete given its specifications.

NOTE Confusingly, logistic regression isn’t a regression algorithm—it’s a classification algorithm.

4.3.1

The California Housing Price dataset

You’ll attempt to predict the median price of homes in different areas of Cal …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.143 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.133 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.147 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.182 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.139 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.140 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.181 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.152 — 4 Classification and regression › 4.3 Predicting house prices: A regression example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.143 (printed 119) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › 
recovered by later layers: each layer can potentially become an information bottleneck. In the previous example, you used 16-dimensional intermediate layers, but a 16-dimensional space may be too limited to learn to separate 46 different classes: such small layers may act as information bottlenecks, permanently dropping relevant information.

For this reason, you’ll use larger intermediate layers. Let’s go with 64 units.

Listing 4.15 Model definition

### PDF p.152 (printed 128) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Building your model
Because so few samples are available, you’ll use a very small model with two intermediate layers, each with 64 units. In general, the less training data you have, the worse overfitting will be, and using a small model is one way to mitigate overfitting.

Listing 4.27 Model definition

### PDF p.133 (printed 109) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › 
Having 16 units means the weight matrix `W` will have shape `(input_dimension,` `16)`: the dot product with `W` will project the input data onto a 16-dimensional representation space (and then you’ll add the bias vector `b` and apply the `relu` operation). You can intuitively understand the dimensionality of your representation space as “how much freedom you’re allowing the model to have when learning internal representations.” Having more units (a higher-dimensional representation space) allows your model to learn more complex representations, but it makes the model more computationally expensive and may lead to learning unwanted patterns (patterns that will improve performance on the training data but not on the test data).

The intermediate layers use `relu` as their activation function, and the final layer uses a sigmoid activation to output a probability (a score between 0 and 1, indicating how likely the review is to be positive). A `relu` (rectified linear unit) is a function meant to zero-out negative values (see figure 4.2), whereas a sigmoid “squashes” arbitrary values into the `[0, 1]` interval (see figure 4.3), outputting something that can be interpreted as a probabili

### PDF p.159 (printed 135) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Summary
The three most common kinds of machine learning tasks on vector data are binary classification, multiclass classification, and scalar regression. Each task uses different loss functions: – `binary_crossentropy` for binary classification – `categorical_crossentropy` for multiclass classification – `mean_squared_error` for scalar regression

You’ll usually need to preprocess raw data before feeding it into a neural network.

When your data has features with different ranges, scale each feature independently as part of preprocessing.

As training progresses, neural networks eventually begin to overfit and obtain worse results on never-before-seen data.

If you don’t have much training data, use a small model with only one or two intermediate layers, to avoid severe overfitting.

If your data is divided into many categories, you may cause information bottlenecks if you make the intermediate layers too small.

When you’re working with little data, K-fold validation can help reliably evaluate your model.

### PDF p.139 (printed 115) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › Further experiments
The following experiments will help convince you that the architecture choices you’ve made are all fairly reasonable, although there’s still room for improvement:

You used two representation layers before the final classification layer. Try using one or three representation layers and see how doing so affects validation and test accuracy.

### PDF p.140 (printed 116) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › 
Try using layers with more units or fewer units: 32 units, 64 units, and so on.

Try using the `mean_squared_error` loss function instead of `binary_crossentropy`.

Try using the `tanh` activation (an activation that was popular in the early days of neural networks) instead of `relu`.

### PDF p.181 (printed 157) — 5 Fundamentals of machine learning › 5.3 Improving model fit › 
you can’t clearly overfit, even after many iterations over the training data. You’re likely to encounter similar curves often in your career.

Remember that it should always be possible to overfit. Much like the problem “the training loss doesn’t go down,” this is an issue that can always be solved. If you can’t seem to be able to overfit, it’s likely a problem with the representational power of your model: you’re going to need a bigger model, one with more capacity—that is, able to store more information. You can increase representational power by adding more layers, using bigger layers (layers with more parameters), or using kinds of layers that are more appropriate for the problem at hand (better architecture priors). Let’s try training a bigger model, one with two intermediate layers with 128 units each:

### PDF p.142 (printed 118) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › Building your model
This topic classification problem looks similar to the previous movie review classification problem: in both cases, you’re trying to classify short snippets of text. But there is a new constraint here: the number of output classes has gone from 2 to 46. The dimensionality of the output space is much larger.

In a stack of `Dense` layers like those you’ve been using, each layer can only access information present in the output of the previous layer. If one layer drops some information relevant to the classification problem, this information can never be
