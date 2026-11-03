# Generation bundle for page_148  (PDF page 148, printed page 124)

Page kind: content
Chapter: 4 Classification and regression
Section: 4.2 Classifying newswires: A multiclass classification example
Subsection: 4.2.6 A different way to handle the labels and the loss
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_148.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_148.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p148_e03] (heading_subsection) 4.2.5
[p148_e04] (heading_subsection) Generating predictions on new data
[p148_e05] (body) Calling the model’s `predict` method on new samples returns a class probability distribution over all 46 topics for each sample. Let’s generate topic predictions for all of the test data:
[p148_e06] (code)
```
predictions = model.predict(x_test)
```
[p148_e07] (body) Each entry in “predictions” is a vector of length 46:
[p148_e08] (code)
```
>>> predictions[0].shape
(46,)
```
[p148_e09] (body) The coefficients in this vector sum to 1, as they form a probability distribution:
[p148_e10] (code)
```
>>> np.sum(predictions[0])
1.0
```
[p148_e11] (body) The largest entry is the predicted class—the class with the highest probability:
[p148_e12] (code)
```
>>> np.argmax(predictions[0])
4
```
[p148_e13] (heading_subsection) 4.2.6
[p148_e14] (heading_subsection) A different way to handle the labels and the loss
[p148_e15] (body) We mentioned earlier that another way to encode the labels would be to leave them untouched as integer tensors, like this:
[p148_e16] (code)
```
y_train = train_labels
y_test = test_labels
```
[p148_e17] (body) The only thing this approach would change is the choice of the loss function. The loss function used in listing 4.22, `categorical_crossentropy`, expects the labels to follow a categorical encoding. With integer labels, you should use `sparse_categorical_crossentropy`:
[p148_e18] (code)
```
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

## Equation candidates (formulas written inline in the text)
- [p148_e06] predictions = model.predict(x_test)
- [p148_e16] y_train = train_labels
y_test = test_labels
- [p148_e18] model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

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

## Previous page (PDF 147) text, for continuity
The model begins to overfit after nine epochs. Let’s train a new model from scratch for nine epochs and then evaluate it on the test set.

Listing 4.22 Retraining a model from scratch

Here are the final results:

This approach reaches an accuracy of approximately 80%. With a balanced binary classification problem, the accuracy reached by a purely random classifier would be 50%. But in this case, we have 46 classes, and they may not be equally represented. What would be the accuracy of a random baseline? We could try quickly implementing one to check this empirically:

As you can see, a random classifier would score around 19% classification accuracy, so the results of our model seem pretty good in that light.

## Next page (PDF 149) text, for continuity
This new loss function is still mathematically the same as `categorical_crossentropy`; it just has a different interface.

4.2.7

The importance of having sufficiently large intermediate layers

We mentioned earlier that because the final outputs are 46-dimensional, you should avoid intermediate layers with much fewer than 46 units. Now let’s see what happens when you introduce an information bottleneck by having intermediate layers that are significantly less than 46-dimensional: for example, 4-dimensional.

Listing 4.23 A model with an information bottleneck

The model now peaks at approximately 71% validation accuracy, an 8% absolute drop. This drop is mostly due to the fact that you’re trying to compress a lot of information (enough information to recover the separation hyperplanes of 46 classes) into an intermediate space that is too low-dimensional. The model is able to cram most of the necessary information into these 4-dimensional representations, but not all of it.

4.2.8

Further experiments

Like in the previous example, we encourage you to try out the following experiments to train your intuition about the kind of configuration decisions you have to make with such models:

Try using larger or smaller layers: 32 units, 128 units, and so on.

You used two intermediate layers before the final softmax classification layer. Now try using a single intermediate layer, or three intermediate layers.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.158 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.139 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.142 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.143 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.150 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.505 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.134 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.44 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.150 (printed 126) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Wrapping up
Here’s what you should take away from this example:

If you’re trying to classify data points among N classes, your model should end with a `Dense` layer of size N.

In a single-label, multiclass classification problem, your model should end with a `softmax` activation so that it will output a probability distribution over the N output classes.

Categorical crossentropy is almost always the loss function you should use for such problems. It minimizes the distance between the probability distributions output by the model and the true distribution of the targets.

There are two ways to handle labels in multiclass classification: – Encoding the labels via categorical encoding (also known as one-hot encoding) and using `categorical_crossentropy` as a loss function – Encoding the labels as integers and using the `sparse_categorical_` `crossentropy` loss function

If you need to classify data into a large number of categories, you should avoid creating information bottlenecks in your model due to intermediate layers that are too small.

### PDF p.143 (printed 119) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › 
There are two other things you should note about this architecture:

You end the model with a `Dense` layer of size 46. This means for each input sample, the network will output a 46-dimensional vector. Each entry in this vector (each dimension) will encode a different output class.

The last layer uses a `softmax` activation. You saw this pattern in the MNIST example. It means the model will output a probability distribution over the 46 different output classes—for every input sample, the model will produce a 46-dimensional output vector, where `output[i]` is the probability that the sample belongs to class `i`. The 46 scores will sum to 1.

The best loss function to use in this case is `categorical_crossentropy`. It measures the distance between two probability distributions—here, between the probability distribution outputted by the model and the true distribution of the labels. By minimizing the distance between these two distributions, you train the model to output something as close as possible to the true labels.

Like last time, we’ll also monitor accuracy. However, accuracy is a bit of a crude metric in this case: if the model has the correct class as its second choice for

### PDF p.158 (printed 134) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Generating predictions on new data
When calling `predict()` on our binary classification model, we retrieved a scalar score between 0 and 1 for each input sample. With our multiclass classification model, we retrieved a probability distribution over all classes for each sample. Now, with this scalar regression model, `predict()` returns the model’s guess for the sample’s price in hundreds of thousands of dollars:

### PDF p.449 (printed 425) — 15 Language models and the Transformer › 15.1 The language model › 
This model outputs a softmax probability for every possible character in our vocabulary, and we will `compile()` it with a crossentropy loss. Note that our model is still training on a classification problem, it’s just that we will make one classification prediction for every token in our sequence. For our batch of 64 samples with 100 characters each, we will predict 6,400 individual labels. Loss and accuracy metrics reported by Keras during training will be averaged first across each sequence and, second, across each batch.

Let’s go ahead and train our language model.

Listing 15.5 Training a miniature language model

### PDF p.140 (printed 116) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › Classifying newswires: A multiclass classification example
In the previous section, you saw how to classify vector inputs into two mutually exclusive classes using a densely connected neural network. But what happens when you have more than two classes?

In this section, you’ll build a model to classify Reuters newswires into 46 mutually exclusive topics. Because you have many classes, this problem is an instance of multiclass classification, and because each data point should be classified into only one category, the problem is more specifically an instance of single-label, multiclass classification. If each data point could belong to multiple categories (in this case, topics), you’d be facing a multilabel, multiclass classification problem.

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

### PDF p.142 (printed 118) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › Preparing the data
To vectorize the labels, there are two possibilities: you can leave the labels untouched as integers, or you can use one-hot encoding. One-hot encoding is a widely used format for categorical data, also called categorical encoding. In this case, one-hot encoding of the labels consists of embedding each label as an all-zero vector with a 1 in the place of the label index. Here’s an example.

Listing 4.14 Encoding the labels

### PDF p.649 (printed 625) — index ›  › 
Task Last-layer activation Loss function Metrics

Binary classification Sigmoid Binary crossentropy Binary accuracy, ROC AUC Multiclass, single-label Softmax Categorical crossentropy Categorical accuracy, classification Top-k categorical accuracy, ROC AUC Multiclass, multi-label Sigmoid Binary crossentropy Binary accuracy, ROC classification AUC Regression None Mean squared error Mean absolute error

Input Output Model

Vector data Class probability, regression value Densely connected network Timeseries Class probability, regression value RNN, Transformer Images Class probability, regression value ConvNet Text Class probability, regression value Transformer Text, images Text Transformer Text, images Images VAE, diffusion model
