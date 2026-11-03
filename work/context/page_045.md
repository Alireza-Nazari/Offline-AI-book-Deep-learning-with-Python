# Generation bundle for page_045  (PDF page 45, printed page 21)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.2 Data representations for neural networks
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_045.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_045.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p045_e03] (code)
```
>>> predictions[0].argmax()
7
>>> predictions[0][7]
0.99999106
```
[p045_e04] (body) We can check that the test label agrees:
[p045_e05] (code)
```
>>> test_labels[0]
7
```
[p045_e06] (body) On average, how good is our model at classifying such never-before-seen digits? Let’s check by computing average accuracy over the entire test set.
[p045_e07] (listing_caption) Listing 2.7 Evaluating the model on new data
[p045_e08] (code)
```
>>> test_loss, test_acc = model.evaluate(test_images, test_labels)
>>> print(f"test_acc: {test_acc}")
test_acc: 0.9785
```
[p045_e09] (body) The test set accuracy turns out to be 97.8%—that’s almost double the error rate of the training set (at 98.9% accuracy). This gap between training accuracy and test accuracy is an example of overfitting: the fact that machine learning models tend to perform worse on new data than on their training data. Overfitting is a central topic in chapter 5.
[p045_e10] (body) This concludes our first example. You just saw how you can build and train a neural network to classify handwritten digits in less than 15 lines of Python code. In this chapter and the next, we’ll go into detail about every moving piece we just previewed and clarify what’s going on behind the scenes. You’ll learn about tensors, the data-storing objects going into the model; tensor operations, which layers are made of; and gradient descent, which allows your model to learn from its training examples.
[p045_e11] (heading_section) 2.2
[p045_e12] (heading_section) Data representations for neural networks
[p045_e13] (body) In the previous example, we started from data stored in multidimensional NumPy arrays, also called tensors. In general, all current machine learning systems use tensors as their basic data structure. Tensors are fundamental to the field—so fundamental that the TensorFlow framework was named after them. So what’s a tensor?
[p045_e14] (body) At its core, a tensor is a container for data—usually numerical data. So it’s a container for numbers. You may already be familiar with matrices, which are rank-2 tensors: tensors are a generalization of matrices to an arbitrary number of dimensions (note that in the context of tensors, a dimension is often called an axis). Going over the details of tensors might seem a bit abstract at first. But it’s well worth it—manipulating tensors will be the bread and butter of any machine learning code you ever write.

## Listings
- Listing 2.7: Evaluating the model on new data (code elements p045_e08)

## Chapter outline
Chapter: 2 The mathematical building blocks of neural networks (PDF pages 40-83)
  - 2.1 A first look at a neural network (p.41)
  - 2.2 Data representations for neural networks (p.45)
      - 2.2.1 Scalars (rank-0 tensors) (p.46)
      - 2.2.2 Vectors (rank-1 tensors) (p.46)
      - 2.2.3 Matrices (rank-2 tensors) (p.46)
      - 2.2.4 Rank-3 tensors and higher-rank tensors (p.47)
      - 2.2.5 Key attributes (p.47)
      - 2.2.6 Manipulating tensors in NumPy (p.49)
      - 2.2.7 The notion of data batches (p.49)
      - 2.2.8 Real-world examples of data tensors (p.50)
  - 2.3 The gears of neural networks: Tensor operations (p.52)
      - 2.3.1 Element-wise operations (p.53)
      - 2.3.2 Broadcasting (p.54)
      - 2.3.3 Tensor product (p.56)
      - 2.3.4 Tensor reshaping (p.58)
      - 2.3.5 Geometric interpretation of tensor operations (p.59)
      - 2.3.6 A geometric interpretation of deep learning (p.62)
  - 2.4 The engine of neural networks: Gradient-based optimization (p.63)
      - 2.4.1 What’s a derivative? (p.65)
      - 2.4.2 Derivative of a tensor operation: The gradient (p.66)
      - 2.4.3 Stochastic gradient descent (p.67)
      - 2.4.4 Chaining derivatives: The Backpropagation algorithm (p.70)
  - 2.5 Looking back at our first example (p.75)
      - 2.5.1 Reimplementing our first example from scratch (p.77)
      - 2.5.2 Running one training step (p.79)
      - 2.5.3 The full training loop (p.81)
      - 2.5.4 Evaluating the model (p.82)

## Previous page (PDF 44) text, for continuity
Before training, we’ll preprocess the data by reshaping it into the shape the model expects and scaling it so that all values are in the `[0,` `1]` interval. Previously, our training images were stored in an array of shape `(60000,` `28,` `28)` of type `uint8` with values in the `[0,` `255]` interval. We transform it into a `float32` array of shape `(60000,` `28` `*` `28)` with values between `0` and `1`.

Listing 2.4 Preparing the image data

We’re now ready to train the model, which in Keras is done via a call to the model’s `fit()` method—we fit the model to its training data.

Listing 2.5 “Fitting” the model

Two quantities are displayed during training: the loss of the model over the training data and the accuracy of the model over the training data. We quickly reach an accuracy of 0.989 (98.9%) on the training data.

Now that we have a trained model, we can use it to predict class probabilities for new digits—images that weren’t part of the training data, like those from the test set.

Listing 2.6 Using the model to make predictions

Each number of index `i` in that array corresponds to the probability that digit image `test_digits[0]` belong to class `i`.

This first test digit has the highest probability score (0.99999106, almost 1) at index 7, so according to our model, it must be a 7:

## Next page (PDF 46) text, for continuity
2.2.1

Scalars (rank-0 tensors)

A tensor that contains only one number is called a scalar (or scalar tensor, rank-0 tensor, or 0D tensor). In NumPy, a `float32` or `float64` number is a scalar tensor (or scalar array). You can display the number of axes of a NumPy tensor via the `ndim` attribute; a scalar tensor has 0 axes (`ndim` `==` `0`). The number of axes of a tensor is also called its rank. Here’s a NumPy scalar:

2.2.2

Vectors (rank-1 tensors)

An array of numbers is called a vector (or rank-1 tensor or 1D tensor). A rank-1 tensor has exactly one axis. The following is a NumPy vector:

This vector has five entries and so is called a 5-dimensional vector. Don’t confuse a 5D vector with a 5D tensor! A 5D vector has only one axis and has five dimensions along its axis, whereas a 5D tensor has five axes (and may have any number of dimensions along each axis). Dimensionality can denote either the number of entries along a specific axis (as in the case of our 5D vector) or the number of axes in a tensor (such as a 5D tensor), which can be confusing at times. In the latter case, it’s technically more correct to talk about a tensor of rank 5 (the rank of a tensor being the number of axes), but the ambiguous notation 5D tensor is common regardless.

2.2.3

Matrices (rank-2 tensors)

An array of vectors is a matrix (or rank-2 tensor or 2D tensor). A matrix has two axes (often referred to as rows and columns). You can visually interpret a matrix as a rectangular grid of numbers. This is a NumPy matrix:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.82 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.42 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network
- PDF p.75 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.50 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.47 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.9 — contents › 
- PDF p.52 — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations
- PDF p.276 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.75 (printed 51) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › Looking back at our first example
Now you understand that the input images are stored in NumPy tensors, which are here formatted as `float32` tensors of shape `(60000,` `784)` (training data) and `(10000, ` `784)` (test data), respectively.

This was our model:

### PDF p.9 (printed ix) — contents ›  › 
2

The mathematical building blocks of neural networks 16 2.1

A first look at a neural network 17 2.2 Data representations for neural networks 21

Scalars (rank-0 tensors) 22 ■ Vectors (rank-1 tensors) 22 Matrices (rank-2 tensors) 22 ■ Rank-3 tensors and higher-rank tensors 23 ■ Key attributes 23 ■ Manipulating tensors in NumPy 25 ■ The notion of data batches 25 Real-world examples of data tensors 26

2.3

The gears of neural networks: Tensor operations 28

Element-wise operations 29 ■ Broadcasting 30 Tensor product 32 ■ Tensor reshaping 34 ■ Geometric interpretation of tensor operations 35 ■ A geometric interpretation of deep learning 38

2.4

The engine of neural networks: Gradient-based optimization 39

What’s a derivative? 41 ■ Derivative of a tensor operation: The gradient 42 ■ Stochastic gradient descent 43 ■ Chaining derivatives: The Backpropagation algorithm 46

2.5

Looking back at our first example 51

Reimplementing our first example from scratch 53 Running one training step 55 ■ The full training loop 57 Evaluating the model 58

3

Introduction to TensorFlow, PyTorch, JAX, and Keras 60 3.1

A brief history of deep learning frameworks 61 3.2 How these frameworks relate 

### PDF p.42 (printed 18) — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network › 
`train_images` and `train_labels` form the training set, the data that the model will learn from. The model will then be tested on the test set, `test_images` and `test_labels`. The images are encoded as NumPy arrays, and the labels are an array of digits, ranging from 0 to 9. The images and labels have a one-to-one correspondence.

NOTE NumPy is a highly popular Python library for numerical computation. You will see it pop up frequently in your machine learning journey. It is rarely used to implement modern machine learning algorithms, due to lacking GPU and autodifferentiation support, but NumPy arrays are often used as a numerical data exchange format—like here, for MNIST digits and their labels.

Let’s look at the training data:

### PDF p.203 (printed 179) — 6 The universal workflow of machine learning › 6.2 Developing a model › Vectorization
All inputs and targets in a neural network must be typically tensors of floating-point data (or, in specific cases, tensors of integers or strings). Whatever data you need to process—sound, images, text—you must first turn into tensors, a step called data vectorization. For instance, in the two previous text classification examples from chapter 4, we started from text represented as lists of integers (standing for sequences of words), and we used multi-hot encoding to turn them into a tensor of `float32` data. In the examples of classifying digits and predicting house prices, the data already came in vectorized form, so you were able to skip this step.

### PDF p.82 (printed 58) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › Summary
Tensors form the foundation of modern machine learning systems. They come in various flavors of `dtype`, `rank`, and `shape`.

You can manipulate numerical tensors via tensor operations (such as addition, tensor product, or element-wise multiplication), which can be interpreted as encoding geometric transformations. In general, everything in deep learning is amenable to a geometric interpretation.

Deep learning models consist of chains of simple tensor operations, parameterized by weights, which are themselves tensors. The weights of a model are where its “knowledge” is stored.

Learning means finding a set of values for the model’s weights that minimizes a loss function for a given set of training data samples and their corresponding targets.

Learning happens by drawing random batches of data samples and their targets and computing the gradient of the model parameters with respect to the loss on the batch. The model parameters are then moved a bit (the magnitude of the

### PDF p.276 (printed 252) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
We get a test accuracy of 78.6% (due to the randomness of neural network initializations, you may get numbers within 1 percentage point of that).

Because you have relatively few training samples (2,000), overfitting will be your number-one concern. You already know about a number of techniques that can help mitigate overfitting, such as dropout and weight decay (L2 regularization). We’re now going to work with a new one, specific to computer vision and used almost universally when processing images with deep learning models: data augmentation.

### PDF p.40 (printed 16) — 2 The mathematical building blocks of neural networks ›  › This chapter covers
A first example of a neural network

Tensors and tensor operations

How neural networks learn via backpropagation

and gradient descent

Understanding deep learning requires familiarity with many simple mathematical

concepts: tensors, tensor operations, differentiation, gradient descent, and so on. Our goal

in this chapter will be to build up your intuition about these notions without getting overly technical. In particular, we’ll steer away from mathematical notation, which can introduce unnecessary barriers for those without any mathematics background, and isn’t necessary to explain things well. The most precise, unambiguous description of a mathematical operation is its executable code.

To provide sufficient context for introducing tensors and gradient descent, we’ll begin the chapter with a practical example of a neural network. Then we’ll go over every new concept that’s been introduced, point by point. Keep in mind that these concepts will be essential for you to understand the practical examples that will come in the following chapters!

### PDF p.138 (printed 114) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › 
Figure 4.5 Training and validation accuracy

As you can see, the training loss decreases with every epoch, and the training accuracy increases with every epoch. That’s what you would expect when running gradient-descent optimization—the quantity you’re trying to minimize should be less with every iteration. But that isn’t the case for the validation loss and accuracy: they seem to peak at the fourth epoch. This is an example of what we warned against earlier: a model that performs better on the training data isn’t necessarily a model that will do better on data it has never seen before. In precise terms, what you’re seeing is overfitting: after the fourth epoch, you’re overoptimizing on the training data, and you end up learning representations that are specific to the training data and don’t generalize to data outside of the training set.

In this case, to prevent overfitting, you could stop training after four epochs. In general, you can use a range of techniques to mitigate overfitting, which we’ll cover in chapter 5. Let’s train a new model from scratch for four epochs and then evaluate it on the test data.

Listing 4.10 Training the model for four epochs
