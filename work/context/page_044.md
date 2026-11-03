# Generation bundle for page_044  (PDF page 44, printed page 20)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.1 A first look at a neural network
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_044.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_044.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p044_e03] (code)
```
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```
[p044_e04] (body) Before training, we’ll preprocess the data by reshaping it into the shape the model expects and scaling it so that all values are in the `[0,` `1]` interval. Previously, our training images were stored in an array of shape `(60000,` `28,` `28)` of type `uint8` with values in the `[0,` `255]` interval. We transform it into a `float32` array of shape `(60000,` `28` `*` `28)` with values between `0` and `1`.
[p044_e05] (listing_caption) Listing 2.4 Preparing the image data
[p044_e06] (code)
```
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
```
[p044_e07] (body) We’re now ready to train the model, which in Keras is done via a call to the model’s `fit()` method—we fit the model to its training data.
[p044_e08] (listing_caption) Listing 2.5 “Fitting” the model
[p044_e09] (code)
```
model.fit(train_images, train_labels, epochs=5, batch_size=128)
```
[p044_e10] (body) Two quantities are displayed during training: the loss of the model over the training data and the accuracy of the model over the training data. We quickly reach an accuracy of 0.989 (98.9%) on the training data.
[p044_e11] (body) Now that we have a trained model, we can use it to predict class probabilities for new digits—images that weren’t part of the training data, like those from the test set.
[p044_e12] (listing_caption) Listing 2.6 Using the model to make predictions
[p044_e13] (code)
```
>>> test_digits = test_images[0:10]
>>> predictions = model.predict(test_digits)
>>> predictions[0]
array([1.0726176e-10, 1.6918376e-10, 6.1314843e-08, 8.4106023e-06,
       2.9967067e-11, 3.0331331e-09, 8.3651971e-14, 9.9999106e-01,
       2.6657624e-08, 3.8127661e-07], dtype=float32)
```
[p044_e14] (body) Each number of index `i` in that array corresponds to the probability that digit image `test_digits[0]` belong to class `i`.
[p044_e15] (body) This first test digit has the highest probability score (0.99999106, almost 1) at index 7, so according to our model, it must be a 7:

## Listings
- Listing 2.4: Preparing the image data (code elements p044_e06)
- Listing 2.5: “Fitting” the model (code elements p044_e09)
- Listing 2.6: Using the model to make predictions (code elements p044_e13)

## Equation candidates (formulas written inline in the text)
- [p044_e03]     loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

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

## Previous page (PDF 43) text, for continuity
labels. Finally, we’ll ask the network to produce predictions for `test_images`, and we’ll verify whether these predictions match the labels from `test_labels`.

Let’s build the network—again, remember that you aren’t expected to understand everything about this example yet.

Listing 2.2 The network architecture

The core building block of neural networks is the layer. You can think of a layer as a filter for data: some data goes in, and it comes out in a more useful form. Specifically, layers extract representations out of the data fed into them—hopefully, representations that are more meaningful for the problem at hand. Most of deep learning consists of chaining together simple layers that will implement a form of progressive data distillation. A deep learning model is like a sieve for data processing, made of a succession of increasingly refined data filters—the layers.

Here, our model consists of a sequence of two `Dense` layers, which are densely connected (also called fully connected) neural layers. The second (and last) layer is a 10-way `softmax` classification layer, which means it will return an array of 10 probability scores (summing to 1). Each score will be the probability that the current digit image belongs to one of our 10 digit classes. To make the model ready for training, we need to pick three more things, as part of the compilation step:

A loss function—How the model will be able to measure its performance on the training data and thus how it will be able to steer itself in the right direction.

An optimizer—The mechanism through which the model will update itself based on the training data it sees, to improve its performance.

Metrics to monitor during training and testing—Here, we’ll only care about accuracy (the fraction of the images that were  …

## Next page (PDF 45) text, for continuity
We can check that the test label agrees:

On average, how good is our model at classifying such never-before-seen digits? Let’s check by computing average accuracy over the entire test set.

Listing 2.7 Evaluating the model on new data

The test set accuracy turns out to be 97.8%—that’s almost double the error rate of the training set (at 98.9% accuracy). This gap between training accuracy and test accuracy is an example of overfitting: the fact that machine learning models tend to perform worse on new data than on their training data. Overfitting is a central topic in chapter 5.

This concludes our first example. You just saw how you can build and train a neural network to classify handwritten digits in less than 15 lines of Python code. In this chapter and the next, we’ll go into detail about every moving piece we just previewed and clarify what’s going on behind the scenes. You’ll learn about tensors, the data-storing objects going into the model; tensor operations, which layers are made of; and gradient descent, which allows your model to learn from its training examples.

2.2

Data representations for neural networks

In the previous example, we started from data stored in multidimensional NumPy arrays, also called tensors. In general, all current machine learning systems use tensors as their basic data structure. Tensors are fundamental to the field—so fundamental that the TensorFlow framework was named after them. So what’s a tensor?

At its core, a tensor is a container for data—usually numerical data. So it’s a container for numbers. You may already be familiar with matrices, which are rank-2 tensors: tensors are a generalization of matrices to an arbitrary number of dimensions (note that in the context of tensors, a dimension is often called an axis). Going ov …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.82 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.75 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.42 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network
- PDF p.81 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.76 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.79 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.152 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.139 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.75 (printed 51) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › Looking back at our first example
Now you understand that the input images are stored in NumPy tensors, which are here formatted as `float32` tensors of shape `(60000,` `784)` (training data) and `(10000, ` `784)` (test data), respectively.

This was our model:

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

### PDF p.76 (printed 52) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › 
Now you understand what happens when you call `fit`: the model will start to iterate on the training data in mini-batches of 128 samples, 5 times over (each iteration over all the training data is called an epoch). For each batch, the model will compute the gradient of the loss with regard to the weights (using the Backpropagation algorithm, which derives from the chain rule in calculus) and move the weights in the direction that will reduce the value of the loss for this batch.

After these 5 epochs, the model will have performed 2,345 gradient updates (469 per epoch), and the loss of the model will be sufficiently low that the model will be capable of classifying handwritten digits with high accuracy.

### PDF p.42 (printed 18) — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network › 
`train_images` and `train_labels` form the training set, the data that the model will learn from. The model will then be tested on the test set, `test_images` and `test_labels`. The images are encoded as NumPy arrays, and the labels are an array of digits, ranging from 0 to 9. The images and labels have a one-to-one correspondence.

NOTE NumPy is a highly popular Python library for numerical computation. You will see it pop up frequently in your machine learning journey. It is rarely used to implement modern machine learning algorithms, due to lacking GPU and autodifferentiation support, but NumPy arrays are often used as a numerical data exchange format—like here, for MNIST digits and their labels.

Let’s look at the training data:

### PDF p.230 (printed 206) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › Using built-in training and evaluation loops
Listing 7.17 The standard workflow: `compile()`, `fit()`, `evaluate()`, `predict()`
from keras.datasets import mnist
def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model
(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]
model = get_mnist_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_images,
    train_labels,
    epochs=3,
    validation_data=(val_images, val_labels),
)
test_metrics = model.evaluate(test_images, test_labels)
predictions = model.predict(test_images)

### PDF p.271 (printed 247) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › Data preprocessing
As you know by now, data should be formatted into appropriately preprocessed floating-point tensors before being fed into the model. Currently, the data sits on a drive as JPEG files, so the steps for getting it into the model are roughly as follows:

1 Read the picture files. 2 Decode the JPEG content to RGB grids of pixels. 3 Convert these into floating-point tensors. 4 Resize them to a shared size (we’ll use 180 x 180). 5 Pack them into batches (we’ll use batches of 32 images). This may seem a bit daunting, but fortunately Keras has utilities to take care of these steps automatically. In particular, Keras features the utility function `image_dataset_` `from_directory`, which lets you quickly set up a data pipeline that can automatically turn image files on disk into batches of preprocessed tensors. This is what you’ll use here. Calling `image_dataset_from_directory(directory)` will first list the subdirectories of `directory` and assume each one contains images from one of your classes. It will then index the image files in each subdirectory. Finally, it will create and return a `tf.data
` `.Dataset` object configured to read these files, shuffle them, decode them to tensors, re

### PDF p.135 (printed 111) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › Validating your approach
You’ll now train the model for 20 epochs (20 iterations over all samples in the training data), in mini-batches of 512 samples. At the same time, you’ll monitor loss and accuracy on the 10,000 samples that you set apart. You do so by passing the validation data as the `validation_data` argument to `model.fit()`.

Listing 4.7 Training your model

### PDF p.222 (printed 198) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › Training a multi-input, multi-output model
You can train your model in much the same way as you would train a Sequential model, by calling `fit()` with lists of input and output data. These lists of data should respect the same order as the inputs you passed to the `Model()` constructor.

Listing 7.10 Training a model by providing lists of input and target arrays
