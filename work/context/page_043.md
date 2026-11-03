# Generation bundle for page_043  (PDF page 43, printed page 19)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.1 A first look at a neural network
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_043.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_043.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p043_e03] (body) labels. Finally, we’ll ask the network to produce predictions for `test_images`, and we’ll verify whether these predictions match the labels from `test_labels`.
[p043_e04] (body) Let’s build the network—again, remember that you aren’t expected to understand everything about this example yet.
[p043_e05] (listing_caption) Listing 2.2 The network architecture
[p043_e06] (code)
```
import keras
from keras import layers
model = keras.Sequential(
    [
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
```
[p043_e07] (body) The core building block of neural networks is the layer. You can think of a layer as a filter for data: some data goes in, and it comes out in a more useful form. Specifically, layers extract representations out of the data fed into them—hopefully, representations that are more meaningful for the problem at hand. Most of deep learning consists of chaining together simple layers that will implement a form of progressive data distillation. A deep learning model is like a sieve for data processing, made of a succession of increasingly refined data filters—the layers.
[p043_e08] (body) Here, our model consists of a sequence of two `Dense` layers, which are densely connected (also called fully connected) neural layers. The second (and last) layer is a 10-way `softmax` classification layer, which means it will return an array of 10 probability scores (summing to 1). Each score will be the probability that the current digit image belongs to one of our 10 digit classes. To make the model ready for training, we need to pick three more things, as part of the compilation step:
[p043_e09] (bullet) A loss function—How the model will be able to measure its performance on the training data and thus how it will be able to steer itself in the right direction.
[p043_e10] (bullet) An optimizer—The mechanism through which the model will update itself based on the training data it sees, to improve its performance.
[p043_e11] (bullet) Metrics to monitor during training and testing—Here, we’ll only care about accuracy (the fraction of the images that were correctly classified).
[p043_e12] (body) The exact purpose of the loss function and the optimizer will be made clear throughout the next two chapters.
[p043_e13] (listing_caption) Listing 2.3 The compilation step
[p043_e14] (code)
```
model.compile(
    optimizer="adam",
```

## Listings
- Listing 2.2: The network architecture (code elements p043_e06)
- Listing 2.3: The compilation step (code elements p043_e14)

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

## Previous page (PDF 42) text, for continuity
NOTE In machine learning, a category in a classification problem is called a class. Data points are called samples. The class associated with a specific sample is called

Figure 2.1 MNIST sample digits

a label.

The MNIST dataset comes preloaded in Keras, in the form of a set of four NumPy arrays.

Listing 2.1 Loading the MNIST dataset in Keras

`train_images` and `train_labels` form the training set, the data that the model will learn from. The model will then be tested on the test set, `test_images` and `test_labels`. The images are encoded as NumPy arrays, and the labels are an array of digits, ranging from 0 to 9. The images and labels have a one-to-one correspondence.

NOTE NumPy is a highly popular Python library for numerical computation. You will see it pop up frequently in your machine learning journey. It is rarely used to implement modern machine learning algorithms, due to lacking GPU and autodifferentiation support, but NumPy arrays are often used as a numerical data exchange format—like here, for MNIST digits and their labels.

Let’s look at the training data:

And here’s the test data:

The workflow will be as follows. First, we’ll feed the neural network the training data, `train_images` and `train_labels`. The network will then learn to associate images and

## Next page (PDF 44) text, for continuity
Before training, we’ll preprocess the data by reshaping it into the shape the model expects and scaling it so that all values are in the `[0,` `1]` interval. Previously, our training images were stored in an array of shape `(60000,` `28,` `28)` of type `uint8` with values in the `[0,` `255]` interval. We transform it into a `float32` array of shape `(60000,` `28` `*` `28)` with values between `0` and `1`.

Listing 2.4 Preparing the image data

We’re now ready to train the model, which in Keras is done via a call to the model’s `fit()` method—we fit the model to its training data.

Listing 2.5 “Fitting” the model

Two quantities are displayed during training: the loss of the model over the training data and the accuracy of the model over the training data. We quickly reach an accuracy of 0.989 (98.9%) on the training data.

Now that we have a trained model, we can use it to predict class probabilities for new digits—images that weren’t part of the training data, like those from the test set.

Listing 2.6 Using the model to make predictions

Each number of index `i` in that array corresponds to the probability that digit image `test_digits[0]` belong to class `i`.

This first test digit has the highest probability score (0.99999106, almost 1) at index 7, so according to our model, it must be a 7:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.76 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.75 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.120 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.624 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.143 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.79 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.45 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.10 — contents › 

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.75 (printed 51) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › Looking back at our first example
You’re nearing the end of this chapter, and you should now have a general understanding of what’s going on behind the scenes in a neural network. What was a magical black box at the start of the chapter has turned into a clearer picture, as illustrated in figure 2.26: the model, composed of layers that are chained together, maps the input data to predictions. The loss function then compares these predictions to the targets, producing a loss value: a measure of how well the model’s predictions match what was expected. The optimizer uses this loss value to update the model’s weights.

Figure 2.26 Relationship between the network, layers, loss function, and optimizer

Let’s go back to the first example and review each piece of it in the light of what you’ve learned in the previous sections.

This was the input data:

### PDF p.120 (printed 96) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › From layers to models
A deep learning model is a graph of layers. In Keras, that’s the `Model` class. For now, you’ve only seen `Sequential` models (a subclass of `Model`), which are simple stacks of layers, mapping a single input to a single output. But as you move forward, you’ll be exposed to a much broader variety of network topologies. Some common ones are

Two-branch networks

Multihead networks

Residual connections

Network topology can get quite involved. For instance, figure 3.5 shows topology of the graph of layers of a Transformer, a common architecture designed to process text data.

There are generally two ways of building such models in Keras: you can directly subclass the `Model` class, or you can use the Functional API, which lets you do more with less code. We’ll cover both approaches in chapter 7. The topology of a model defines a hypothesis space. You may remember that in chapter 1, we described machine learning as “searching for useful representations of some input data, within a predefined space of possibilities, using guidance from a feedback signal.” By choosing a network topology, you constrain your space of possibilities (hypothesis space) to a specific series of tensor operati

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

### PDF p.32 (printed 8) — 1 What is deep learning? › 1.6 Understanding how deep learning works, in three figures › Understanding how deep learning works, in three figures
At this point, you know that machine learning is about mapping inputs (such as images) to targets (such as the label “cat”), which is done by observing many examples of inputs and targets. You also know that deep neural networks do this input-to-target mapping via a deep sequence of simple data transformations (layers) and that these data transformations are learned by exposure to examples. Now let’s look at how this learning happens, concretely.

The specification of what a layer does to its input data is stored in the layer’s weights, which in essence are a bunch of numbers. In technical terms, we’d say that the transformation implemented by a layer is parameterized by its weights (see figure 1.7). (Weights are also sometimes called the parameters of a layer.) In this context, learning means finding a set of values for the weights of all layers in a network, such that the network will correctly map example inputs to their associated targets. But here’s the thing: a deep neural network can contain tens of millions of parameters. Finding the correct

### PDF p.76 (printed 52) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › 
Now you understand that this model consists of a chain of two `Dense` layers, that each layer applies a few simple tensor operations to the input data, and that these operations involve weight tensors. Weight tensors, which are attributes of the layers, are where the knowledge of the model persists.

This was the model-compilation step:

### PDF p.33 (printed 9) — 1 What is deep learning? › 1.6 Understanding how deep learning works, in three figures › 
value for all of them may seem like a daunting task, especially given that modifying the value of one parameter will affect the behavior of all the others!

Figure 1.7 A neural network is parameterized by its weights.

To control something, first you need to be able to observe it. To control the output of a neural network, you need to be able to measure how far this output is from what you expected. This is the job of the loss function of the network, also sometimes called the objective function or cost function. The loss function takes the predictions of the network and the true target (what you wanted the network to output) and computes a distance score, capturing how well the network has done on this specific example (see figure 1.8).

The fundamental trick in deep learning is to use this score as a feedback signal to adjust the value of the weights a little, in a direction that will lower the loss score for the current example (see figure 1.9). This adjustment is the job of the optimizer, which implements what’s called the Backpropagation algorithm: the central algorithm

Figure 1.8 A loss function measures the quality of the

in deep learning. The next chap

network’s output.


### PDF p.143 (printed 119) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › 
There are two other things you should note about this architecture:

You end the model with a `Dense` layer of size 46. This means for each input sample, the network will output a 46-dimensional vector. Each entry in this vector (each dimension) will encode a different output class.

The last layer uses a `softmax` activation. You saw this pattern in the MNIST example. It means the model will output a probability distribution over the 46 different output classes—for every input sample, the model will produce a 46-dimensional output vector, where `output[i]` is the probability that the sample belongs to class `i`. The 46 scores will sum to 1.

The best loss function to use in this case is `categorical_crossentropy`. It measures the distance between two probability distributions—here, between the probability distribution outputted by the model and the true distribution of the labels. By minimizing the distance between these two distributions, you train the model to output something as close as possible to the true labels.

Like last time, we’ll also monitor accuracy. However, accuracy is a bit of a crude metric in this case: if the model has the correct class as its second choice for

### PDF p.116 (printed 92) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Layers: The building blocks of deep learning
The fundamental data structure in neural networks is the layer, to which you were introduced in chapter 2. A layer is a data processing module that takes as input one or more tensors and that outputs one or more tensors. Some layers are stateless, but more frequently layers have a state: the layer’s weights, one or several tensors learned with stochastic gradient descent, which together contain the network’s knowledge.
