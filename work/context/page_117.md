# Generation bundle for page_117  (PDF page 117, printed page 93)

Page kind: content
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 3.6 Introduction to Keras
Subsection: 3.6.2 Layers: The building blocks of deep learning
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_117.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_117.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p117_e03] (body) Different types of layers are appropriate for different tensor formats and different types of data processing. For instance, simple vector data, stored in 2D tensors of shape `(samples,` `features)`, is often processed by densely connected layers, also called fully connected or dense layers (the `Dense` class in Keras). Sequence data, stored in 3D tensors of shape `(samples,` `timesteps,` `features)`, is typically processed by recurrent layers, such as an LSTM layer, or 1D convolution layers (`Conv1D`). Image data, stored in rank-4 tensors, is usually processed by 2D convolution layers (`Conv2D`). You can think of layers as the LEGO bricks of deep learning, a metaphor that is made explicit by Keras. Building deep learning models in Keras is done by clipping together compatible layers to form useful data transformation pipelines.
[p117_e04] (heading_subsubsection) The base Layer class in Keras
[p117_e05] (body) A simple API should have a single abstraction around which everything is centered. In Keras, that’s the `Layer` class. Everything in Keras is either a `Layer` or something that closely interacts with a `Layer`.
[p117_e06] (body) A `Layer` is an object that encapsulates some state (weights) and some computation (a forward pass). The weights are typically defined in a `build()` (although they could also be created in the constructor `__init__()`), and the computation is defined in the `call()` method. In the previous chapter, we implemented a `NaiveDense` class that contained two weights `W` and `b` and applied the computation `output = activation(matmul(input, W) + ` `b)`. The following is what the same layer would look like in Keras.
[p117_e07] (listing_caption) Listing 3.33 A simple dense layer from scratch in Keras
[p117_e08] (code)
```
import keras
class SimpleDense(keras.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation
    def build(self, input_shape):
        batch_dim, input_dim = input_shape
        self.W = self.add_weight(
            shape=(input_dim, self.units), initializer="random_normal"
        )
        self.b = self.add_weight(shape=(self.units,), initializer="zeros")
    def call(self, inputs):
        y = keras.ops.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y
```

## Code annotations printed next to listings (callouts, in order)
- All Keras layers inherit
- from the base Layer class.
- Weight creation takes place
- in the build() method.
- We define the
- forward pass
- computation in
- the call() method.
- add_weight is a shortcut method for creating weights. It’s also
- possible to create standalone variables and assign them as layer
- attributes, like self.W = keras.Variable(shape=…, initializer=…).

## Listings
- Listing 3.33: A simple dense layer from scratch in Keras (code elements p117_e08)

## Equation candidates (formulas written inline in the text)
- [p117_e06] In the previous chapter, we implemented a `NaiveDense` class that contained two weights `W` and `b` and applied the computation `output = activation(matmul(input, W) + ` `b)`.

## Chapter outline
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras (PDF pages 84-127)
  - 3.1 A brief history of deep learning frameworks (p.85)
  - 3.2 How these frameworks relate to each other (p.87)
  - 3.3 Introduction to TensorFlow (p.87)
      - 3.3.1 First steps with TensorFlow (p.88)
      - 3.3.2 An end-to-end example: A linear classifier in pure TensorFlow (p.93)
      - 3.3.3 What makes the TensorFlow approach unique (p.98)
  - 3.4 Introduction to PyTorch (p.98)
      - 3.4.1 First steps with PyTorch (p.99)
      - 3.4.2 An end-to-end example: A linear classifier in pure PyTorch (p.102)
      - 3.4.3 What makes the PyTorch approach unique (p.105)
  - 3.5 Introduction to JAX (p.106)
      - 3.5.1 First steps with JAX (p.106)
      - 3.5.2 Tensors in JAX (p.107)
      - 3.5.3 Random number generation in JAX (p.107)
      - 3.5.4 An end-to-end example: A linear classifier in pure JAX (p.112)
      - 3.5.5 What makes the JAX approach unique (p.114)
  - 3.6 Introduction to Keras (p.114)
      - 3.6.1 First steps with Keras (p.115)
      - 3.6.2 Layers: The building blocks of deep learning (p.116)
      - 3.6.3 From layers to models (p.120)
      - 3.6.4 The “compile” step: Configuring the learning process (p.121)
      - 3.6.5 Picking a loss function (p.123)
      - 3.6.6 Understanding the fit method (p.124)
      - 3.6.7 Monitoring loss and metrics on validation data (p.125)
      - 3.6.8 Inference: Using a model after training (p.126)

## Previous page (PDF 116) text, for continuity
at the top of your Python file or notebook (note that it must imperatively go before the first `import keras`):

Edit your local Keras configuration file at `~/.keras/keras.json`. If you have already imported Keras once, this file has already been created with default settings. You can use any text editor to open and modify it—it’s a human-readable JSON file. It should look like this:

NOTE When configuring the Keras backend, you should use the string `"torch"` to refer to the PyTorch backend, rather than the string `"pytorch"`, which would be invalid. This is because the PyTorch package name is `torch` (as in `import ` `torch` or `pip` `install` `torch`). Now, you may ask, which backend should I be picking? It’s really your own choice: all Keras code examples in the rest of the book will be compatible with all three backends. If the need for backend-specific code arises (as in chapter 7, for instance), we will show you all three versions—TensorFlow, PyTorch, JAX. If you have no particular backend preference, our personal recommendation is JAX. It’s usually the most performant backend. Once your backend is configured, you can start actually building and training Keras models. Let’s take a look.

3.6.2

Layers: The building blocks of deep learning

The fundamental data structure in neural networks is the layer, to which you were introduced in chapter 2. A layer is a data processing module that takes as input one or more tensors and that outputs one or more tensors. Some layers are stateless, but more frequently layers have a state: the layer’s weights, one or several tensors learned with stochastic gradient descent, which together contain the network’s knowledge.

## Next page (PDF 118) text, for continuity
In the next section, we’ll cover in detail the purpose of these `build()` and `call()` methods. Don’t worry if you don’t understand everything just yet!

Once instantiated, a layer like this can be used just like a function, taking as input a tensor:

Now, you’re probably wondering, why did we have to implement `call()` and `build()`, since we ended up using our layer by plainly calling it, that is to say, by using its `__call__()` method? It’s because we want to be able to create the state just in time. Let’s see how that works.

Automatic shape inference: Building layers on the fly

Just like with LEGO bricks, you can only “clip” together layers that are compatible. The notion of layer compatibility here refers specifically to the fact that every layer will only accept input tensors of a certain shape and will return output tensors of a certain shape. Consider the following example:

This layer will return a tensor whose non-batch dimension is 32. It can only be connected to a downstream layer that expects 32-dimensional vectors as its input.

When using Keras, you don’t have to worry about size compatibility most of the time because the layers you add to your models are dynamically built to match the shape of the incoming inputs. For instance, suppose you write the following:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.119 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.77 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.625 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.219 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.229 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.120 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.127 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.220 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.43 (printed 19) — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network › 
The core building block of neural networks is the layer. You can think of a layer as a filter for data: some data goes in, and it comes out in a more useful form. Specifically, layers extract representations out of the data fed into them—hopefully, representations that are more meaningful for the problem at hand. Most of deep learning consists of chaining together simple layers that will implement a form of progressive data distillation. A deep learning model is like a sieve for data processing, made of a succession of increasingly refined data filters—the layers.

Here, our model consists of a sequence of two `Dense` layers, which are densely connected (also called fully connected) neural layers. The second (and last) layer is a 10-way `softmax` classification layer, which means it will return an array of 10 probability scores (summing to 1). Each score will be the probability that the current digit image belongs to one of our 10 digit classes. To make the model ready for training, we need to pick three more things, as part of the compilation step:

A loss function—How the model will be able to measure its performance on the training data and thus how it will be able to steer itse

### PDF p.120 (printed 96) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › From layers to models
A deep learning model is a graph of layers. In Keras, that’s the `Model` class. For now, you’ve only seen `Sequential` models (a subclass of `Model`), which are simple stacks of layers, mapping a single input to a single output. But as you move forward, you’ll be exposed to a much broader variety of network topologies. Some common ones are

Two-branch networks

Multihead networks

Residual connections

Network topology can get quite involved. For instance, figure 3.5 shows topology of the graph of layers of a Transformer, a common architecture designed to process text data.

There are generally two ways of building such models in Keras: you can directly subclass the `Model` class, or you can use the Functional API, which lets you do more with less code. We’ll cover both approaches in chapter 7. The topology of a model defines a hypothesis space. You may remember that in chapter 1, we described machine learning as “searching for useful representations of some input data, within a predefined space of possibilities, using guidance from a feedback signal.” By choosing a network topology, you constrain your space of possibilities (hypothesis space) to a specific series of tensor operati

### PDF p.226 (printed 202) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › Subclassing the Model class
The last model-building pattern you should know about is the most advanced one: `Model` subclassing. You’ve already learned in chapter 3 how to subclass the `Layer` class to create custom layers. Subclassing `Model` is pretty similar:

In the `__init__()` method, define the layers the model will use.

In the `call` method, define the forward pass of the model, reusing the layers previously created.

Instantiate your subclass and call it on data to create its weights.

### PDF p.76 (printed 52) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › 
Now you understand that this model consists of a chain of two `Dense` layers, that each layer applies a few simple tensor operations to the input data, and that these operations involve weight tensors. Weight tensors, which are attributes of the layers, are where the knowledge of the model persists.

This was the model-compilation step:

### PDF p.392 (printed 368) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3.2 A recurrent layer in Keras
The process you just naively implemented in NumPy corresponds to an actual Keras layer—the `SimpleRNN` layer.

There is one minor difference: `SimpleRNN` processes batches of sequences, like all other Keras layers, not a single sequence as in the NumPy example. This means it takes inputs of shape `(batch_size,` `timesteps,` `input_features)` rather than `(timesteps, ` `input_features)`. When specifying the `shape` argument of your initial `Input()`, note that you can set the `timesteps` entry to `None`, which enables your network to process sequences of arbitrary length.

Listing 13.16 An RNN layer that can process sequences of any length

### PDF p.219 (printed 195) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › A simple example
Let’s start with something simple: the two-layer stack we used in the previous section. Its Functional API version looks like the following listing.

Listing 7.8 A simple Functional model with two `Dense` layers

### PDF p.220 (printed 196) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › 
All Keras layers can be called both on real tensors of data or on these symbolic tensors. In the latter case, they return a new symbolic tensor, with updated shape and dtype information:

### PDF p.626 (printed 602) — 20 Conclusions › 20.1 Key concepts in review › ConvNets
Convolution layers look at spatially local patterns by applying the same geometric transformation to different spatial locations (patches) in an input tensor. This results in representations that are translation invariant, making convolution layers highly data efficient and modular. This idea is applicable to spaces of any dimensionality: 1D (continuous sequences), 2D (images), 3D (volumes), and so on. You can use the `Conv1D` layer to process sequences, the `Conv2D` layer to process images, and the `Conv3D` layer to process volumes. As a leaner, more efficient alternative to convolution layers, you can also use depthwise separable convolution layers, such as `SeparableConv2D`.

ConvNets, or convolutional networks, consist of stacks of convolution and max-pooling layers. The pooling layers let you spatially downsample the data, which is required to keep feature maps to a reasonable size as the number of features grows and to allow subsequent convolution layers to “see” a greater spatial extent of the inputs. ConvNets are often ended with either a `Flatten` operation or a global pooling layer, turning spatial feature maps into vectors, followed by `Dense` layers to achieve classific
