# Generation bundle for page_120  (PDF page 120, printed page 96)

Page kind: content
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 3.6 Introduction to Keras
Subsection: 3.6.3 From layers to models
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_120.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_120.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p120_e03] (code)
```
        SimpleDense(10, activation="softmax"),
    ]
)
```
[p120_e04] (body) Note that automatic shape inference is not the only thing that the `Layer` class’s `__call__()` method handles. It takes care of many more things, in particular routing between eager and graph execution, and input masking (which we cover in chapter 14). For now, just remember: when implementing your own layers, put the forward pass in the `call()` method.
[p120_e05] (heading_subsection) 3.6.3
[p120_e06] (heading_subsection) From layers to models
[p120_e07] (body) A deep learning model is a graph of layers. In Keras, that’s the `Model` class. For now, you’ve only seen `Sequential` models (a subclass of `Model`), which are simple stacks of layers, mapping a single input to a single output. But as you move forward, you’ll be exposed to a much broader variety of network topologies. Some common ones are
[p120_e08] (bullet) Two-branch networks
[p120_e09] (bullet) Multihead networks
[p120_e10] (bullet) Residual connections
[p120_e11] (body) Network topology can get quite involved. For instance, figure 3.5 shows topology of the graph of layers of a Transformer, a common architecture designed to process text data.
[p120_e12] (body) There are generally two ways of building such models in Keras: you can directly subclass the `Model` class, or you can use the Functional API, which lets you do more with less code. We’ll cover both approaches in chapter 7. The topology of a model defines a hypothesis space. You may remember that in chapter 1, we described machine learning as “searching for useful representations of some input data, within a predefined space of possibilities, using guidance from a feedback signal.” By choosing a network topology, you constrain your space of possibilities (hypothesis space) to a specific series of tensor operations, mapping input data to output data. What you’ll then be searching for is a good set of values for the weight tensors involved in these tensor operations. To learn from data, you have to make assumptions about it. These assumptions define what can be learned. As such, the structure of your hypothesis space—the architecture of your model—is extremely important. It encodes the assumptions you make about your problem, the prior knowledge that the model starts with. For instance, if you’re working on a two-class classification problem with a model made of a single `Dense` layer with no activation (a pure affine transformation), you are assuming that your two classes are linearly separable. Picking the right network architecture is more an art than a science, and although there are some best practices and principles you can rely on, only practice can help you become a proper neural network architect. The next few chapters will both teach you explicit principles for building neural networks and help you develop intuition as to

## Equation candidates (formulas written inline in the text)
- [p120_e03]         SimpleDense(10, activation="softmax"),
    ]
)

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

## Previous page (PDF 119) text, for continuity
The layers didn’t receive any information about the shape of their inputs. Instead, they automatically inferred their input shape as being the shape of the first inputs they see.

In the toy version of a `Dense` layer that we’ve implemented in chapter 2, we had to pass the layer’s input size explicitly to the constructor in order to be able to create its weights. That’s not ideal, because it would lead to models that look like this, where each new layer needs to be made aware of the shape of the layer before it:

It would be even worse when the rules used by a layer to produce its output shape are complex. For instance, what if our layer returned outputs of shape `(batch,` `input_size ` `* 2 if input_size % 2 == 0 else input_size * 3)?`

If we were to reimplement our `NaiveDense` layer as a Keras layer capable of automatic shape inference, it would look like the `SimpleDense` layer, with its `build()` and `call()` methods. In the Keras `SimpleDense`, we no longer create weights in the constructor like in the previous example. Instead, we create them in a dedicated state-creation method `build()`, which receives as argument the first input shape seen by the layer. The `build()` method is called automatically the first time the layer is called (via its `__call__()` method). In fact, that’s why we defined the computation in a separate `call()` method rather than in the `__call__()` method directly! The `__call__()` method of the base layer schematically looks like this:

With automatic shape inference, our previous example becomes simple and neat:

## Next page (PDF 121) text, for continuity
Figure 3.5 The Transformer architecture. There’s a lot going on here. Throughout the next few chapters, you’ll climb your way up to understanding it (in chapter 15).

what works or doesn’t work for specific problems. You’ll build a solid intuition about what type of model architectures work for different kinds of problems, how to build these networks in practice, how to pick the right learning configuration, and how to tweak a model until it yields the results you want to see.

3.6.4

The “compile” step: Configuring the learning process

Once the model architecture is defined, you still have to choose three more things:

Loss function (objective function)—The quantity that will be minimized during training. It represents a measure of success for the task at hand.

Optimizer—Determines how the network will be updated based on the loss function. It implements a specific variant of stochastic gradient descent (SGD).

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.116 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.219 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.117 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.627 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.223 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.76 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.104 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.226 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.292 (printed 268) — 9 ConvNet architecture patterns ›  › This chapter covers
The modularity-hierarchy-reuse formula for model

architecture

An overview of standard best practices for

building ConvNets: residual connections,

batch normalization, and depthwise separable

convolutions

Ongoing design trends for computer vision

models

A model’s “architecture” is the sum of the choices that went into creating it: which layers to use, how to configure them, in what arrangement to connect them. These choices define the hypothesis space of your model: the space of possible functions that gradient descent can search over, parameterized by the model’s weights. Like feature engineering, a good hypothesis space encodes prior knowledge that you have about the problem at hand and its solution. For instance, using convolution layers means that you know in advance that the relevant patterns present in your input

### PDF p.226 (printed 202) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › Subclassing the Model class
The last model-building pattern you should know about is the most advanced one: `Model` subclassing. You’ve already learned in chapter 3 how to subclass the `Layer` class to create custom layers. Subclassing `Model` is pretty similar:

In the `__init__()` method, define the layers the model will use.

In the `call` method, define the forward pass of the model, reusing the layers previously created.

Instantiate your subclass and call it on data to create its weights.

### PDF p.228 (printed 204) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › Beware: What subclassed models don’t support
This freedom comes at a cost: with subclassed models, you are responsible for more of the model logic, which means your potential error surface is much larger. As a result, you will have more debugging work to do. You are developing a new Python object, not just snapping together LEGO bricks.

Functional and subclassed models are also substantially different in nature: a Functional model is an explicit data structure—a graph of layers, which you can view, inspect, and modify. Meanwhile, a subclassed model is a piece of bytecode—a Python class with a `call()` method that contains raw code. This is the source of the subclassing workflow’s flexibility—you can just code up whatever functionality you like—but it introduces new limitations. For instance, because the way layers are connected to each other is hidden inside the body of the `call()` method, you cannot access that information. Calling `summary()` will not display layer connectivity, and you cannot plot the model topology via `plot_` `model()`. Likewise, if you have a subclassed model, you cannot access the nodes of the graph of layers to do feature extraction—because there is simply no graph. Once the model is instantiated, i

### PDF p.223 (printed 199) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › The power of the Functional API: Access to layer connectivity
A Functional model is an explicit graph data structure. This makes it possible to inspect how layers are connected and reuse previous graph nodes (which are layer outputs) as part of new models. It also nicely fits the “mental model” that most researchers use when thinking about a deep neural network: a graph of layers.

This enables two important use cases: model visualization and feature extraction. Let’s take a look.

### PDF p.227 (printed 203) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › 
So far, everything looks very similar to `Layer` subclassing, a workflow you’ve already encountered in chapter 3. What, then, is the difference between a `Layer` subclass and a `Model` subclass? It’s simple: a layer is a building block you use to create models, and a model is the top-level object that you will actually train, export for inference, etc. In short, a `Model` has a `fit()`, `evaluate()`, and `predict()` method. Layers don’t. Other than that, the two classes are virtually identical (another difference is that you can save a model to a file on disk—which we will cover in a few sections).

You can compile and train a `Model` subclass just like a Sequential or Functional model:

### PDF p.117 (printed 93) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
Different types of layers are appropriate for different tensor formats and different types of data processing. For instance, simple vector data, stored in 2D tensors of shape `(samples,` `features)`, is often processed by densely connected layers, also called fully connected or dense layers (the `Dense` class in Keras). Sequence data, stored in 3D tensors of shape `(samples,` `timesteps,` `features)`, is typically processed by recurrent layers, such as an LSTM layer, or 1D convolution layers (`Conv1D`). Image data, stored in rank-4 tensors, is usually processed by 2D convolution layers (`Conv2D`). You can think of layers as the LEGO bricks of deep learning, a metaphor that is made explicit by Keras. Building deep learning models in Keras is done by clipping together compatible layers to form useful data transformation pipelines.

### PDF p.216 (printed 192) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › Different ways to build Keras models
There are three APIs for building models in Keras, as shown in figure 7.1:

The Sequential model is the most approachable API—it’s basically a Python list. As such, it’s limited to simple stacks of layers.

The Functional API, which focuses on graph-like model architectures. It represents a nice mid-point between usability and flexibility, and as such, it’s the most commonly used model-building API.

Model subclassing, a low-level option where you write everything yourself from scratch. This is ideal if you want full control over every little thing. However, you won’t get access to many built-in Keras features, and you will be more at risk of making mistakes.

Figure 7.1 Progressive disclosure of complexity for model building

### PDF p.118 (printed 94) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Automatic shape inference: Building layers on the fly
from keras import models
from keras import layers
model = models.Sequential(
    [
        layers.Dense(32, activation="relu"),
        layers.Dense(32),
    ]
)
