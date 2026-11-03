# Generation bundle for page_220  (PDF page 220, printed page 196)

Page kind: content
Chapter: 7 A deep dive on Keras
Section: 7.2 Different ways to build Keras models
Subsection: 7.2.2 The Functional API
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_220.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_220.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p220_e03] (body) This `inputs` object holds information about the shape and `dtype` of the data that the model will process:
[p220_e04] (code)
```
>>> inputs.shape
(None, 3)
>>> inputs.dtype
"float32"
```
[p220_e05] (body) We call such an object a symbolic tensor. It doesn’t contain any actual data, but it encodes the specifications of the actual tensors of data that the model will see when you use it. It stands for future tensors of data.
[p220_e06] (body) Next, we created a layer and called it on the input:
[p220_e07] (code)
```
features = layers.Dense(64, activation="relu")(inputs)
```
[p220_e08] (body) All Keras layers can be called both on real tensors of data or on these symbolic tensors. In the latter case, they return a new symbolic tensor, with updated shape and dtype information:
[p220_e09] (code)
```
>>> features.shape
(None, 64)
```
[p220_e10] (body) After obtaining the final outputs, we instantiated the model by specifying its inputs and outputs in the `Model` constructor:
[p220_e11] (code)
```
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs, name="my_functional_model")
```
[p220_e12] (body) Here’s the summary of our model:
[p220_e13] (code)
```
>>> model.summary()
Model: "my_functional_model"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ my_input (InputLayer)             │ (None, 3)                │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ dense_8 (Dense)                   │ (None, 64)               │           256 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ dense_9 (Dense)                   │ (None, 10)               │           650 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
```

## Code annotations printed next to listings (callouts, in order)
- The model will process batches where each sample
- has shape (3,). The number of samples per batch is
- variable (indicated by the None batch size).
- These batches will have dtype float32.

## Equation candidates (formulas written inline in the text)
- [p220_e07] features = layers.Dense(64, activation="relu")(inputs)
- [p220_e11] outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs, name="my_functional_model")

## Chapter outline
Chapter: 7 A deep dive on Keras (PDF pages 214-254)
  - 7.1 A spectrum of workflows (p.215)
  - 7.2 Different ways to build Keras models (p.216)
      - 7.2.1 The Sequential model (p.216)
      - 7.2.2 The Functional API (p.219)
      - 7.2.3 Subclassing the Model class (p.226)
      - 7.2.4 Mixing and matching different components (p.228)
      - 7.2.5 Remember: Use the right tool for the job (p.229)
  - 7.3 Using built-in training and evaluation loops (p.230)
      - 7.3.1 Writing your own metrics (p.231)
      - 7.3.2 Using callbacks (p.232)
      - 7.3.3 Writing your own callbacks (p.234)
      - 7.3.4 Monitoring and visualization with TensorBoard (p.236)
  - 7.4 Writing your own training and evaluation loops (p.238)
      - 7.4.1 Training vs. inference (p.239)
      - 7.4.2 Writing custom training step functions (p.240)
      - 7.4.3 Low-level usage of metrics (p.245)
      - 7.4.4 Using fit() with a custom training loop (p.246)
      - 7.4.5 Handling metrics in a custom train_step() (p.250)

## Previous page (PDF 219) text, for continuity
This is a pretty common debugging workflow when dealing with layers that transform their inputs in complex ways, such as the convolutional layers you’ll learn about in chapter 8.

7.2.2

The Functional API

The Sequential model is easy to use, but its applicability is extremely limited: it can only express models with a single input and a single output, applying one layer after the other in a sequential fashion. In practice, it’s pretty common to encounter models with multiple inputs (say, an image and its metadata), multiple outputs (different things you want to predict about the data), or a nonlinear topology.

In such cases, you’d build your model using the Functional API. This is what most Keras models you’ll encounter in the wild use. It’s fun and powerful—it feels like playing with LEGO bricks.

A simple example

Let’s start with something simple: the two-layer stack we used in the previous section. Its Functional API version looks like the following listing.

Listing 7.8 A simple Functional model with two `Dense` layers

Let’s go over this step by step. We started by declaring an `Input` (note that you can also give names to these input objects, like everything else):

## Next page (PDF 221) text, for continuity
Multi-input, multi-output models

Unlike this toy model, most deep learning models don’t look like lists—they look like graphs. They may, for instance, have multiple inputs or multiple outputs. It’s for this kind of model that the Functional API really shines.

Let’s say you’re building a system to rank customer support tickets by priority and route them to the appropriate department. Your model has three inputs:

The title of the ticket (text input)

The text body of the ticket (text input)

Any tags added by the user (categorical input, assumed here to be multi-hot encoded)

We can encode the text inputs as arrays of 1s and 0s of size `vocabulary_size` (see chapter 14 for detailed information about text encoding techniques).

Your model also has two outputs:

The priority score of the ticket, a scalar between 0 and 1 (sigmoid output)

The department that should handle the ticket (a softmax over the set of departments)

You can build this model in a few lines with the Functional API.

Listing 7.9 A multi-input, multi-output Functional model

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.248 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.218 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.229 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.225 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.251 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.252 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.626 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.625 — 20 Conclusions › 20.1 Key concepts in review

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.117 (printed 93) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
Different types of layers are appropriate for different tensor formats and different types of data processing. For instance, simple vector data, stored in 2D tensors of shape `(samples,` `features)`, is often processed by densely connected layers, also called fully connected or dense layers (the `Dense` class in Keras). Sequence data, stored in 3D tensors of shape `(samples,` `timesteps,` `features)`, is typically processed by recurrent layers, such as an LSTM layer, or 1D convolution layers (`Conv1D`). Image data, stored in rank-4 tensors, is usually processed by 2D convolution layers (`Conv2D`). You can think of layers as the LEGO bricks of deep learning, a metaphor that is made explicit by Keras. Building deep learning models in Keras is done by clipping together compatible layers to form useful data transformation pipelines.

### PDF p.116 (printed 92) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Layers: The building blocks of deep learning
The fundamental data structure in neural networks is the layer, to which you were introduced in chapter 2. A layer is a data processing module that takes as input one or more tensors and that outputs one or more tensors. Some layers are stateless, but more frequently layers have a state: the layer’s weights, one or several tensors learned with stochastic gradient descent, which together contain the network’s knowledge.

### PDF p.119 (printed 95) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
It would be even worse when the rules used by a layer to produce its output shape are complex. For instance, what if our layer returned outputs of shape `(batch,` `input_size ` `* 2 if input_size % 2 == 0 else input_size * 3)?`

If we were to reimplement our `NaiveDense` layer as a Keras layer capable of automatic shape inference, it would look like the `SimpleDense` layer, with its `build()` and `call()` methods. In the Keras `SimpleDense`, we no longer create weights in the constructor like in the previous example. Instead, we create them in a dedicated state-creation method `build()`, which receives as argument the first input shape seen by the layer. The `build()` method is called automatically the first time the layer is called (via its `__call__()` method). In fact, that’s why we defined the computation in a separate `call()` method rather than in the `__call__()` method directly! The `__call__()` method of the base layer schematically looks like this:

### PDF p.76 (printed 52) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › 
Now you understand that this model consists of a chain of two `Dense` layers, that each layer applies a few simple tensor operations to the input data, and that these operations involve weight tensors. Weight tensors, which are attributes of the layers, are where the knowledge of the model persists.

This was the model-compilation step:

### PDF p.119 (printed 95) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
The layers didn’t receive any information about the shape of their inputs. Instead, they automatically inferred their input shape as being the shape of the first inputs they see.

In the toy version of a `Dense` layer that we’ve implemented in chapter 2, we had to pass the layer’s input size explicitly to the constructor in order to be able to create its weights. That’s not ideal, because it would lead to models that look like this, where each new layer needs to be made aware of the shape of the layer before it:

### PDF p.75 (printed 51) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › Looking back at our first example
Now you understand that the input images are stored in NumPy tensors, which are here formatted as `float32` tensors of shape `(60000,` `784)` (training data) and `(10000, ` `784)` (test data), respectively.

This was our model:

### PDF p.217 (printed 193) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › 
You’ve seen in chapter 3 that layers only get built (which is to say, create their weights) when they are called for the first time. That’s because the shape of the layers’ weights depends on the shape of their input: until the input shape is known, they can’t be created.

As such, the previous Sequential model does not have any weights until you actually call it on some data, or call its `build()` method with an input shape.

Listing 7.3 Models that aren’t yet built have no weights

### PDF p.229 (printed 205) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › 
Listing 7.16 Creating a subclassed model that includes a Functional model
inputs = keras.Input(shape=(64,))
outputs = layers.Dense(1, activation="sigmoid")(inputs)
binary_classifier = keras.Model(inputs=inputs, outputs=outputs)
class MyModel(keras.Model):
    def __init__(self, num_classes=2):
        super().__init__()
        self.dense = layers.Dense(64, activation="relu")
        self.classifier = binary_classifier
    def call(self, inputs):
        features = self.dense(inputs)
        return self.classifier(features)
model = MyModel()
