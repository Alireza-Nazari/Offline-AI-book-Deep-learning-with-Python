# Generation bundle for page_218  (PDF page 218, printed page 194)

Page kind: content
Chapter: 7 A deep dive on Keras
Section: 7.2 Different ways to build Keras models
Subsection: 7.2.1 The Sequential model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_218.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_218.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p218_e03] (body) As you can see, your model happens to be named `sequential_1`. You can actually give names to everything in Keras—every model, every layer.
[p218_e04] (listing_caption) Listing 7.6 Naming models and layers with the `name` argument
[p218_e05] (code)
```
>>> model = keras.Sequential(name="my_example_model")
>>> model.add(layers.Dense(64, activation="relu", name="my_first_layer"))
>>> model.add(layers.Dense(10, activation="softmax", name="my_last_layer"))
>>> model.build((None, 3))
>>> model.summary()
Model: "my_example_model"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ my_first_layer (Dense)            │ (None, 64)               │           256 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ my_last_layer (Dense)             │ (None, 10)               │           650 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
 Total params: 906 (3.54 KB)
 Trainable params: 906 (3.54 KB)
 Non-trainable params: 0 (0.00 B)
```
[p218_e06] (body) When building a Sequential model incrementally, it’s useful to be able to print a summary of what the current model looks like after you add each layer. But you can’t print a summary until the model is built! There’s actually a way to have your Sequential model get built on the fly: just declare the shape of the model’s inputs in advance. You can do this via the `Input` class.
[p218_e07] (listing_caption) Listing 7.7 Specifying the input shape of your model in advance
[p218_e08] (code)
```
model = keras.Sequential()
model.add(keras.Input(shape=(3,)))
model.add(layers.Dense(64, activation="relu"))
```
[p218_e09] (body) Now you can use `summary()` to follow how the output shape of your model changes as you add more layers:
[p218_e10] (code)
```
>>> model.summary()
Model: "sequential_2"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_4 (Dense)                   │ (None, 64)               │           256 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
 Total params: 256 (1.00 KB)
```

## Code annotations printed next to listings (callouts, in order)
- Use an Input to declare the shape of
- the inputs. Note that the shape
- argument must be the shape of each
- sample, not the shape of one batch.

## Listings
- Listing 7.6: Naming models and layers with the `name` argument (code elements p218_e05)
- Listing 7.7: Specifying the input shape of your model in advance (code elements p218_e08)

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

## Previous page (PDF 217) text, for continuity
Listing 7.2 Incrementally building a Sequential model

You’ve seen in chapter 3 that layers only get built (which is to say, create their weights) when they are called for the first time. That’s because the shape of the layers’ weights depends on the shape of their input: until the input shape is known, they can’t be created.

As such, the previous Sequential model does not have any weights until you actually call it on some data, or call its `build()` method with an input shape.

Listing 7.3 Models that aren’t yet built have no weights

Listing 7.4 Calling a model for the first time to build it

After the model is built, you can display its contents via the `summary()` method, which comes in handy for debugging.

Listing 7.5 The summary method

## Next page (PDF 219) text, for continuity
This is a pretty common debugging workflow when dealing with layers that transform their inputs in complex ways, such as the convolutional layers you’ll learn about in chapter 8.

7.2.2

The Functional API

The Sequential model is easy to use, but its applicability is extremely limited: it can only express models with a single input and a single output, applying one layer after the other in a sequential fashion. In practice, it’s pretty common to encounter models with multiple inputs (say, an image and its metadata), multiple outputs (different things you want to predict about the data), or a nonlinear topology.

In such cases, you’d build your model using the Functional API. This is what most Keras models you’ll encounter in the wild use. It’s fun and powerful—it feels like playing with LEGO bricks.

A simple example

Let’s start with something simple: the two-layer stack we used in the previous section. Its Functional API version looks like the following listing.

Listing 7.8 A simple Functional model with two `Dense` layers

Let’s go over this step by step. We started by declaring an `Input` (note that you can also give names to these input objects, like everything else):

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.220 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.225 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.216 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.221 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.257 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.118 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.143 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.119 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.216 (printed 192) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › Different ways to build Keras models
There are three APIs for building models in Keras, as shown in figure 7.1:

The Sequential model is the most approachable API—it’s basically a Python list. As such, it’s limited to simple stacks of layers.

The Functional API, which focuses on graph-like model architectures. It represents a nice mid-point between usability and flexibility, and as such, it’s the most commonly used model-building API.

Model subclassing, a low-level option where you write everything yourself from scratch. This is ideal if you want full control over every little thing. However, you won’t get access to many built-in Keras features, and you will be more at risk of making mistakes.

Figure 7.1 Progressive disclosure of complexity for model building

### PDF p.216 (printed 192) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › The Sequential model
The simplest way to build a Keras model is the Sequential model, which you already know about.

Listing 7.1 The `Sequential` class

### PDF p.214 (printed 190) — 7 A deep dive on Keras ›  › This chapter covers
The different ways to create Keras models: the

`Sequential` class, the Functional API, and model

subclassing

How to use the built-in Keras training and

evaluation loops, including how to use custom

metrics and custom losses

Using Keras callbacks to further customize how

training proceeds

Using TensorBoard for monitoring your training

and evaluation metrics over time

How to write your own training and evaluation

loops from scratch

You’re starting to have some amount of experience with Keras. You’re familiar with the Sequential model, `Dense` layers, and built-in APIs for training, evaluation, and inference—`compile()`, `fit()`, `evaluate()`, and `predict()`. You’ve even learned in chapter 3 how to inherit from the `Layer` class to create custom layers, and how to

### PDF p.227 (printed 203) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › 
So far, everything looks very similar to `Layer` subclassing, a workflow you’ve already encountered in chapter 3. What, then, is the difference between a `Layer` subclass and a `Model` subclass? It’s simple: a layer is a building block you use to create models, and a model is the top-level object that you will actually train, export for inference, etc. In short, a `Model` has a `fit()`, `evaluate()`, and `predict()` method. Layers don’t. Other than that, the two classes are virtually identical (another difference is that you can save a model to a file on disk—which we will cover in a few sections).

You can compile and train a `Model` subclass just like a Sequential or Functional model:

### PDF p.254 (printed 230) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Summary
Keras offers a spectrum of different workflows, based on the principle of progressive disclosure of complexity. They all smoothly interoperate.

You can build models via the `Sequential` class, via the Functional API, or by subclassing the `Model` class. Most of the time, you’ll be using the Functional API.

The simplest way to train and evaluate a model is via the default `fit()` and `evaluate()` methods.

Keras callbacks provide a simple way to monitor models during your call to `fit()` and automatically take action based on the state of the model.

You can also fully control what `fit()` does by overriding the `train_step()` method, using APIs from your backend of choice—JAX, TensorFlow, or PyTorch.

Beyond `fit()`, you can also write your own training loops entirely from scratch, in a backend-native way. This is useful for researchers implementing brand-new training algorithms.

### PDF p.118 (printed 94) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Automatic shape inference: Building layers on the fly
from keras import models
from keras import layers
model = models.Sequential(
    [
        layers.Dense(32, activation="relu"),
        layers.Dense(32),
    ]
)

### PDF p.228 (printed 204) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › Mixing and matching different components
Crucially, choosing one of these patterns—the Sequential model, the Functional API, `Model` subclassing—does not lock you out of the others. All models in the Keras API can smoothly interoperate with each other, whether they’re Sequential models, Functional models, or subclassed models written from scratch. They’re all part of the same spectrum of workflows. For instance, you can use a subclassed layer or model in a Functional model.

Listing 7.15 Creating a Functional model that includes a subclassed model

### PDF p.225 (printed 201) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › 
 <InputLayer name=text_body, built=True>,
 <InputLayer name=tags, built=True>,
 <Concatenate name=concatenate, built=True>,
 <Dense name=dense_10, built=True>,
 <Dense name=priority, built=True>,
 <Dense name=department, built=True>]
>>> model.layers[3].input
[<KerasTensor shape=(None, 10000), dtype=float32, sparse=None,
name=title>,
 <KerasTensor shape=(None, 10000), dtype=float32, sparse=None, name=text_body>,
 <KerasTensor shape=(None, 100), dtype=float32, sparse=None, name=tags>]
>>> model.layers[3].output
<KerasTensor shape=(None, 20100), dtype=float32, sparse=False>
