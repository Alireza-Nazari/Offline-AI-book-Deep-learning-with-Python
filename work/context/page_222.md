# Generation bundle for page_222  (PDF page 222, printed page 198)

Page kind: content
Chapter: 7 A deep dive on Keras
Section: 7.2 Different ways to build Keras models
Subsection: 7.2.2 The Functional API
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_222.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_222.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p222_e03] (code)
```
model = keras.Model(
    inputs=[title, text_body, tags],
    outputs=[priority, department],
)
```
[p222_e04] (body) The Functional API is a simple, LEGO-like, yet very flexible way to define arbitrary graphs of layers like these.
[p222_e05] (heading_subsubsection) Training a multi-input, multi-output model
[p222_e06] (body) You can train your model in much the same way as you would train a Sequential model, by calling `fit()` with lists of input and output data. These lists of data should respect the same order as the inputs you passed to the `Model()` constructor.
[p222_e07] (listing_caption) Listing 7.10 Training a model by providing lists of input and target arrays
[p222_e08] (code)
```
import numpy as np
num_samples = 1280
title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))
priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, num_departments, size=(num_samples, 1))
model.compile(
    optimizer="adam",
    loss=["mean_squared_error", "sparse_categorical_crossentropy"],
    metrics=[["mean_absolute_error"], ["accuracy"]],
)
model.fit(
    [title_data, text_body_data, tags_data],
    [priority_data, department_data],
    epochs=1,
)
model.evaluate(
    [title_data, text_body_data, tags_data], [priority_data, department_data]
)
priority_preds, department_preds = model.predict(
    [title_data, text_body_data, tags_data]
)
```
[p222_e09] (body) If you don’t want to rely on input order (for instance, because you have many inputs or outputs), you can also use the names you gave to the `Input` objects and to the output layers, and pass data via dictionaries.

## Code annotations printed next to listings (callouts, in order)
- Creates the model by specifying
- its inputs and outputs
- Dummy input data
- Dummy target data

## Listings
- Listing 7.10: Training a model by providing lists of input and target arrays (code elements p222_e08)

## Equation candidates (formulas written inline in the text)
- [p222_e03] model = keras.Model(
    inputs=[title, text_body, tags],
    outputs=[priority, department],
)

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

## Previous page (PDF 221) text, for continuity
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

## Next page (PDF 223) text, for continuity
Listing 7.11 Training a model by providing dicts of input and target arrays

The power of the Functional API: Access to layer connectivity

A Functional model is an explicit graph data structure. This makes it possible to inspect how layers are connected and reuse previous graph nodes (which are layer outputs) as part of new models. It also nicely fits the “mental model” that most researchers use when thinking about a deep neural network: a graph of layers.

This enables two important use cases: model visualization and feature extraction. Let’s take a look.

Plotting layer connectivity

Let’s visualize the connectivity of the

Figure 7.2 Plot generated by `plot_model()` on

model we just defined (the topology of

our ticket classifier model

the model). You can plot a Functional model as a graph with the `plot_model()` utility, as shown in figure 7.2:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.227 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.219 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.225 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.228 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.230 — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops
- PDF p.119 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.287 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.218 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.219 (printed 195) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › The Functional API
The Sequential model is easy to use, but its applicability is extremely limited: it can only express models with a single input and a single output, applying one layer after the other in a sequential fashion. In practice, it’s pretty common to encounter models with multiple inputs (say, an image and its metadata), multiple outputs (different things you want to predict about the data), or a nonlinear topology.

In such cases, you’d build your model using the Functional API. This is what most Keras models you’ll encounter in the wild use. It’s fun and powerful—it feels like playing with LEGO bricks.

### PDF p.254 (printed 230) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Summary
Keras offers a spectrum of different workflows, based on the principle of progressive disclosure of complexity. They all smoothly interoperate.

You can build models via the `Sequential` class, via the Functional API, or by subclassing the `Model` class. Most of the time, you’ll be using the Functional API.

The simplest way to train and evaluate a model is via the default `fit()` and `evaluate()` methods.

Keras callbacks provide a simple way to monitor models during your call to `fit()` and automatically take action based on the state of the model.

You can also fully control what `fit()` does by overriding the `train_step()` method, using APIs from your backend of choice—JAX, TensorFlow, or PyTorch.

Beyond `fit()`, you can also write your own training loops entirely from scratch, in a backend-native way. This is useful for researchers implementing brand-new training algorithms.

### PDF p.216 (printed 192) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › Different ways to build Keras models
There are three APIs for building models in Keras, as shown in figure 7.1:

The Sequential model is the most approachable API—it’s basically a Python list. As such, it’s limited to simple stacks of layers.

The Functional API, which focuses on graph-like model architectures. It represents a nice mid-point between usability and flexibility, and as such, it’s the most commonly used model-building API.

Model subclassing, a low-level option where you write everything yourself from scratch. This is ideal if you want full control over every little thing. However, you won’t get access to many built-in Keras features, and you will be more at risk of making mistakes.

Figure 7.1 Progressive disclosure of complexity for model building

### PDF p.228 (printed 204) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › Mixing and matching different components
Crucially, choosing one of these patterns—the Sequential model, the Functional API, `Model` subclassing—does not lock you out of the others. All models in the Keras API can smoothly interoperate with each other, whether they’re Sequential models, Functional models, or subclassed models written from scratch. They’re all part of the same spectrum of workflows. For instance, you can use a subclassed layer or model in a Functional model.

Listing 7.15 Creating a Functional model that includes a subclassed model

### PDF p.230 (printed 206) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › 
Going forward, all examples in this book will use the Functional API—simply because all of the models we will work with are expressible as graphs of layers. We will, however, make frequent use of subclassed layers. In general, using Functional models that include subclassed layers provides the best of both worlds: high development flexibility while retaining the advantages of the Functional API.

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

### PDF p.219 (printed 195) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › A simple example
Let’s start with something simple: the two-layer stack we used in the previous section. Its Functional API version looks like the following listing.

Listing 7.8 A simple Functional model with two `Dense` layers

### PDF p.120 (printed 96) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › From layers to models
A deep learning model is a graph of layers. In Keras, that’s the `Model` class. For now, you’ve only seen `Sequential` models (a subclass of `Model`), which are simple stacks of layers, mapping a single input to a single output. But as you move forward, you’ll be exposed to a much broader variety of network topologies. Some common ones are

Two-branch networks

Multihead networks

Residual connections

Network topology can get quite involved. For instance, figure 3.5 shows topology of the graph of layers of a Transformer, a common architecture designed to process text data.

There are generally two ways of building such models in Keras: you can directly subclass the `Model` class, or you can use the Functional API, which lets you do more with less code. We’ll cover both approaches in chapter 7. The topology of a model defines a hypothesis space. You may remember that in chapter 1, we described machine learning as “searching for useful representations of some input data, within a predefined space of possibilities, using guidance from a feedback signal.” By choosing a network topology, you constrain your space of possibilities (hypothesis space) to a specific series of tensor operati
