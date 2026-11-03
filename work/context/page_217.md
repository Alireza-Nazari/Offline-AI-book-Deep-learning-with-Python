# Generation bundle for page_217  (PDF page 217, printed page 193)

Page kind: content
Chapter: 7 A deep dive on Keras
Section: 7.2 Different ways to build Keras models
Subsection: 7.2.1 The Sequential model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_217.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_217.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p217_e03] (listing_caption) Listing 7.2 Incrementally building a Sequential model
[p217_e04] (code)
```
model = keras.Sequential()
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))
```
[p217_e05] (body) You’ve seen in chapter 3 that layers only get built (which is to say, create their weights) when they are called for the first time. That’s because the shape of the layers’ weights depends on the shape of their input: until the input shape is known, they can’t be created.
[p217_e06] (body) As such, the previous Sequential model does not have any weights until you actually call it on some data, or call its `build()` method with an input shape.
[p217_e07] (listing_caption) Listing 7.3 Models that aren’t yet built have no weights
[p217_e08] (code)
```
>>> model.weights
[]
```
[p217_e09] (listing_caption) Listing 7.4 Calling a model for the first time to build it
[p217_e10] (code)
```
>>> model.build(input_shape=(None, 3))
>>> model.weights
[<Variable shape=(3, 64), dtype=float32, path=sequential/dense_2/kernel ...>,
 <Variable shape=(64,), dtype=float32, path=sequential/dense_2/bias ...>,
 <Variable shape=(64, 10), dtype=float32, path=sequential/dense_3/kernel ...>,
 <Variable shape=(10,), dtype=float32, path=sequential/dense_3/bias ...>>]
```
[p217_e11] (body) After the model is built, you can display its contents via the `summary()` method, which comes in handy for debugging.
[p217_e12] (listing_caption) Listing 7.5 The summary method
[p217_e13] (code)
```
>>> model.summary()
Model: "sequential_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_2 (Dense)                   │ (None, 64)               │           256 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ dense_3 (Dense)                   │ (None, 10)               │           650 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
 Total params: 906 (3.54 KB)
 Trainable params: 906 (3.54 KB)
 Non-trainable params: 0 (0.00 B)
```

## Code annotations printed next to listings (callouts, in order)
- At that point, the
- model isn’t built yet.
- Builds the model. Now the model will expect
- samples of shape (3,). The None in the input shape
- signals that the batch size could be anything.
- Now you can retrieve
- the model’s weights.

## Listings
- Listing 7.2: Incrementally building a Sequential model (code elements p217_e04)
- Listing 7.3: Models that aren’t yet built have no weights (code elements p217_e08)
- Listing 7.4: Calling a model for the first time to build it (code elements p217_e10)
- Listing 7.5: The summary method (code elements p217_e13)

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

## Previous page (PDF 216) text, for continuity
7.2

Different ways to build Keras models

There are three APIs for building models in Keras, as shown in figure 7.1:

The Sequential model is the most approachable API—it’s basically a Python list. As such, it’s limited to simple stacks of layers.

The Functional API, which focuses on graph-like model architectures. It represents a nice mid-point between usability and flexibility, and as such, it’s the most commonly used model-building API.

Model subclassing, a low-level option where you write everything yourself from scratch. This is ideal if you want full control over every little thing. However, you won’t get access to many built-in Keras features, and you will be more at risk of making mistakes.

Figure 7.1 Progressive disclosure of complexity for model building

7.2.1

The Sequential model

The simplest way to build a Keras model is the Sequential model, which you already know about.

Listing 7.1 The `Sequential` class

Note that it’s possible to build the same model incrementally via the `add()` method, similar to the `append()` method of a Python list.

## Next page (PDF 218) text, for continuity
As you can see, your model happens to be named `sequential_1`. You can actually give names to everything in Keras—every model, every layer.

Listing 7.6 Naming models and layers with the `name` argument

When building a Sequential model incrementally, it’s useful to be able to print a summary of what the current model looks like after you add each layer. But you can’t print a summary until the model is built! There’s actually a way to have your Sequential model get built on the fly: just declare the shape of the model’s inputs in advance. You can do this via the `Input` class.

Listing 7.7 Specifying the input shape of your model in advance

Now you can use `summary()` to follow how the output shape of your model changes as you add more layers:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.219 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.225 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.220 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.119 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.578 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.222 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.226 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.227 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.119 (printed 95) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
It would be even worse when the rules used by a layer to produce its output shape are complex. For instance, what if our layer returned outputs of shape `(batch,` `input_size ` `* 2 if input_size % 2 == 0 else input_size * 3)?`

If we were to reimplement our `NaiveDense` layer as a Keras layer capable of automatic shape inference, it would look like the `SimpleDense` layer, with its `build()` and `call()` methods. In the Keras `SimpleDense`, we no longer create weights in the constructor like in the previous example. Instead, we create them in a dedicated state-creation method `build()`, which receives as argument the first input shape seen by the layer. The `build()` method is called automatically the first time the layer is called (via its `__call__()` method). In fact, that’s why we defined the computation in a separate `call()` method rather than in the `__call__()` method directly! The `__call__()` method of the base layer schematically looks like this:

### PDF p.226 (printed 202) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › Subclassing the Model class
The last model-building pattern you should know about is the most advanced one: `Model` subclassing. You’ve already learned in chapter 3 how to subclass the `Layer` class to create custom layers. Subclassing `Model` is pretty similar:

In the `__init__()` method, define the layers the model will use.

In the `call` method, define the forward pass of the model, reusing the layers previously created.

Instantiate your subclass and call it on data to create its weights.

### PDF p.227 (printed 203) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › 
Once you’ve defined the model, you can instantiate it. Note that it will only create its weights the first time you call it on some data—much like `Layer` subclasses:

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

### PDF p.119 (printed 95) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
The layers didn’t receive any information about the shape of their inputs. Instead, they automatically inferred their input shape as being the shape of the first inputs they see.

In the toy version of a `Dense` layer that we’ve implemented in chapter 2, we had to pass the layer’s input size explicitly to the constructor in order to be able to create its weights. That’s not ideal, because it would lead to models that look like this, where each new layer needs to be made aware of the shape of the layer before it:

### PDF p.78 (printed 54) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › A simple Sequential class
Now, let’s create a `NaiveSequential` class to chain these layers. It wraps a list of layers and exposes a `__call__()` method that simply calls the underlying layers on the inputs, in order. It also features a `weights` property to easily keep track of the layers’ parameters:

### PDF p.118 (printed 94) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
Now, you’re probably wondering, why did we have to implement `call()` and `build()`, since we ended up using our layer by plainly calling it, that is to say, by using its `__call__()` method? It’s because we want to be able to create the state just in time. Let’s see how that works.

### PDF p.228 (printed 204) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › 
The `Model` subclassing workflow is the most flexible way to build a model: it enables you to build models that cannot be expressed as directed acyclic graphs of layers— imagine, for instance, a model where the `call()` method uses layers inside a `for` loop, or even calls them recursively. Anything is possible—you’re in charge.
