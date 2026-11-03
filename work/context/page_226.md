# Generation bundle for page_226  (PDF page 226, printed page 202)

Page kind: content
Chapter: 7 A deep dive on Keras
Section: 7.2 Different ways to build Keras models
Subsection: 7.2.3 Subclassing the Model class
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_226.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_226.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p226_e03] (figure_caption) Figure 7.4 Plot of our new model
[p226_e04] (heading_subsection) 7.2.3
[p226_e05] (heading_subsection) Subclassing the Model class
[p226_e06] (body) The last model-building pattern you should know about is the most advanced one: `Model` subclassing. You’ve already learned in chapter 3 how to subclass the `Layer` class to create custom layers. Subclassing `Model` is pretty similar:
[p226_e07] (bullet) In the `__init__()` method, define the layers the model will use.
[p226_e08] (bullet) In the `call` method, define the forward pass of the model, reusing the layers previously created.
[p226_e09] (bullet) Instantiate your subclass and call it on data to create its weights.
[p226_e10] (heading_subsubsection) Rewriting our previous example as a subclassed model
[p226_e11] (body) Let’s take a look at a simple example: we will reimplement the customer support ticket management model using a `Model` subclass.
[p226_e12] (listing_caption) Listing 7.14 A simple subclassed model
[p226_e13] (code)
```
class CustomerTicketModel(keras.Model):
    def __init__(self, num_departments):
        super().__init__()
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation="relu")
        self.priority_scorer = layers.Dense(1, activation="sigmoid")
        self.department_classifier = layers.Dense(
            num_departments, activation="softmax"
        )
    def call(self, inputs):
        title = inputs["title"]
```

## Code annotations printed next to listings (callouts, in order)
- Don’t forget to call the
- super constructor!
- Defines sublayers in the constructor
- Defines the forward pass
- in the call() method

## Listings
- Listing 7.14: A simple subclassed model (code elements p226_e13)

## Figures
- Figure 7.4: Plot of our new model  | labels inside the figure: (none - image)

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

## Previous page (PDF 225) text, for continuity
This enables you to do feature extraction: creating models that reuse intermediate features from another model.

Let’s say you want to add another output to the model we previously defined—you want to also predict an estimate of how long a given issue ticket will take to resolve, a kind of difficulty rating. You could do this via a classification layer over three categories—”quick,” “medium,” and “difficult.” You don’t need to recreate and retrain a model from scratch! You can just start from the intermediate features of your previous model, since you have access to them.

Listing 7.13 Creating a new model by reusing intermediate layer outputs

Let’s plot our new model, as shown in figure 7.4:

## Next page (PDF 227) text, for continuity
Once you’ve defined the model, you can instantiate it. Note that it will only create its weights the first time you call it on some data—much like `Layer` subclasses:

So far, everything looks very similar to `Layer` subclassing, a workflow you’ve already encountered in chapter 3. What, then, is the difference between a `Layer` subclass and a `Model` subclass? It’s simple: a layer is a building block you use to create models, and a model is the top-level object that you will actually train, export for inference, etc. In short, a `Model` has a `fit()`, `evaluate()`, and `predict()` method. Layers don’t. Other than that, the two classes are virtually identical (another difference is that you can save a model to a file on disk—which we will cover in a few sections).

You can compile and train a `Model` subclass just like a Sequential or Functional model:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.229 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.228 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.219 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.216 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.217 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.117 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.248 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.565 — 18 Best practices for the real world › 18.1 Getting the most out of your models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.228 (printed 204) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › Beware: What subclassed models don’t support
This freedom comes at a cost: with subclassed models, you are responsible for more of the model logic, which means your potential error surface is much larger. As a result, you will have more debugging work to do. You are developing a new Python object, not just snapping together LEGO bricks.

Functional and subclassed models are also substantially different in nature: a Functional model is an explicit data structure—a graph of layers, which you can view, inspect, and modify. Meanwhile, a subclassed model is a piece of bytecode—a Python class with a `call()` method that contains raw code. This is the source of the subclassing workflow’s flexibility—you can just code up whatever functionality you like—but it introduces new limitations. For instance, because the way layers are connected to each other is hidden inside the body of the `call()` method, you cannot access that information. Calling `summary()` will not display layer connectivity, and you cannot plot the model topology via `plot_` `model()`. Likewise, if you have a subclassed model, you cannot access the nodes of the graph of layers to do feature extraction—because there is simply no graph. Once the model is instantiated, i

### PDF p.228 (printed 204) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › 
The `Model` subclassing workflow is the most flexible way to build a model: it enables you to build models that cannot be expressed as directed acyclic graphs of layers— imagine, for instance, a model where the `call()` method uses layers inside a `for` loop, or even calls them recursively. Anything is possible—you’re in charge.

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

### PDF p.565 (printed 541) — 18 Best practices for the real world › 18.1 Getting the most out of your models › 
If you want to adopt a more modular and configurable approach to model-building, you can also subclass the `HyperModel` class and define a `build` method.

Listing 18.2 A KerasTuner `HyperModel`

### PDF p.216 (printed 192) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › Different ways to build Keras models
There are three APIs for building models in Keras, as shown in figure 7.1:

The Sequential model is the most approachable API—it’s basically a Python list. As such, it’s limited to simple stacks of layers.

The Functional API, which focuses on graph-like model architectures. It represents a nice mid-point between usability and flexibility, and as such, it’s the most commonly used model-building API.

Model subclassing, a low-level option where you write everything yourself from scratch. This is ideal if you want full control over every little thing. However, you won’t get access to many built-in Keras features, and you will be more at risk of making mistakes.

Figure 7.1 Progressive disclosure of complexity for model building

### PDF p.217 (printed 193) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › 
You’ve seen in chapter 3 that layers only get built (which is to say, create their weights) when they are called for the first time. That’s because the shape of the layers’ weights depends on the shape of their input: until the input shape is known, they can’t be created.

As such, the previous Sequential model does not have any weights until you actually call it on some data, or call its `build()` method with an input shape.

Listing 7.3 Models that aren’t yet built have no weights

### PDF p.229 (printed 205) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › Remember: Use the right tool for the job
You’ve learned about the spectrum of workflows for building Keras models, from the simplest workflow—the Sequential model—to the most advanced one, model subclassing. When should you use one over the other? Each one has its pros and cons—pick the one most suitable for the job at hand.

In general, the Functional API provides you with a pretty good tradeoff between ease of use and flexibility. It also gives you direct access to layer connectivity, which is very powerful for use cases such as model plotting or feature extraction. If you can use the Functional API—that is, if your model can be expressed as a directed acyclic graph of layers—we recommend using it over model subclassing.

### PDF p.120 (printed 96) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › From layers to models
A deep learning model is a graph of layers. In Keras, that’s the `Model` class. For now, you’ve only seen `Sequential` models (a subclass of `Model`), which are simple stacks of layers, mapping a single input to a single output. But as you move forward, you’ll be exposed to a much broader variety of network topologies. Some common ones are

Two-branch networks

Multihead networks

Residual connections

Network topology can get quite involved. For instance, figure 3.5 shows topology of the graph of layers of a Transformer, a common architecture designed to process text data.

There are generally two ways of building such models in Keras: you can directly subclass the `Model` class, or you can use the Functional API, which lets you do more with less code. We’ll cover both approaches in chapter 7. The topology of a model defines a hypothesis space. You may remember that in chapter 1, we described machine learning as “searching for useful representations of some input data, within a predefined space of possibilities, using guidance from a feedback signal.” By choosing a network topology, you constrain your space of possibilities (hypothesis space) to a specific series of tensor operati
