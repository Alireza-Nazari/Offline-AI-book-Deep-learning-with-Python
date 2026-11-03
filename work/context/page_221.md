# Generation bundle for page_221  (PDF page 221, printed page 197)

Page kind: content
Chapter: 7 A deep dive on Keras
Section: 7.2 Different ways to build Keras models
Subsection: 7.2.2 The Functional API
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_221.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_221.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p221_e03] (code)
```
 Total params: 906 (3.54 KB)
 Trainable params: 906 (3.54 KB)
 Non-trainable params: 0 (0.00 B)
```
[p221_e04] (heading_subsubsection) Multi-input, multi-output models
[p221_e05] (body) Unlike this toy model, most deep learning models don’t look like lists—they look like graphs. They may, for instance, have multiple inputs or multiple outputs. It’s for this kind of model that the Functional API really shines.
[p221_e06] (body) Let’s say you’re building a system to rank customer support tickets by priority and route them to the appropriate department. Your model has three inputs:
[p221_e07] (bullet) The title of the ticket (text input)
[p221_e08] (bullet) The text body of the ticket (text input)
[p221_e09] (bullet) Any tags added by the user (categorical input, assumed here to be multi-hot encoded)
[p221_e10] (body) We can encode the text inputs as arrays of 1s and 0s of size `vocabulary_size` (see chapter 14 for detailed information about text encoding techniques).
[p221_e11] (body) Your model also has two outputs:
[p221_e12] (bullet) The priority score of the ticket, a scalar between 0 and 1 (sigmoid output)
[p221_e13] (bullet) The department that should handle the ticket (a softmax over the set of departments)
[p221_e14] (body) You can build this model in a few lines with the Functional API.
[p221_e15] (listing_caption) Listing 7.9 A multi-input, multi-output Functional model
[p221_e16] (code)
```
vocabulary_size = 10000
num_tags = 100
num_departments = 4
title = keras.Input(shape=(vocabulary_size,), name="title")
text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
tags = keras.Input(shape=(num_tags,), name="tags")
features = layers.Concatenate()([title, text_body, tags])
features = layers.Dense(64, activation="relu", name="dense_features")(features)
priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department = layers.Dense(
    num_departments, activation="softmax", name="department"
)(features)
```

## Code annotations printed next to listings (callouts, in order)
- Applies intermediate layer to recombine
- input features into richer representations
- Combines input features into a single
- tensor, features, by concatenating them
- Defines model inputs
- Defines model outputs

## Listings
- Listing 7.9: A multi-input, multi-output Functional model (code elements p221_e16)

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

## Previous page (PDF 220) text, for continuity
This `inputs` object holds information about the shape and `dtype` of the data that the model will process:

We call such an object a symbolic tensor. It doesn’t contain any actual data, but it encodes the specifications of the actual tensors of data that the model will see when you use it. It stands for future tensors of data.

Next, we created a layer and called it on the input:

All Keras layers can be called both on real tensors of data or on these symbolic tensors. In the latter case, they return a new symbolic tensor, with updated shape and dtype information:

After obtaining the final outputs, we instantiated the model by specifying its inputs and outputs in the `Model` constructor:

Here’s the summary of our model:

## Next page (PDF 222) text, for continuity
The Functional API is a simple, LEGO-like, yet very flexible way to define arbitrary graphs of layers like these.

Training a multi-input, multi-output model

You can train your model in much the same way as you would train a Sequential model, by calling `fit()` with lists of input and output data. These lists of data should respect the same order as the inputs you passed to the `Model()` constructor.

Listing 7.10 Training a model by providing lists of input and target arrays

If you don’t want to rely on input order (for instance, because you have many inputs or outputs), you can also use the names you gave to the `Input` objects and to the output layers, and pass data via dictionaries.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.227 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.219 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.225 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.223 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.229 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.218 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.251 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.625 — 20 Conclusions › 20.1 Key concepts in review

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.219 (printed 195) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › The Functional API
The Sequential model is easy to use, but its applicability is extremely limited: it can only express models with a single input and a single output, applying one layer after the other in a sequential fashion. In practice, it’s pretty common to encounter models with multiple inputs (say, an image and its metadata), multiple outputs (different things you want to predict about the data), or a nonlinear topology.

In such cases, you’d build your model using the Functional API. This is what most Keras models you’ll encounter in the wild use. It’s fun and powerful—it feels like playing with LEGO bricks.

### PDF p.216 (printed 192) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › Different ways to build Keras models
There are three APIs for building models in Keras, as shown in figure 7.1:

The Sequential model is the most approachable API—it’s basically a Python list. As such, it’s limited to simple stacks of layers.

The Functional API, which focuses on graph-like model architectures. It represents a nice mid-point between usability and flexibility, and as such, it’s the most commonly used model-building API.

Model subclassing, a low-level option where you write everything yourself from scratch. This is ideal if you want full control over every little thing. However, you won’t get access to many built-in Keras features, and you will be more at risk of making mistakes.

Figure 7.1 Progressive disclosure of complexity for model building

### PDF p.224 (printed 200) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › Feature extraction with a Functional model
Listing 7.12 Retrieving the inputs or outputs of a layer in a Functional model
>>> model.layers
[<InputLayer name=title, built=True>,

### PDF p.228 (printed 204) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › Mixing and matching different components
Crucially, choosing one of these patterns—the Sequential model, the Functional API, `Model` subclassing—does not lock you out of the others. All models in the Keras API can smoothly interoperate with each other, whether they’re Sequential models, Functional models, or subclassed models written from scratch. They’re all part of the same spectrum of workflows. For instance, you can use a subclassed layer or model in a Functional model.

Listing 7.15 Creating a Functional model that includes a subclassed model

### PDF p.225 (printed 201) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › 
Listing 7.13 Creating a new model by reusing intermediate layer outputs
features = model.layers[4].output
difficulty = layers.Dense(3, activation="softmax", name="difficulty")(features)
new_model = keras.Model(
    inputs=[title, text_body, tags], outputs=[priority, department, difficulty]
)

### PDF p.223 (printed 199) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › 
Listing 7.11 Training a model by providing dicts of input and target arrays
model.compile(
    optimizer="adam",
    loss={
        "priority": "mean_squared_error",
        "department": "sparse_categorical_crossentropy",
    },
    metrics={
        "priority": ["mean_absolute_error"],
        "department": ["accuracy"],
    },
)
model.fit(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data},
    {"priority": priority_data, "department": department_data},
    epochs=1,
)
model.evaluate(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data},
    {"priority": priority_data, "department": department_data},
)
priority_preds, department_preds = model.predict(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data}
)

### PDF p.219 (printed 195) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › A simple example
Listing 7.8 A simple Functional model with two `Dense` layers
inputs = keras.Input(shape=(3,), name="my_input")
features = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs, name="my_functional_model")

### PDF p.230 (printed 206) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › 
Going forward, all examples in this book will use the Functional API—simply because all of the models we will work with are expressible as graphs of layers. We will, however, make frequent use of subclassed layers. In general, using Functional models that include subclassed layers provides the best of both worlds: high development flexibility while retaining the advantages of the Functional API.
