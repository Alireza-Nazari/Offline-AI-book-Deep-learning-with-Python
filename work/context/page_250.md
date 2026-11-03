# Generation bundle for page_250  (PDF page 250, printed page 226)

Page kind: content
Chapter: 7 A deep dive on Keras
Section: 7.4 Writing your own training and evaluation loops
Subsection: 7.4.5 Handling metrics in a custom train_step()
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_250.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_250.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p250_e03] (code)
```
            targets,
            training=True,
        )
        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )
        logs = {"loss": loss}
        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        )
        return logs, state
```
[p250_e04] (body) Let’s try it out:
[p250_e05] (code)
```
model = get_custom_model()
model.fit(train_images, train_labels, epochs=3)
```
[p250_e06] (heading_subsection) 7.4.5
[p250_e07] (heading_subsection) Handling metrics in a custom train_step()
[p250_e08] (body) Finally, what about the `loss` and `metrics` that you can pass to `compile()`? After you’ve called `compile()`, you get access to
[p250_e09] (bullet) `self.compute_loss`—This combines the loss function you passed to `compile()` together with regularization losses that may be added by certain layers.
[p250_e10] (bullet) `self.metrics`—The list of metrics you passed to `compile()`. Note that it also includes a metric that tracks the loss.
[p250_e11] (heading_subsubsection) train_step() metrics handling with TensorFlow
[p250_e12] (body) Here’s what it looks like with TensorFlow:
[p250_e13] (code)
```
import keras
from keras import layers
class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.compute_loss(y=targets, y_pred=predictions)
```

## Code annotations printed next to listings (callouts, in order)
- Computes gradients
- and updates to
- non-trainable variables
- Updates trainable
- variables and
- optimizer variables
- We aren’t computing a moving
- average of the loss, instead
- returning the per-batch value.
- Returns metric logs and
- updated state variables
- Computes the loss via
- self.compute_loss

## Equation candidates (formulas written inline in the text)
- [p250_e05] model = get_custom_model()
model.fit(train_images, train_labels, epochs=3)

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

## Previous page (PDF 249) text, for continuity
Customizing fit() with JAX

Finally, let’s write the JAX version. First we need to define a `compute_loss_and_` `updates()` method, similar to the `compute_loss_and_updates()` function we used in our custom training step example:

Note we aren’t computing a moving average of the loss like we did for the other two backends. Instead we just return the per-batch loss value, which is less useful. We do this to simplify metric state management in the example: the code would get very verbose if we included it (you will learn about metric management in the next section):

## Next page (PDF 251) text, for continuity
Let’s try it:

train_step() metrics handling with PyTorch

And here’s what it looks like with PyTorch—it’s exactly the same code change!

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.253 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.245 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.252 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.248 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.244 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.241 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.247 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.254 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.252 (printed 228) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › train_step() metrics handling with JAX
Finally, here’s what it looks like with JAX. To start with, you can use `compute_loss()` in your `compute_loss_and_updates()` method to hit the loss passed to `compile()`:

### PDF p.122 (printed 98) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
Metrics—The measures of success you want to monitor during training and validation, such as classification accuracy. Unlike the loss, training will not optimize directly for these metrics. As such, metrics don’t need to be differentiable.

Once you’ve picked your loss, optimizer, and metrics, you can use the built-in `compile()` and `fit()` methods to start training your model. Alternatively, you can write your own custom training loops—we cover how to do this in chapter 7. It’s a lot more work! For now, let’s take a look at `compile()` and `fit()`.

The `compile()` method configures the training process—you’ve already been introduced to it in your very first neural network example in chapter 2. It takes the arguments `optimizer`, `loss`, and `metrics` (a list):

### PDF p.122 (printed 98) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
In the previous call to `compile()`, we passed the optimizer, loss, and metrics as strings (such as `"rmsprop"`). These strings are actually shortcuts that get converted to Python objects. For instance, `"rmsprop"` becomes `keras.optimizers.RMSprop()`. Importantly, it’s also possible to specify these arguments as object instances, like this:

### PDF p.247 (printed 223) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Customizing fit() with TensorFlow
We can now instantiate our custom model, compile it (we only pass the optimizer, since the loss is already defined outside of the model), and train it using `fit()` as usual.

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

### PDF p.247 (printed 223) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Customizing fit() with TensorFlow
Listing 7.21 Customizing `fit()`: TensorFlow version
import keras
from keras import layers
loss_fn = keras.losses.SparseCategoricalCrossentropy()
loss_tracker = keras.metrics.Mean(name="loss")
class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply(gradients, self.trainable_weights)
        loss_tracker.update_state(loss)
        return {"loss": loss_tracker.result()}
    @property
    def metrics(self):
        return [loss_tracker]

### PDF p.126 (printed 102) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
`evaluate()` will iterate in batches (of size `batch_size`) over the data passed and return a list of scalars, where the first entry is the validation loss and the following entries are the validation metrics. If the model has no metrics, only the validation loss is returned (rather than a list).

### PDF p.123 (printed 99) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
In chapter 7, we cover how to create custom losses and metrics. In general, you won’t have to create your own losses, metrics, or optimizers from scratch because Keras offers a wide range of built-in options that is likely to include what you need:

Optimizers – `SGD()` (with or without momentum) – `RMSprop()` – `Adam()` – Etc.

Losses – `CategoricalCrossentropy()` – `SparseCategoricalCrossentropy()` – `BinaryCrossentropy()` – `MeanSquaredError()` – `KLDivergence()` – `CosineSimilarity()` – `Etc.`

Metrics – `CategoricalAccuracy()` – `SparseCategoricalAccuracy()` – `BinaryAccuracy()` – `AUC()` – `Precision()` – `Recall()` – Etc.

Throughout this book, you’ll see concrete applications of many of these options.
