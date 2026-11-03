# Generation bundle for page_232  (PDF page 232, printed page 208)

Page kind: content
Chapter: 7 A deep dive on Keras
Section: 7.3 Using built-in training and evaluation loops
Subsection: 7.3.2 Using callbacks
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_232.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_232.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p232_e03] (body) You use the `result()` method to return the current value of the metric:
[p232_e04] (code)
```
def result(self):
    return ops.sqrt(self.mse_sum / self.total_samples)
```
[p232_e05] (body) Meanwhile, you also need to expose a way to reset the metric state without having to reinstantiate it—this enables the same metric objects to be used across different epochs of training or across both training and evaluation. You do this in the `reset_state()` method:
[p232_e06] (code)
```
def reset_state(self):
    self.mse_sum.assign(0.)
    self.total_samples.assign(0.)
```
[p232_e07] (body) Custom metrics can be used just like built-in ones. Let’s test-drive our own metric:
[p232_e08] (code)
```
model = get_mnist_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", RootMeanSquaredError()],
)
model.fit(
    train_images,
    train_labels,
    epochs=3,
    validation_data=(val_images, val_labels),
)
test_metrics = model.evaluate(test_images, test_labels)
```
[p232_e09] (body) You can now see the `fit()` progress bar display the RMSE of your model.
[p232_e10] (heading_subsection) 7.3.2
[p232_e11] (heading_subsection) Using callbacks
[p232_e12] (body) Launching a training run on a large dataset for tens of epochs using `model.fit()` can be a bit like launching a paper airplane: past the initial impulse, you don’t have any control over its trajectory or its landing spot. If you want to avoid bad outcomes (and thus wasted paper airplanes), it’s smarter to use, not a paper plane, but a drone that can sense its environment, send data back to its operator, and automatically make steering decisions based on its current state. The Keras callbacks API will help you transform your call to `model.fit()` from a paper airplane into a smart, autonomous drone that can self-introspect and dynamically take action.
[p232_e13] (body) A callback is an object (a class instance implementing specific methods) that is passed to the model in the call to `fit()` and that is called by the model at various points during

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

## Previous page (PDF 231) text, for continuity
There are a couple of ways you can customize this simple workflow:

By providing your own custom metrics

By passing callbacks to the `fit()` method to schedule actions to be taken at specific points during training

Let’s take a look at these.

7.3.1

Writing your own metrics

Metrics are key to measuring the performance of your model—in particular, to measure the difference between its performance on the training data and its performance on the test data. Commonly used metrics for classification and regression are already part of the built-in `keras.metrics` module—most of the time, that’s what you will use. But if you’re doing anything out of the ordinary, you will need to be able to write your own metrics. It’s simple!

A Keras metric is a subclass of the `keras.metrics.Metric` class. Similarly to layers, a metric has an internal state stored in Keras variables. Unlike layers, these variables aren’t updated via backpropagation, so you have to write the state update logic yourself—which happens in the `update_state()` method. For example, here’s a simple custom metric that measures the root mean squared error (RMSE).

Listing 7.18 Implementing a custom metric by subclassing the `Metric` class

## Next page (PDF 233) text, for continuity
training. It has access to all the available data about the state of the model and its performance, and it can take action: interrupt training, save a model, load a different weight set, or otherwise alter the state of the model.

Here are some examples of ways you can use callbacks:

Model checkpointing—Saving the current state of the model at different points during training.

Early stopping—Interrupting training when the validation loss is no longer improving (and of course, saving the best model obtained during training).

Dynamically adjusting the value of certain parameters during training—Such as the learning rate of the optimizer.

Logging training and validation metrics during training, or visualizing the representa

tions learned by the model as they’re updated—The `fit()` progress bar that you’re

familiar with is in fact a callback! The `keras.callbacks` module includes a number of built-in callbacks (this is not an exhaustive list):

Let’s review two of them to give you an idea of how to use them: `EarlyStopping` and `ModelCheckpoint`.

The EarlyStopping and ModelCheckpoint callbacks

When you’re training a model, there are many things you can’t predict at the start. In particular, you can’t tell how many epochs will be needed to get to an optimal validation loss. Our examples so far have adopted the strategy of training for enough epochs that you begin overfitting, using the first run to figure out the optimal number of epochs, and then finally launching a new training run from scratch using this optimal number. Of course, this approach is wasteful. A much better way to handle this is to stop training when you measure that the validation loss is no longer improving. This can be achieved using the `EarlyStopping` callback.

The `EarlyStopping` callback inte …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.245 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.234 — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops
- PDF p.246 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.254 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.230 — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops
- PDF p.235 — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops
- PDF p.237 — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops
- PDF p.251 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.254 (printed 230) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Summary
Keras offers a spectrum of different workflows, based on the principle of progressive disclosure of complexity. They all smoothly interoperate.

You can build models via the `Sequential` class, via the Functional API, or by subclassing the `Model` class. Most of the time, you’ll be using the Functional API.

The simplest way to train and evaluate a model is via the default `fit()` and `evaluate()` methods.

Keras callbacks provide a simple way to monitor models during your call to `fit()` and automatically take action based on the state of the model.

You can also fully control what `fit()` does by overriding the `train_step()` method, using APIs from your backend of choice—JAX, TensorFlow, or PyTorch.

Beyond `fit()`, you can also write your own training loops entirely from scratch, in a backend-native way. This is useful for researchers implementing brand-new training algorithms.

### PDF p.245 (printed 221) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Low-level usage of metrics
In a low-level training loop, you will probably want to use Keras metrics (whether custom ones or the built-in ones). You’ve already learned about the metrics API: simply call `update_state(y_true, y_pred)` for each batch of targets and predictions, and then use `result()` to query the current metric value:

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

### PDF p.234 (printed 210) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › 
Listing 7.19 Using the `callbacks` argument in the `fit()` method
callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor="accuracy",
        patience=1,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="checkpoint_path.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
]
model = get_mnist_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_images,
    train_labels,
    epochs=10,
    callbacks=callbacks_list,
    validation_data=(val_images, val_labels),
)

### PDF p.246 (printed 222) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › 
Remember to use `metric.reset_state()` when you want to reset the current results (at the start of a training epoch or at the start of evaluation).

Now, if you’re using JAX, state-modifying methods like `update_state()` or `reset()` can’t be used inside a stateless function. Instead, you can use the stateless metrics API, which is similar to the `model.stateless_call()` and `optimizer.stateless_apply()` methods you’ve already learned about. Here’s how it works:

### PDF p.246 (printed 222) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Using fit() with a custom training loop
In the previous sections, we were writing our own training logic entirely from scratch. Doing so provides you with the most flexibility, but you end up writing a lot of code, while simultaneously missing out on many convenient features of `fit()`, such as callbacks, performance optimizations, or built-in support for distributed training.

What if you need a custom training algorithm, but you still want to use the power of the built-in Keras training loop? There’s actually a middle ground between `fit()` and a training loop written from scratch: you can provide a custom training step function and let the framework do the rest. You can do this by overriding the `train_step()` method of the `Model` class. This is the function that is called by `fit()` for every batch of data. You will then be able to call `fit()` as usual—and it will be running your own learning algorithm under the hood. Here’s how it works:

Create a new class that subclasses `keras.Model`.

### PDF p.234 (printed 210) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › 
Listing 7.19 Using the `callbacks` argument in the `fit()` method

### PDF p.122 (printed 98) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
Metrics—The measures of success you want to monitor during training and validation, such as classification accuracy. Unlike the loss, training will not optimize directly for these metrics. As such, metrics don’t need to be differentiable.

Once you’ve picked your loss, optimizer, and metrics, you can use the built-in `compile()` and `fit()` methods to start training your model. Alternatively, you can write your own custom training loops—we cover how to do this in chapter 7. It’s a lot more work! For now, let’s take a look at `compile()` and `fit()`.

The `compile()` method configures the training process—you’ve already been introduced to it in your very first neural network example in chapter 2. It takes the arguments `optimizer`, `loss`, and `metrics` (a list):
