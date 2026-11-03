# Generation bundle for page_234  (PDF page 234, printed page 210)

Page kind: content
Chapter: 7 A deep dive on Keras
Section: 7.3 Using built-in training and evaluation loops
Subsection: 7.3.3 Writing your own callbacks
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_234.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_234.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p234_e03] (listing_caption) Listing 7.19 Using the `callbacks` argument in the `fit()` method
[p234_e04] (code)
```
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
```
[p234_e05] (body) Note that you can always save models manually after training as well—just call `model
` `.save("my_checkpoint_path.keras")`. To reload the model you’ve saved, use
[p234_e06] (code)
```
model = keras.models.load_model("checkpoint_path.keras")
```
[p234_e07] (heading_subsection) 7.3.3
[p234_e08] (heading_subsection) Writing your own callbacks
[p234_e09] (body) If you need to take a specific action during training that isn’t covered by one of the built-in callbacks, you can write your own callback. Callbacks are implemented by subclassing the class `keras.callbacks.Callback`. You can then implement any number of the following transparently named methods, which are called at various points during training:
[p234_e10] (code)
```
on_epoch_begin(epoch, logs)
on_epoch_end(epoch, logs)
```

## Code annotations printed next to listings (callouts, in order)
- Interrupts training when
- Callbacks are passed to the model
- improvement stops
- via the callbacks argument in fit(),
- which takes a list of callbacks. You
- Monitors the model’s
- can pass any number of callbacks.
- validation accuracy
- Interrupts training when accuracy
- has stopped improving for more
- than one epoch (that is, two epochs)
- Saves the current weights after every epoch
- Path to the destination model file
- These two arguments mean you won’t
- overwrite the model file unless val_loss
- has improved, which allows you to keep
- the best model seen during training.
- You monitor accuracy, so it should
- be part of the model’s metrics.
- Because the callback will monitor
- validation loss and validation
- accuracy, you need to pass
- validation_data to the call to fit().
- Called at the start of every epoch
- Called at the end of every epoch

## Listings
- Listing 7.19: Using the `callbacks` argument in the `fit()` method (code elements p234_e04)

## Equation candidates (formulas written inline in the text)
- [p234_e06] model = keras.models.load_model("checkpoint_path.keras")

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

## Previous page (PDF 233) text, for continuity
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

## Next page (PDF 235) text, for continuity
These methods are all called with a `logs` argument, which is a dictionary containing information about the previous batch, epoch, or training run: training and validation metrics, and so on. The `on_epoch_*` and `on_batch_*` methods also take the epoch or batch index as first argument (an integer).

Here’s a simple example callback that saves a list of per-batch loss values during training and plots these values at the end of each epoch.

Listing 7.20 Creating a custom callback by subclassing the `Callback` class

Let’s test-drive it:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.232 — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops
- PDF p.236 — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops
- PDF p.237 — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops
- PDF p.254 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.287 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.246 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.274 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.230 — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

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

### PDF p.232 (printed 208) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › Using callbacks
Launching a training run on a large dataset for tens of epochs using `model.fit()` can be a bit like launching a paper airplane: past the initial impulse, you don’t have any control over its trajectory or its landing spot. If you want to avoid bad outcomes (and thus wasted paper airplanes), it’s smarter to use, not a paper plane, but a drone that can sense its environment, send data back to its operator, and automatically make steering decisions based on its current state. The Keras callbacks API will help you transform your call to `model.fit()` from a paper airplane into a smart, autonomous drone that can self-introspect and dynamically take action.

A callback is an object (a class instance implementing specific methods) that is passed to the model in the call to `fit()` and that is called by the model at various points during

### PDF p.246 (printed 222) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Using fit() with a custom training loop
In the previous sections, we were writing our own training logic entirely from scratch. Doing so provides you with the most flexibility, but you end up writing a lot of code, while simultaneously missing out on many convenient features of `fit()`, such as callbacks, performance optimizations, or built-in support for distributed training.

What if you need a custom training algorithm, but you still want to use the power of the built-in Keras training loop? There’s actually a middle ground between `fit()` and a training loop written from scratch: you can provide a custom training step function and let the framework do the rest. You can do this by overriding the `train_step()` method of the `Model` class. This is the function that is called by `fit()` for every batch of data. You will then be able to call `fit()` as usual—and it will be running your own learning algorithm under the hood. Here’s how it works:

Create a new class that subclasses `keras.Model`.

### PDF p.254 (printed 230) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Summary
Keras offers a spectrum of different workflows, based on the principle of progressive disclosure of complexity. They all smoothly interoperate.

You can build models via the `Sequential` class, via the Functional API, or by subclassing the `Model` class. Most of the time, you’ll be using the Functional API.

The simplest way to train and evaluate a model is via the default `fit()` and `evaluate()` methods.

Keras callbacks provide a simple way to monitor models during your call to `fit()` and automatically take action based on the state of the model.

You can also fully control what `fit()` does by overriding the `train_step()` method, using APIs from your backend of choice—JAX, TensorFlow, or PyTorch.

Beyond `fit()`, you can also write your own training loops entirely from scratch, in a backend-native way. This is useful for researchers implementing brand-new training algorithms.

### PDF p.231 (printed 207) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › 
There are a couple of ways you can customize this simple workflow:

By providing your own custom metrics

By passing callbacks to the `fit()` method to schedule actions to be taken at specific points during training

Let’s take a look at these.

### PDF p.185 (printed 161) — 5 Fundamentals of machine learning › 5.4 Improving generalization › Using early stopping
In deep learning, we always use models that are vastly overparameterized: they have way more degrees of freedom than the minimum necessary to fit to the latent manifold of the data. This overparameterization is not an issue because you never fully fit a deep learning model. Such a fit wouldn’t generalize at all. You will always interrupt training long before you’ve reached the minimum possible training loss.

Finding the exact point during training where you’ve reached the most generalizable fit—the exact boundary between an underfit curve and an overfit curve—is one of the most effective things you can do to improve generalization. In the examples from the previous chapter, we would start by training our models for longer than needed to figure out the number of epochs that yielded the best validation metrics, then we would retrain a new model for exactly that number of epochs. This is pretty standard. However, it requires you to do redundant work, which can sometimes be expensive. Naturally, you could just save your model at the end of each epoch, then once you’ve found the best epoch, reuse the closest saved model you have. In Keras, it’s typical to do this with an `EarlyStopping

### PDF p.274 (printed 250) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › Fitting the model
Listing 8.15 Fitting the model using a `Dataset`
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=callbacks,
)

### PDF p.287 (printed 263) — 8 Image classification › 8.3 Using a pretrained model › 
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="feature_extraction_with_data_augmentation.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]
history = model.fit(
    augmented_train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks,
)
