# Generation bundle for page_248  (PDF page 248, printed page 224)

Page kind: content
Chapter: 7 A deep dive on Keras
Section: 7.4 Writing your own training and evaluation loops
Subsection: 7.4.4 Using fit() with a custom training loop
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_248.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_248.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p248_e03] (body) Let’s put the model definition in its own reusable function:
[p248_e04] (code)
```
def get_custom_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = CustomModel(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam())
    return model
```
[p248_e05] (body) Let’s give it a whirl:
[p248_e06] (code)
```
model = get_custom_model()
model.fit(train_images, train_labels, epochs=3)
```
[p248_e07] (heading_subsubsection) Customizing fit() with PyTorch
[p248_e08] (body) Next, the PyTorch version:
[p248_e09] (code)
```
import keras
from keras import layers
loss_fn = keras.losses.SparseCategoricalCrossentropy()
loss_tracker = keras.metrics.Mean(name="loss")
class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        predictions = self(inputs, training=True)
        loss = loss_fn(targets, predictions)
        loss.backward()
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)
        loss_tracker.update_state(loss)
        return {"loss": loss_tracker.result()}
    @property
    def metrics(self):
        return [loss_tracker]
```
[p248_e10] (body) Let’s try it:
[p248_e11] (code)
```
model = get_custom_model()
model.fit(train_images, train_labels, epochs=3)
```

## Code annotations printed next to listings (callouts, in order)
- Runs the forward pass
- Retrieves
- the gradients
- Updates
- weights
- Updates loss
- tracker metric
- Returns the average
- loss so far by querying
- the loss tracker metric

## Equation candidates (formulas written inline in the text)
- [p248_e06] model = get_custom_model()
model.fit(train_images, train_labels, epochs=3)
- [p248_e11] model = get_custom_model()
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

## Previous page (PDF 247) text, for continuity
Override the `train_step()` method. Its contents are nearly identical to what we used in the previous section.

Return a dictionary mapping metric names (including the loss) to their current value.

Note the following:

This pattern does not prevent you from building models with the Functional API. You can do this whether you’re building Sequential models, Functional API models, or subclassed models.

You don’t need to use a `@tf.function` or `@jax.jit` decorator when you override `train_step()`—the framework does it for you.

Customizing fit() with TensorFlow

Let’s start by coding a custom TensorFlow train step.

Listing 7.21 Customizing `fit()`: TensorFlow version

We can now instantiate our custom model, compile it (we only pass the optimizer, since the loss is already defined outside of the model), and train it using `fit()` as usual.

## Next page (PDF 249) text, for continuity
Customizing fit() with JAX

Finally, let’s write the JAX version. First we need to define a `compute_loss_and_` `updates()` method, similar to the `compute_loss_and_updates()` function we used in our custom training step example:

Note we aren’t computing a moving average of the loss like we did for the other two backends. Instead we just return the per-batch loss value, which is less useful. We do this to simplify metric state management in the example: the code would get very verbose if we included it (you will learn about metric management in the next section):

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.251 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.252 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.250 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.242 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.241 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.230 — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops
- PDF p.220 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.219 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.246 (printed 222) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Using fit() with a custom training loop
In the previous sections, we were writing our own training logic entirely from scratch. Doing so provides you with the most flexibility, but you end up writing a lot of code, while simultaneously missing out on many convenient features of `fit()`, such as callbacks, performance optimizations, or built-in support for distributed training.

What if you need a custom training algorithm, but you still want to use the power of the built-in Keras training loop? There’s actually a middle ground between `fit()` and a training loop written from scratch: you can provide a custom training step function and let the framework do the rest. You can do this by overriding the `train_step()` method of the `Model` class. This is the function that is called by `fit()` for every batch of data. You will then be able to call `fit()` as usual—and it will be running your own learning algorithm under the hood. Here’s how it works:

Create a new class that subclasses `keras.Model`.

### PDF p.239 (printed 215) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › 
Whenever you find yourself in a situation where the built-in `fit()` is not enough, you will need to write your own custom training logic. You’ve already seen simple examples of low-level training loops in chapters 2 and 3. As a reminder, the contents of a typical training loop look like this: 1 Run the “forward pass” (compute the model’s output) to obtain a loss value for

the current batch of data. 2 Retrieve the gradients of the loss with regard to the model’s weights. 3 Update the model’s weights so as to lower the loss value on the current batch of data. These steps are repeated for as many batches as necessary. This is essentially what `fit()` does under the hood. In this section, you will learn to reimplement `fit()` from scratch, which will give you all the knowledge you need to write any training algorithm you may come up with. Let’s go over the details. Throughout the next few sections, you’ll work your way up to writing a fully featured custom training loop in TensorFlow, PyTorch, and JAX.

### PDF p.124 (printed 100) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Understanding the fit method
After `compile()` comes `fit()`. The `fit` method implements the training loop itself. Its key arguments are

The data (inputs and targets) to train on. It will typically be passed either in the form of NumPy arrays or a TensorFlow `Dataset` object. You’ll learn more about the `Dataset` API in the next chapters.

The number of epochs to train for: how many times the training loop should iterate over the data passed.

The batch size to use within each epoch of mini-batch gradient descent: the number of training examples considered to compute the gradients for one weight update step.

Listing 3.34 Calling `fit` with NumPy data

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

### PDF p.238 (printed 214) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Writing your own training and evaluation loops
The `fit()` workflow strikes a nice balance between ease of use and flexibility. It’s what you will use most of the time. However, it isn’t meant to support everything a deep learning researcher may want to do—even with custom metrics, custom losses, and custom callbacks.

After all, the built-in `fit()` workflow is solely focused on supervised learning: a setup where there are known targets (also called labels or annotations) associated with your input data and where you compute your loss as a function of these targets and the model’s predictions. However, not every form of machine learning falls into this category. There are other setups where no explicit targets are present, such as generative learning (which we will introduce in chapter 16), self-supervised learning (where targets are obtained from the inputs), or reinforcement learning (where learning is driven by occasional “rewards”—much like training a dog). And even if you’re doing regular supervised learning, as a researcher, you may want to add some novel bells and whistles that require low-level flexibility.

### PDF p.254 (printed 230) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Summary
Keras offers a spectrum of different workflows, based on the principle of progressive disclosure of complexity. They all smoothly interoperate.

You can build models via the `Sequential` class, via the Functional API, or by subclassing the `Model` class. Most of the time, you’ll be using the Functional API.

The simplest way to train and evaluate a model is via the default `fit()` and `evaluate()` methods.

Keras callbacks provide a simple way to monitor models during your call to `fit()` and automatically take action based on the state of the model.

You can also fully control what `fit()` does by overriding the `train_step()` method, using APIs from your backend of choice—JAX, TensorFlow, or PyTorch.

Beyond `fit()`, you can also write your own training loops entirely from scratch, in a backend-native way. This is useful for researchers implementing brand-new training algorithms.

### PDF p.578 (printed 554) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › 
Once the distribution configuration is set, you can create your model and `fit()` it. No other part of your code changes—your model definition code is the same, and your training code is the same. That’s true whether you’re using built-in APIs like `fit()` and `evaluate()` or your own training logic. Assuming that you have the right `Layout` `Map` for your variables, the little code snippets you just saw are enough to distribute

### PDF p.240 (printed 216) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Writing custom training step functions
This snippet is pseudocode rather than real code because it includes an imaginary function, `get_gradients_of()`. In reality, retrieving gradients is done in a way that is specific to your current backend—JAX, TensorFlow, or PyTorch.

Let’s use what you learned about each framework in chapter 3 to implement a real version of this `train_step()` function. We’ll start with TensorFlow and PyTorch because these two make the job relatively easy, so they’re a good place to start. We’ll end with JAX, which is quite a bit more complex.
