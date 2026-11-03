# Generation bundle for page_251  (PDF page 251, printed page 227)

Page kind: content
Chapter: 7 A deep dive on Keras
Section: 7.4 Writing your own training and evaluation loops
Subsection: 7.4.5 Handling metrics in a custom train_step()
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_251.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_251.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p251_e03] (code)
```
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply(gradients, self.trainable_weights)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(targets, predictions)
        return {m.name: m.result() for m in self.metrics}
```
[p251_e04] (body) Let’s try it:
[p251_e05] (code)
```
def get_custom_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = CustomModel(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model
model = get_custom_model()
model.fit(train_images, train_labels, epochs=3)
```
[p251_e06] (heading_subsubsection) train_step() metrics handling with PyTorch
[p251_e07] (body) And here’s what it looks like with PyTorch—it’s exactly the same code change!
[p251_e08] (code)
```
import keras
from keras import layers
class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        predictions = self(inputs, training=True)
        loss = self.compute_loss(y=targets, y_pred=predictions)
        loss.backward()
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]
```

## Code annotations printed next to listings (callouts, in order)
- Updates the
- model’s metrics,
- including the one
- that tracks the loss
- Returns a dict mapping metric
- names to their current value

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

## Previous page (PDF 250) text, for continuity
Let’s try it out:

7.4.5

Handling metrics in a custom train_step()

Finally, what about the `loss` and `metrics` that you can pass to `compile()`? After you’ve called `compile()`, you get access to

`self.compute_loss`—This combines the loss function you passed to `compile()` together with regularization losses that may be added by certain layers.

`self.metrics`—The list of metrics you passed to `compile()`. Note that it also includes a metric that tracks the loss.

train_step() metrics handling with TensorFlow

Here’s what it looks like with TensorFlow:

## Next page (PDF 252) text, for continuity
Let’s see how it runs:

train_step() metrics handling with JAX

Finally, here’s what it looks like with JAX. To start with, you can use `compute_loss()` in your `compute_loss_and_updates()` method to hit the loss passed to `compile()`:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.248 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.241 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.242 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.247 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.240 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.253 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.219 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.245 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.240 (printed 216) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Writing custom training step functions
This snippet is pseudocode rather than real code because it includes an imaginary function, `get_gradients_of()`. In reality, retrieving gradients is done in a way that is specific to your current backend—JAX, TensorFlow, or PyTorch.

Let’s use what you learned about each framework in chapter 3 to implement a real version of this `train_step()` function. We’ll start with TensorFlow and PyTorch because these two make the job relatively easy, so they’re a good place to start. We’ll end with JAX, which is quite a bit more complex.

### PDF p.122 (printed 98) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
Metrics—The measures of success you want to monitor during training and validation, such as classification accuracy. Unlike the loss, training will not optimize directly for these metrics. As such, metrics don’t need to be differentiable.

Once you’ve picked your loss, optimizer, and metrics, you can use the built-in `compile()` and `fit()` methods to start training your model. Alternatively, you can write your own custom training loops—we cover how to do this in chapter 7. It’s a lot more work! For now, let’s take a look at `compile()` and `fit()`.

The `compile()` method configures the training process—you’ve already been introduced to it in your very first neural network example in chapter 2. It takes the arguments `optimizer`, `loss`, and `metrics` (a list):

### PDF p.248 (printed 224) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Customizing fit() with PyTorch
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

### PDF p.245 (printed 221) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › 
It’s definitely a bit more work than TensorFlow and PyTorch, but the speed and scalability benefits of JAX more than make up for it.

Next, let’s take a look at another important element of a custom training loop: metrics.

### PDF p.246 (printed 222) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Using fit() with a custom training loop
In the previous sections, we were writing our own training logic entirely from scratch. Doing so provides you with the most flexibility, but you end up writing a lot of code, while simultaneously missing out on many convenient features of `fit()`, such as callbacks, performance optimizations, or built-in support for distributed training.

What if you need a custom training algorithm, but you still want to use the power of the built-in Keras training loop? There’s actually a middle ground between `fit()` and a training loop written from scratch: you can provide a custom training step function and let the framework do the rest. You can do this by overriding the `train_step()` method of the `Model` class. This is the function that is called by `fit()` for every batch of data. You will then be able to call `fit()` as usual—and it will be running your own learning algorithm under the hood. Here’s how it works:

Create a new class that subclasses `keras.Model`.

### PDF p.113 (printed 89) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › 
Next, we can set up our training step function. It looks straightforward. Be mindful that, unlike its TensorFlow and PyTorch equivalents, it needs to be stateless, and so it must return the updated values of the `W` and `b` tensors:

### PDF p.245 (printed 221) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Low-level usage of metrics
In a low-level training loop, you will probably want to use Keras metrics (whether custom ones or the built-in ones). You’ve already learned about the metrics API: simply call `update_state(y_true, y_pred)` for each batch of targets and predictions, and then use `result()` to query the current metric value:

### PDF p.240 (printed 216) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Writing custom training step functions
def train_step(inputs, targets):
    predictions = model(inputs, training=True)
    loss = loss_fn(targets, predictions)
    gradients = get_gradients_of(loss, wrt=model.trainable_weights)
    optimizer.apply(gradients, model.trainable_weights)
