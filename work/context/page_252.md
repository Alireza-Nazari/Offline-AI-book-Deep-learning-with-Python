# Generation bundle for page_252  (PDF page 252, printed page 228)

Page kind: content
Chapter: 7 A deep dive on Keras
Section: 7.4 Writing your own training and evaluation loops
Subsection: 7.4.5 Handling metrics in a custom train_step()
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_252.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_252.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p252_e03] (code)
```
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(targets, predictions)
        return {m.name: m.result() for m in self.metrics}
```
[p252_e04] (body) Let’s see how it runs:
[p252_e05] (code)
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
[p252_e06] (heading_subsubsection) train_step() metrics handling with JAX
[p252_e07] (body) Finally, here’s what it looks like with JAX. To start with, you can use `compute_loss()` in your `compute_loss_and_updates()` method to hit the loss passed to `compile()`:
[p252_e08] (code)
```
import keras
from keras import layers
class CustomModel(keras.Model):
    def compute_loss_and_updates(
        self,
        trainable_variables,
        non_trainable_variables,
        inputs,
        targets,
        training=False,
    ):
        predictions, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
```

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

## Previous page (PDF 251) text, for continuity
Let’s try it:

train_step() metrics handling with PyTorch

And here’s what it looks like with PyTorch—it’s exactly the same code change!

## Next page (PDF 253) text, for continuity
Next up: metric management. As usual, it’s a tad more complicated due to JAX’s statelessness requirements:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.248 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.243 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.250 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.249 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.245 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.242 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.244 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.247 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.250 (printed 226) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Handling metrics in a custom train_step()
Finally, what about the `loss` and `metrics` that you can pass to `compile()`? After you’ve called `compile()`, you get access to

`self.compute_loss`—This combines the loss function you passed to `compile()` together with regularization losses that may be added by certain layers.

`self.metrics`—The list of metrics you passed to `compile()`. Note that it also includes a metric that tracks the loss.

### PDF p.249 (printed 225) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Customizing fit() with JAX
Finally, let’s write the JAX version. First we need to define a `compute_loss_and_` `updates()` method, similar to the `compute_loss_and_updates()` function we used in our custom training step example:

### PDF p.122 (printed 98) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
Metrics—The measures of success you want to monitor during training and validation, such as classification accuracy. Unlike the loss, training will not optimize directly for these metrics. As such, metrics don’t need to be differentiable.

Once you’ve picked your loss, optimizer, and metrics, you can use the built-in `compile()` and `fit()` methods to start training your model. Alternatively, you can write your own custom training loops—we cover how to do this in chapter 7. It’s a lot more work! For now, let’s take a look at `compile()` and `fit()`.

The `compile()` method configures the training process—you’ve already been introduced to it in your very first neural network example in chapter 2. It takes the arguments `optimizer`, `loss`, and `metrics` (a list):

### PDF p.250 (printed 226) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › train_step() metrics handling with TensorFlow
Here’s what it looks like with TensorFlow:

### PDF p.240 (printed 216) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Writing custom training step functions
This snippet is pseudocode rather than real code because it includes an imaginary function, `get_gradients_of()`. In reality, retrieving gradients is done in a way that is specific to your current backend—JAX, TensorFlow, or PyTorch.

Let’s use what you learned about each framework in chapter 3 to implement a real version of this `train_step()` function. We’ll start with TensorFlow and PyTorch because these two make the job relatively easy, so they’re a good place to start. We’ll end with JAX, which is quite a bit more complex.

### PDF p.245 (printed 221) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › 
It’s definitely a bit more work than TensorFlow and PyTorch, but the speed and scalability benefits of JAX more than make up for it.

Next, let’s take a look at another important element of a custom training loop: metrics.

### PDF p.249 (printed 225) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Customizing fit() with JAX
    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        inputs, targets = data
        grad_fn = jax.value_and_grad(
            self.compute_loss_and_updates, has_aux=True
        )
        (loss, non_trainable_variables), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            inputs,

### PDF p.247 (printed 223) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › 
Override the `train_step()` method. Its contents are nearly identical to what we used in the previous section.

Return a dictionary mapping metric names (including the loss) to their current value.

Note the following:

This pattern does not prevent you from building models with the Functional API. You can do this whether you’re building Sequential models, Functional API models, or subclassed models.

You don’t need to use a `@tf.function` or `@jax.jit` decorator when you override `train_step()`—the framework does it for you.
