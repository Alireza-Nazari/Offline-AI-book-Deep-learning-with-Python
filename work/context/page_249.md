# Generation bundle for page_249  (PDF page 249, printed page 225)

Page kind: content
Chapter: 7 A deep dive on Keras
Section: 7.4 Writing your own training and evaluation loops
Subsection: 7.4.4 Using fit() with a custom training loop
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_249.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_249.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p249_e03] (heading_subsubsection) Customizing fit() with JAX
[p249_e04] (body) Finally, let’s write the JAX version. First we need to define a `compute_loss_and_` `updates()` method, similar to the `compute_loss_and_updates()` function we used in our custom training step example:
[p249_e05] (code)
```
import keras
from keras import layers
loss_fn = keras.losses.SparseCategoricalCrossentropy()
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
            inputs,
            training=training,
        )
        loss = loss_fn(targets, predictions)
        return loss, non_trainable_variables
```
[p249_e06] (body) Note we aren’t computing a moving average of the loss like we did for the other two backends. Instead we just return the per-batch loss value, which is less useful. We do this to simplify metric state management in the example: the code would get very verbose if we included it (you will learn about metric management in the next section):
[p249_e07] (code)
```
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
```

## Code annotations printed next to listings (callouts, in order)
- Returns both the loss
- and the updated
- non-trainable variables
- Unpacks the state.
- metrics_variables are
- part of it, although we
- won’t use them here.
- Gets the gradient
- function
- Computes gradients
- and updates to
- non-trainable variables

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

## Previous page (PDF 248) text, for continuity
Let’s put the model definition in its own reusable function:

Let’s give it a whirl:

Customizing fit() with PyTorch

Next, the PyTorch version:

Let’s try it:

## Next page (PDF 250) text, for continuity
Let’s try it out:

7.4.5

Handling metrics in a custom train_step()

Finally, what about the `loss` and `metrics` that you can pass to `compile()`? After you’ve called `compile()`, you get access to

`self.compute_loss`—This combines the loss function you passed to `compile()` together with regularization losses that may be added by certain layers.

`self.metrics`—The list of metrics you passed to `compile()`. Note that it also includes a metric that tracks the loss.

train_step() metrics handling with TensorFlow

Here’s what it looks like with TensorFlow:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.252 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.253 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.243 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.247 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.245 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.244 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.113 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.112 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.239 (printed 215) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › 
Whenever you find yourself in a situation where the built-in `fit()` is not enough, you will need to write your own custom training logic. You’ve already seen simple examples of low-level training loops in chapters 2 and 3. As a reminder, the contents of a typical training loop look like this: 1 Run the “forward pass” (compute the model’s output) to obtain a loss value for

the current batch of data. 2 Retrieve the gradients of the loss with regard to the model’s weights. 3 Update the model’s weights so as to lower the loss value on the current batch of data. These steps are repeated for as many batches as necessary. This is essentially what `fit()` does under the hood. In this section, you will learn to reimplement `fit()` from scratch, which will give you all the knowledge you need to write any training algorithm you may come up with. Let’s go over the details. Throughout the next few sections, you’ll work your way up to writing a fully featured custom training loop in TensorFlow, PyTorch, and JAX.

### PDF p.636 (printed 612) — index ›  › 
training process 525

filters 236 U-Net denoising autoencoder 522 filter visualization loop 296–299 visualizing results with custom callback 528 fit() function 191, 223, 342, 489 distance function 585

customizing with JAX 225 distributed training 551–555 customizing with PyTorch 224 accessing multiple GPUs 551 customizing with TensorFlow 223 DeviceMesh API 552 using with custom training loops 222 LayoutMap API 553–555 fitting model 250 using model parallelism with JAX 552 Flatten operation 602 Dogs vs. Cats dataset, downloading data 242 float8 training 560 dot-product attention 439 float16 inference 558 downsampling stage 522 float32 weights 188 dropout 167–168 floating-point precision 557 argument 215, 373 flops (floating-point operations) 505 Dropout layer 168, 215, 254 forecasting, defined 352 dynamics, defined 352 for loop 591 forward pass 40

### PDF p.242 (printed 218) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › A JAX training step function
When it comes to low-level training code, JAX tends to be the most complex of the three backends because of its fully stateless nature. Statelessness makes JAX highly performant and scalable, making it amenable to compilation and automatic performance optimizations. However, writing stateless code requires you to jump through some hoops.

Since the gradient function is obtained via metaprogramming, you first need to define the function that returns your loss. Further, this function needs to be stateless, so it needs to take as arguments all the variables it’s going to be using, and it needs to return the value of any variable it has updated. Remember those non-trainable weights that can get modified during the forward pass? Those are the variables we need to return. To make it easier to work with the stateless programming paradigm of JAX, Keras models make available a stateless forward pass method: the `stateless_call()` method. It behaves just like `call`, except that

It takes as input the model’s trainable weights and non-trainable weights, in addition to the `inputs` and `training` arguments.

### PDF p.252 (printed 228) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › train_step() metrics handling with JAX
Finally, here’s what it looks like with JAX. To start with, you can use `compute_loss()` in your `compute_loss_and_updates()` method to hit the loss passed to `compile()`:

### PDF p.122 (printed 98) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
Metrics—The measures of success you want to monitor during training and validation, such as classification accuracy. Unlike the loss, training will not optimize directly for these metrics. As such, metrics don’t need to be differentiable.

Once you’ve picked your loss, optimizer, and metrics, you can use the built-in `compile()` and `fit()` methods to start training your model. Alternatively, you can write your own custom training loops—we cover how to do this in chapter 7. It’s a lot more work! For now, let’s take a look at `compile()` and `fit()`.

The `compile()` method configures the training process—you’ve already been introduced to it in your very first neural network example in chapter 2. It takes the arguments `optimizer`, `loss`, and `metrics` (a list):

### PDF p.110 (printed 86) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Computing gradients with JAX
Unlike TensorFlow and PyTorch, JAX takes a metaprogramming approach to gradient computation. Metaprogramming refers to the idea of having functions that return functions—you could call them “meta-functions.” In practice, JAX lets you turn a loss-computation function into a gradient-computation function. So computing gradients in JAX is a three-step process:

1 Define a loss function, `compute_loss()`. 2 Call `grad_fn = jax.grad(compute_loss)` to retrieve a gradient-computation

function. 3 Call `grad_fn` to retrieve the gradient values. The loss function should verify the following properties:

It should return a scalar loss value.

Its first argument (which, in the following example, is also the only argument) should contain the state arrays we need gradients for. This argument is usually named `state`. For instance, this first argument could be a single array, a list of arrays, or a dict of arrays.

Let’s take a look at a simple example. Here’s a loss-computation function that takes a single scalar, `input_var` and returns a scalar loss value—just the square of the input:

### PDF p.246 (printed 222) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Using fit() with a custom training loop
In the previous sections, we were writing our own training logic entirely from scratch. Doing so provides you with the most flexibility, but you end up writing a lot of code, while simultaneously missing out on many convenient features of `fit()`, such as callbacks, performance optimizations, or built-in support for distributed training.

What if you need a custom training algorithm, but you still want to use the power of the built-in Keras training loop? There’s actually a middle ground between `fit()` and a training loop written from scratch: you can provide a custom training step function and let the framework do the rest. You can do this by overriding the `train_step()` method of the `Model` class. This is the function that is called by `fit()` for every batch of data. You will then be able to call `fit()` as usual—and it will be running your own learning algorithm under the hood. Here’s how it works:

Create a new class that subclasses `keras.Model`.

### PDF p.114 (printed 90) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
That’s it! You’re now able to write a custom training loop in JAX.
