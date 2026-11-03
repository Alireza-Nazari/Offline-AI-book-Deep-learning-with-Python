# Generation bundle for page_584  (PDF page 584, printed page 560)

Page kind: content
Chapter: 18 Best practices for the real world
Section: 18.3 Speeding up training and inference with lower-precision computation
Subsection: 18.3.5 Beyond mixed precision: float8 training
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_584.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_584.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p584_e03] (body) of representable numbers can cause small gradients to be rounded down to zero. This prevents the model from learning effectively.
[p584_e04] (body) Gradient values are proportional to the loss value, so to encourage gradients to be larger, a simple trick is to multiply the loss by a large scalar factor. Your gradients will then be much less likely to get rounded to zero. Keras makes this easy. If you want to use a fixed loss scaling factor, you can simply pass a `loss_scale_factor` argument to your optimizer like this:
[p584_e05] (code)
```
optimizer = keras.optimizers.Adam(learning_rate=1e-3, loss_scale_factor=10)
```
[p584_e06] (body) If you would like for the optimizer to automatically figure out the right scaling factor, you can also use the `LossScaleOptimizer` wrapper:
[p584_e07] (code)
```
optimizer = keras.optimizers.LossScaleOptimizer(
    keras.optimizers.Adam(learning_rate=1e-3)
)
```
[p584_e08] (body) Using `LossScaleOptimizer` is usually your best option: the right scaling value can change over the course of training!
[p584_e09] (heading_subsection) 18.3.5 Beyond mixed precision: float8 training
[p584_e10] (body) If running your forward pass in 16-bit precision yields such neat performance benefits, you might want to ask: Could we go even lower? What about 8-bit precision? Four bits, maybe? Two bits? The answer is, it’s complicated.
[p584_e11] (body) Mixed precision training using `float16` in the forward pass is that last level of precision that “just works”—`float16` precision has enough bits to represent all intermediate tensors (except for gradient updates, which is why we use `float32` for those). This is no longer true if you go down to `float8` precision: you are simply losing too much information. It is still possible to use `float8` in some computations, but this requires you to make considerable modifications to your forward pass. You will not be able to simply set your `compute_dtype` to `float8` and run. The Keras framework provides a built-in implementation for `float8` training. Because it specifically targets Transformer use cases, it only covers a restricted set of layers: `Dense`, `EinsumDense` (the version of `Dense` that is used by the `MultiHeadAttention` layer), and `Embedding` layers. The way it works is not simple—it keeps track of past activation values to rescale activations at each step so as to utilize the full range of values representable in `float8`. It also needs to override part of the backward pass to do the same with gradient values. Importantly, this added overhead has a computational cost. If your model is too small or if your GPU isn’t powerful enough, that cost will exceed the benefits of doing certain operations in `float8`, and you will see a slowdown instead of a speedup. `float8` training

## Equation candidates (formulas written inline in the text)
- [p584_e05] optimizer = keras.optimizers.Adam(learning_rate=1e-3, loss_scale_factor=10)
- [p584_e07] optimizer = keras.optimizers.LossScaleOptimizer(
    keras.optimizers.Adam(learning_rate=1e-3)
)

## Chapter outline
Chapter: 18 Best practices for the real world (PDF pages 562-587)
  - 18.1 Getting the most out of your models (p.563)
      - 18.1.1 Hyperparameter optimization (p.563)
      - 18.1.2 Model ensembling (p.570)
  - 18.2 Scaling up model training with multiple devices (p.572)
      - 18.2.1 Multi-GPU training (p.572)
      - 18.2.2 Distributed training in practice (p.574)
      - 18.2.3 TPU training (p.579)
  - 18.3 Speeding up training and inference with lower-precision computation (p.580)
      - 18.3.1 Understanding floating-point precision (p.580)
      - 18.3.2 Float16 inference (p.582)
      - 18.3.3 Mixed-precision training (p.583)
      - 18.3.4 Using loss scaling with mixed precision (p.583)
      - 18.3.5 Beyond mixed precision: float8 training (p.584)
      - 18.3.6 Faster inference with quantization (p.585)

## Previous page (PDF 583) text, for continuity
Table 18.1 Difference between `float16` and `bfloat16`

dtype

Exponent bits 5 8 Mantissa bits 10 7 Sign bits 1 1

18.3.3 Mixed-precision training

Setting your default float precision to 16 bits is a great way to speed up inference. Now, when it comes to training, there’s a significant complication. The gradient descent process wouldn’t run smoothly in `float16` or `bfloa16`, since we couldn’t represent small gradient updates of around 1e-5 or 1e-6, which are quite common.

You can, however, use a hybrid approach: that’s what mixed-precision training is about. The idea is to use 16-bit computation in places where precision isn’t an issue, while working with 32-bit values in other places to maintain numerical stability—in particular, when handling gradients and variable updates. By maintaining the precision-sensitive parts of the model in full precision, you can get most of the speed benefits of 16-bit computation without meaningfully impacting model quality. You can turn on mixed precision like this:

Typically, most of the forward pass of the model will be done in `float16` (with the exception of numerically unstable operations like softmax), while the weights of the model will be stored and updated in `float32`. Your `float16` gradients will be cast to `float32` before updating the `float32` variables.

Keras layers have a `variable_dtype` and a `compute_dtype` attribute. By default, both of these are set to `float32`. When you turn on mixed precision, the `compute_dtype` of most layers switches to `float16`. As a result, those layer will cast their inputs to `float16` and will perform their computation in `float16` (using half-precision copies of the weights). However, since their `variable_dtype` is still `float32`, their weights will be able to receive accurate `f …

## Next page (PDF 585) text, for continuity
is only viable for very large models (typically over 5B parameters) and large, recent GPUs such as the NVDIA H100. `float8` is rarely used in practice, except in foundation model training runs.

18.3.6 Faster inference with quantization

Running inference in `float16`—or even `float8`—will result in a nice speedup for your models. But there’s also another trick you can use: `int8` quantization. The big idea is to take an already trained model with weights in `float32` and convert these weights to a lower-precision dtype (typically `int8`) while preserving the numerical correctness of the forward pass as much as possible.

If you want to implement quantization from scratch, the math is simple: the general idea is to scale all `matmul` input tensors by a certain factor so that their coefficients fit in the range representable with `int8`, which is `[-127, 127]`—a total of 256 possible values. After scaling the inputs, you cast them to `int8` and perform the `matmul` operation in `int8` precision, which should be quite a bit faster than `float16`. Finally, you cast the output back to `float32`, and you divide it by the product of the input scaling factors. Since `matmul` is a linear operation, this final unscaling cancels out the initial scaling, and you should get the same output as if you used the original values—any loss of accuracy only comes from the value rounding that happens when you cast the inputs to `int8`. Let’s make this concrete with an example. Let’s say you want to perform `matmul(x, ` `kernel)`, with the following values:

If you were to naively cast these values to `int8` without scaling first, that would be very destructive—for instance, your `x` would become `[[0, 0], [1, 0]]`. So let’s apply the “abs-max” scaling scheme, which spreads out the values of …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.241 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.244 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.586 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation
- PDF p.96 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.242 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.122 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.103 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.582 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.582 (printed 558) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 18.3.2 Float16 inference
Every model you’ve seen in this book so far has used single-precision numbers: it stored its state as `float32` weight variables and ran its computations on `float32` inputs. That’s enough precision to run the forward and backwards pass of a model without losing any information—in particular when it comes to small gradient updates (recall that the typically learning rate is 1e-3, and it’s pretty common to see weight updates on the order of 1e-6).

Modern GPUs and TPUs feature specialized hardware that can run 16-bit operations much faster and using less memory than equivalent 32-bit operations. By using these lower-precision operations whenever possible, you can speed up training on those devices by a significant factor. You can set the default floating point precision to `float16` in Keras via

### PDF p.586 (printed 562) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 
Pretty accurate! For a large `matmul`, doing this will save you a lot of compute, since `int8` computation can be considerably faster than even `float16` computation, and you only had to add fairly fast elementwise ops to the computation graphs—`abs`, `max`, `clip`, `cast`, `divide`, `multiply`.

Now, of course, I don’t expect you to ever implement quantization by hand—that would be tremendously impractical. Similarly to `float8`, `int8` quantization is built directly into specific Keras layers: `Dense`, `EinsumDense`, and `Embedding`. This unlocks `int8` inference support for any Transformer-based model. Here’s how to use it with any Keras model that includes such layers:

### PDF p.239 (printed 215) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Training vs. inference
In the low-level training loop examples you’ve seen so far, step 1 (the forward pass) was done via `predictions = model(inputs)`, and step 2 (retrieving the gradients computed by the gradient tape) was done via a backend-specific API, such as

`gradients = tape.gradient(loss, model.weights)` in TensorFlow

`loss.backward()` in PyTorch

`jax.value_and_grad()` in JAX

In the general case, there are actually two subtleties you need to take into account.

Some Keras layers, such as the `Dropout` layer, have different behaviors during training and during inference (when you use them to generate predictions). Such layers expose a `training` Boolean argument in their `call()` method. Calling `dropout(inputs, ` `training=True)` will drop some activation entries, while calling `dropout(inputs, ` `training=False)` does nothing. By extension, Functional models and Sequential models also expose this `training` argument in their `call()` methods. Remember to pass `training=True` when you call a Keras model during the forward pass! Our forward pass thus becomes `predictions = model(inputs, training=True)`. In addition, note that when you retrieve the gradients of the weights of your model, you s

### PDF p.68 (printed 44) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
Instead, you can use the four-step algorithm outlined at the beginning of this section: modify the parameters little by little based on the current loss value on a random batch of data. Because you’re dealing with a differentiable function, you can compute its gradient, which gives you an efficient way to implement step 4. If you update the weights in the opposite direction from the gradient, the loss will be a little less every time: 1 Draw a batch of training samples `x` and corresponding targets `y_true`. 2 Run the model on `x` to obtain predictions `y_pred` (this is called the forward pass). 3 Compute the loss of the model on the batch, a measure of the mismatch between

`y_pred` and `y_true`. 4 Compute the gradient of the loss with regard to the model’s parameters (this is

called the backward pass).

5 Move the parameters a little in the opposite direction from the gradient—for

example, `W -= learning_rate * gradient`—thus reducing the loss on the batch a bit. The learning rate (`learning_rate` here) would be a scalar factor modulating the “speed” of the gradient descent process. Easy enough! What we just described is called mini-batch stochastic gradient descent (mini-batch

### PDF p.212 (printed 188) — 6 The universal workflow of machine learning › 6.3 Deploying your model › 
the memory and compute footprint of your model at a small cost in performance metrics. By tuning how much pruning you want to apply, you are in control of the tradeoff between size and accuracy.

Weight quantization—Deep learning models are trained with single-precision floating-point (`float32`) weights. However, it’s possible to quantize weights to 8-bit signed integers (`int8`) to get an inference-only model that’s four times smaller but remains near the accuracy of the original model. Keras models come with a built-in `quantize()` API that can help with this. Simply call `model
` `.quantize("int8")` to compress each weight in your model to a single byte.

### PDF p.64 (printed 40) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
`y_pred` and `y_true`. 4 Update all weights of the model in a way that slightly reduces the loss on this batch. You’ll eventually end up with a model that has a very low loss on its training data: a low mismatch between predictions `y_pred` and expected targets `y_true`. The model has “learned” to map its inputs to correct targets. From afar, it may look like magic, but when you reduce it to elementary steps, it turns out to be simple. Step 1 sounds easy enough—it’s just I/O code. Steps 2 and 3 are merely the application of a handful of tensor operations, so you could implement these steps purely from what you learned in the previous section. The difficult part is step 4: updating the model’s weights. Given an individual weight coefficient in the model, how can you compute whether the coefficient should be increased or decreased, and by how much? One naive solution would be to freeze all weights in the model except the one scalar coefficient being considered and try different values for this coefficient. Let’s say the initial value of the coefficient is 0.3. After the forward pass on a batch of data, the loss of the model on the batch is 0.5. If you change the coefficient’s value t

### PDF p.521 (printed 497) — 16 Text generation › 16.4 Going further with LLMs › Choosing a dtype for large models
You might have noticed we passed `dtype="float32"` when creating our Gemma model the first time and `dtype="bfloat16"` now. What is going on? For models like Gemma with more than a billion parameters, the number of bytes used for each floating-point number is an important consideration. When you are training a model, it is often a good idea to use 32 bits (4 bytes) per parameter. 32-bit floats can represent very small values, which can help keep training gradients stable. Here we aren’t doing any training, so we pass `bfloat16`, which uses only 2 bytes per parameter. We don’t need to worry about gradient stability, and we will save many gigabytes of memory by using a lower precision. There’s a detailed discussion of floating-point precision coming up in chapter 18.

Like the earlier Gemma model we fine-tuned ourselves, this instruction-tuned checkpoint comes with a specific template for formatting its input. Again, the exact text does not matter, what is important is that our prompt template matches what was used to tune the model:

### PDF p.582 (printed 558) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 18.3.2 Float16 inference
Note that this should be done before you define your model. Doing this will net you a nice speedup for model inference, for instance, via `model.predict()`. You should expect a nearly 2× speed boost on GPU and TPU.

There’s also an alternative to `float16` that works better on some devices, in particular TPUs: `bfloat16`. `bfloat16` is also a 16-bit precision floating-point type, but it differs from `float16` in its structure: it uses 8 exponent bits instead of 5, and 7 mantissa bits instead of 10 (see table 18.1). This means it can cover a much wider range of values, but it has a lower “resolution” over this range. Some devices are better optimized for `bfloat16` compared to `float16`, so it can be a good idea to try both before settling for the option that turns out to be the fastest.
