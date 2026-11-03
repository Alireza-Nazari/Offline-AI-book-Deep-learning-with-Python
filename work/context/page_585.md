# Generation bundle for page_585  (PDF page 585, printed page 561)

Page kind: content
Chapter: 18 Best practices for the real world
Section: 18.3 Speeding up training and inference with lower-precision computation
Subsection: 18.3.6 Faster inference with quantization
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_585.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_585.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p585_e03] (body) is only viable for very large models (typically over 5B parameters) and large, recent GPUs such as the NVDIA H100. `float8` is rarely used in practice, except in foundation model training runs.
[p585_e04] (heading_subsection) 18.3.6 Faster inference with quantization
[p585_e05] (body) Running inference in `float16`—or even `float8`—will result in a nice speedup for your models. But there’s also another trick you can use: `int8` quantization. The big idea is to take an already trained model with weights in `float32` and convert these weights to a lower-precision dtype (typically `int8`) while preserving the numerical correctness of the forward pass as much as possible.
[p585_e06] (body) If you want to implement quantization from scratch, the math is simple: the general idea is to scale all `matmul` input tensors by a certain factor so that their coefficients fit in the range representable with `int8`, which is `[-127, 127]`—a total of 256 possible values. After scaling the inputs, you cast them to `int8` and perform the `matmul` operation in `int8` precision, which should be quite a bit faster than `float16`. Finally, you cast the output back to `float32`, and you divide it by the product of the input scaling factors. Since `matmul` is a linear operation, this final unscaling cancels out the initial scaling, and you should get the same output as if you used the original values—any loss of accuracy only comes from the value rounding that happens when you cast the inputs to `int8`. Let’s make this concrete with an example. Let’s say you want to perform `matmul(x, ` `kernel)`, with the following values:
[p585_e07] (code)
```
from keras import ops
x = ops.array([[0.1, 0.9], [1.2, -0.8]])
kernel = ops.array([[-0.1, -2.2], [1.1, 0.7]])
```
[p585_e08] (body) If you were to naively cast these values to `int8` without scaling first, that would be very destructive—for instance, your `x` would become `[[0, 0], [1, 0]]`. So let’s apply the “abs-max” scaling scheme, which spreads out the values of each tensor across the `[-127, 127]` range:
[p585_e09] (code)
```
def abs_max_quantize(value):
    abs_max = ops.max(ops.abs(value), keepdims=True)
    scale = ops.divide(127, abs_max + 1e-7)
    scaled_value = value * scale
    scaled_value = ops.clip(ops.round(scaled_value), -127, 127)
```

## Code annotations printed next to listings (callouts, in order)
- Scales the value
- Scale is max of int range divided by max of
- tensor (1e-7 is to avoid dividing by 0).
- Rounding and clipping first is more
- accurate than directly casting.
- Max of absolute value of the tensor

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

## Previous page (PDF 584) text, for continuity
of representable numbers can cause small gradients to be rounded down to zero. This prevents the model from learning effectively.

Gradient values are proportional to the loss value, so to encourage gradients to be larger, a simple trick is to multiply the loss by a large scalar factor. Your gradients will then be much less likely to get rounded to zero. Keras makes this easy. If you want to use a fixed loss scaling factor, you can simply pass a `loss_scale_factor` argument to your optimizer like this:

If you would like for the optimizer to automatically figure out the right scaling factor, you can also use the `LossScaleOptimizer` wrapper:

Using `LossScaleOptimizer` is usually your best option: the right scaling value can change over the course of training!

18.3.5 Beyond mixed precision: float8 training

If running your forward pass in 16-bit precision yields such neat performance benefits, you might want to ask: Could we go even lower? What about 8-bit precision? Four bits, maybe? Two bits? The answer is, it’s complicated.

Mixed precision training using `float16` in the forward pass is that last level of precision that “just works”—`float16` precision has enough bits to represent all intermediate tensors (except for gradient updates, which is why we use `float32` for those). This is no longer true if you go down to `float8` precision: you are simply losing too much information. It is still possible to use `float8` in some computations, but this requires you to make considerable modifications to your forward pass. You will not be able to simply set your `compute_dtype` to `float8` and run. The Keras framework provides a built-in implementation for `float8` training. Because it specifically targets Transformer use cases, it only covers a restricted set of layers: `D …

## Next page (PDF 586) text, for continuity
Now we can perform a faster `matmul` and unscale the output:

How accurate is it? Let’s compare our `y` with the output of the `float32` `matmul`:

Pretty accurate! For a large `matmul`, doing this will save you a lot of compute, since `int8` computation can be considerably faster than even `float16` computation, and you only had to add fairly fast elementwise ops to the computation graphs—`abs`, `max`, `clip`, `cast`, `divide`, `multiply`.

Now, of course, I don’t expect you to ever implement quantization by hand—that would be tremendously impractical. Similarly to `float8`, `int8` quantization is built directly into specific Keras layers: `Dense`, `EinsumDense`, and `Embedding`. This unlocks `int8` inference support for any Transformer-based model. Here’s how to use it with any Keras model that includes such layers:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.583 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation
- PDF p.582 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation
- PDF p.212 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.587 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation
- PDF p.44 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network
- PDF p.153 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.152 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.574 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.212 (printed 188) — 6 The universal workflow of machine learning › 6.3 Deploying your model › 
the memory and compute footprint of your model at a small cost in performance metrics. By tuning how much pruning you want to apply, you are in control of the tradeoff between size and accuracy.

Weight quantization—Deep learning models are trained with single-precision floating-point (`float32`) weights. However, it’s possible to quantize weights to 8-bit signed integers (`int8`) to get an inference-only model that’s four times smaller but remains near the accuracy of the original model. Keras models come with a built-in `quantize()` API that can help with this. Simply call `model
` `.quantize("int8")` to compress each weight in your model to a single byte.

### PDF p.582 (printed 558) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 18.3.2 Float16 inference
Every model you’ve seen in this book so far has used single-precision numbers: it stored its state as `float32` weight variables and ran its computations on `float32` inputs. That’s enough precision to run the forward and backwards pass of a model without losing any information—in particular when it comes to small gradient updates (recall that the typically learning rate is 1e-3, and it’s pretty common to see weight updates on the order of 1e-6).

Modern GPUs and TPUs feature specialized hardware that can run 16-bit operations much faster and using less memory than equivalent 32-bit operations. By using these lower-precision operations whenever possible, you can speed up training on those devices by a significant factor. You can set the default floating point precision to `float16` in Keras via

### PDF p.582 (printed 558) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 18.3.2 Float16 inference
Note that this should be done before you define your model. Doing this will net you a nice speedup for model inference, for instance, via `model.predict()`. You should expect a nearly 2× speed boost on GPU and TPU.

There’s also an alternative to `float16` that works better on some devices, in particular TPUs: `bfloat16`. `bfloat16` is also a 16-bit precision floating-point type, but it differs from `float16` in its structure: it uses 8 exponent bits instead of 5, and 7 mantissa bits instead of 10 (see table 18.1). This means it can cover a much wider range of values, but it has a lower “resolution” over this range. Some devices are better optimized for `bfloat16` compared to `float16`, so it can be a good idea to try both before settling for the option that turns out to be the fastest.

### PDF p.583 (printed 559) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 18.3.3 Mixed-precision training
Typically, most of the forward pass of the model will be done in `float16` (with the exception of numerically unstable operations like softmax), while the weights of the model will be stored and updated in `float32`. Your `float16` gradients will be cast to `float32` before updating the `float32` variables.

Keras layers have a `variable_dtype` and a `compute_dtype` attribute. By default, both of these are set to `float32`. When you turn on mixed precision, the `compute_dtype` of most layers switches to `float16`. As a result, those layer will cast their inputs to `float16` and will perform their computation in `float16` (using half-precision copies of the weights). However, since their `variable_dtype` is still `float32`, their weights will be able to receive accurate `float32` updates from the optimizer, as opposed to half-precision updates. Some operations may be numerically unstable in `float16` (in particular, softmax and crossentropy). If you need to opt out of mixed precision for a specific layer, just pass the argument `dtype="float32"` to the constructor of this layer.

### PDF p.587 (printed 563) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › Summary
You can use hyperparameter tuning and KerasTuner to automate the tedium out of finding the best model configuration. But be mindful of validation-set overfitting!

An ensemble of diverse models can often significantly improve the quality of your predictions.

To further scale your workflows, you can use data parallelism to train a model on multiple devices, as long as the model is small enough to fit on a single device.

For larger models, you can also use model parallelism to split your model’s variables and computation across several devices.

You can speed up model training on GPUs or TPUs by turning on mixed precision—you’ll generally get a nice speed boost at virtually no cost.

You can also speed up inference by using `float16` precision or even `int8` quantization.

### PDF p.583 (printed 559) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 18.3.3 Mixed-precision training
Setting your default float precision to 16 bits is a great way to speed up inference. Now, when it comes to training, there’s a significant complication. The gradient descent process wouldn’t run smoothly in `float16` or `bfloa16`, since we couldn’t represent small gradient updates of around 1e-5 or 1e-6, which are quite common.

You can, however, use a hybrid approach: that’s what mixed-precision training is about. The idea is to use 16-bit computation in places where precision isn’t an issue, while working with 32-bit values in other places to maintain numerical stability—in particular, when handling gradients and variable updates. By maintaining the precision-sensitive parts of the model in full precision, you can get most of the speed benefits of 16-bit computation without meaningfully impacting model quality. You can turn on mixed precision like this:

### PDF p.641 (printed 617) — index ›  › Q
REST API (application programming quantization, faster inference with 561

interface) 185 quantize() API 188 result() method 208

### PDF p.521 (printed 497) — 16 Text generation › 16.4 Going further with LLMs › Choosing a dtype for large models
You might have noticed we passed `dtype="float32"` when creating our Gemma model the first time and `dtype="bfloat16"` now. What is going on? For models like Gemma with more than a billion parameters, the number of bytes used for each floating-point number is an important consideration. When you are training a model, it is often a good idea to use 32 bits (4 bytes) per parameter. 32-bit floats can represent very small values, which can help keep training gradients stable. Here we aren’t doing any training, so we pass `bfloat16`, which uses only 2 bytes per parameter. We don’t need to worry about gradient stability, and we will save many gigabytes of memory by using a lower precision. There’s a detailed discussion of floating-point precision coming up in chapter 18.

Like the earlier Gemma model we fine-tuned ourselves, this instruction-tuned checkpoint comes with a specific template for formatting its input. Again, the exact text does not matter, what is important is that our prompt template matches what was used to tune the model:
