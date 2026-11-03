# Generation bundle for page_582  (PDF page 582, printed page 558)

Page kind: content
Chapter: 18 Best practices for the real world
Section: 18.3 Speeding up training and inference with lower-precision computation
Subsection: 18.3.2 Float16 inference
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_582.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_582.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p582_e03] (figure_caption) (continued)
[p582_e04] (sidebar) For this reason, the numerical error incurred when converting a number to its floating-point representation can vary wildly depending on the exact value considered, and the error tends to get larger for numbers with a large absolute value.
[p582_e05] (body) The way to think about the resolution of floating-point numbers is in terms of the smallest distance between two arbitrary numbers that you’ll be able to safely process. In single precision, that’s around 1e-7. In double precision, that’s around 1e-16. And in half precision, it’s only 1e-3.
[p582_e06] (heading_subsection) 18.3.2 Float16 inference
[p582_e07] (body) Every model you’ve seen in this book so far has used single-precision numbers: it stored its state as `float32` weight variables and ran its computations on `float32` inputs. That’s enough precision to run the forward and backwards pass of a model without losing any information—in particular when it comes to small gradient updates (recall that the typically learning rate is 1e-3, and it’s pretty common to see weight updates on the order of 1e-6).
[p582_e08] (body) Modern GPUs and TPUs feature specialized hardware that can run 16-bit operations much faster and using less memory than equivalent 32-bit operations. By using these lower-precision operations whenever possible, you can speed up training on those devices by a significant factor. You can set the default floating point precision to `float16` in Keras via
[p582_e09] (code)
```
import keras
keras.config.set_dtype_policy("float16")
```
[p582_e10] (body) Note that this should be done before you define your model. Doing this will net you a nice speedup for model inference, for instance, via `model.predict()`. You should expect a nearly 2× speed boost on GPU and TPU.
[p582_e11] (body) There’s also an alternative to `float16` that works better on some devices, in particular TPUs: `bfloat16`. `bfloat16` is also a 16-bit precision floating-point type, but it differs from `float16` in its structure: it uses 8 exponent bits instead of 5, and 7 mantissa bits instead of 10 (see table 18.1). This means it can cover a much wider range of values, but it has a lower “resolution” over this range. Some devices are better optimized for `bfloat16` compared to `float16`, so it can be a good idea to try both before settling for the option that turns out to be the fastest.

## Figures
- Figure None: (continued)  | labels inside the figure: (none - image)

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

## Previous page (PDF 581) text, for continuity
For instance, you may be familiar with `uint8` integers, which are integers encoded on eight bits: `00000000` represents `0` in `uint8`, and `11111111` represents 255. To represent integers beyond 255, you’d need to add more bits—eight isn’t enough. Most integers are stored on 32 bits, with which we can represent signed integers ranging from −2147483648 to 2147483647.

Floating-point numbers are the same. In mathematics, real numbers form a continuous axis: there’s an infinite number of points in between any two numbers. You can always zoom in on the axis of reals. In computer science, this isn’t true: there’s only a finite number of intermediate points between 3 and 4, for instance. How many? Well, it depends on the precision you’re working with: the number of bits you’re using to store a number. You can only zoom up to a certain resolution. There are three levels of precision you’d typically use:

Half precision, or `float16`, where numbers are stored on 16 bits

Single precision, or `float32`, where numbers are stored on 32 bits

Double precision, or `float64`, where numbers are stored on 64 bits

You could even go up to `float8`, as you’ll see in a bit.

On floating-point encoding

A counterintuitive fact about floating-point numbers is that representable numbers are not uniformly distributed. Larger numbers have lower precision: there’s the same number of representable values between `2 ** N` and `2 ** (N + 1)` as there is between 1 and 2, for any `N`. That’s because floating-point numbers are encoded in three parts—the sign, the significant value (called the mantissa), and the exponent in the form

For example, figure 18.3 demonstrates how you would encode the closest `float32` value approximating Pi:

Figure 18.3 The number Pi encoded in single precision via a si …

## Next page (PDF 583) text, for continuity
Table 18.1 Difference between `float16` and `bfloat16`

dtype

Exponent bits 5 8 Mantissa bits 10 7 Sign bits 1 1

18.3.3 Mixed-precision training

Setting your default float precision to 16 bits is a great way to speed up inference. Now, when it comes to training, there’s a significant complication. The gradient descent process wouldn’t run smoothly in `float16` or `bfloa16`, since we couldn’t represent small gradient updates of around 1e-5 or 1e-6, which are quite common.

You can, however, use a hybrid approach: that’s what mixed-precision training is about. The idea is to use 16-bit computation in places where precision isn’t an issue, while working with 32-bit values in other places to maintain numerical stability—in particular, when handling gradients and variable updates. By maintaining the precision-sensitive parts of the model in full precision, you can get most of the speed benefits of 16-bit computation without meaningfully impacting model quality. You can turn on mixed precision like this:

Typically, most of the forward pass of the model will be done in `float16` (with the exception of numerically unstable operations like softmax), while the weights of the model will be stored and updated in `float32`. Your `float16` gradients will be cast to `float32` before updating the `float32` variables.

Keras layers have a `variable_dtype` and a `compute_dtype` attribute. By default, both of these are set to `float32`. When you turn on mixed precision, the `compute_dtype` of most layers switches to `float16`. As a result, those layer will cast their inputs to `float16` and will perform their computation in `float16` (using half-precision copies of the weights). However, since their `variable_dtype` is still `float32`, their weights will be able to receive accurate `f …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.580 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation
- PDF p.585 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation
- PDF p.584 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation
- PDF p.212 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.499 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.521 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.203 — 6 The universal workflow of machine learning › 6.2 Developing a model
- PDF p.586 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.584 (printed 560) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 18.3.5 Beyond mixed precision: float8 training
If running your forward pass in 16-bit precision yields such neat performance benefits, you might want to ask: Could we go even lower? What about 8-bit precision? Four bits, maybe? Two bits? The answer is, it’s complicated.

Mixed precision training using `float16` in the forward pass is that last level of precision that “just works”—`float16` precision has enough bits to represent all intermediate tensors (except for gradient updates, which is why we use `float32` for those). This is no longer true if you go down to `float8` precision: you are simply losing too much information. It is still possible to use `float8` in some computations, but this requires you to make considerable modifications to your forward pass. You will not be able to simply set your `compute_dtype` to `float8` and run. The Keras framework provides a built-in implementation for `float8` training. Because it specifically targets Transformer use cases, it only covers a restricted set of layers: `Dense`, `EinsumDense` (the version of `Dense` that is used by the `MultiHeadAttention` layer), and `Embedding` layers. The way it works is not simple—it keeps track of past activation values to rescale activations at each

### PDF p.585 (printed 561) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 18.3.6 Faster inference with quantization
Running inference in `float16`—or even `float8`—will result in a nice speedup for your models. But there’s also another trick you can use: `int8` quantization. The big idea is to take an already trained model with weights in `float32` and convert these weights to a lower-precision dtype (typically `int8`) while preserving the numerical correctness of the forward pass as much as possible.

If you want to implement quantization from scratch, the math is simple: the general idea is to scale all `matmul` input tensors by a certain factor so that their coefficients fit in the range representable with `int8`, which is `[-127, 127]`—a total of 256 possible values. After scaling the inputs, you cast them to `int8` and perform the `matmul` operation in `int8` precision, which should be quite a bit faster than `float16`. Finally, you cast the output back to `float32`, and you divide it by the product of the input scaling factors. Since `matmul` is a linear operation, this final unscaling cancels out the initial scaling, and you should get the same output as if you used the original values—any loss of accuracy only comes from the value rounding that happens when you cast the inputs to `int8`.

### PDF p.212 (printed 188) — 6 The universal workflow of machine learning › 6.3 Deploying your model › 
the memory and compute footprint of your model at a small cost in performance metrics. By tuning how much pruning you want to apply, you are in control of the tradeoff between size and accuracy.

Weight quantization—Deep learning models are trained with single-precision floating-point (`float32`) weights. However, it’s possible to quantize weights to 8-bit signed integers (`int8`) to get an inference-only model that’s four times smaller but remains near the accuracy of the original model. Keras models come with a built-in `quantize()` API that can help with this. Simply call `model
` `.quantize("int8")` to compress each weight in your model to a single byte.

### PDF p.521 (printed 497) — 16 Text generation › 16.4 Going further with LLMs › Choosing a dtype for large models
You might have noticed we passed `dtype="float32"` when creating our Gemma model the first time and `dtype="bfloat16"` now. What is going on? For models like Gemma with more than a billion parameters, the number of bytes used for each floating-point number is an important consideration. When you are training a model, it is often a good idea to use 32 bits (4 bytes) per parameter. 32-bit floats can represent very small values, which can help keep training gradients stable. Here we aren’t doing any training, so we pass `bfloat16`, which uses only 2 bytes per parameter. We don’t need to worry about gradient stability, and we will save many gigabytes of memory by using a lower precision. There’s a detailed discussion of floating-point precision coming up in chapter 18.

Like the earlier Gemma model we fine-tuned ourselves, this instruction-tuned checkpoint comes with a specific template for formatting its input. Again, the exact text does not matter, what is important is that our prompt template matches what was used to tune the model:

### PDF p.587 (printed 563) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › Summary
You can use hyperparameter tuning and KerasTuner to automate the tedium out of finding the best model configuration. But be mindful of validation-set overfitting!

An ensemble of diverse models can often significantly improve the quality of your predictions.

To further scale your workflows, you can use data parallelism to train a model on multiple devices, as long as the model is small enough to fit on a single device.

For larger models, you can also use model parallelism to split your model’s variables and computation across several devices.

You can speed up model training on GPUs or TPUs by turning on mixed precision—you’ll generally get a nice speed boost at virtually no cost.

You can also speed up inference by using `float16` precision or even `int8` quantization.

### PDF p.580 (printed 556) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 18.3.1 Understanding floating-point precision
Precision is to numbers what resolution is to images. Because computers can only process 1s and 0s, any number seen by a computer has to be encoded as a binary string.

### PDF p.580 (printed 556) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › computation
What if I told you there’s a simple technique you could use to speed up training and inference of almost any model by up to 2×, basically for free? It seems too good to be true, and yet, such a trick does exist. To understand how it works, first, we need to take a look at the notion of “precision” in computer science.

### PDF p.586 (printed 562) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 
Pretty accurate! For a large `matmul`, doing this will save you a lot of compute, since `int8` computation can be considerably faster than even `float16` computation, and you only had to add fairly fast elementwise ops to the computation graphs—`abs`, `max`, `clip`, `cast`, `divide`, `multiply`.

Now, of course, I don’t expect you to ever implement quantization by hand—that would be tremendously impractical. Similarly to `float8`, `int8` quantization is built directly into specific Keras layers: `Dense`, `EinsumDense`, and `Embedding`. This unlocks `int8` inference support for any Transformer-based model. Here’s how to use it with any Keras model that includes such layers:
