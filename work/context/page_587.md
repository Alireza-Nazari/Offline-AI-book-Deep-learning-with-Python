# Generation bundle for page_587  (PDF page 587, printed page 563)

Page kind: content
Chapter: 18 Best practices for the real world
Section: 18.3 Speeding up training and inference with lower-precision computation
Subsection: 18.3.6 Faster inference with quantization
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_587.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_587.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p587_e03] (heading_section) Summary
[p587_e04] (bullet) You can use hyperparameter tuning and KerasTuner to automate the tedium out of finding the best model configuration. But be mindful of validation-set overfitting!
[p587_e05] (bullet) An ensemble of diverse models can often significantly improve the quality of your predictions.
[p587_e06] (bullet) To further scale your workflows, you can use data parallelism to train a model on multiple devices, as long as the model is small enough to fit on a single device.
[p587_e07] (bullet) For larger models, you can also use model parallelism to split your model’s variables and computation across several devices.
[p587_e08] (bullet) You can speed up model training on GPUs or TPUs by turning on mixed precision—you’ll generally get a nice speed boost at virtually no cost.
[p587_e09] (bullet) You can also speed up inference by using `float16` precision or even `int8` quantization.

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

## Previous page (PDF 586) text, for continuity
Now we can perform a faster `matmul` and unscale the output:

How accurate is it? Let’s compare our `y` with the output of the `float32` `matmul`:

Pretty accurate! For a large `matmul`, doing this will save you a lot of compute, since `int8` computation can be considerably faster than even `float16` computation, and you only had to add fairly fast elementwise ops to the computation graphs—`abs`, `max`, `clip`, `cast`, `divide`, `multiply`.

Now, of course, I don’t expect you to ever implement quantization by hand—that would be tremendously impractical. Similarly to `float8`, `int8` quantization is built directly into specific Keras layers: `Dense`, `EinsumDense`, and `Embedding`. This unlocks `int8` inference support for any Transformer-based model. Here’s how to use it with any Keras model that includes such layers:

## Next page (PDF 588) text, for continuity
The future of AI

This chapter covers

The limitations of deep learning

The nature of intelligence

What’s missing from current approaches

What the future might look like

To use a tool appropriately, you should not only understand what it can do but also be aware of what it can’t do. I’m going to present an overview of some key limitations of deep learning. Then, I’ll offer some speculative thoughts about the future evolution of AI and what it would take to get to human-level general intelligence. This should be especially interesting to you if you’d like to get into fundamental research.

19.1 The limitations of deep learning

There are infinitely many things you can do with deep learning. But deep learning can’t do everything. To use a tool well, you should be aware of its limitations, not just its strengths. So where does deep learning fall short?

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.572 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.563 — 18 Best practices for the real world › 18.1 Getting the most out of your models
- PDF p.574 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.562 — 18 Best practices for the real world › 
- PDF p.573 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.580 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation
- PDF p.585 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation
- PDF p.569 — 18 Best practices for the real world › 18.1 Getting the most out of your models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.585 (printed 561) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 18.3.6 Faster inference with quantization
Running inference in `float16`—or even `float8`—will result in a nice speedup for your models. But there’s also another trick you can use: `int8` quantization. The big idea is to take an already trained model with weights in `float32` and convert these weights to a lower-precision dtype (typically `int8`) while preserving the numerical correctness of the forward pass as much as possible.

If you want to implement quantization from scratch, the math is simple: the general idea is to scale all `matmul` input tensors by a certain factor so that their coefficients fit in the range representable with `int8`, which is `[-127, 127]`—a total of 256 possible values. After scaling the inputs, you cast them to `int8` and perform the `matmul` operation in `int8` precision, which should be quite a bit faster than `float16`. Finally, you cast the output back to `float32`, and you divide it by the product of the input scaling factors. Since `matmul` is a linear operation, this final unscaling cancels out the initial scaling, and you should get the same output as if you used the original values—any loss of accuracy only comes from the value rounding that happens when you cast the inputs to `int8`.

### PDF p.572 (printed 548) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › 18.2.1 Multi-GPU training
While GPUs are getting more powerful every year, deep learning models are also getting increasingly larger, requiring ever more computational resources. Training on a single GPU puts a hard bound on how fast you can move. The solution? You could simply add more GPUs and start doing multi-GPU distributed training.

There are two ways to distribute computation across multiple devices: data parallelism and model parallelism. With data parallelism, a single model gets replicated on multiple devices or multiple machines. Each of the model replicas processes different batches of data, and then they merge their results. With model parallelism, different parts of a single model run on different devices, processing a single batch of data together at the same time. This works best with models that have a naturally parallel architecture, such as models that feature multiple branches. In practice, model parallelism is only used in the case of models that are too large to fit on any single device: it isn’t used as a way to speed up training of regular models but as a way to train larger models. Then, of course, you can also mix both data parallelism and model parallelism: a single model can be 

### PDF p.582 (printed 558) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 18.3.2 Float16 inference
Note that this should be done before you define your model. Doing this will net you a nice speedup for model inference, for instance, via `model.predict()`. You should expect a nearly 2× speed boost on GPU and TPU.

There’s also an alternative to `float16` that works better on some devices, in particular TPUs: `bfloat16`. `bfloat16` is also a 16-bit precision floating-point type, but it differs from `float16` in its structure: it uses 8 exponent bits instead of 5, and 7 mantissa bits instead of 10 (see table 18.1). This means it can cover a much wider range of values, but it has a lower “resolution” over this range. Some devices are better optimized for `bfloat16` compared to `float16`, so it can be a good idea to try both before settling for the option that turns out to be the fastest.

### PDF p.562 (printed 538) — 18 Best practices for the real world ›  › This chapter covers
Hyperparameter tuning

Model ensembling

Training Keras models on multiple GPUs or on TPU

Mixed-precision training

Quantization

You’ve come quite far since the beginning of this book. You can now train image classification models, image segmentation models, models for classification or regression on vector data, timeseries forecasting models, text classification models, sequence-to-sequence models, and even generative models for text and images. You’ve got all the bases covered.

However, your models so far have all been trained at a small scale—on small datasets, with a single GPU—and they generally haven’t reached the best achievable performance on each dataset we’ve looked at. This book is, after all, an introductory book. If you are to go out into the real world and achieve state-of-the-art results on brand new problems, there’s still a bit of a chasm that you’ll need to cross.

### PDF p.580 (printed 556) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › computation
What if I told you there’s a simple technique you could use to speed up training and inference of almost any model by up to 2×, basically for free? It seems too good to be true, and yet, such a trick does exist. To understand how it works, first, we need to take a look at the notion of “precision” in computer science.

### PDF p.582 (printed 558) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 18.3.2 Float16 inference
Every model you’ve seen in this book so far has used single-precision numbers: it stored its state as `float32` weight variables and ran its computations on `float32` inputs. That’s enough precision to run the forward and backwards pass of a model without losing any information—in particular when it comes to small gradient updates (recall that the typically learning rate is 1e-3, and it’s pretty common to see weight updates on the order of 1e-6).

Modern GPUs and TPUs feature specialized hardware that can run 16-bit operations much faster and using less memory than equivalent 32-bit operations. By using these lower-precision operations whenever possible, you can speed up training on those devices by a significant factor. You can set the default floating point precision to `float16` in Keras via

### PDF p.583 (printed 559) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 18.3.3 Mixed-precision training
Setting your default float precision to 16 bits is a great way to speed up inference. Now, when it comes to training, there’s a significant complication. The gradient descent process wouldn’t run smoothly in `float16` or `bfloa16`, since we couldn’t represent small gradient updates of around 1e-5 or 1e-6, which are quite common.

You can, however, use a hybrid approach: that’s what mixed-precision training is about. The idea is to use 16-bit computation in places where precision isn’t an issue, while working with 32-bit values in other places to maintain numerical stability—in particular, when handling gradients and variable updates. By maintaining the precision-sensitive parts of the model in full precision, you can get most of the speed benefits of 16-bit computation without meaningfully impacting model quality. You can turn on mixed precision like this:

### PDF p.572 (printed 548) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › 18.2 Scaling up model training with multiple devices
Recall the “loop of progress” concept we introduced in chapter 7: the quality of your ideas is a function of how many refinement cycles they’ve been through (figure 18.1). And the speed at which you can iterate on an idea is a function of how fast you can set up an experiment, how fast you can run that experiment, and, finally, how well you can analyze the resulting data.

Figure 18.1 The loop of progress

As you develop your expertise in the Keras API, how fast you can code up your deep learning experiments will cease to be the bottleneck of this progress cycle. The next bottleneck will become the speed at which you can train your models. Fast training infrastructure means that you can get your results back in 10 or 15 minutes and, hence, that you can go through dozens of iterations every day. Faster training directly improves the quality of your deep learning solutions. In this section, you’ll learn about how to scale up your training runs by using multiple GPUs or TPUs.
