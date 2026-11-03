# Generation bundle for page_580  (PDF page 580, printed page 556)

Page kind: content
Chapter: 18 Best practices for the real world
Section: 18.3 Speeding up training and inference with lower-precision computation
Subsection: 18.3.1 Understanding floating-point precision
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_580.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_580.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p580_e03] (body) TPU runtime in addition to the GPU runtime. For more serious training runs, Google Cloud also makes available TPU v3 through v5, which are even faster.
[p580_e04] (body) When running Keras code with the JAX backend on a TPU-enabled notebook, you don’t need anything more than calling `keras.distribution.set_distribution
` `(distribution)` with a `DataParallel` or `ModelParallel` distribution instance to start using your TPU cores. Make sure to call it before creating your model! NOTE Because TPUs can process batches of data extremely quickly, the speed at which you can read data from Google Cloud Storage (GCS) can easily become a bottleneck. If your dataset is small enough, you should keep it in the memory of the virtual machine. You can do so by calling `dataset.cache()` on your `tf.data.Dataset` instance. That way, the data will only be read from GCS once.
[p580_e05] (heading_subsubsection) Using step fusing to improve TPU utilization
[p580_e06] (body) Because a TPU has a lot of compute power available, you need to train with very large batches to keep the TPU cores busy. For small models, the batch size required can get extraordinarily large—upward of 10,000 samples per batch. When working with enormous batches, you should make sure to increase your optimizer learning rate accordingly: you’re going to be making fewer updates to your weights, but each update will be more accurate (since the gradients are computed using more data points); hence, you should move the weights by a greater magnitude with each update.
[p580_e07] (body) There is, however, a simple trick you can use to keep reasonably sized batches while maintaining full TPU utilization: step fusing. The idea is to run multiple steps of training during each TPU execution step. Basically, do more work in between two roundtrips from the virtual machine memory to the TPU. To do this, simply specify the `steps_per_` `execution` argument in `compile()`—for instance, `steps_per_execution=8` to run eight steps of training during each TPU execution. For small models that are underutilizing the TPU, this can result in a dramatic speedup:
[p580_e08] (code)
```
model.compile(..., steps_per_execution=8)
```
[p580_e09] (heading_section) 18.3 Speeding up training and inference with lower-precision
[p580_e10] (heading_section) computation
[p580_e11] (body) What if I told you there’s a simple technique you could use to speed up training and inference of almost any model by up to 2×, basically for free? It seems too good to be true, and yet, such a trick does exist. To understand how it works, first, we need to take a look at the notion of “precision” in computer science.
[p580_e12] (heading_subsection) 18.3.1 Understanding floating-point precision
[p580_e13] (body) Precision is to numbers what resolution is to images. Because computers can only process 1s and 0s, any number seen by a computer has to be encoded as a binary string.

## Equation candidates (formulas written inline in the text)
- [p580_e07] To do this, simply specify the `steps_per_` `execution` argument in `compile()`—for instance, `steps_per_execution=8` to run eight steps of training during each TPU execution.
- [p580_e08] model.compile(..., steps_per_execution=8)

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

## Previous page (PDF 579) text, for continuity
computation for any large language model training run—it scales to as many devices as you have available and arbitrary model sizes.

To check how your variables were sharded, you can inspect the `variable.value
` `.sharding` property, like this:

You can even visualize it via the JAX utility `jax.debug.visualize_sharding`:

tf.data performance tips

When doing distributed training, always provide your data as a `tf.data.Dataset` object to guarantee best performance (passing your data as NumPy arrays also works since those get converted to `Dataset` objects by `fit()`). You should also make sure to use data prefetching: before passing the dataset to `fit()`, call `dataset
` `.prefetch(buffer_size)`. If you aren’t sure what buffer size to pick, try the `dataset
` `.prefetch(tf.data.AUTOTUNE)` option, which will pick a buffer size for you.

18.2.3 TPU training

Beyond just GPUs, there is generally a trend in the deep learning world toward moving workflows to increasingly specialized hardware designed specifically for deep learning workflows; such single-purpose chips are known as ASICs (application-specific integrated circuits). Various companies big and small are working on new chips, but today the most prominent effort along these lines is Google’s Tensor Processing Unit (TPU), which is available on Google Cloud and via Google Colab.

Training on TPU does involve jumping through some hoops. But it’s worth the extra work: TPUs are really, really fast. Training on a TPU v2 (available on Colab) will typically be 15× faster than training a NVIDIA P100 GPU. For most models, TPU training ends up being 3× more cost-effective than GPU training on average. You can actually use TPU v2 for free in Colab. In the Colab menu, under the Runtime tab, in the Change Runtime Type option, y …

## Next page (PDF 581) text, for continuity
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

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.582 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation
- PDF p.563 — 18 Best practices for the real world › 18.1 Getting the most out of your models
- PDF p.583 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation
- PDF p.96 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.572 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.573 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.576 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.584 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.582 (printed 558) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 18.3.2 Float16 inference
Note that this should be done before you define your model. Doing this will net you a nice speedup for model inference, for instance, via `model.predict()`. You should expect a nearly 2× speed boost on GPU and TPU.

There’s also an alternative to `float16` that works better on some devices, in particular TPUs: `bfloat16`. `bfloat16` is also a 16-bit precision floating-point type, but it differs from `float16` in its structure: it uses 8 exponent bits instead of 5, and 7 mantissa bits instead of 10 (see table 18.1). This means it can cover a much wider range of values, but it has a lower “resolution” over this range. Some devices are better optimized for `bfloat16` compared to `float16`, so it can be a good idea to try both before settling for the option that turns out to be the fastest.

### PDF p.96 (printed 72) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › 
For simplicity, we’ll do batch training instead of mini-batch training: we’ll run each training step (gradient computation and weight update) on the entire data, rather than iterate over the data in small batches. On one hand, this means that each training step will take much longer to run, since we compute the forward pass and the gradients for 2,000 samples at once. On the other hand, each gradient update will be much more effective at reducing the loss on the training data, since it will encompass information from all training samples instead of, say, only 128 random samples. As a result, we will need many fewer steps of training, and we should use a larger learning rate than what we would typically use for mini-batch training (we’ll use `learning_rate = 0.1`, as previously defined).

Listing 3.21 The batch training loop

### PDF p.582 (printed 558) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 18.3.2 Float16 inference
Every model you’ve seen in this book so far has used single-precision numbers: it stored its state as `float32` weight variables and ran its computations on `float32` inputs. That’s enough precision to run the forward and backwards pass of a model without losing any information—in particular when it comes to small gradient updates (recall that the typically learning rate is 1e-3, and it’s pretty common to see weight updates on the order of 1e-6).

Modern GPUs and TPUs feature specialized hardware that can run 16-bit operations much faster and using less memory than equivalent 32-bit operations. By using these lower-precision operations whenever possible, you can speed up training on those devices by a significant factor. You can set the default floating point precision to `float16` in Keras via

### PDF p.587 (printed 563) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › Summary
You can use hyperparameter tuning and KerasTuner to automate the tedium out of finding the best model configuration. But be mindful of validation-set overfitting!

An ensemble of diverse models can often significantly improve the quality of your predictions.

To further scale your workflows, you can use data parallelism to train a model on multiple devices, as long as the model is small enough to fit on a single device.

For larger models, you can also use model parallelism to split your model’s variables and computation across several devices.

You can speed up model training on GPUs or TPUs by turning on mixed precision—you’ll generally get a nice speed boost at virtually no cost.

You can also speed up inference by using `float16` precision or even `int8` quantization.

### PDF p.115 (printed 91) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Picking a backend framework
Keras can be used together with JAX, TensorFlow, or PyTorch. They’re the “backend frameworks” of Keras. Through these backend frameworks, Keras can run on top of different types of hardware (see figure 3.4)—GPU, TPU, or plain CPU—can be seamlessly scaled to thousands of machines, and can be deployed to a variety of platforms.

Backend frameworks are pluggable: you can switch to a different backend framework after you’ve written some Keras code. You aren’t locked into a single framework and a single ecosystem—you can move your models from JAX to TensorFlow to PyTorch depending on your current needs. For instance, when you develop a Keras model, you could debug it with PyTorch, train it on TPU with JAX for maximum efficiency, and finally run inference with the excellent tooling from the TensorFlow ecosystem.

Figure 3.4 Keras and its backends. A backend is a low-level tensor-computing platform; Keras is a high-level deep learning API.

The default backend for Keras right now is TensorFlow, so if you run `import` `keras` in a fresh environment, without having configured anything, you will be running on top of TensorFlow. There are two ways to pick a different backend:

Set the environ

### PDF p.562 (printed 538) — 18 Best practices for the real world ›  › This chapter covers
Hyperparameter tuning

Model ensembling

Training Keras models on multiple GPUs or on TPU

Mixed-precision training

Quantization

You’ve come quite far since the beginning of this book. You can now train image classification models, image segmentation models, models for classification or regression on vector data, timeseries forecasting models, text classification models, sequence-to-sequence models, and even generative models for text and images. You’ve got all the bases covered.

However, your models so far have all been trained at a small scale—on small datasets, with a single GPU—and they generally haven’t reached the best achievable performance on each dataset we’ve looked at. This book is, after all, an introductory book. If you are to go out into the real world and achieve state-of-the-art results on brand new problems, there’s still a bit of a chasm that you’ll need to cross.

### PDF p.126 (printed 102) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Inference: Using a model after training
However, this will process all inputs in `new_inputs` at once, which may not be feasible if you’re looking at a lot of data (in particular, it may require more memory than your GPU has).

A better way to do inference is to use the `predict()` method. It will iterate over the data in small batches and return a NumPy array of predictions. And unlike `__call__()`, it can also process TensorFlow `Dataset` objects:

### PDF p.573 (printed 549) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › Data parallelism: Replicating your model on each GPU
Data parallelism is the most common form of distributed training. It operates on a simple principle: divide and conquer. Each GPU receives a copy of the entire model, called a replica. Incoming batches of data are split into N sub-batches, which are processed by one model replica each, in parallel. This is why it’s called data parallelism: different samples (data points) are processed in parallel. For instance, with two GPUs, a batch of size 128 would be split into two sub-batches of size 64, which would be processed by two model replicas. Then

In inference—We would retrieve the predictions for each sub-batch and concatenate them to obtain the predictions for the full batch.

In training—We would retrieve the gradients for each sub-batch, average them, and update all model replicas based on the gradient average. The state of the model would then be the same as if you had trained it on the full batch of 128 samples. This is called synchronous training, since all replicas are kept in sync— their weights have the same value at all times. Nonsynchronous alternatives exist, but they are less efficient and aren’t used anymore in practice.

Data parallelism is a simple and highly scalabl
