# Generation bundle for page_578  (PDF page 578, printed page 554)

Page kind: content
Chapter: 18 Best practices for the real world
Section: 18.2 Scaling up model training with multiple devices
Subsection: 18.2.2 Distributed training in practice
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_578.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_578.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p578_e03] (body) This is the first time you encountered the concept of a variable path—it is simply a string identifier that looks like `"sequential/dense_1/kernel"`. It’s a useful way to refer to a variable without keeping a handle on the actual variable instance.
[p578_e04] (body) Here’s how you can print the paths for all variables in a model:
[p578_e05] (code)
```
for v in model.variables:
    print(v.path)
```
[p578_e06] (body) On the example model from listing 18.4, here’s what we get:
[p578_e07] (code)
```
sequential/dense/kernel
sequential/dense/bias
sequential/dense_1/kernel
sequential/dense_1/bias
```
[p578_e08] (body) Now let’s shard and replicate these variables. In the case of a simple model like this one, your go-to rule of thumb for variable sharding should be as follows:
[p578_e09] (bullet) Shard the last dimension of the variable along the `"model"` mesh axis.
[p578_e10] (bullet) Leave all other dimensions as replicated.
[p578_e11] (body) Simple enough, right? Like this:
[p578_e12] (code)
```
layout_map = keras.distribution.LayoutMap(device_mesh)
layout_map["sequential/dense/kernel"] = (None, "model")
layout_map["sequential/dense/bias"] = ("model",)
layout_map["sequential/dense_1/kernel"] = (None, "model")
layout_map["sequential/dense_1/bias"] = ("model",)
```
[p578_e13] (body) Finally, we tell Keras to refer to this sharding layout when instantiating the variables by setting the distribution configuration like this:
[p578_e14] (code)
```
model_parallel = keras.distribution.ModelParallel(
    layout_map=layout_map,
    batch_dim_name="data",
)
keras.distribution.set_distribution(model_parallel)
```
[p578_e15] (body) Once the distribution configuration is set, you can create your model and `fit()` it. No other part of your code changes—your model definition code is the same, and your training code is the same. That’s true whether you’re using built-in APIs like `fit()` and `evaluate()` or your own training logic. Assuming that you have the right `Layout` `Map` for your variables, the little code snippets you just saw are enough to distribute

## Code annotations printed next to listings (callouts, in order)
- This argument tells
- Keras to use the mesh
- axis named "data" for
- data parallelism.

## Equation candidates (formulas written inline in the text)
- [p578_e14] model_parallel = keras.distribution.ModelParallel(
    layout_map=layout_map,
    batch_dim_name="data",
)
keras.distribution.set_distribution(model_parallel)

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

## Previous page (PDF 577) text, for continuity
As you may have guessed from the `axis_names` argument, we intend to use the devices along axis 0 for data parallelism and the devices along axis 1 for model parallelism. Since there are two devices along axis 0 and four along axis 1, we’ll split our model’s computation across four GPUs, and we’ll make two copies of our split model, running each copy on a different sub-batch of data in parallel.

Now that we have our mesh, we need to tell Keras how to split different pieces of computation across our devices. For that, we’ll use the `LayoutMap` API.

The LayoutMap API

To specify where different bits of computation should take place, we use variables as our frame of reference. We will split or replicate variables across our devices, and we will let the compiler move all computation associated with that part of the variable to the corresponding device.

Consider a variable. Its shape is, let’s say, `(32, 64)`. There are two things you could do with this variable:

You could replicate it (copy it) across an axis of your mesh so each device along that axis sees the same value.

You could shard it (split it) across an axis of your mesh—for instance, you could shard it into four chunks of shape `(32,` `16)`—so that each device along that axis sees one different chunk.

Now, do note that our variable has two dimensions. Importantly, “sharding” or “replicating” are decisions that you can make independently for each dimension of the variable.

The API you will use to tell Keras about such decisions is the `LayoutMap` class. A `LayoutMap` is similar to a dictionary. It maps model variables (for instance, the kernel variable of the first dense layer in your model) to a bit of information about how that variable should be replicated or sharded over a device mesh. Specifically, it m …

## Next page (PDF 579) text, for continuity
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

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.574 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.217 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.227 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.218 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.219 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.228 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.216 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.222 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.217 (printed 193) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › 
Listing 7.4 Calling a model for the first time to build it
>>> model.build(input_shape=(None, 3))
>>> model.weights
[<Variable shape=(3, 64), dtype=float32, path=sequential/dense_2/kernel ...>,
 <Variable shape=(64,), dtype=float32, path=sequential/dense_2/bias ...>,
 <Variable shape=(64, 10), dtype=float32, path=sequential/dense_3/kernel ...>,
 <Variable shape=(10,), dtype=float32, path=sequential/dense_3/bias ...>>]

### PDF p.89 (printed 65) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › Tensor assignment and the Variable class
To train a model, we’ll need to update its state, which is a set of tensors. If tensors aren’t assignable, how do we do it, then? That’s where variables come in. `tf.Variable` is the class meant to manage modifiable state in TensorFlow.

To create a variable, you need to provide some initial value, such as a random tensor.

Listing 3.5 Creating a `tf.Variable`

### PDF p.246 (printed 222) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Using fit() with a custom training loop
In the previous sections, we were writing our own training logic entirely from scratch. Doing so provides you with the most flexibility, but you end up writing a lot of code, while simultaneously missing out on many convenient features of `fit()`, such as callbacks, performance optimizations, or built-in support for distributed training.

What if you need a custom training algorithm, but you still want to use the power of the built-in Keras training loop? There’s actually a middle ground between `fit()` and a training loop written from scratch: you can provide a custom training step function and let the framework do the rest. You can do this by overriding the `train_step()` method of the `Model` class. This is the function that is called by `fit()` for every batch of data. You will then be able to call `fit()` as usual—and it will be running your own learning algorithm under the hood. Here’s how it works:

Create a new class that subclasses `keras.Model`.

### PDF p.89 (printed 65) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › Tensor assignment and the Variable class
Listing 3.5 Creating a `tf.Variable`
>>> v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))
>>> print(v)

### PDF p.254 (printed 230) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Summary
Keras offers a spectrum of different workflows, based on the principle of progressive disclosure of complexity. They all smoothly interoperate.

You can build models via the `Sequential` class, via the Functional API, or by subclassing the `Model` class. Most of the time, you’ll be using the Functional API.

The simplest way to train and evaluate a model is via the default `fit()` and `evaluate()` methods.

Keras callbacks provide a simple way to monitor models during your call to `fit()` and automatically take action based on the state of the model.

You can also fully control what `fit()` does by overriding the `train_step()` method, using APIs from your backend of choice—JAX, TensorFlow, or PyTorch.

Beyond `fit()`, you can also write your own training loops entirely from scratch, in a backend-native way. This is useful for researchers implementing brand-new training algorithms.

### PDF p.230 (printed 206) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › Using built-in training and evaluation loops
The principle of progressive disclosure of complexity—access to a spectrum of workflows that go from dead easy to arbitrarily flexible, one step at a time—also applies to model training. Keras provides you with different workflows for training models—it can be as simple as calling `fit()` on your data or as advanced as writing a new training algorithm from scratch.

You are already familiar with the `compile()`, `fit()`, `evaluate()`, `predict()` workflow. As a reminder, it looks like the following listing.

Listing 7.17 The standard workflow: `compile()`, `fit()`, `evaluate()`, `predict()`

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

### PDF p.239 (printed 215) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › 
Whenever you find yourself in a situation where the built-in `fit()` is not enough, you will need to write your own custom training logic. You’ve already seen simple examples of low-level training loops in chapters 2 and 3. As a reminder, the contents of a typical training loop look like this: 1 Run the “forward pass” (compute the model’s output) to obtain a loss value for

the current batch of data. 2 Retrieve the gradients of the loss with regard to the model’s weights. 3 Update the model’s weights so as to lower the loss value on the current batch of data. These steps are repeated for as many batches as necessary. This is essentially what `fit()` does under the hood. In this section, you will learn to reimplement `fit()` from scratch, which will give you all the knowledge you need to write any training algorithm you may come up with. Let’s go over the details. Throughout the next few sections, you’ll work your way up to writing a fully featured custom training loop in TensorFlow, PyTorch, and JAX.
