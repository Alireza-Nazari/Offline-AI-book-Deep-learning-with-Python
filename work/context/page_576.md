# Generation bundle for page_576  (PDF page 576, printed page 552)

Page kind: content
Chapter: 18 Best practices for the real world
Section: 18.2 Scaling up model training with multiple devices
Subsection: 18.2.2 Distributed training in practice
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_576.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_576.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p576_e03] (body) In an ideal world, training on N GPUs would result in a speedup of factor N. In practice, however, distribution introduces some overhead—in particular, merging the weight deltas originating from different devices takes some time. The effective speedup you get is a function of the number of GPUs used:
[p576_e04] (bullet) With two GPUs, the speedup stays close to 2×.
[p576_e05] (bullet) With four, the speedup is around 3.8×.
[p576_e06] (bullet) With eight, it’s around 7.3×.
[p576_e07] (body) This assumes that you’re using a large-enough global batch size to keep each GPU utilized at full capacity. If your batch size is too small, the local batch size won’t be enough to keep your GPUs busy.
[p576_e08] (heading_subsubsection) Using model parallelism with JAX
[p576_e09] (body) Keras also provides powerful tools for fully customizing how you want to do distributed training, including model parallel training and any mixture of data parallel and model parallel training you can imagine. Let’s dive in.
[p576_e10] (heading_subsubsection) The DeviceMesh API
[p576_e11] (body) First, you need to understand the concept of a device mesh. A device mesh is simply a grid of devices. Consider this example, with eight GPUs:
[p576_e12] (code)
```
gpu:0   |   gpu:4
--------|---------
gpu:1   |   gpu:5
--------|---------
gpu:2   |   gpu:6
--------|---------
gpu:3   |   gpu:7
```
[p576_e13] (body) The big idea is to separate devices into groups, organized along axes. Typically, one axis will be responsible for data parallelism, and one axis will be responsible for model parallelism (like in figure 18.2, your devices form a grid, where the horizontal axis handles data parallelism and the vertical axis handles model parallelism).
[p576_e14] (body) A device mesh doesn’t have to be 2D—it could be any shape you want. In practice, however, you will only ever see 1D and 2D meshes. Let’s make a 2 × 4 device mesh in Keras:
[p576_e15] (code)
```
device_mesh = keras.distribution.DeviceMesh(
    shape=(2, 4),
    axis_names=["data", "model"],
)
```
[p576_e16] (body) Mind you, you can also explicitly specify the devices you want to use:

## Code annotations printed next to listings (callouts, in order)
- We assume eight devices,
- organized as a 2 × 4 grid.
- It’s convenient to give your
- axes meaningful names!

## Equation candidates (formulas written inline in the text)
- [p576_e15] device_mesh = keras.distribution.DeviceMesh(
    shape=(2, 4),
    axis_names=["data", "model"],
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

## Previous page (PDF 575) text, for continuity
using JAX, you’re making a mistake— and wasting your dollars burning way more compute than you actually need.

Getting your hands on two or more GPUs

First, you need to get access to several GPUs. As of now, Google Colab only lets you use a single GPU, so you will need to do one of two things:

Acquire two to eight GPUs, mount them on a single machine (it will require a beefy power supply), and install CUDA drivers, cuDNN,

Figure 18.2 Distributing a model across eight

etc. For most people, this isn’t the

devices: two model replicas, each handled by a group of four devices

best option.

Rent a multi-GPU virtual machine (VM) on Google Cloud, Azure, or AWS. You’ll be able to use VM images with pre-installed drivers and software, and you’ll have very little setup overhead. This is likely the best option for anyone who isn’t training models 24/7.

We won’t cover the details of how to spin up multi-GPU cloud VMs because such instructions would be relatively short-lived, and this information is readily available online.

Using data parallelism with JAX

Using data parallelism with Keras and JAX is very simple: before building your model, just add the following line of code:

That’s it.

If you want more granular control, you can specify which devices you want to use. You can list available devices via

It will return a list of strings—the names of your devices, such as `"gpu:0"`, `"gpu:1"`, and so on. You can then pass these to the `DataParallel` constructor:

## Next page (PDF 577) text, for continuity
As you may have guessed from the `axis_names` argument, we intend to use the devices along axis 0 for data parallelism and the devices along axis 1 for model parallelism. Since there are two devices along axis 0 and four along axis 1, we’ll split our model’s computation across four GPUs, and we’ll make two copies of our split model, running each copy on a different sub-batch of data in parallel.

Now that we have our mesh, we need to tell Keras how to split different pieces of computation across our devices. For that, we’ll use the `LayoutMap` API.

The LayoutMap API

To specify where different bits of computation should take place, we use variables as our frame of reference. We will split or replicate variables across our devices, and we will let the compiler move all computation associated with that part of the variable to the corresponding device.

Consider a variable. Its shape is, let’s say, `(32, 64)`. There are two things you could do with this variable:

You could replicate it (copy it) across an axis of your mesh so each device along that axis sees the same value.

You could shard it (split it) across an axis of your mesh—for instance, you could shard it into four chunks of shape `(32,` `16)`—so that each device along that axis sees one different chunk.

Now, do note that our variable has two dimensions. Importantly, “sharding” or “replicating” are decisions that you can make independently for each dimension of the variable.

The API you will use to tell Keras about such decisions is the `LayoutMap` class. A `LayoutMap` is similar to a dictionary. It maps model variables (for instance, the kernel variable of the first dense layer in your model) to a bit of information about how that variable should be replicated or sharded over a device mesh. Specifically, it m …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.574 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.573 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.572 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.580 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation
- PDF p.579 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.211 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.578 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.584 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.572 (printed 548) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › 18.2.1 Multi-GPU training
While GPUs are getting more powerful every year, deep learning models are also getting increasingly larger, requiring ever more computational resources. Training on a single GPU puts a hard bound on how fast you can move. The solution? You could simply add more GPUs and start doing multi-GPU distributed training.

There are two ways to distribute computation across multiple devices: data parallelism and model parallelism. With data parallelism, a single model gets replicated on multiple devices or multiple machines. Each of the model replicas processes different batches of data, and then they merge their results. With model parallelism, different parts of a single model run on different devices, processing a single batch of data together at the same time. This works best with models that have a naturally parallel architecture, such as models that feature multiple branches. In practice, model parallelism is only used in the case of models that are too large to fit on any single device: it isn’t used as a way to speed up training of regular models but as a way to train larger models. Then, of course, you can also mix both data parallelism and model parallelism: a single model can be 

### PDF p.573 (printed 549) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › Data parallelism: Replicating your model on each GPU
Data parallelism is the most common form of distributed training. It operates on a simple principle: divide and conquer. Each GPU receives a copy of the entire model, called a replica. Incoming batches of data are split into N sub-batches, which are processed by one model replica each, in parallel. This is why it’s called data parallelism: different samples (data points) are processed in parallel. For instance, with two GPUs, a batch of size 128 would be split into two sub-batches of size 64, which would be processed by two model replicas. Then

In inference—We would retrieve the predictions for each sub-batch and concatenate them to obtain the predictions for the full batch.

In training—We would retrieve the gradients for each sub-batch, average them, and update all model replicas based on the gradient average. The state of the model would then be the same as if you had trained it on the full batch of 128 samples. This is called synchronous training, since all replicas are kept in sync— their weights have the same value at all times. Nonsynchronous alternatives exist, but they are less efficient and aren’t used anymore in practice.

Data parallelism is a simple and highly scalabl

### PDF p.574 (printed 550) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › 
parameters, and the last one has about 512 million parameters. If you’re working with two small devices, you won’t be able to use data parallelism, since you can’t fit the model on a single device. What you can do is split a single instance of the model across multiple devices. This is often called sharding or partitioning a model. There are two main ways to split a model across devices: horizontal partitioning and vertical partitioning.

In horizontal partitioning, each device processes different layers of the model. For example, in the previous model, one GPU would handle the first `Dense` layer, and the other one would handle the second `Dense` layer. The main drawback of this approach is that it can introduce communication overhead. For example, the output of the first layer needs to be copied to the second device before it can be processed by the second layer. This can become a bottleneck, especially if the output of the first layer is large— you’d risk keeping your GPUs idle. In vertical partitioning, each layer is split across all available devices. Since layers are usually implemented in terms of `matmul` or `convolution` operations, which are highly parallelizable, this st

### PDF p.587 (printed 563) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › Summary
You can use hyperparameter tuning and KerasTuner to automate the tedium out of finding the best model configuration. But be mindful of validation-set overfitting!

An ensemble of diverse models can often significantly improve the quality of your predictions.

To further scale your workflows, you can use data parallelism to train a model on multiple devices, as long as the model is small enough to fit on a single device.

For larger models, you can also use model parallelism to split your model’s variables and computation across several devices.

You can speed up model training on GPUs or TPUs by turning on mixed precision—you’ll generally get a nice speed boost at virtually no cost.

You can also speed up inference by using `float16` precision or even `int8` quantization.

### PDF p.572 (printed 548) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › 18.2 Scaling up model training with multiple devices
Recall the “loop of progress” concept we introduced in chapter 7: the quality of your ideas is a function of how many refinement cycles they’ve been through (figure 18.1). And the speed at which you can iterate on an idea is a function of how fast you can set up an experiment, how fast you can run that experiment, and, finally, how well you can analyze the resulting data.

Figure 18.1 The loop of progress

As you develop your expertise in the Keras API, how fast you can code up your deep learning experiments will cease to be the bottleneck of this progress cycle. The next bottleneck will become the speed at which you can train your models. Fast training infrastructure means that you can get your results back in 10 or 15 minutes and, hence, that you can go through dozens of iterations every day. Faster training directly improves the quality of your deep learning solutions. In this section, you’ll learn about how to scale up your training runs by using multiple GPUs or TPUs.

### PDF p.574 (printed 550) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › 
In reality, you will want to mix data parallelism and model parallelism. You will split your model across, say, four devices, and you will replicate that split model across multiple groups of two devices—let’s say two—each processing one sub-batch of data in parallel. You will then have two replicas, each running on four devices, for a total of eight devices used (figure 18.2).

### PDF p.573 (printed 549) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › Model parallelism: Splitting your model across multiple GPUs
That’s where model parallelism comes in. While data parallelism works by splitting your batches of data into sub-batches and processing the sub-batches in parallel, model parallelism works by splitting your model into submodels and running each one on a different device—in parallel. For instance, consider the following model.

Listing 18.4 A large densely connected model

### PDF p.582 (printed 558) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 18.3.2 Float16 inference
Every model you’ve seen in this book so far has used single-precision numbers: it stored its state as `float32` weight variables and ran its computations on `float32` inputs. That’s enough precision to run the forward and backwards pass of a model without losing any information—in particular when it comes to small gradient updates (recall that the typically learning rate is 1e-3, and it’s pretty common to see weight updates on the order of 1e-6).

Modern GPUs and TPUs feature specialized hardware that can run 16-bit operations much faster and using less memory than equivalent 32-bit operations. By using these lower-precision operations whenever possible, you can speed up training on those devices by a significant factor. You can set the default floating point precision to `float16` in Keras via
