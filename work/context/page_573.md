# Generation bundle for page_573  (PDF page 573, printed page 549)

Page kind: content
Chapter: 18 Best practices for the real world
Section: 18.2 Scaling up model training with multiple devices
Subsection: 18.2.1 Multi-GPU training
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_573.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_573.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p573_e03] (body) replicated across multiple groups of devices (e.g., twice, for a total of 2 * 4 = 8 devices used).
[p573_e04] (body) Let’s see how that works in detail.
[p573_e05] (heading_subsubsection) Data parallelism: Replicating your model on each GPU
[p573_e06] (body) Data parallelism is the most common form of distributed training. It operates on a simple principle: divide and conquer. Each GPU receives a copy of the entire model, called a replica. Incoming batches of data are split into N sub-batches, which are processed by one model replica each, in parallel. This is why it’s called data parallelism: different samples (data points) are processed in parallel. For instance, with two GPUs, a batch of size 128 would be split into two sub-batches of size 64, which would be processed by two model replicas. Then
[p573_e07] (bullet) In inference—We would retrieve the predictions for each sub-batch and concatenate them to obtain the predictions for the full batch.
[p573_e08] (bullet) In training—We would retrieve the gradients for each sub-batch, average them, and update all model replicas based on the gradient average. The state of the model would then be the same as if you had trained it on the full batch of 128 samples. This is called synchronous training, since all replicas are kept in sync— their weights have the same value at all times. Nonsynchronous alternatives exist, but they are less efficient and aren’t used anymore in practice.
[p573_e09] (body) Data parallelism is a simple and highly scalable way to train your models faster. If you get more devices, just increase your batch size, and your training throughput increases accordingly. It has one limitation, though: it requires your model to be able to fit into one of your devices. However, it is now common to train foundation models that have tens of billions of parameters, which wouldn’t fit on any single GPU.
[p573_e10] (heading_subsubsection) Model parallelism: Splitting your model across multiple GPUs
[p573_e11] (body) That’s where model parallelism comes in. While data parallelism works by splitting your batches of data into sub-batches and processing the sub-batches in parallel, model parallelism works by splitting your model into submodels and running each one on a different device—in parallel. For instance, consider the following model.
[p573_e12] (listing_caption) Listing 18.4 A large densely connected model
[p573_e13] (code)
```
model = keras.Sequential(
    [
        keras.layers.Input(shape=(16000,)),
        keras.layers.Dense(64000, activation="relu"),
        keras.layers.Dense(8000, activation="sigmoid"),
    ]
)
```
[p573_e14] (body) Each sample has 16,000 features and gets classified into 8,000 potentially overlapping categories by two `Dense` layers. Those are large layers—the first one has about 1 billion

## Listings
- Listing 18.4: A large densely connected model (code elements p573_e13)

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

## Previous page (PDF 572) text, for continuity
was so different from every other model: it provided information that the other models didn’t have access to. That’s precisely the point of ensembling. It’s not so much about how good your best model is; it’s about the diversity of your set of candidate models.

18.2 Scaling up model training with multiple devices

Recall the “loop of progress” concept we introduced in chapter 7: the quality of your ideas is a function of how many refinement cycles they’ve been through (figure 18.1). And the speed at which you can iterate on an idea is a function of how fast you can set up an experiment, how fast you can run that experiment, and, finally, how well you can analyze the resulting data.

Figure 18.1 The loop of progress

As you develop your expertise in the Keras API, how fast you can code up your deep learning experiments will cease to be the bottleneck of this progress cycle. The next bottleneck will become the speed at which you can train your models. Fast training infrastructure means that you can get your results back in 10 or 15 minutes and, hence, that you can go through dozens of iterations every day. Faster training directly improves the quality of your deep learning solutions. In this section, you’ll learn about how to scale up your training runs by using multiple GPUs or TPUs.

18.2.1 Multi-GPU training

While GPUs are getting more powerful every year, deep learning models are also getting increasingly larger, requiring ever more computational resources. Training on a single GPU puts a hard bound on how fast you can move. The solution? You could simply add more GPUs and start doing multi-GPU distributed training.

There are two ways to distribute computation across multiple devices: data parallelism and model parallelism. With data parallelism, a single model get …

## Next page (PDF 574) text, for continuity
parameters, and the last one has about 512 million parameters. If you’re working with two small devices, you won’t be able to use data parallelism, since you can’t fit the model on a single device. What you can do is split a single instance of the model across multiple devices. This is often called sharding or partitioning a model. There are two main ways to split a model across devices: horizontal partitioning and vertical partitioning.

In horizontal partitioning, each device processes different layers of the model. For example, in the previous model, one GPU would handle the first `Dense` layer, and the other one would handle the second `Dense` layer. The main drawback of this approach is that it can introduce communication overhead. For example, the output of the first layer needs to be copied to the second device before it can be processed by the second layer. This can become a bottleneck, especially if the output of the first layer is large— you’d risk keeping your GPUs idle. In vertical partitioning, each layer is split across all available devices. Since layers are usually implemented in terms of `matmul` or `convolution` operations, which are highly parallelizable, this strategy is easy to implement in practice and is almost always the best fit for large models. For example, in the previous model, you could split the kernel and bias of the first `Dense` layer into two halves so that each device only receives a kernel of shape `(16000, 32000)` (split along its last axis) and a bias of shape `(32000,)`. You’d compute `matmul(inputs, kernel) + bias` with this half-kernel and half-bias for each device, and you’d merge the two outputs by concatenating them like this:

In reality, you will want to mix data parallelism and model parallelism. You will split your model  …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.577 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.576 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.587 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation
- PDF p.580 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation
- PDF p.575 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.76 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.625 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.570 — 18 Best practices for the real world › 18.1 Getting the most out of your models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.577 (printed 553) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › 
As you may have guessed from the `axis_names` argument, we intend to use the devices along axis 0 for data parallelism and the devices along axis 1 for model parallelism. Since there are two devices along axis 0 and four along axis 1, we’ll split our model’s computation across four GPUs, and we’ll make two copies of our split model, running each copy on a different sub-batch of data in parallel.

Now that we have our mesh, we need to tell Keras how to split different pieces of computation across our devices. For that, we’ll use the `LayoutMap` API.

### PDF p.587 (printed 563) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › Summary
You can use hyperparameter tuning and KerasTuner to automate the tedium out of finding the best model configuration. But be mindful of validation-set overfitting!

An ensemble of diverse models can often significantly improve the quality of your predictions.

To further scale your workflows, you can use data parallelism to train a model on multiple devices, as long as the model is small enough to fit on a single device.

For larger models, you can also use model parallelism to split your model’s variables and computation across several devices.

You can speed up model training on GPUs or TPUs by turning on mixed precision—you’ll generally get a nice speed boost at virtually no cost.

You can also speed up inference by using `float16` precision or even `int8` quantization.

### PDF p.576 (printed 552) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › The DeviceMesh API
The big idea is to separate devices into groups, organized along axes. Typically, one axis will be responsible for data parallelism, and one axis will be responsible for model parallelism (like in figure 18.2, your devices form a grid, where the horizontal axis handles data parallelism and the vertical axis handles model parallelism).

A device mesh doesn’t have to be 2D—it could be any shape you want. In practice, however, you will only ever see 1D and 2D meshes. Let’s make a 2 × 4 device mesh in Keras:

### PDF p.576 (printed 552) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › Using model parallelism with JAX
Keras also provides powerful tools for fully customizing how you want to do distributed training, including model parallel training and any mixture of data parallel and model parallel training you can imagine. Let’s dive in.

### PDF p.575 (printed 551) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › Getting your hands on two or more GPUs
First, you need to get access to several GPUs. As of now, Google Colab only lets you use a single GPU, so you will need to do one of two things:

Acquire two to eight GPUs, mount them on a single machine (it will require a beefy power supply), and install CUDA drivers, cuDNN,

Figure 18.2 Distributing a model across eight

etc. For most people, this isn’t the

devices: two model replicas, each handled by a group of four devices

best option.

Rent a multi-GPU virtual machine (VM) on Google Cloud, Azure, or AWS. You’ll be able to use VM images with pre-installed drivers and software, and you’ll have very little setup overhead. This is likely the best option for anyone who isn’t training models 24/7.

We won’t cover the details of how to spin up multi-GPU cloud VMs because such instructions would be relatively short-lived, and this information is readily available online.

### PDF p.575 (printed 551) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › Using data parallelism with JAX
keras.distribution.set_distribution(
    keras.distribution.DataParallel(["gpu:0", "gpu:1"])
)

### PDF p.49 (printed 25) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › The notion of data batches
In general, the first axis (axis 0, because indexing starts at 0) in all data tensors you’ll come across in deep learning will be the samples axis. In the MNIST example, “samples” are images of digits.

In addition, deep learning models don’t process an entire dataset at once; rather, they break the data into small “batches,” or groups of samples with a fixed size. Concretely, here’s one batch of our MNIST digits, with a batch size of 128:

### PDF p.580 (printed 556) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › Using step fusing to improve TPU utilization
Because a TPU has a lot of compute power available, you need to train with very large batches to keep the TPU cores busy. For small models, the batch size required can get extraordinarily large—upward of 10,000 samples per batch. When working with enormous batches, you should make sure to increase your optimizer learning rate accordingly: you’re going to be making fewer updates to your weights, but each update will be more accurate (since the gradients are computed using more data points); hence, you should move the weights by a greater magnitude with each update.

There is, however, a simple trick you can use to keep reasonably sized batches while maintaining full TPU utilization: step fusing. The idea is to run multiple steps of training during each TPU execution step. Basically, do more work in between two roundtrips from the virtual machine memory to the TPU. To do this, simply specify the `steps_per_` `execution` argument in `compile()`—for instance, `steps_per_execution=8` to run eight steps of training during each TPU execution. For small models that are underutilizing the TPU, this can result in a dramatic speedup:
