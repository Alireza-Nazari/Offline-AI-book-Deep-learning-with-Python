# Generation bundle for page_572  (PDF page 572, printed page 548)

Page kind: content
Chapter: 18 Best practices for the real world
Section: 18.2 Scaling up model training with multiple devices
Subsection: 18.2.1 Multi-GPU training
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_572.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_572.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p572_e03] (body) was so different from every other model: it provided information that the other models didn’t have access to. That’s precisely the point of ensembling. It’s not so much about how good your best model is; it’s about the diversity of your set of candidate models.
[p572_e04] (heading_section) 18.2 Scaling up model training with multiple devices
[p572_e05] (body) Recall the “loop of progress” concept we introduced in chapter 7: the quality of your ideas is a function of how many refinement cycles they’ve been through (figure 18.1). And the speed at which you can iterate on an idea is a function of how fast you can set up an experiment, how fast you can run that experiment, and, finally, how well you can analyze the resulting data.
[p572_e06] (figure_caption) Figure 18.1 The loop of progress
[p572_e07] (body) As you develop your expertise in the Keras API, how fast you can code up your deep learning experiments will cease to be the bottleneck of this progress cycle. The next bottleneck will become the speed at which you can train your models. Fast training infrastructure means that you can get your results back in 10 or 15 minutes and, hence, that you can go through dozens of iterations every day. Faster training directly improves the quality of your deep learning solutions. In this section, you’ll learn about how to scale up your training runs by using multiple GPUs or TPUs.
[p572_e08] (heading_subsection) 18.2.1 Multi-GPU training
[p572_e09] (body) While GPUs are getting more powerful every year, deep learning models are also getting increasingly larger, requiring ever more computational resources. Training on a single GPU puts a hard bound on how fast you can move. The solution? You could simply add more GPUs and start doing multi-GPU distributed training.
[p572_e10] (body) There are two ways to distribute computation across multiple devices: data parallelism and model parallelism. With data parallelism, a single model gets replicated on multiple devices or multiple machines. Each of the model replicas processes different batches of data, and then they merge their results. With model parallelism, different parts of a single model run on different devices, processing a single batch of data together at the same time. This works best with models that have a naturally parallel architecture, such as models that feature multiple branches. In practice, model parallelism is only used in the case of models that are too large to fit on any single device: it isn’t used as a way to speed up training of regular models but as a way to train larger models. Then, of course, you can also mix both data parallelism and model parallelism: a single model can be split across multiple devices (e.g., 4), and that split model can be

## Figures
- Figure 18.1: The loop of progress  | labels inside the figure: Idea, Visualization, framework:, TensorBoard, Deep learning, framework:, Keras, GPUs, TPUs, Results, Experiment

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

## Previous page (PDF 571) text, for continuity
However, this will work only if the classifiers are more or less equally good. If one of them is significantly worse than the others, the final predictions may not be as good as the best classifier of the group.

A smarter way to ensemble classifiers is to do a weighted average, where the weights are learned on the validation data—typically, the better classifiers are given a higher weight, and the worse classifiers are given a lower weight. To search for a good set of ensembling weights, you can use random search or a simple optimization algorithm, such as the Nelder-Mead algorithm:

There are many possible variants: you can do an average of an exponential of the predictions, for instance. In general, a simple weighted average with weights optimized on the validation data provides a very strong baseline.

The key to making ensembling work is the diversity of the set of classifiers. Diversity is strength. If all the blind men only touched the elephant’s trunk, they would agree that elephants are like snakes, and they would forever stay ignorant of the truth of the elephant. Diversity is what makes ensembling work. In machine learning terms, if all of your models are biased in the same way, then your ensemble will retain this same bias. If your models are biased in different ways, the biases will cancel each other out, and the ensemble will be more robust and more accurate. For this reason, you should ensemble models that are as good as possible while being as different as possible. This typically means using very different architectures or even different brands of machine learning approaches. One thing that is largely not worth doing is ensembling the same network trained several times independently, from different random initializations. If the only difference between  …

## Next page (PDF 573) text, for continuity
replicated across multiple groups of devices (e.g., twice, for a total of 2 * 4 = 8 devices used).

Let’s see how that works in detail.

Data parallelism: Replicating your model on each GPU

Data parallelism is the most common form of distributed training. It operates on a simple principle: divide and conquer. Each GPU receives a copy of the entire model, called a replica. Incoming batches of data are split into N sub-batches, which are processed by one model replica each, in parallel. This is why it’s called data parallelism: different samples (data points) are processed in parallel. For instance, with two GPUs, a batch of size 128 would be split into two sub-batches of size 64, which would be processed by two model replicas. Then

In inference—We would retrieve the predictions for each sub-batch and concatenate them to obtain the predictions for the full batch.

In training—We would retrieve the gradients for each sub-batch, average them, and update all model replicas based on the gradient average. The state of the model would then be the same as if you had trained it on the full batch of 128 samples. This is called synchronous training, since all replicas are kept in sync— their weights have the same value at all times. Nonsynchronous alternatives exist, but they are less efficient and aren’t used anymore in practice.

Data parallelism is a simple and highly scalable way to train your models faster. If you get more devices, just increase your batch size, and your training throughput increases accordingly. It has one limitation, though: it requires your model to be able to fit into one of your devices. However, it is now common to train foundation models that have tens of billions of parameters, which wouldn’t fit on any single GPU.

Model parallelism: Splitting your  …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.574 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.587 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation
- PDF p.576 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.570 — 18 Best practices for the real world › 18.1 Getting the most out of your models
- PDF p.562 — 18 Best practices for the real world › 
- PDF p.577 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.563 — 18 Best practices for the real world › 18.1 Getting the most out of your models
- PDF p.575 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.587 (printed 563) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › Summary
You can use hyperparameter tuning and KerasTuner to automate the tedium out of finding the best model configuration. But be mindful of validation-set overfitting!

An ensemble of diverse models can often significantly improve the quality of your predictions.

To further scale your workflows, you can use data parallelism to train a model on multiple devices, as long as the model is small enough to fit on a single device.

For larger models, you can also use model parallelism to split your model’s variables and computation across several devices.

You can speed up model training on GPUs or TPUs by turning on mixed precision—you’ll generally get a nice speed boost at virtually no cost.

You can also speed up inference by using `float16` precision or even `int8` quantization.

### PDF p.574 (printed 550) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › 
parameters, and the last one has about 512 million parameters. If you’re working with two small devices, you won’t be able to use data parallelism, since you can’t fit the model on a single device. What you can do is split a single instance of the model across multiple devices. This is often called sharding or partitioning a model. There are two main ways to split a model across devices: horizontal partitioning and vertical partitioning.

In horizontal partitioning, each device processes different layers of the model. For example, in the previous model, one GPU would handle the first `Dense` layer, and the other one would handle the second `Dense` layer. The main drawback of this approach is that it can introduce communication overhead. For example, the output of the first layer needs to be copied to the second device before it can be processed by the second layer. This can become a bottleneck, especially if the output of the first layer is large— you’d risk keeping your GPUs idle. In vertical partitioning, each layer is split across all available devices. Since layers are usually implemented in terms of `matmul` or `convolution` operations, which are highly parallelizable, this st

### PDF p.236 (printed 212) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › Monitoring and visualization with TensorBoard
To do good research or develop good models, you need rich, frequent feedback about what’s going on inside your models during your experiments. That’s the point of running experiments: to get information about how well a model performs—as much information as possible. Making progress is an iterative process, a loop: you start with an idea and express it as an experiment, attempting to validate or invalidate your idea. You run this experiment and process the information it generates, as shown in figure 7.6. This inspires your next idea. The more iterations of this loop you’re able to run, the more refined and powerful your ideas become. Keras helps you go from idea to experiment in the least possible time, and fast GPUs can help you get from experiment to result as quickly as possible. But what about processing the experiment results? That’s where TensorBoard comes in.

Figure 7.6 The loop of progress

### PDF p.562 (printed 538) — 18 Best practices for the real world ›  › This chapter covers
Hyperparameter tuning

Model ensembling

Training Keras models on multiple GPUs or on TPU

Mixed-precision training

Quantization

You’ve come quite far since the beginning of this book. You can now train image classification models, image segmentation models, models for classification or regression on vector data, timeseries forecasting models, text classification models, sequence-to-sequence models, and even generative models for text and images. You’ve got all the bases covered.

However, your models so far have all been trained at a small scale—on small datasets, with a single GPU—and they generally haven’t reached the best achievable performance on each dataset we’ve looked at. This book is, after all, an introductory book. If you are to go out into the real world and achieve state-of-the-art results on brand new problems, there’s still a bit of a chasm that you’ll need to cross.

### PDF p.575 (printed 551) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › Getting your hands on two or more GPUs
First, you need to get access to several GPUs. As of now, Google Colab only lets you use a single GPU, so you will need to do one of two things:

Acquire two to eight GPUs, mount them on a single machine (it will require a beefy power supply), and install CUDA drivers, cuDNN,

Figure 18.2 Distributing a model across eight

etc. For most people, this isn’t the

devices: two model replicas, each handled by a group of four devices

best option.

Rent a multi-GPU virtual machine (VM) on Google Cloud, Azure, or AWS. You’ll be able to use VM images with pre-installed drivers and software, and you’ll have very little setup overhead. This is likely the best option for anyone who isn’t training models 24/7.

We won’t cover the details of how to spin up multi-GPU cloud VMs because such instructions would be relatively short-lived, and this information is readily available online.

### PDF p.577 (printed 553) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › 
As you may have guessed from the `axis_names` argument, we intend to use the devices along axis 0 for data parallelism and the devices along axis 1 for model parallelism. Since there are two devices along axis 0 and four along axis 1, we’ll split our model’s computation across four GPUs, and we’ll make two copies of our split model, running each copy on a different sub-batch of data in parallel.

Now that we have our mesh, we need to tell Keras how to split different pieces of computation across our devices. For that, we’ll use the `LayoutMap` API.

### PDF p.574 (printed 550) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › 
In reality, you will want to mix data parallelism and model parallelism. You will split your model across, say, four devices, and you will replicate that split model across multiple groups of two devices—let’s say two—each processing one sub-batch of data in parallel. You will then have two replicas, each running on four devices, for a total of eight devices used (figure 18.2).

### PDF p.576 (printed 552) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › 
In an ideal world, training on N GPUs would result in a speedup of factor N. In practice, however, distribution introduces some overhead—in particular, merging the weight deltas originating from different devices takes some time. The effective speedup you get is a function of the number of GPUs used:

With two GPUs, the speedup stays close to 2×.

With four, the speedup is around 3.8×.

With eight, it’s around 7.3×.

This assumes that you’re using a large-enough global batch size to keep each GPU utilized at full capacity. If your batch size is too small, the local batch size won’t be enough to keep your GPUs busy.
