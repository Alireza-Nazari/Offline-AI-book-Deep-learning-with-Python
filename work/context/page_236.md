# Generation bundle for page_236  (PDF page 236, printed page 212)

Page kind: content
Chapter: 7 A deep dive on Keras
Section: 7.3 Using built-in training and evaluation loops
Subsection: 7.3.4 Monitoring and visualization with TensorBoard
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_236.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_236.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p236_e03] (code)
```
    train_images,
    train_labels,
    epochs=10,
    callbacks=[LossHistory()],
    validation_data=(val_images, val_labels),
)
```
[p236_e04] (body) We get plots that look like figure 7.5.
[p236_e05] (figure_caption) Figure 7.5 The output of our custom history– plotting callback
[p236_e06] (heading_subsection) 7.3.4
[p236_e07] (heading_subsection) Monitoring and visualization with TensorBoard
[p236_e08] (body) To do good research or develop good models, you need rich, frequent feedback about what’s going on inside your models during your experiments. That’s the point of running experiments: to get information about how well a model performs—as much information as possible. Making progress is an iterative process, a loop: you start with an idea and express it as an experiment, attempting to validate or invalidate your idea. You run this experiment and process the information it generates, as shown in figure 7.6. This inspires your next idea. The more iterations of this loop you’re able to run, the more refined and powerful your ideas become. Keras helps you go from idea to experiment in the least possible time, and fast GPUs can help you get from experiment to result as quickly as possible. But what about processing the experiment results? That’s where TensorBoard comes in.
[p236_e09] (figure_caption) Figure 7.6 The loop of progress

## Figures
- Figure 7.5: The output of our custom history– plotting callback  | labels inside the figure: (none - image)
- Figure 7.6: The loop of progress  | labels inside the figure: Idea, Visualization, framework:, TensorBoard, Deep learning, framework:, Keras, GPUs, TPUs, Results, Experiment

## Equation candidates (formulas written inline in the text)
- [p236_e03]     train_images,
    train_labels,
    epochs=10,
    callbacks=[LossHistory()],
    validation_data=(val_images, val_labels),
)

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

## Previous page (PDF 235) text, for continuity
These methods are all called with a `logs` argument, which is a dictionary containing information about the previous batch, epoch, or training run: training and validation metrics, and so on. The `on_epoch_*` and `on_batch_*` methods also take the epoch or batch index as first argument (an integer).

Here’s a simple example callback that saves a list of per-batch loss values during training and plots these values at the end of each epoch.

Listing 7.20 Creating a custom callback by subclassing the `Callback` class

Let’s test-drive it:

## Next page (PDF 237) text, for continuity
TensorBoard is a browser-based application that you can run locally. It’s the best way to monitor everything that goes on inside your model during training. With TensorBoard, you can

Visually monitor metrics during training

Visualize your model architecture

Visualize histograms of activations and gradients

Explore embeddings in 3D

If you’re monitoring more information than just the model’s final loss, you can develop a clearer vision of what the model does and doesn’t do, and you can make progress more quickly.

The easiest way to use TensorBoard with a Keras model and the `fit()` method is the `keras.callbacks.TensorBoard` callback. In the simplest case, just specify where you want the callback to write logs, and you’re good to go:

Once the model starts running, it will write logs at the target location. If you are running your Python script on a local machine, you can then launch the local TensorBoard server using the following command (note that the `tensorboard` executable should already be available if you have installed TensorFlow via `pip`; if not, you can install TensorBoard manually via `pip install tensorboard`):

You can then navigate to the URL that the command returns to access the TensorBoard interface.

If you are running your script in a Colab notebook, you can run an embedded TensorBoard instance as part of your notebook, using the following commands:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.234 — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops
- PDF p.232 — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops
- PDF p.238 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.233 — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops
- PDF p.554 — 17 Image generation › 17.2 Diffusion models
- PDF p.215 — 7 A deep dive on Keras › 7.1 A spectrum of workflows
- PDF p.572 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.246 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.572 (printed 548) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › 18.2 Scaling up model training with multiple devices
Recall the “loop of progress” concept we introduced in chapter 7: the quality of your ideas is a function of how many refinement cycles they’ve been through (figure 18.1). And the speed at which you can iterate on an idea is a function of how fast you can set up an experiment, how fast you can run that experiment, and, finally, how well you can analyze the resulting data.

Figure 18.1 The loop of progress

As you develop your expertise in the Keras API, how fast you can code up your deep learning experiments will cease to be the bottleneck of this progress cycle. The next bottleneck will become the speed at which you can train your models. Fast training infrastructure means that you can get your results back in 10 or 15 minutes and, hence, that you can go through dozens of iterations every day. Faster training directly improves the quality of your deep learning solutions. In this section, you’ll learn about how to scale up your training runs by using multiple GPUs or TPUs.

### PDF p.238 (printed 214) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › 
In the TensorBoard interface, you will be able to monitor live graphs of your training and evaluation metrics, as shown in figure 7.7.

Figure 7.7 TensorBoard can be used for easy monitoring of training and evaluation metrics.

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

### PDF p.11 (printed xi) — contents ›  › 
5.4

Improving generalization 158

Dataset curation 159 ■ Feature engineering 159 ■ Using early stopping 161 ■ Regularizing your model 161

6

The universal workflow of machine learning 171 6.1

Defining the task 172

Framing the problem 172 ■ Collecting a dataset 174 Understanding your data 178 ■ Choosing a measure of success 178

6.2

Developing a model 179

Preparing the data 179 ■ Choosing an evaluation protocol 180 Beating a baseline 181 ■ Scaling up: Developing a model that overfits 182 ■ Regularizing and tuning your model 183

6.3

Deploying your model 183

Explaining your work to stakeholders and setting expectations 184 Shipping an inference model 184 ■ Monitoring your model in the wild 188 ■ Maintaining your model 188

7

A deep dive on Keras 190 7.1

A spectrum of workflows 191 7.2 Different ways to build Keras models 192

The Sequential model 192 ■ The Functional API 195 Subclassing the Model class 202 ■ Mixing and matching different components 204 ■ Remember: Use the right tool for the job 205

7.3

Using built-in training and evaluation loops 206

Writing your own metrics 207 ■ Using callbacks 208 Writing your own callbacks 210 ■ Monitoring and visualization with Tens

### PDF p.246 (printed 222) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Using fit() with a custom training loop
In the previous sections, we were writing our own training logic entirely from scratch. Doing so provides you with the most flexibility, but you end up writing a lot of code, while simultaneously missing out on many convenient features of `fit()`, such as callbacks, performance optimizations, or built-in support for distributed training.

What if you need a custom training algorithm, but you still want to use the power of the built-in Keras training loop? There’s actually a middle ground between `fit()` and a training loop written from scratch: you can provide a custom training step function and let the framework do the rest. You can do this by overriding the `train_step()` method of the `Model` class. This is the function that is called by `fit()` for every batch of data. You will then be able to call `fit()` as usual—and it will be running your own learning algorithm under the hood. Here’s how it works:

Create a new class that subclasses `keras.Model`.

### PDF p.239 (printed 215) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › 
Whenever you find yourself in a situation where the built-in `fit()` is not enough, you will need to write your own custom training logic. You’ve already seen simple examples of low-level training loops in chapters 2 and 3. As a reminder, the contents of a typical training loop look like this: 1 Run the “forward pass” (compute the model’s output) to obtain a loss value for

the current batch of data. 2 Retrieve the gradients of the loss with regard to the model’s weights. 3 Update the model’s weights so as to lower the loss value on the current batch of data. These steps are repeated for as many batches as necessary. This is essentially what `fit()` does under the hood. In this section, you will learn to reimplement `fit()` from scratch, which will give you all the knowledge you need to write any training algorithm you may come up with. Let’s go over the details. Throughout the next few sections, you’ll work your way up to writing a fully featured custom training loop in TensorFlow, PyTorch, and JAX.

### PDF p.41 (printed 17) — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network › Running the code in this book
This book is full of runnable Python code. Each chapter is paired with a Jupyter notebook that contains all of the code from the chapter. A Jupyter notebook is a live Python scratch pad of sorts, where you can interactively run code, graph data, view images, and a lot more. You will gain a lot more practical knowledge from this book if you run and experiment with the code as you read. By far the easiest way to set up a deep learning environment to run these notebooks is Google Colaboratory (or Colab for short), a hosted environment for Jupyter notebooks that has become the industry standard for ML practitioners. With Colab, you can run the code for this book interactively in the browser, connecting to cloud runtimes with configurable hardware. By default, the notebooks in this book will run on Colab’s free GPU runtime. If you would like, you can also run these notebooks locally on your own machine. A GPU is recommended, especially as you get to the larger and more compute-intensive models later in this book. Instructions for running locally and on Colab, along with the code, can be found at https://github.com/fchollet/deep-learning-with-python-notebooks.

### PDF p.233 (printed 209) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › The EarlyStopping and ModelCheckpoint callbacks
When you’re training a model, there are many things you can’t predict at the start. In particular, you can’t tell how many epochs will be needed to get to an optimal validation loss. Our examples so far have adopted the strategy of training for enough epochs that you begin overfitting, using the first run to figure out the optimal number of epochs, and then finally launching a new training run from scratch using this optimal number. Of course, this approach is wasteful. A much better way to handle this is to stop training when you measure that the validation loss is no longer improving. This can be achieved using the `EarlyStopping` callback.

The `EarlyStopping` callback interrupts training once a target metric being monitored has stopped improving for a fixed number of epochs. For instance, this callback allows you to interrupt training as soon as you start overfitting, thus avoiding having to retrain your model for a smaller number of epochs. This callback is typically used in combination with `ModelCheckpoint`, which lets you continually save the model during training (and, optionally, save only the current best model so far: the version of the model that achieved the best perf
