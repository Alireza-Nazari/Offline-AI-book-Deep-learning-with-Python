# Generation bundle for page_254  (PDF page 254, printed page 230)

Page kind: content
Chapter: 7 A deep dive on Keras
Section: 7.4 Writing your own training and evaluation loops
Subsection: 7.4.5 Handling metrics in a custom train_step()
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_254.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_254.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p254_e03] (code)
```
            new_metrics_vars += current_vars
        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metrics_vars,
        )
        return logs, state
```
[p254_e04] (body) That was a lot of information, but by now you know enough to use Keras to do almost anything!
[p254_e05] (heading_section) Summary
[p254_e06] (bullet) Keras offers a spectrum of different workflows, based on the principle of progressive disclosure of complexity. They all smoothly interoperate.
[p254_e07] (bullet) You can build models via the `Sequential` class, via the Functional API, or by subclassing the `Model` class. Most of the time, you’ll be using the Functional API.
[p254_e08] (bullet) The simplest way to train and evaluate a model is via the default `fit()` and `evaluate()` methods.
[p254_e09] (bullet) Keras callbacks provide a simple way to monitor models during your call to `fit()` and automatically take action based on the state of the model.
[p254_e10] (bullet) You can also fully control what `fit()` does by overriding the `train_step()` method, using APIs from your backend of choice—JAX, TensorFlow, or PyTorch.
[p254_e11] (bullet) Beyond `fit()`, you can also write your own training loops entirely from scratch, in a backend-native way. This is useful for researchers implementing brand-new training algorithms.

## Code annotations printed next to listings (callouts, in order)
- Returns the new metrics
- variables as part of the state

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

## Previous page (PDF 253) text, for continuity
Next up: metric management. As usual, it’s a tad more complicated due to JAX’s statelessness requirements:

## Next page (PDF 255) text, for continuity
Image classification

This chapter covers

Understanding convolutional neural networks

(ConvNets)

Using data augmentation to mitigate overfitting

Using a pretrained ConvNet for feature extraction

Fine-tuning a pretrained ConvNet

Computer vision was the first big success story of deep learning. It led to the initial rise of deep learning between 2011 and 2015. A type of deep learning called convolutional neural networks started getting remarkably good results on image classification competitions around that time, first with Dan Ciresan winning two niche competitions (the ICDAR 2011 Chinese character recognition competition and the IJCNN 2011 German traffic signs recognition competition) and then, more notably, in fall 2012, with Hinton’s group winning the high-profile ImageNet large-scale visual recognition challenge. Many more promising results quickly started bubbling up in other computer vision tasks.

Interestingly, these early successes weren’t quite enough to make deep learning mainstream at the time—it took a few years. The computer vision research

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.232 — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops
- PDF p.250 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.230 — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops
- PDF p.246 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.245 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.214 — 7 A deep dive on Keras › 
- PDF p.215 — 7 A deep dive on Keras › 7.1 A spectrum of workflows
- PDF p.244 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

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

### PDF p.246 (printed 222) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Using fit() with a custom training loop
In the previous sections, we were writing our own training logic entirely from scratch. Doing so provides you with the most flexibility, but you end up writing a lot of code, while simultaneously missing out on many convenient features of `fit()`, such as callbacks, performance optimizations, or built-in support for distributed training.

What if you need a custom training algorithm, but you still want to use the power of the built-in Keras training loop? There’s actually a middle ground between `fit()` and a training loop written from scratch: you can provide a custom training step function and let the framework do the rest. You can do this by overriding the `train_step()` method of the `Model` class. This is the function that is called by `fit()` for every batch of data. You will then be able to call `fit()` as usual—and it will be running your own learning algorithm under the hood. Here’s how it works:

Create a new class that subclasses `keras.Model`.

### PDF p.230 (printed 206) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › Using built-in training and evaluation loops
The principle of progressive disclosure of complexity—access to a spectrum of workflows that go from dead easy to arbitrarily flexible, one step at a time—also applies to model training. Keras provides you with different workflows for training models—it can be as simple as calling `fit()` on your data or as advanced as writing a new training algorithm from scratch.

You are already familiar with the `compile()`, `fit()`, `evaluate()`, `predict()` workflow. As a reminder, it looks like the following listing.

Listing 7.17 The standard workflow: `compile()`, `fit()`, `evaluate()`, `predict()`

### PDF p.239 (printed 215) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › 
Whenever you find yourself in a situation where the built-in `fit()` is not enough, you will need to write your own custom training logic. You’ve already seen simple examples of low-level training loops in chapters 2 and 3. As a reminder, the contents of a typical training loop look like this: 1 Run the “forward pass” (compute the model’s output) to obtain a loss value for

the current batch of data. 2 Retrieve the gradients of the loss with regard to the model’s weights. 3 Update the model’s weights so as to lower the loss value on the current batch of data. These steps are repeated for as many batches as necessary. This is essentially what `fit()` does under the hood. In this section, you will learn to reimplement `fit()` from scratch, which will give you all the knowledge you need to write any training algorithm you may come up with. Let’s go over the details. Throughout the next few sections, you’ll work your way up to writing a fully featured custom training loop in TensorFlow, PyTorch, and JAX.

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

### PDF p.232 (printed 208) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › Using callbacks
Launching a training run on a large dataset for tens of epochs using `model.fit()` can be a bit like launching a paper airplane: past the initial impulse, you don’t have any control over its trajectory or its landing spot. If you want to avoid bad outcomes (and thus wasted paper airplanes), it’s smarter to use, not a paper plane, but a drone that can sense its environment, send data back to its operator, and automatically make steering decisions based on its current state. The Keras callbacks API will help you transform your call to `model.fit()` from a paper airplane into a smart, autonomous drone that can self-introspect and dynamically take action.

A callback is an object (a class instance implementing specific methods) that is passed to the model in the call to `fit()` and that is called by the model at various points during

### PDF p.122 (printed 98) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
Metrics—The measures of success you want to monitor during training and validation, such as classification accuracy. Unlike the loss, training will not optimize directly for these metrics. As such, metrics don’t need to be differentiable.

Once you’ve picked your loss, optimizer, and metrics, you can use the built-in `compile()` and `fit()` methods to start training your model. Alternatively, you can write your own custom training loops—we cover how to do this in chapter 7. It’s a lot more work! For now, let’s take a look at `compile()` and `fit()`.

The `compile()` method configures the training process—you’ve already been introduced to it in your very first neural network example in chapter 2. It takes the arguments `optimizer`, `loss`, and `metrics` (a list):

### PDF p.215 (printed 191) — 7 A deep dive on Keras › 7.1 A spectrum of workflows › A spectrum of workflows
As such, there’s not a single “true” way of using Keras. Rather, Keras offers a spectrum of workflows, from the very simple to the very flexible. There are different ways to build Keras models, and different ways to train them, answering different needs. For instance, you have a range of ways to build models and an array of ways to train them, each representing a certain tradeoff between usability and flexibility. You could be using Keras like you would use scikit-learn—just calling `fit()` and letting the framework do its thing—or you could be using it like NumPy—taking full control of every little detail. Because all these workflows are based on shared APIs, such as `Layer` and `Model`, components from any workflow can be used in any other workflow: they can all talk to each other. This means that everything you’re learning now as you’re getting started will still be relevant once you’ve become an expert. You can get started easily and then gradually dive into workflows where you’re writing more and more logic from scratch. You won’t have to switch to an entirely different framework as you go from student to researcher, or from data scientist to deep learning engineer. This philo
