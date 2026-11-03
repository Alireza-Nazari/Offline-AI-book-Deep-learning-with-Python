# Generation bundle for page_083  (PDF page 83, printed page 59)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.5 Looking back at our first example
Subsection: 2.5.4 Evaluating the model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_083.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_083.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p083_e03] (body) move is defined by the learning rate) in the opposite direction from the gradient.
[p083_e04] (quote) This is called mini-batch gradient descent.
[p083_e05] (bullet) The entire learning process is made possible by the fact that all tensor operations in neural networks are differentiable, and thus it’s possible to apply the chain rule of derivation to find the gradient function mapping the current parameters and current batch of data to a gradient value. This is called backpropagation.
[p083_e06] (bullet) Two key concepts you’ll see frequently in future chapters are loss and optimizers. These are the two things you need to define before you begin feeding data into a model: – The loss is the quantity you’ll attempt to minimize during training, so it should represent a measure of success for the task you’re trying to solve. – The optimizer specifies the exact way in which the gradient of the loss will be used to update parameters: for instance, it could be the RMSProp optimizer, SGD with momentum, and so on.

## Chapter outline
Chapter: 2 The mathematical building blocks of neural networks (PDF pages 40-83)
  - 2.1 A first look at a neural network (p.41)
  - 2.2 Data representations for neural networks (p.45)
      - 2.2.1 Scalars (rank-0 tensors) (p.46)
      - 2.2.2 Vectors (rank-1 tensors) (p.46)
      - 2.2.3 Matrices (rank-2 tensors) (p.46)
      - 2.2.4 Rank-3 tensors and higher-rank tensors (p.47)
      - 2.2.5 Key attributes (p.47)
      - 2.2.6 Manipulating tensors in NumPy (p.49)
      - 2.2.7 The notion of data batches (p.49)
      - 2.2.8 Real-world examples of data tensors (p.50)
  - 2.3 The gears of neural networks: Tensor operations (p.52)
      - 2.3.1 Element-wise operations (p.53)
      - 2.3.2 Broadcasting (p.54)
      - 2.3.3 Tensor product (p.56)
      - 2.3.4 Tensor reshaping (p.58)
      - 2.3.5 Geometric interpretation of tensor operations (p.59)
      - 2.3.6 A geometric interpretation of deep learning (p.62)
  - 2.4 The engine of neural networks: Gradient-based optimization (p.63)
      - 2.4.1 What’s a derivative? (p.65)
      - 2.4.2 Derivative of a tensor operation: The gradient (p.66)
      - 2.4.3 Stochastic gradient descent (p.67)
      - 2.4.4 Chaining derivatives: The Backpropagation algorithm (p.70)
  - 2.5 Looking back at our first example (p.75)
      - 2.5.1 Reimplementing our first example from scratch (p.77)
      - 2.5.2 Running one training step (p.79)
      - 2.5.3 The full training loop (p.81)
      - 2.5.4 Evaluating the model (p.82)

## Previous page (PDF 82) text, for continuity
2.5.4

Evaluating the model

We can evaluate the model by taking the `argmax` of its predictions over the test images, and comparing it to the expected labels:

All done! As you can see, it’s quite a bit of work to do “by hand” what you can do in a few lines of Keras code. But because you’ve gone through these steps, you should now have a crystal-clear understanding of what goes on inside a neural network when you call `fit()`. Having this low-level mental model of what your code is doing behind the scenes will make you better able to take advantage of the high-level features of the Keras API.

2.6

Summary

Tensors form the foundation of modern machine learning systems. They come in various flavors of `dtype`, `rank`, and `shape`.

You can manipulate numerical tensors via tensor operations (such as addition, tensor product, or element-wise multiplication), which can be interpreted as encoding geometric transformations. In general, everything in deep learning is amenable to a geometric interpretation.

Deep learning models consist of chains of simple tensor operations, parameterized by weights, which are themselves tensors. The weights of a model are where its “knowledge” is stored.

Learning means finding a set of values for the model’s weights that minimizes a loss function for a given set of training data samples and their corresponding targets.

Learning happens by drawing random batches of data samples and their targets and computing the gradient of the model parameters with respect to the loss on the batch. The model parameters are then moved a bit (the magnitude of the

## Next page (PDF 84) text, for continuity
Introduction to

TensorFlow, PyTorch,

JAX, and Keras

This chapter covers

A closer look at all major deep learning

frameworks and their relationships

An overview of how core deep learning concepts

translate to code across all frameworks

This chapter is meant to give you everything you need to start doing deep learning in practice. First, you’ll get familiar with three popular deep learning frameworks that can be used with Keras:

TensorFlow (https://tensorflow.org)

PyTorch (https://pytorch.org/)

JAX (https://jax.readthedocs.io/)

Then, building on top of the first contact you’ve had with Keras in chapter 2, we’ll review the core components of neural networks and how they translate to Keras APIs.

By the end of this chapter, you’ll be ready to move on to practical, real-world applications—which will start with chapter 4.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.68 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.76 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.64 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.79 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.66 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.70 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.80 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.69 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.68 (printed 44) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
Instead, you can use the four-step algorithm outlined at the beginning of this section: modify the parameters little by little based on the current loss value on a random batch of data. Because you’re dealing with a differentiable function, you can compute its gradient, which gives you an efficient way to implement step 4. If you update the weights in the opposite direction from the gradient, the loss will be a little less every time: 1 Draw a batch of training samples `x` and corresponding targets `y_true`. 2 Run the model on `x` to obtain predictions `y_pred` (this is called the forward pass). 3 Compute the loss of the model on the batch, a measure of the mismatch between

`y_pred` and `y_true`. 4 Compute the gradient of the loss with regard to the model’s parameters (this is

called the backward pass).

5 Move the parameters a little in the opposite direction from the gradient—for

example, `W -= learning_rate * gradient`—thus reducing the loss on the batch a bit. The learning rate (`learning_rate` here) would be a scalar factor modulating the “speed” of the gradient descent process. Easy enough! What we just described is called mini-batch stochastic gradient descent (mini-batch

### PDF p.64 (printed 40) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
In particular, this applies to the function that maps the model’s coefficients to the loss of the model on a batch of data: a small change of the model’s coefficients results in a small, predictable change of the loss value. This enables you to use a mathematical operator called the gradient to describe how the loss varies as you move the model’s coefficients in different directions. If you compute this gradient, you can use it to move the coefficients (all at once in a single update, rather than one at a time) in a direction that decreases the loss. If you already know what differentiable means and what a gradient is, you can skip the next two sections. Otherwise, the following will help you understand these concepts.

### PDF p.121 (printed 97) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › The “compile” step: Configuring the learning process
Once the model architecture is defined, you still have to choose three more things:

Loss function (objective function)—The quantity that will be minimized during training. It represents a measure of success for the task at hand.

Optimizer—Determines how the network will be updated based on the loss function. It implements a specific variant of stochastic gradient descent (SGD).

### PDF p.76 (printed 52) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › 
Now you understand what happens when you call `fit`: the model will start to iterate on the training data in mini-batches of 128 samples, 5 times over (each iteration over all the training data is called an epoch). For each batch, the model will compute the gradient of the loss with regard to the weights (using the Backpropagation algorithm, which derives from the chain rule in calculus) and move the weights in the direction that will reduce the value of the loss for this batch.

After these 5 epochs, the model will have performed 2,345 gradient updates (469 per epoch), and the loss of the model will be sufficiently low that the model will be capable of classifying handwritten digits with high accuracy.

### PDF p.70 (printed 46) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › The chain rule
Backpropagation is a way to use the derivative of simple operations (such as addition, `relu`, or tensor product) to easily compute the gradient of arbitrarily complex combinations of these atomic operations. Crucially, a neural network consists of many tensor operations chained together, each of which has a simple, known derivative. For instance, the model from our first example can be expressed as a function parameterized by the variables `W1`, `b1`, `W2`, and `b2` (belonging to the first and second `Dense` layers, respectively), involving the atomic operations `matmul`, `relu`, `softmax`, and `+`, as well as our loss function, `loss`, which are all easily differentiable:

### PDF p.68 (printed 44) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
local minimum. If `learning_rate` is too large, your updates may end up taking you to completely random locations on the curve.

Note that a variant of the mini-batch SGD algorithm would be to draw a single sample and target at each iteration, rather than drawing a batch of data. This would be true SGD (as opposed to mini-batch SGD). Alternatively, going to the opposite extreme, you could run every step on all data available, which is called batch gradient descent. Each update would then be more accurate, but far more expensive. The efficient compromise between these two extremes is to use mini-batches of reasonable size. Although figure 2.18 illustrates gradient descent in a 1D parameter space, in practice, you’ll use gradient descent in highly dimensional spaces: every weight coefficient in a neural network is a free dimension in the space, and there may be tens of thousands or even millions of them. To help you build intuition about loss surfaces, you can also visualize gradient descent along a 2D loss surface, as shown in figure 2.19. But

### PDF p.76 (printed 52) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › 
Now you understand that `"sparse_categorical_crossentropy"` is the loss function that’s used as a feedback signal for learning the weight tensors, which the training phase will attempt to minimize. You also know that this reduction of the loss happens via mini-batch stochastic gradient descent. The exact rules governing a specific use of gradient descent are defined by the `"adam"` optimizer passed as the first argument.

Finally, this was the training loop:

### PDF p.69 (printed 45) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
you can’t possibly visualize what the actual process of training a neural network looks like—you can’t represent a 1,000,000-dimensional space in a way that makes sense to humans. As such, it’s good to keep in mind that the intuitions you develop through these low-dimensional representations may not always be accurate in practice. This has historically been a source of issues in the world of deep learning research.

Figure 2.19 Gradient descent down a 2D loss surface (two learnable parameters)

Additionally, there exist multiple variants of SGD that differ by taking into account previous weight updates when computing the next weight update, rather than just looking at the current value of the gradients. There is, for instance, SGD with momentum, as well as Adagrad, RMSprop, and several others. Such variants are known as optimization methods or optimizers. In particular, the concept of momentum, which is used in many of these variants, deserves your attention. Momentum addresses two issues with SGD: convergence speed and local minima. Consider figure 2.20, which shows the curve of a loss as a function of a model parameter.

As you can see, around a certain parameter value, there is 
