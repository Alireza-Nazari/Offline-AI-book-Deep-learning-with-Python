# Generation bundle for page_096  (PDF page 96, printed page 72)

Page kind: content
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 3.3 Introduction to TensorFlow
Subsection: 3.3.2 An end-to-end example: A linear classifier in pure TensorFlow
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_096.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_096.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p096_e03] (body) Now, we move to the training step, which receives some training data and updates the weights `W` and `b` to minimize the loss on the data.
[p096_e04] (listing_caption) Listing 3.20 The training-step function
[p096_e05] (code)
```
learning_rate = 0.1
@tf.function(jit_compile=True)
def training_step(inputs, targets, W, b):
    with tf.GradientTape() as tape:
        predictions = model(inputs, W, b)
        loss = mean_squared_error(predictions, targets)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss
```
[p096_e06] (body) For simplicity, we’ll do batch training instead of mini-batch training: we’ll run each training step (gradient computation and weight update) on the entire data, rather than iterate over the data in small batches. On one hand, this means that each training step will take much longer to run, since we compute the forward pass and the gradients for 2,000 samples at once. On the other hand, each gradient update will be much more effective at reducing the loss on the training data, since it will encompass information from all training samples instead of, say, only 128 random samples. As a result, we will need many fewer steps of training, and we should use a larger learning rate than what we would typically use for mini-batch training (we’ll use `learning_rate = 0.1`, as previously defined).
[p096_e07] (listing_caption) Listing 3.21 The batch training loop
[p096_e08] (code)
```
for step in range(40):
    loss = training_step(inputs, targets, W, b)
    print(f"Loss at step {step}: {loss:.4f}")
```
[p096_e09] (body) After 40 steps, the training loss seems to have stabilized around 0.025. Let’s plot how our linear model classifies the training data points, as shown in figure 3.2. Because our targets are 0s and 1s a given input point will be classified as “0” if its prediction value is below 0.5, and as “1” if it is above 0.5:
[p096_e10] (code)
```
predictions = model(inputs, W, b)
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
```

## Code annotations printed next to listings (callouts, in order)
- Wraps the function in a tf.function
- decorator to speed it up
- Forward pass, inside of
- a gradient tape scope
- Retrieves the
- gradient of the
- Updates the weights
- loss with regard
- to weights

## Listings
- Listing 3.20: The training-step function (code elements p096_e05)
- Listing 3.21: The batch training loop (code elements p096_e08)

## Equation candidates (formulas written inline in the text)
- [p096_e06] As a result, we will need many fewer steps of training, and we should use a larger learning rate than what we would typically use for mini-batch training (we’ll use `learning_rate = 0.1`, as previously defined).
- [p096_e10] predictions = model(inputs, W, b)
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()

## Chapter outline
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras (PDF pages 84-127)
  - 3.1 A brief history of deep learning frameworks (p.85)
  - 3.2 How these frameworks relate to each other (p.87)
  - 3.3 Introduction to TensorFlow (p.87)
      - 3.3.1 First steps with TensorFlow (p.88)
      - 3.3.2 An end-to-end example: A linear classifier in pure TensorFlow (p.93)
      - 3.3.3 What makes the TensorFlow approach unique (p.98)
  - 3.4 Introduction to PyTorch (p.98)
      - 3.4.1 First steps with PyTorch (p.99)
      - 3.4.2 An end-to-end example: A linear classifier in pure PyTorch (p.102)
      - 3.4.3 What makes the PyTorch approach unique (p.105)
  - 3.5 Introduction to JAX (p.106)
      - 3.5.1 First steps with JAX (p.106)
      - 3.5.2 Tensors in JAX (p.107)
      - 3.5.3 Random number generation in JAX (p.107)
      - 3.5.4 An end-to-end example: A linear classifier in pure JAX (p.112)
      - 3.5.5 What makes the JAX approach unique (p.114)
  - 3.6 Introduction to Keras (p.114)
      - 3.6.1 First steps with Keras (p.115)
      - 3.6.2 Layers: The building blocks of deep learning (p.116)
      - 3.6.3 From layers to models (p.120)
      - 3.6.4 The “compile” step: Configuring the learning process (p.121)
      - 3.6.5 Picking a loss function (p.123)
      - 3.6.6 Understanding the fit method (p.124)
      - 3.6.7 Monitoring loss and metrics on validation data (p.125)
      - 3.6.8 Inference: Using a model after training (p.126)

## Previous page (PDF 95) text, for continuity
Now, let’s create a linear classifier that can learn to separate these two blobs. A linear classifier is an affine transformation (`prediction` `=` `matmul(input,` `W)` `+` `b`) trained to minimize the square of the difference between predictions and the targets.

As you’ll see, it’s actually a much simpler example than the end-to-end example of a toy two-layer neural network from the end of chapter 2. However, this time, you should be able to understand everything about the code, line by line. Let’s create our variables `W` and `b`, initialized with random values and with zeros, respectively.

Listing 3.17 Creating the linear classifier variables

Here’s our forward pass function.

Listing 3.18 The forward pass function

Because our linear classifier operates on 2D inputs, `W` is really just two scalar coefficients: `W` `=` `[[w1],` `[w2]]`. Meanwhile, `b` is a single scalar coefficient. As such, for given input point `[x,` `y]`, its prediction value is `prediction` `=` `[[w1],` `[w2]]` `•` `[x,` `y]` `+` `b` `=` `w1` `*` `x + w2` `*` `y` `+` `b`.

Here’s our loss function.

Listing 3.19 The mean squared error loss function

## Next page (PDF 97) text, for continuity
Figure 3.2 Our model’s predictions on the training inputs: pretty similar to the training targets

Recall that the prediction value for a given point `[x, y]` is simply `prediction == [[w1], ` `[w2]] • [x, y] + b == w1 * x + w2 * y + b`. Thus, class “0” is defined as `w1 * x + w2 * ` `y + b < 0.5` and class “1” is defined as `w1 * x + w2 * y + b > 0.5`. You’ll notice that what you’re looking at is really the equation of a line in the 2D plane: `w1 * x + w2 * y + b ` `= 0.5`. Class 1 is above the line; class 0 is below the line. You may be used to seeing line equations in the format `y = a * x + b`; in the same format, our line becomes `y = - w1 ` `/ w2 * x + (0.5 - b) / w2`.

Let’s plot this line, as shown in figure 3.3:

Figure 3.3 Our model, visualized as a line

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.103 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.240 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.113 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.241 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.104 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.114 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.124 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.250 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.68 (printed 44) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
Instead, you can use the four-step algorithm outlined at the beginning of this section: modify the parameters little by little based on the current loss value on a random batch of data. Because you’re dealing with a differentiable function, you can compute its gradient, which gives you an efficient way to implement step 4. If you update the weights in the opposite direction from the gradient, the loss will be a little less every time: 1 Draw a batch of training samples `x` and corresponding targets `y_true`. 2 Run the model on `x` to obtain predictions `y_pred` (this is called the forward pass). 3 Compute the loss of the model on the batch, a measure of the mismatch between

`y_pred` and `y_true`. 4 Compute the gradient of the loss with regard to the model’s parameters (this is

called the backward pass).

5 Move the parameters a little in the opposite direction from the gradient—for

example, `W -= learning_rate * gradient`—thus reducing the loss on the batch a bit. The learning rate (`learning_rate` here) would be a scalar factor modulating the “speed” of the gradient descent process. Easy enough! What we just described is called mini-batch stochastic gradient descent (mini-batch

### PDF p.124 (printed 100) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Understanding the fit method
After `compile()` comes `fit()`. The `fit` method implements the training loop itself. Its key arguments are

The data (inputs and targets) to train on. It will typically be passed either in the form of NumPy arrays or a TensorFlow `Dataset` object. You’ll learn more about the `Dataset` API in the next chapters.

The number of epochs to train for: how many times the training loop should iterate over the data passed.

The batch size to use within each epoch of mini-batch gradient descent: the number of training examples considered to compute the gradients for one weight update step.

Listing 3.34 Calling `fit` with NumPy data

### PDF p.79 (printed 55) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › Running one training step
The most difficult part of the process is the “training step”: updating the weights of the model after running it on one batch of data. We need to

Compute the predictions of the model for the images in the batch

Compute the loss value for these predictions given the actual labels

Compute the gradient of the loss with regard to the model’s weights

Move the weights by a small amount in the direction opposite to the gradient

Listing 2.9 A single step of training

### PDF p.239 (printed 215) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › 
Whenever you find yourself in a situation where the built-in `fit()` is not enough, you will need to write your own custom training logic. You’ve already seen simple examples of low-level training loops in chapters 2 and 3. As a reminder, the contents of a typical training loop look like this: 1 Run the “forward pass” (compute the model’s output) to obtain a loss value for

the current batch of data. 2 Retrieve the gradients of the loss with regard to the model’s weights. 3 Update the model’s weights so as to lower the loss value on the current batch of data. These steps are repeated for as many batches as necessary. This is essentially what `fit()` does under the hood. In this section, you will learn to reimplement `fit()` from scratch, which will give you all the knowledge you need to write any training algorithm you may come up with. Let’s go over the details. Throughout the next few sections, you’ll work your way up to writing a fully featured custom training loop in TensorFlow, PyTorch, and JAX.

### PDF p.76 (printed 52) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › 
Now you understand what happens when you call `fit`: the model will start to iterate on the training data in mini-batches of 128 samples, 5 times over (each iteration over all the training data is called an epoch). For each batch, the model will compute the gradient of the loss with regard to the weights (using the Backpropagation algorithm, which derives from the chain rule in calculus) and move the weights in the direction that will reduce the value of the loss for this batch.

After these 5 epochs, the model will have performed 2,345 gradient updates (469 per epoch), and the loss of the model will be sufficiently low that the model will be capable of classifying handwritten digits with high accuracy.

### PDF p.68 (printed 44) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
local minimum. If `learning_rate` is too large, your updates may end up taking you to completely random locations on the curve.

Note that a variant of the mini-batch SGD algorithm would be to draw a single sample and target at each iteration, rather than drawing a batch of data. This would be true SGD (as opposed to mini-batch SGD). Alternatively, going to the opposite extreme, you could run every step on all data available, which is called batch gradient descent. Each update would then be more accurate, but far more expensive. The efficient compromise between these two extremes is to use mini-batches of reasonable size. Although figure 2.18 illustrates gradient descent in a 1D parameter space, in practice, you’ll use gradient descent in highly dimensional spaces: every weight coefficient in a neural network is a free dimension in the space, and there may be tens of thousands or even millions of them. To help you build intuition about loss surfaces, you can also visualize gradient descent along a 2D loss surface, as shown in figure 2.19. But

### PDF p.64 (printed 40) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
`y_pred` and `y_true`. 4 Update all weights of the model in a way that slightly reduces the loss on this batch. You’ll eventually end up with a model that has a very low loss on its training data: a low mismatch between predictions `y_pred` and expected targets `y_true`. The model has “learned” to map its inputs to correct targets. From afar, it may look like magic, but when you reduce it to elementary steps, it turns out to be simple. Step 1 sounds easy enough—it’s just I/O code. Steps 2 and 3 are merely the application of a handful of tensor operations, so you could implement these steps purely from what you learned in the previous section. The difficult part is step 4: updating the model’s weights. Given an individual weight coefficient in the model, how can you compute whether the coefficient should be increased or decreased, and by how much? One naive solution would be to freeze all weights in the model except the one scalar coefficient being considered and try different values for this coefficient. Let’s say the initial value of the coefficient is 0.3. After the forward pass on a batch of data, the loss of the model on the batch is 0.5. If you change the coefficient’s value t

### PDF p.83 (printed 59) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › 
move is defined by the learning rate) in the opposite direction from the gradient.

This is called mini-batch gradient descent.

The entire learning process is made possible by the fact that all tensor operations in neural networks are differentiable, and thus it’s possible to apply the chain rule of derivation to find the gradient function mapping the current parameters and current batch of data to a gradient value. This is called backpropagation.

Two key concepts you’ll see frequently in future chapters are loss and optimizers. These are the two things you need to define before you begin feeding data into a model: – The loss is the quantity you’ll attempt to minimize during training, so it should represent a measure of success for the task you’re trying to solve. – The optimizer specifies the exact way in which the gradient of the loss will be used to update parameters: for instance, it could be the RMSProp optimizer, SGD with momentum, and so on.
