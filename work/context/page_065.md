# Generation bundle for page_065  (PDF page 65, printed page 41)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.4 The engine of neural networks: Gradient-based optimization
Subsection: 2.4.1 What’s a derivative?
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_065.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_065.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p065_e03] (heading_subsection) 2.4.1
[p065_e04] (heading_subsection) What’s a derivative?
[p065_e05] (body) Consider a continuous, smooth function `f(x) = y`, mapping a number `x` to a new number `y`. We can use the function in figure 2.15 as an example.
[p065_e06] (body) Because the function is continuous, a small change in `x` can only result in a small change in `y`—that’s the intuition behind continuity. Let’s say you increase `x` by a small factor `epsilon_x`: this results in a small `epsilon_y` change to `y`, as shown in figure 2.16.
[p065_e07] (figure_caption) Figure 2.15 A continuous, smooth Figure 2.16 With a continuous function, function a small change in `x` results in a small change in `y`.
[p065_e08] (body) In addition, because the function is smooth (its curve doesn’t have any abrupt angles), when `epsilon_x` is small enough, around a certain point `p`, it’s possible to approximate `f` as a linear function of slope `a`, so that `epsilon_y` becomes `a` `*` `epsilon_x`:
[p065_e09] (code)
```
f(x + epsilon_x) = y + a * epsilon_x
```
[p065_e10] (body) Obviously, this linear approximation is valid only when `x` is close enough to `p`.
[p065_e11] (body) The slope `a` is called the derivative of `f` in `p`. If `a` is negative, it means a small increase in `x` around `p` will result in a decrease of `f(x)`, as shown in figure 2.17, and if `a` is positive, a small increase in `x` will result in an increase of `f(x)`. Further, the absolute value of `a` (the magnitude of the derivative) tells you how quickly this increase or decrease will happen.
[p065_e12] (figure_caption) Figure 2.17 Derivative of `f` in `p`
[p065_e13] (body) For every differentiable function `f(x)` (differentiable means “can be derived”: for example, smooth, continuous functions can be derived), there exists a derivative function `f'(x)` that maps values of `x` to the slope of the local linear approximation of `f` in those points. For instance, the derivative of `cos(x) is -sin(x)`, the derivative of `f(x) = a * x `is` f'(x) ` `= a`, and so on.

## Figures
- Figure 2.15: A continuous, smooth Figure 2.16 With a continuous function, function a small change in `x` results in a small change in `y`.  | labels inside the figure: y = f(x), x, y, y = f(x), epsilon_x, epsilon_y, x, y
- Figure 2.17: Derivative of `f` in `p`  | labels inside the figure: y = f(x), x, y, Local linear, approximation of , with, slope a, f

## Equation candidates (formulas written inline in the text)
- [p065_e09] f(x + epsilon_x) = y + a * epsilon_x
- [p065_e13] For instance, the derivative of `cos(x) is -sin(x)`, the derivative of `f(x) = a * x `is` f'(x) ` `= a`, and so on.

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

## Previous page (PDF 64) text, for continuity
1 Draw a batch of training samples `x` and corresponding targets `y_true`. 2 Run the model on `x` (a step called the forward pass) to obtain predictions `y_pred`. 3 Compute the loss of the model on the batch, a measure of the mismatch between

`y_pred` and `y_true`. 4 Update all weights of the model in a way that slightly reduces the loss on this batch. You’ll eventually end up with a model that has a very low loss on its training data: a low mismatch between predictions `y_pred` and expected targets `y_true`. The model has “learned” to map its inputs to correct targets. From afar, it may look like magic, but when you reduce it to elementary steps, it turns out to be simple. Step 1 sounds easy enough—it’s just I/O code. Steps 2 and 3 are merely the application of a handful of tensor operations, so you could implement these steps purely from what you learned in the previous section. The difficult part is step 4: updating the model’s weights. Given an individual weight coefficient in the model, how can you compute whether the coefficient should be increased or decreased, and by how much? One naive solution would be to freeze all weights in the model except the one scalar coefficient being considered and try different values for this coefficient. Let’s say the initial value of the coefficient is 0.3. After the forward pass on a batch of data, the loss of the model on the batch is 0.5. If you change the coefficient’s value to 0.35 and rerun the forward pass, the loss increases to 0.6. But if you lower the coefficient to 0.25, the loss falls to 0.4. In this case, it seems that updating the coefficient by –0.05 would contribute to minimizing the loss. This would have to be repeated for all coefficients in the model. But such an approach would be horribly inefficient because y …

## Next page (PDF 66) text, for continuity
Being able to derive functions is a very powerful tool when it comes to optimization, the task of finding values of `x` that minimize the value of `f(x)`. If you’re trying to update `x` by a factor `epsilon_x` to minimize `f(x)` and you know the derivative of `f`, then your job is done: the derivative completely describes how `f(x)` evolves as you change `x`. If you want to reduce the value of `f(x)`, you just need to move `x` a little in the opposite direction from the derivative.

2.4.2

Derivative of a tensor operation: The gradient

The function we were just looking at turned a scalar value `x` into another scalar value `y`: you could plot it as a curve in a 2D plane. Now, imagine a function that turns a tuple of scalars `(x,` `y)` into a scalar value `z`: that would be a vector operation. You could plot it as a 2D surface in a 3D space (indexed by coordinates `x,` `y,` `z`). Likewise, you can imagine functions that take as input matrices, functions that take as input rank-3 tensors, etc.

The concept of derivation can be applied to any such function, as long as the surfaces they describe are continuous and smooth. The derivative of a tensor operation (or tensor function) is called a gradient. Gradients are just the generalization of the concept of derivatives to functions that take tensors as inputs. Remember how, for a scalar function, the derivative represents the local slope of the curve of the function? In just the same way, the gradient of a tensor function represents the curvature of the multidimensional surface described by the function. It characterizes how the output of the function varies when its input parameters vary. Let’s look at an example grounded in machine learning. Consider

An input vector `x` (a sample in a dataset)

A matrix `W` (the weights o …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.67 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.70 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.71 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.80 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.69 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.153 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.170 — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning
- PDF p.134 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.67 (printed 43) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
Let’s say the current value of `W` is `W0`. Then the derivative of `f` in the point `W0` is a tensor `grad(loss_value, W0)`, with the same shape as `W`, where each coefficient `grad(loss_` `value, W0)[i, j]` indicates the direction and magnitude of the change in `loss_value` you observe when modifying `W0[i,` `j]`. That tensor `grad(loss_value, W0)` is the gradient of the function `f(W) = loss_value` in `W0`, also called “gradient of `loss_value` with respect to `W` around `W0`.”

NOTE The tensor operation `grad(f(W), W)` (which takes as input a matrix `W`) can be expressed as a combination of scalar functions `grad_ij(f(W), w_ij)`, each of which would return the derivative of `loss_value = f(W)` with respect to the coefficient `W[i, j]` of `W`, assuming all other coefficients are constant. `grad_ij` is called the partial derivative of `f` with respect to `W[i, j]`.

Concretely, what does `grad(loss_value, W0)` represent? You saw earlier that the derivative of a function `f(x)` of a single coefficient can be interpreted as the slope of the curve of `f`. Likewise, `grad(loss_value, W0)` can be interpreted as the tensor describing the curvature of `loss_value = f(W)` around `W0`. Eac

### PDF p.67 (printed 43) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › Stochastic gradient descent
Given a differentiable function, it’s theoretically possible to find its minimum analytically: it’s known that a function’s minimum is a point where the derivative is 0, so all you have to do is find all the points where the derivative goes to 0 and check for which of these points the function has the lowest value.

Applied to a neural network, that means finding analytically the combination of weight values that yields the smallest possible loss function. This can be done by solving the equation `grad(f(W), W) = 0` for `W`. This is a polynomial equation of `N` variables, where `N` is the number of coefficients in the model. Although it would be possible to solve such an equation for `N = 2` or `N = 3`, doing so is intractable for real neural networks, where the number of parameters is never less than a few thousand and can sometimes be in the billions.

### PDF p.69 (printed 45) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
you can’t possibly visualize what the actual process of training a neural network looks like—you can’t represent a 1,000,000-dimensional space in a way that makes sense to humans. As such, it’s good to keep in mind that the intuitions you develop through these low-dimensional representations may not always be accurate in practice. This has historically been a source of issues in the world of deep learning research.

Figure 2.19 Gradient descent down a 2D loss surface (two learnable parameters)

Additionally, there exist multiple variants of SGD that differ by taking into account previous weight updates when computing the next weight update, rather than just looking at the current value of the gradients. There is, for instance, SGD with momentum, as well as Adagrad, RMSprop, and several others. Such variants are known as optimization methods or optimizers. In particular, the concept of momentum, which is used in many of these variants, deserves your attention. Momentum addresses two issues with SGD: convergence speed and local minima. Consider figure 2.20, which shows the curve of a loss as a function of a model parameter.

As you can see, around a certain parameter value, there is 

### PDF p.80 (printed 56) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › 
“learning rate,” typically a small quantity. The simplest way to implement this `update_` `weights()` function is to subtract `gradient` `*` `learning_rate` from each weight:

### PDF p.70 (printed 46) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › The chain rule
Backpropagation is a way to use the derivative of simple operations (such as addition, `relu`, or tensor product) to easily compute the gradient of arbitrarily complex combinations of these atomic operations. Crucially, a neural network consists of many tensor operations chained together, each of which has a simple, known derivative. For instance, the model from our first example can be expressed as a function parameterized by the variables `W1`, `b1`, `W2`, and `b2` (belonging to the first and second `Dense` layers, respectively), involving the atomic operations `matmul`, `relu`, `softmax`, and `+`, as well as our loss function, `loss`, which are all easily differentiable:

### PDF p.153 (printed 129) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › 
The model ends with a single unit and no activation: it will be a linear layer. This is a typical setup for scalar regression—a regression where you’re trying to predict a single continuous value. Applying an activation function would constrain the range the output can take; for instance, if you applied a `sigmoid` activation function to the last layer, the model could only learn to predict values between 0 and 1. Here, because the last layer is purely linear, the model is free to learn to predict values in any range.

Note that you compile the model with the `mean_squared_error` loss function—mean squared error, the square of the difference between the predictions and the targets. This is a widely used loss function for regression problems. You’re also monitoring a new metric during training: mean absolute error (MAE). It’s the absolute value of the difference between the predictions and the targets. For instance, an MAE of 0.5 on this problem would mean your predictions are off by $50,000 on average (remember the target scaling of factor 100,000).

### PDF p.70 (printed 46) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › The chain rule
Calculus tells us that such a chain of functions can be derived using the following identity, called the chain rule. Consider two functions `f` and `g`, as well as the composed function `fg` such that `y = fg(x) == f(g(x))`:

### PDF p.71 (printed 47) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
Then the chain rule states that `grad(y, x) == grad(y, x1) * grad(x1, x)`. This enables you to compute the derivative of `fg` as long as you know the derivatives of `f` and `g`. The chain rule is named like this because when you add more intermediate functions, it starts looking like a chain:
