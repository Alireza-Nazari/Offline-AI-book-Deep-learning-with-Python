# Generation bundle for page_071  (PDF page 71, printed page 47)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.4 The engine of neural networks: Gradient-based optimization
Subsection: 2.4.4 Chaining derivatives: The Backpropagation algorithm
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_071.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_071.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p071_e03] (code)
```
def fg(x):
    x1 = g(x)
    y = f(x1)
    return y
```
[p071_e04] (body) Then the chain rule states that `grad(y, x) == grad(y, x1) * grad(x1, x)`. This enables you to compute the derivative of `fg` as long as you know the derivatives of `f` and `g`. The chain rule is named like this because when you add more intermediate functions, it starts looking like a chain:
[p071_e05] (code)
```
def fghj(x):
    x1 = j(x)
    x2 = h(x1)
    x3 = g(x2)
    y = f(x3)
    return y
grad(y, x) == grad(y, x3) * grad(x3, x2) * grad(x2, x1) * grad(x1, x)
```
[p071_e06] (body) Applying the chain rule to the computation of the gradient values of a neural network gives rise to an algorithm called backpropagation. Let’s see how that works, concretely.
[p071_e07] (heading_subsubsection) Automatic differentiation with computation graphs
[p071_e08] (body) A useful way to think about backpropagation is in terms of computation graphs. A computation graph is the data structure at the heart of the deep learning revolution. It’s a directed acyclic graph of operations—in our case, tensor operations. For instance, figure 2.21 is the graph representation of our first model.
[p071_e09] (body) Computation graphs have been an extremely successful abstraction in computer science because they enable us to treat computation as data: a computable expression is encoded as a machine-readable data structure that can be used as the input or output of another program. For instance, you could imagine a program that receives a computation graph and returns a new computation graph that implements a large-scale distributed version of the same computation—this would mean that you could distribute any computation without having to write the distribution logic yourself. Or imagine … a program that receives a computation graph and can automatically generate the derivative of the expression it represents. It’s much easier to do these things if your computation is expressed as an explicit graph data structure rather than, say, lines of ASCII characters in a `.py` file. To explain backpropagation clearly, let’s look at a really basic example of a computation graph. We’ll consider a simplified version of the graph in figure 2.21, where we only have one linear layer and where all variables are scalar, shown in figure 2.22. We’ll take two scalar variables `w`, `b`, a scalar input `x`, and apply some operations to them to combine into an output `y`. Finally, we’ll apply an absolute value error loss function: `loss_val = ` `abs(y_true - y)`. Since we want to update `w` and `b` in a way that would minimize `loss_val`, we are interested in computing `grad(loss_val, b)` and `grad(loss_val, w)`.

## Equation candidates (formulas written inline in the text)
- [p071_e09] Finally, we’ll apply an absolute value error loss function: `loss_val = ` `abs(y_true - y)`.

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

## Previous page (PDF 70) text, for continuity
momentum, the ball won’t get stuck in a ravine and will end up at the global minimum. Momentum is implemented by moving the ball at each step based not only on the current slope value (current acceleration) but also on the current velocity (resulting from past acceleration). In practice, this means updating the parameter `w` based not only on the current gradient value but also on the previous parameter update, such as in this naive implementation:

2.4.4

Chaining derivatives: The Backpropagation algorithm

In the previously discussed algorithm, we casually assumed that because a function is differentiable, we can easily compute its gradient. But is that true? How can we compute the gradient of complex expressions in practice? In our two-layer network example, how can we get the gradient of the loss with regard to the weights? That’s where the Backpropagation algorithm comes in.

The chain rule

Backpropagation is a way to use the derivative of simple operations (such as addition, `relu`, or tensor product) to easily compute the gradient of arbitrarily complex combinations of these atomic operations. Crucially, a neural network consists of many tensor operations chained together, each of which has a simple, known derivative. For instance, the model from our first example can be expressed as a function parameterized by the variables `W1`, `b1`, `W2`, and `b2` (belonging to the first and second `Dense` layers, respectively), involving the atomic operations `matmul`, `relu`, `softmax`, and `+`, as well as our loss function, `loss`, which are all easily differentiable:

Calculus tells us that such a chain of functions can be derived using the following identity, called the chain rule. Consider two functions `f` and `g`, as well as the composed function `fg` such that `y =  …

## Next page (PDF 72) text, for continuity
Figure 2.22 A basic example of a

computation graph

Figure 2.21 The computation graph representation of our two-layer model

Let’s set concrete values for the “input nodes” in the graph—that is, the input `x`, the target `y_true`,

`w` and `b` (figure 2.23). We propagate these values

to all nodes in the graph, from top to bottom, until we reach `loss_val`. This is the forward pass.

Now let’s “reverse” the graph: for each edge in the graph going from `A` to `B`, we will create

an opposite edge from `B` to `A`, and ask, “How much does `B` vary when `A` varies?” That is, what is

Figure 2.23 Running a forward pass

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.73 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.74 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.80 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.67 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.66 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.103 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.77 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.76 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.74 (printed 50) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
Figure 2.25 Path from `loss_val` to `w` in the backward graph

NOTE If there are multiple paths linking the two nodes of interest `a`, `b` in the backward graph, we would obtain `grad(b, a)` by summing the contributions of all the paths.

And with that, you just saw backpropagation in action! Backpropagation is simply the application of the chain rule to a computation graph. There’s nothing more to it. Backpropagation starts with the final loss value and works backward from the top layers to the bottom layers, computing the contribution that each parameter had in the loss value. That’s where the name “backpropagation” comes from: we “back propagate” the loss contributions of different nodes in a computation graph.

Nowadays, people implement neural networks in modern frameworks that are capable of automatic differentiation, such as JAX, TensorFlow, and PyTorch. Automatic differentiation is implemented with the kind of computation graph previously presented. Automatic differentiation makes it possible to retrieve the gradients of arbitrary compositions of differentiable tensor operations without doing any extra work besides writing down the forward pass. When I wrote my first neural

### PDF p.73 (printed 49) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
Figure 2.24 Running a backward pass

We have

`grad(loss_val, x2) = 1` because as `x2` varies by an amount epsilon, `loss_val` `=` `abs(4` `-` `x2)` varies by the same amount.

`grad(x2, x1) = 1` because as `x1` varies by an amount epsilon, `x2 = x1 + b = x1 + 1` varies by the same amount.

`grad(x2, b) = 1` because as `b` varies by an amount epsilon, `x2 = x1 + b = 6 + b` varies by the same amount.

`grad(x1, w) = 2` because as `w` varies by an amount epsilon, `x1 = x * w = 2 * w` varies by `2 * epsilon`.

What the chain rule says about this backward graph is that you can obtain the derivative of a node with respect to another node by multiplying the derivatives for each edge along the path linking the two nodes. For instance, `grad(loss_val, w) = grad(loss_val, ` `x2) * grad(x2, x1) * grad(x1, w)` (see figure 2.25).

By applying the chain rule to our graph, we obtain what we were looking for:

`grad(loss_val, w) = 1 * 1 * 2 = 2`

`grad(loss_val, b) = 1 * 1 = 1`

### PDF p.80 (printed 56) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › Gradient computation
Now, there’s just one thing we’re still missing: gradient computation (represented by the `get_gradients_of_loss_wrt_weights()` function in listing 2.9). In the previous section, we outlined how we could use the chain rule to obtain the gradients of a chain of functions given their individual derivatives, a process known as backpropagation. We could reimplement backpropagation from scratch here, but that would be rather cumbersome, especially since we’re using a `softmax` operation and a crossentropy loss, which have fairly verbose derivatives.

Instead, we can rely on the automatic differentiation mechanism that’s built into one of the low-level frameworks supported by Keras, such as TensorFlow, JAX, or PyTorch. For the sake of the example, let’s go with TensorFlow here. You’ll learn more about TensorFlow, JAX, and PyTorch in the next chapter. The API through which you can use TensorFlow’s automatic differentiation capabilities is the `tf.GradientTape` object. It’s a Python scope that will “record” the tensor operations that run inside it, in the form of a computation graph (sometimes called a tape). This graph can then be used to retrieve the gradient of any scalar value with res

### PDF p.83 (printed 59) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › 
move is defined by the learning rate) in the opposite direction from the gradient.

This is called mini-batch gradient descent.

The entire learning process is made possible by the fact that all tensor operations in neural networks are differentiable, and thus it’s possible to apply the chain rule of derivation to find the gradient function mapping the current parameters and current batch of data to a gradient value. This is called backpropagation.

Two key concepts you’ll see frequently in future chapters are loss and optimizers. These are the two things you need to define before you begin feeding data into a model: – The loss is the quantity you’ll attempt to minimize during training, so it should represent a measure of success for the task you’re trying to solve. – The optimizer specifies the exact way in which the gradient of the loss will be used to update parameters: for instance, it could be the RMSProp optimizer, SGD with momentum, and so on.

### PDF p.66 (printed 42) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › Derivative of a tensor operation: The gradient
The function we were just looking at turned a scalar value `x` into another scalar value `y`: you could plot it as a curve in a 2D plane. Now, imagine a function that turns a tuple of scalars `(x,` `y)` into a scalar value `z`: that would be a vector operation. You could plot it as a 2D surface in a 3D space (indexed by coordinates `x,` `y,` `z`). Likewise, you can imagine functions that take as input matrices, functions that take as input rank-3 tensors, etc.

The concept of derivation can be applied to any such function, as long as the surfaces they describe are continuous and smooth. The derivative of a tensor operation (or tensor function) is called a gradient. Gradients are just the generalization of the concept of derivatives to functions that take tensors as inputs. Remember how, for a scalar function, the derivative represents the local slope of the curve of the function? In just the same way, the gradient of a tensor function represents the curvature of the multidimensional surface described by the function. It characterizes how the output of the function varies when its input parameters vary. Let’s look at an example grounded in machine learning. Consider

An input vector

### PDF p.73 (printed 49) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
`grad(B, A)`? We’ll annotate each inverted edge with this value (figure 2.24). This backward graph represents the backward pass.

### PDF p.9 (printed ix) — contents ›  › 
2

The mathematical building blocks of neural networks 16 2.1

A first look at a neural network 17 2.2 Data representations for neural networks 21

Scalars (rank-0 tensors) 22 ■ Vectors (rank-1 tensors) 22 Matrices (rank-2 tensors) 22 ■ Rank-3 tensors and higher-rank tensors 23 ■ Key attributes 23 ■ Manipulating tensors in NumPy 25 ■ The notion of data batches 25 Real-world examples of data tensors 26

2.3

The gears of neural networks: Tensor operations 28

Element-wise operations 29 ■ Broadcasting 30 Tensor product 32 ■ Tensor reshaping 34 ■ Geometric interpretation of tensor operations 35 ■ A geometric interpretation of deep learning 38

2.4

The engine of neural networks: Gradient-based optimization 39

What’s a derivative? 41 ■ Derivative of a tensor operation: The gradient 42 ■ Stochastic gradient descent 43 ■ Chaining derivatives: The Backpropagation algorithm 46

2.5

Looking back at our first example 51

Reimplementing our first example from scratch 53 Running one training step 55 ■ The full training loop 57 Evaluating the model 58

3

Introduction to TensorFlow, PyTorch, JAX, and Keras 60 3.1

A brief history of deep learning frameworks 61 3.2 How these frameworks relate 

### PDF p.103 (printed 79) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › 
1 `loss.backward()` runs backpropagation starting from the `loss` output node and

populates the `tensor.grad` attribute on all tensors that were involved in the computation of `loss`. `tensor.grad` represents the gradient of the loss with regard to that tensor. 2 We use the `.grad` attribute to recover the gradients of the loss with regard to `W` and `b`. 3 We update `W` and `b` using those gradients. Because these updates are not intended to be part of the backward pass, we do them inside a `torch.no_grad()` scope, which skips gradient computation for everything inside it. 4 We reset the contents of the `.grad` property of our `W` and `b` parameters, by setting it to `None`. If we didn’t do this, gradient values would accumulate across multiple calls to `training_step()`, resulting in invalid values:
