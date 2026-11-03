# Generation bundle for page_074  (PDF page 74, printed page 50)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.4 The engine of neural networks: Gradient-based optimization
Subsection: 2.4.4 Chaining derivatives: The Backpropagation algorithm
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_074.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_074.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p074_e03] (code)
```
2
x
```
[p074_e04] (code)
```
grad(x1, w) = 2
3
w
*
```
[p074_e05] (code)
```
x1
grad(x2, x1) = 1
1
grad(x2, b) = 1
b
+
```
[p074_e06] (code)
```
x2
grad(loss_val, x2) = 1
4
y_true
abs_diff
```
[p074_e07] (code)
```
loss_val
```
[p074_e08] (figure_caption) Figure 2.25 Path from `loss_val` to `w` in the backward graph
[p074_e09] (body) NOTE If there are multiple paths linking the two nodes of interest `a`, `b` in the backward graph, we would obtain `grad(b, a)` by summing the contributions of all the paths.
[p074_e10] (body) And with that, you just saw backpropagation in action! Backpropagation is simply the application of the chain rule to a computation graph. There’s nothing more to it. Backpropagation starts with the final loss value and works backward from the top layers to the bottom layers, computing the contribution that each parameter had in the loss value. That’s where the name “backpropagation” comes from: we “back propagate” the loss contributions of different nodes in a computation graph.
[p074_e11] (body) Nowadays, people implement neural networks in modern frameworks that are capable of automatic differentiation, such as JAX, TensorFlow, and PyTorch. Automatic differentiation is implemented with the kind of computation graph previously presented. Automatic differentiation makes it possible to retrieve the gradients of arbitrary compositions of differentiable tensor operations without doing any extra work besides writing down the forward pass. When I wrote my first neural networks in C in the 2000s, I had to write my gradients by hand. Now, thanks to modern automatic differentiation tools, you’ll never have to implement backpropagation yourself. Consider yourself lucky!

## Figures
- Figure 2.25: Path from `loss_val` to `w` in the backward graph  | labels inside the figure: (none - image)

## Equation candidates (formulas written inline in the text)
- [p074_e04] grad(x1, w) = 2
3
w
*
- [p074_e05] x1
grad(x2, x1) = 1
1
grad(x2, b) = 1
b
+
- [p074_e06] x2
grad(loss_val, x2) = 1
4
y_true
abs_diff

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

## Previous page (PDF 73) text, for continuity
`grad(B, A)`? We’ll annotate each inverted edge with this value (figure 2.24). This backward graph represents the backward pass.

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

## Next page (PDF 75) text, for continuity
2.5

Looking back at our first example

You’re nearing the end of this chapter, and you should now have a general understanding of what’s going on behind the scenes in a neural network. What was a magical black box at the start of the chapter has turned into a clearer picture, as illustrated in figure 2.26: the model, composed of layers that are chained together, maps the input data to predictions. The loss function then compares these predictions to the targets, producing a loss value: a measure of how well the model’s predictions match what was expected. The optimizer uses this loss value to update the model’s weights.

Figure 2.26 Relationship between the network, layers, loss function, and optimizer

Let’s go back to the first example and review each piece of it in the light of what you’ve learned in the previous sections.

This was the input data:

Now you understand that the input images are stored in NumPy tensors, which are here formatted as `float32` tensors of shape `(60000,` `784)` (training data) and `(10000, ` `784)` (test data), respectively.

This was our model:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.71 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.72 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.67 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.80 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.70 — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization
- PDF p.76 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.9 — contents › 
- PDF p.103 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.80 (printed 56) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › Gradient computation
Now, there’s just one thing we’re still missing: gradient computation (represented by the `get_gradients_of_loss_wrt_weights()` function in listing 2.9). In the previous section, we outlined how we could use the chain rule to obtain the gradients of a chain of functions given their individual derivatives, a process known as backpropagation. We could reimplement backpropagation from scratch here, but that would be rather cumbersome, especially since we’re using a `softmax` operation and a crossentropy loss, which have fairly verbose derivatives.

Instead, we can rely on the automatic differentiation mechanism that’s built into one of the low-level frameworks supported by Keras, such as TensorFlow, JAX, or PyTorch. For the sake of the example, let’s go with TensorFlow here. You’ll learn more about TensorFlow, JAX, and PyTorch in the next chapter. The API through which you can use TensorFlow’s automatic differentiation capabilities is the `tf.GradientTape` object. It’s a Python scope that will “record” the tensor operations that run inside it, in the form of a computation graph (sometimes called a tape). This graph can then be used to retrieve the gradient of any scalar value with res

### PDF p.71 (printed 47) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › Automatic differentiation with computation graphs
A useful way to think about backpropagation is in terms of computation graphs. A computation graph is the data structure at the heart of the deep learning revolution. It’s a directed acyclic graph of operations—in our case, tensor operations. For instance, figure 2.21 is the graph representation of our first model.

Computation graphs have been an extremely successful abstraction in computer science because they enable us to treat computation as data: a computable expression is encoded as a machine-readable data structure that can be used as the input or output of another program. For instance, you could imagine a program that receives a computation graph and returns a new computation graph that implements a large-scale distributed version of the same computation—this would mean that you could distribute any computation without having to write the distribution logic yourself. Or imagine … a program that receives a computation graph and can automatically generate the derivative of the expression it represents. It’s much easier to do these things if your computation is expressed as an explicit graph data structure rather than, say, lines of ASCII characters in a `.py` file. To explai

### PDF p.70 (printed 46) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › Chaining derivatives: The Backpropagation algorithm
In the previously discussed algorithm, we casually assumed that because a function is differentiable, we can easily compute its gradient. But is that true? How can we compute the gradient of complex expressions in practice? In our two-layer network example, how can we get the gradient of the loss with regard to the weights? That’s where the Backpropagation algorithm comes in.

### PDF p.71 (printed 47) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
Applying the chain rule to the computation of the gradient values of a neural network gives rise to an algorithm called backpropagation. Let’s see how that works, concretely.

### PDF p.70 (printed 46) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › The chain rule
Backpropagation is a way to use the derivative of simple operations (such as addition, `relu`, or tensor product) to easily compute the gradient of arbitrarily complex combinations of these atomic operations. Crucially, a neural network consists of many tensor operations chained together, each of which has a simple, known derivative. For instance, the model from our first example can be expressed as a function parameterized by the variables `W1`, `b1`, `W2`, and `b2` (belonging to the first and second `Dense` layers, respectively), involving the atomic operations `matmul`, `relu`, `softmax`, and `+`, as well as our loss function, `loss`, which are all easily differentiable:

### PDF p.72 (printed 48) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
to all nodes in the graph, from top to bottom, until we reach `loss_val`. This is the forward pass.

### PDF p.103 (printed 79) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › 
1 `loss.backward()` runs backpropagation starting from the `loss` output node and

populates the `tensor.grad` attribute on all tensors that were involved in the computation of `loss`. `tensor.grad` represents the gradient of the loss with regard to that tensor. 2 We use the `.grad` attribute to recover the gradients of the loss with regard to `W` and `b`. 3 We update `W` and `b` using those gradients. Because these updates are not intended to be part of the backward pass, we do them inside a `torch.no_grad()` scope, which skips gradient computation for everything inside it. 4 We reset the contents of the `.grad` property of our `W` and `b` parameters, by setting it to `None`. If we didn’t do this, gradient values would accumulate across multiple calls to `training_step()`, resulting in invalid values:

### PDF p.83 (printed 59) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › 
move is defined by the learning rate) in the opposite direction from the gradient.

This is called mini-batch gradient descent.

The entire learning process is made possible by the fact that all tensor operations in neural networks are differentiable, and thus it’s possible to apply the chain rule of derivation to find the gradient function mapping the current parameters and current batch of data to a gradient value. This is called backpropagation.

Two key concepts you’ll see frequently in future chapters are loss and optimizers. These are the two things you need to define before you begin feeding data into a model: – The loss is the quantity you’ll attempt to minimize during training, so it should represent a measure of success for the task you’re trying to solve. – The optimizer specifies the exact way in which the gradient of the loss will be used to update parameters: for instance, it could be the RMSProp optimizer, SGD with momentum, and so on.
