# Generation bundle for page_090  (PDF page 90, printed page 66)

Page kind: content
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 3.3 Introduction to TensorFlow
Subsection: 3.3.1 First steps with TensorFlow
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_090.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_090.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p090_e03] (code)
```
array([[-0.75133973],
       [-0.4872893 ],
       [ 1.6626885 ]], dtype=float32)>
```
[p090_e04] (body) The state of a variable can be modified via its `assign` method.
[p090_e05] (listing_caption) Listing 3.6 Assigning a value to a `Variable`
[p090_e06] (code)
```
>>> v.assign(tf.ones((3, 1)))
array([[1.],
       [1.],
       [1.]], dtype=float32)>
```
[p090_e07] (body) Assignment also works for a subset of the coefficients.
[p090_e08] (listing_caption) Listing 3.7 Assigning a value to a subset of a `Variable`
[p090_e09] (code)
```
>>> v[0, 0].assign(3.)
array([[3.],
       [1.],
       [1.]], dtype=float32)>
```
[p090_e10] (body) Similarly, `assign_add` and `assign_sub` are efficient equivalents of `+=` and `-=`.
[p090_e11] (listing_caption) Listing 3.8 Using `assign_add`
[p090_e12] (code)
```
>>> v.assign_add(tf.ones((3, 1)))
array([[2.],
       [2.],
       [2.]], dtype=float32)>
```
[p090_e13] (heading_subsubsection) Tensor operations: Doing math in TensorFlow
[p090_e14] (body) Just like NumPy, TensorFlow offers a large collection of tensor operations to express mathematical formulas. Here are a few examples.
[p090_e15] (listing_caption) Listing 3.9 A few basic math operations in TensorFlow
[p090_e16] (code)
```
a = tf.ones((2, 2))
b = tf.square(a)
c = tf.sqrt(a)
d = b + c
e = tf.matmul(a, b)
f = tf.concat((a, b), axis=0)
```

## Code annotations printed next to listings (callouts, in order)
- Takes the square, same as np.square
- Takes the square root, same as np.sqrt
- Adds two tensors (element-wise)
- Takes the product of two tensors
- (see chapter 2), same as np.matmul
- Concatenates a and b along axis 0,
- same as np.concatenate

## Listings
- Listing 3.6: Assigning a value to a `Variable` (code elements p090_e06)
- Listing 3.7: Assigning a value to a subset of a `Variable` (code elements p090_e09)
- Listing 3.8: Using `assign_add` (code elements p090_e12)
- Listing 3.9: A few basic math operations in TensorFlow (code elements p090_e16)

## Equation candidates (formulas written inline in the text)
- [p090_e03] array([[-0.75133973],
       [-0.4872893 ],
       [ 1.6626885 ]], dtype=float32)>

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

## Previous page (PDF 89) text, for continuity
Listing 3.2 Random tensors

Tensor assignment and the Variable class

A significant difference between NumPy arrays and TensorFlow tensors is that TensorFlow tensors aren’t assignable: they’re constant. For instance, in NumPy, you can do the following.

Listing 3.3 NumPy arrays are assignable

Try to do the same thing in TensorFlow: you will get an error, `EagerTensor` `object` `does ` `not` `support` `item` `assignment`.

Listing 3.4 TensorFlow tensors are not assignable

To train a model, we’ll need to update its state, which is a set of tensors. If tensors aren’t assignable, how do we do it, then? That’s where variables come in. `tf.Variable` is the class meant to manage modifiable state in TensorFlow.

To create a variable, you need to provide some initial value, such as a random tensor.

Listing 3.5 Creating a `tf.Variable`

## Next page (PDF 91) text, for continuity
Here’s an equivalent of the `Dense` layer we saw in chapter 2:

Gradients in TensorFlow: A second look at the GradientTape API

So far, TensorFlow seems to look a lot like NumPy. But here’s something NumPy can’t do: retrieve the gradient of any differentiable expression with respect to any of its inputs. Just open a `GradientTape` scope, apply some computation to one or several input tensors, and retrieve the gradient of the result with respect to the inputs.

Listing 3.10 Using the `GradientTape`

This is most commonly used to retrieve the gradients of the loss of a model with respect to its weights: `gradients` `=` `tape.gradient(loss,` `weights)`.

In chapter 2, you saw how the `GradientTape` works on either a single input or a list of inputs and how inputs could be either scalars or high-dimensional tensors. So far, you’ve only seen the case where the input tensors in `tape.gradient()` were TensorFlow variables. It’s actually possible for these inputs to be any arbitrary tensor. However, only trainable variables are being tracked by default. With a constant tensor, you’d have to manually mark it as being tracked, by calling `tape.watch()` on it.

Listing 3.11 Using the `GradientTape` with constant tensor inputs

Why? Because it would be too expensive to preemptively store the information required to compute the gradient of anything with respect to anything. To avoid wasting resources, the tape needs to know what to watch. Trainable variables are watched by default because computing the gradient of a loss with regard to a list of trainable variables is the most common use case of the gradient tape.

The gradient tape is a powerful utility, even capable of computing second-order gradients—that is, the gradient of a gradient. For instance, the gradient of the position  …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.109 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.88 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.101 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.94 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.100 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.107 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.108 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.111 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.101 (printed 77) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › Tensor operations: Doing math in PyTorch
Math in PyTorch works just the same as math in NumPy or TensorFlow, although much like TensorFlow, the PyTorch API often diverges in subtle ways from the NumPy API.

Listing 3.25 A few basic math operations in PyTorch

### PDF p.100 (printed 76) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › Tensor assignment and the Parameter class
Like NumPy arrays, but unlike TensorFlow tensors, PyTorch tensors are assignable. You can do operations like this:

### PDF p.109 (printed 85) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Tensor operations: Doing math in JAX
Doing math in JAX looks exactly the same as it does in NumPy. No need to learn anything new this time!

Listing 3.32 A few basic math operations in JAX

### PDF p.109 (printed 85) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Tensor assignment
The second difference between JAX and NumPy is tensor assignment. Like in TensorFlow, JAX arrays are not assignable in place. That’s because any sort of in-place modification would go against JAX’s stateless design. Instead, if you need to update a tensor, you must create a new tensor with the desired value. JAX makes this easy by providing the `at()`/`set()` API. These methods allow you to create a new tensor with an updated element at a specific index. Here’s an example of how you would update the first element of a JAX array to a new value.

Listing 3.31 Modifying values in a JAX array

### PDF p.101 (printed 77) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › Tensor operations: Doing math in PyTorch
Listing 3.25 A few basic math operations in PyTorch
a = torch.ones((2, 2))
b = torch.square(a)
c = torch.sqrt(a)
d = b + c
e = torch.matmul(a, b)
f = torch.cat((a, b), dim=0)

### PDF p.88 (printed 64) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › First steps with TensorFlow
Over the next paragraphs, you’ll get familiar with all the basics of TensorFlow. We’ll cover the following key concepts:

Tensors and variables

Numerical operations in TensorFlow

Computing gradients with a `GradientTape`

Making TensorFlow functions fast by using just-in-time compilation

We’ll then conclude the introduction with an end-to-end example: a pure-TensorFlow implementation of linear regression.

Let’s get those tensors flowing.

### PDF p.92 (printed 68) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › Making TensorFlow functions fast using compilation
All the TensorFlow code you’ve written so far has been executing “eagerly.” This means operations are executed one after the other in the Python runtime, much like any Python code or NumPy code. Eager execution is great for debugging, but it is typically quite slow. It can often be beneficial to parallelize some computation, or “fuse” operations—replacing two consecutive operations, like `matmul` followed by `relu`, with a single, more efficient operation that does the same thing without materializing the intermediate output.

This can be achieved via compilation. The general idea of compilation is to take certain functions you’ve written in Python, lift them out of Python, automatically rewrite them into a faster and more efficient “compiled program,” and then call that program from the Python runtime. The main benefit of compilation is improved performance. There’s a drawback too: the code you write is no longer the code that gets executed, which can make the debugging experience painful. Only turn on compilation after you’ve already debugged your code in the Python runtime. You can apply compilation to any TensorFlow function by wrapping it in a `tf.function` decorator, like thi

### PDF p.100 (printed 76) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › Tensor assignment and the Parameter class
While you can just use a regular `torch.Tensor` to store the trainable state of a model, PyTorch does provide a specialized tensor subclass for that purpose, the `torch
` `.nn.parameter.Parameter` class. Compared to a regular tensor, it provides semantic clarity—if you see a `Parameter`, you’ll know it’s a piece of trainable state, whereas a `Tensor` could be anything. As a result, it enables PyTorch to automatically track and retrieve the `Parameters` you assign to PyTorch models—similar to what Keras does with Keras `Variable` instances.

Here’s a `Parameter`.

Listing 3.24 Creating a PyTorch parameter
