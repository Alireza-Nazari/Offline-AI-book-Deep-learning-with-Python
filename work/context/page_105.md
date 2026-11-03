# Generation bundle for page_105  (PDF page 105, printed page 81)

Page kind: content
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 3.4 Introduction to PyTorch
Subsection: 3.4.3 What makes the PyTorch approach unique
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_105.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_105.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p105_e03] (body) Similarly, previously we needed to reset parameter gradients by hand by doing `tensor.grad = None` on each one. Now we can just do `model.zero_grad()`. Overall, this may feel a bit confusing—somehow the loss tensor, the optimizer, and the `Module` instance all seem to be aware of each other through some hidden background mechanism. They’re all interacting with one another via spooky action at a distance. Don’t worry though—you can just treat this sequence of steps (`loss.backward()` - `optimizer.step()` - `model.zero_grad()`) as a magic incantation to be recited any time you need to write a training step function. Just make sure not to forget `model.zero_` `grad()`. That would be a major bug (and it is unfortunately quite common)!
[p105_e04] (heading_subsubsection) Making PyTorch modules fast using compilation
[p105_e05] (body) One last thing. Similarly to how TensorFlow lets you compile functions for better performance, PyTorch lets you compile functions or even `Module` instances via the `torch
` `.compile()` utility. This API uses PyTorch’s very own compiler, named Dynamo.
[p105_e06] (body) Let’s try it on our linear regression `Module`:
[p105_e07] (code)
```
compiled_model = torch.compile(model)
```
[p105_e08] (body) The resulting object is intended to work identically to the original—except the forward and backward pass should run faster.
[p105_e09] (body) You can also use `torch.compile()` as a function decorator:
[p105_e10] (code)
```
@torch.compile
def dense(inputs, W, b):
    return torch.nn.relu(torch.matmul(inputs, W) + b)
```
[p105_e11] (body) In practice, most PyTorch code out there does not use compilation and simply runs eagerly, as the compiler may not always work with all models and may not always result in a speedup when it does work. Unlike in TensorFlow and Jax where compilation was built in from the inception of the library, PyTorch’s compiler is a relatively recent addition.
[p105_e12] (heading_subsection) 3.4.3
[p105_e13] (heading_subsection) What makes the PyTorch approach unique
[p105_e14] (body) Compared to TensorFlow and JAX, which we will cover next, what makes PyTorch stand out? Why should you use it or not use it?
[p105_e15] (body) Here are PyTorch’s two key strengths:
[p105_e16] (bullet) PyTorch code executes eagerly by default, making it easy to debug. Note that this is also the case for TensorFlow code and JAX code, but a big difference is that PyTorch is generally intended to be run eagerly at all times, whereas any serious TensorFlow or JAX project will inevitably need compilation at some point, which can significantly hurt the debugging experience.

## Equation candidates (formulas written inline in the text)
- [p105_e03] Similarly, previously we needed to reset parameter gradients by hand by doing `tensor.grad = None` on each one.
- [p105_e07] compiled_model = torch.compile(model)

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

## Previous page (PDF 104) text, for continuity
We can now instantiate our `LinearModel`:

When using an instance of `torch.nn.Module`, rather than calling the `forward()` method directly, you’d use `__call__()` (i.e., directly call the model class on inputs), which redirects to `forward()` but adds a few framework hooks to it:

Now, let’s get our hands on a PyTorch optimizer. To instantiate it, you will need to provide the list of parameters that the optimizer is intended to update. You can retrieve it from our `Module` instance via `.parameters()`:

Using our `Module` instance and the PyTorch `SGD` optimizer, we can run a simplified training step:

Previously, updating the model parameters looked like this:

Now we can just do `optimizer.step()`.

## Next page (PDF 106) text, for continuity
The popular pretrained model-sharing platform Hugging Face has first-class support for PyTorch, which means that any model you’d like to use is likely available in PyTorch. This is the primary drive behind PyTorch adoption today.

Meanwhile, there are also some downsides to using PyTorch:

Like with TensorFlow, the PyTorch API is inconsistent with NumPy. Further, it’s also internally inconsistent. For instance, the commonly used keyword `axis` is occasionally named `dim` instead, depending on the function. Some pseudo-random number generation operations take a `seed` argument; others don’t. And so on. This can make PyTorch frustrating to learn, especially when coming from NumPy.

Due to its focus on eager execution, PyTorch is quite slow—it’s the slowest of all the major frameworks by a large margin. For most models, you may see a 20% or 30% speedup with JAX. For some models—especially large ones—you may even see a 3× or a 5× speedup with JAX, even after using `torch.compile()`.

While it is possible to make PyTorch code faster via `torch.compile()`, the PyTorch Dynamo compiler remains at this time (in 2025) quite ineffective and full of trapdoors. As a result, only a very small percentage of the PyTorch user base uses compilation. Perhaps this will be improved in future versions!

3.5

Introduction to JAX

JAX is an open source library for differentiable computation, primarily developed by Google. After its release in 2018, JAX quickly gained traction in the research community, particularly for its ability to use Google’s TPUs at scale. Today, JAX is in use by most of the top players in the generative AI space—companies like DeepMind, Apple, Midjourney, Anthropic, Cohere, and so on.

JAX embraces a stateless approach to computation, meaning that functions in JAX do not …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.99 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.92 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.114 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.112 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.98 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.116 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.102 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.101 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.241 (printed 217) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › A PyTorch training step function
When you use the PyTorch backend, all of your Keras layers and models inherit from the PyTorch `torch.nn.Module` class and expose the native `Module` API. As a result, your model, its trainable weights, and your loss tensor are all aware of each other and interact via three methods: `loss.backward()`, `weight.value.grad`, and `model
` `.zero_grad()`.

As a reminder from chapter 3, the mental model you’ve got to keep in mind is this:

With each forward pass, PyTorch builds up a one-time computation graph that keeps track of the computation that just happened.

Calling `.backward()` on any given scalar node of this graph (like your loss) will run the graph backward starting from that node, automatically populating a `tensor.grad` attribute on all tensors involved (if they satisfy `requires_` `grad=True`), containing the gradient of the output node with respect to that tensor. In particular, it will populate the `grad` attribute of your trainable parameters.

To clear the contents of that `tensor.grad` attribute, you should call `tensor.grad ` `= None` on all your tensors. Because it would be a bit cumbersome to do this on all model variables individually, you can just do it at the mo

### PDF p.99 (printed 75) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › First steps with PyTorch
Over the next paragraphs, you’ll get familiar with all the basics of PyTorch. We’ll cover the following key concepts:

Tensors and parameters

Numerical operations in PyTorch

Computing gradients with the `backward()` method

Packaging computation with the `Module` class

Speeding up PyTorch by using compilation

We’ll conclude the introduction by reimplementing our linear regression end-to-end example in pure PyTorch.

### PDF p.102 (printed 78) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › An end-to-end example: A linear classifier in pure PyTorch
You now know enough to rewrite our linear classifier in PyTorch. It will stay very similar to the TensorFlow one—the only major difference is how we compute the gradients.

Let’s start by creating our model variables. Don’t forget to pass `requires_grad=True` so we can compute gradients with respect to them:

### PDF p.92 (printed 68) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › Making TensorFlow functions fast using compilation
All the TensorFlow code you’ve written so far has been executing “eagerly.” This means operations are executed one after the other in the Python runtime, much like any Python code or NumPy code. Eager execution is great for debugging, but it is typically quite slow. It can often be beneficial to parallelize some computation, or “fuse” operations—replacing two consecutive operations, like `matmul` followed by `relu`, with a single, more efficient operation that does the same thing without materializing the intermediate output.

This can be achieved via compilation. The general idea of compilation is to take certain functions you’ve written in Python, lift them out of Python, automatically rewrite them into a faster and more efficient “compiled program,” and then call that program from the Python runtime. The main benefit of compilation is improved performance. There’s a drawback too: the code you write is no longer the code that gets executed, which can make the debugging experience painful. Only turn on compilation after you’ve already debugged your code in the Python runtime. You can apply compilation to any TensorFlow function by wrapping it in a `tf.function` decorator, like thi

### PDF p.98 (printed 74) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › What makes the TensorFlow approach unique
Now, let’s move on to PyTorch.

### PDF p.114 (printed 90) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › What makes the JAX approach unique
The main thing that makes JAX unique among modern machine learning frameworks is its functional, stateless philosophy. While it may seem to cause friction at first, it is what unlocks the power of JAX—its ability to compile to extremely fast code and to scale to arbitrarily large models and arbitrarily many devices.

There’s a lot to like about JAX:

It’s fast. For most models, it is the fastest of all frameworks you’ve seen so far.

Its numerical API is fully consistent with NumPy, making it pleasant to learn.

It’s the best fit for training models on TPUs, as it was developed from the ground up for XLA and TPUs.

Using JAX can also come with some amount of developer friction:

Its use of metaprogramming and compilation can make it significantly harder to debug compared to pure eager execution.

Low-level training loops tend to be more verbose and more difficult to write than in TensorFlow or PyTorch.

At this point, you know the basics of TensorFlow, PyTorch, and JAX, and you can use these frameworks to implement a basic linear classifier from scratch. That’s a solid foundation to build upon. It’s now time to move on to a more productive path to deep learning: the Keras API.

### PDF p.100 (printed 76) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › Tensor assignment and the Parameter class
While you can just use a regular `torch.Tensor` to store the trainable state of a model, PyTorch does provide a specialized tensor subclass for that purpose, the `torch
` `.nn.parameter.Parameter` class. Compared to a regular tensor, it provides semantic clarity—if you see a `Parameter`, you’ll know it’s a piece of trainable state, whereas a `Tensor` could be anything. As a result, it enables PyTorch to automatically track and retrieve the `Parameters` you assign to PyTorch models—similar to what Keras does with Keras `Variable` instances.

Here’s a `Parameter`.

Listing 3.24 Creating a PyTorch parameter

### PDF p.240 (printed 216) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Writing custom training step functions
This snippet is pseudocode rather than real code because it includes an imaginary function, `get_gradients_of()`. In reality, retrieving gradients is done in a way that is specific to your current backend—JAX, TensorFlow, or PyTorch.

Let’s use what you learned about each framework in chapter 3 to implement a real version of this `train_step()` function. We’ll start with TensorFlow and PyTorch because these two make the job relatively easy, so they’re a good place to start. We’ll end with JAX, which is quite a bit more complex.
