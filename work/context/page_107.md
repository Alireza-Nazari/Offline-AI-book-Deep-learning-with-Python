# Generation bundle for page_107  (PDF page 107, printed page 83)

Page kind: content
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 3.5 Introduction to JAX
Subsection: 3.5.3 Random number generation in JAX
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_107.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_107.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p107_e03] (bullet) Computing gradients via `jax.grad` and `jax.value_and_grad`
[p107_e04] (bullet) Making JAX functions fast by leveraging just-in-time compilation
[p107_e05] (body) Let’s get started.
[p107_e06] (heading_subsection) 3.5.2
[p107_e07] (heading_subsection) Tensors in JAX
[p107_e08] (body) One of the best features of JAX is that it doesn’t try to implement its own independent, similar-to-NumPy-but-slightly-divergent numerical API. Instead, it just implements the NumPy API, as is. It is available as the `jax.numpy` namespace, and you will often see it imported as `jnp` for short.
[p107_e09] (body) Here are some JAX arrays.
[p107_e10] (listing_caption) Listing 3.28 All-ones or all-zeros tensors
[p107_e11] (code)
```
>>> from jax import numpy as jnp
>>> jnp.ones(shape=(2, 1))
Array([[1.],
       [1.]], dtype=float32)
>>> jnp.zeros(shape=(2, 1))
Array([[0.],
       [0.]], dtype=float32)
>>> jnp.array([1, 2, 3], dtype="float32")
Array([1., 2., 3.], dtype=float32)
```
[p107_e12] (body) There are, however, two minor differences between `jax.numpy` and the actual NumPy API: random number generation and array assignment. Let’s take a look.
[p107_e13] (heading_subsection) 3.5.3
[p107_e14] (heading_subsection) Random number generation in JAX
[p107_e15] (body) The first difference between JAX and NumPy has to do with the way JAX handles random operations—what is known as “PRNG” (Pseudo-Random Number Generation) operations. We said earlier that JAX is stateless, which implies that JAX code can’t rely on any hidden global state. Consider the following NumPy code.
[p107_e16] (listing_caption) Listing 3.29 Random tensors
[p107_e17] (code)
```
>>> np.random.normal(size=(3,))
array([-1.68856166,  0.16489586,  0.67707523])
>>> np.random.normal(size=(3,))
array([-0.73671259,  0.3053194 ,  0.84124895])
```
[p107_e18] (body) How did the second call to `np.random.normal()` know to return a different value from the first call? That’s right—it’s a hidden piece of global state. You can actually retrieve that global state via `np.random.get_state()` and set it via `np.random.seed(seed)`.
[p107_e19] (body) In a stateless framework, we can’t have any such global state. The same API call must always return the same value. As a result, in a stateless version of NumPy, you would have to rely on passing different seed arguments to your `np.random` calls to get different values.

## Listings
- Listing 3.28: All-ones or all-zeros tensors (code elements p107_e11)
- Listing 3.29: Random tensors (code elements p107_e17)

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

## Previous page (PDF 106) text, for continuity
The popular pretrained model-sharing platform Hugging Face has first-class support for PyTorch, which means that any model you’d like to use is likely available in PyTorch. This is the primary drive behind PyTorch adoption today.

Meanwhile, there are also some downsides to using PyTorch:

Like with TensorFlow, the PyTorch API is inconsistent with NumPy. Further, it’s also internally inconsistent. For instance, the commonly used keyword `axis` is occasionally named `dim` instead, depending on the function. Some pseudo-random number generation operations take a `seed` argument; others don’t. And so on. This can make PyTorch frustrating to learn, especially when coming from NumPy.

Due to its focus on eager execution, PyTorch is quite slow—it’s the slowest of all the major frameworks by a large margin. For most models, you may see a 20% or 30% speedup with JAX. For some models—especially large ones—you may even see a 3× or a 5× speedup with JAX, even after using `torch.compile()`.

While it is possible to make PyTorch code faster via `torch.compile()`, the PyTorch Dynamo compiler remains at this time (in 2025) quite ineffective and full of trapdoors. As a result, only a very small percentage of the PyTorch user base uses compilation. Perhaps this will be improved in future versions!

3.5

Introduction to JAX

JAX is an open source library for differentiable computation, primarily developed by Google. After its release in 2018, JAX quickly gained traction in the research community, particularly for its ability to use Google’s TPUs at scale. Today, JAX is in use by most of the top players in the generative AI space—companies like DeepMind, Apple, Midjourney, Anthropic, Cohere, and so on.

JAX embraces a stateless approach to computation, meaning that functions in JAX do not …

## Next page (PDF 108) text, for continuity
Now, it’s often the case that your PRNG calls are going to be in functions that get called multiple times and that are intended to use different random values each time. If you don’t want to rely on any global state, this requires you to manage your seed state outside of the target function, like this:

It’s basically the same in JAX. However, JAX doesn’t use integer seeds. It uses special array structures called keys. You can create one from an integer value, like this:

To force you to always provide a seed “key” to PRNG calls, all JAX PRNG-using operations take `key` (the random seed) as their first positional argument. Here’s how to use `random.normal()`:

Two calls to `random.normal()` that receive the same seed key will always return the same value.

Listing 3.30 Using a random seed in Jax

If you need a new seed key, you can simply create a new one from an existing one using the `jax.random.split()` function. It is deterministic, so the same sequence of splits will always result in the same final seed key:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.109 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.112 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.99 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch
- PDF p.88 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.110 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.114 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.89 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.111 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.109 (printed 85) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Tensor assignment
The second difference between JAX and NumPy is tensor assignment. Like in TensorFlow, JAX arrays are not assignable in place. That’s because any sort of in-place modification would go against JAX’s stateless design. Instead, if you need to update a tensor, you must create a new tensor with the desired value. JAX makes this easy by providing the `at()`/`set()` API. These methods allow you to create a new tensor with an updated element at a specific index. Here’s an example of how you would update the first element of a JAX array to a new value.

Listing 3.31 Modifying values in a JAX array

### PDF p.109 (printed 85) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › 
This is definitely more work than `np.random`! But the benefits of statelessness far outweigh the costs: it makes your code vectorizable (i.e., the JAX compiler can automatically turn it into highly parallel code) while maintaining determinism (i.e., you can run the same code twice with the same results). That is impossible to achieve with a global PRNG state.

### PDF p.112 (printed 88) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › An end-to-end example: A linear classifier in pure JAX
Now you know enough JAX to write the JAX version of our linear classifier example. There are two major differences from the TensorFlow and PyTorch versions you’ve already seen:

All functions we will create will be stateless. That means the state (the arrays `W` and `b`) will be provided as function arguments, and if they get modified by the function, their new value will be returned by the function.

Gradients are computed using the JAX `value_and_grad()` utility.

Let’s get started. The model function and the mean squared error function should look familiar:

### PDF p.9 (printed ix) — contents ›  › 
3.4

Introduction to PyTorch 74

First steps with PyTorch 75 ■ An end-to-end example: A linear classifier in pure PyTorch 78 ■ What makes the PyTorch approach unique 81

3.5

Introduction to JAX 82

First steps with JAX 82 ■ Tensors in JAX 83 ■ Random number generation in JAX 83 ■ An end-to-end example: A linear classifier in pure JAX 88 ■ What makes the JAX approach unique 90

### PDF p.112 (printed 88) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Making JAX functions fast with @jax.jit
Be mindful that you can only decorate a stateless function—any tensors that get updated by the function should be part of its return values.

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

### PDF p.110 (printed 86) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX › Computing gradients with JAX
Unlike TensorFlow and PyTorch, JAX takes a metaprogramming approach to gradient computation. Metaprogramming refers to the idea of having functions that return functions—you could call them “meta-functions.” In practice, JAX lets you turn a loss-computation function into a gradient-computation function. So computing gradients in JAX is a three-step process:

1 Define a loss function, `compute_loss()`. 2 Call `grad_fn = jax.grad(compute_loss)` to retrieve a gradient-computation

function. 3 Call `grad_fn` to retrieve the gradient values. The loss function should verify the following properties:

It should return a scalar loss value.

Its first argument (which, in the following example, is also the only argument) should contain the state arrays we need gradients for. This argument is usually named `state`. For instance, this first argument could be a single array, a list of arrays, or a dict of arrays.

Let’s take a look at a simple example. Here’s a loss-computation function that takes a single scalar, `input_var` and returns a scalar loss value—just the square of the input:

### PDF p.521 (printed 497) — 16 Text generation › 16.4 Going further with LLMs › Choosing a dtype for large models
>>> prompt = "Why can't you assign values in Jax tensors? Be brief!"
>>> gemma_lm.generate(PROMPT_TEMPLATE.format(prompt), max_length=512)
<start_of_turn>user
Why can't you assign values in Jax tensors? Be brief!<end_of_turn>
<start_of_turn>model
Jax tensors are designed for efficient automatic differentiation. Directly
assigning values disrupts this process, making it difficult to track gradients
correctly. Instead, Jax uses operations to modify tensor values, preserving the
differentiation pipeline.<end_of_turn>
