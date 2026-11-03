# Generation bundle for page_503  (PDF page 503, printed page 479)

Page kind: content
Chapter: 16 Text generation
Section: 16.2 Training a mini-GPT
Subsection: 16.2.3 Generative decoding
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_503.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_503.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p503_e03] (body) Let’s try this out with a text prompt:
[p503_e04] (code)
```
>>> prompt = "A piece of advice"
>>> generate(prompt)
A piece of advice, and the best way to get a feel for yourself is to get a sense
of what you are doing.
If you are a business owner, you can get a sense of what you are doing. You can
get a sense of what you are doing, and you can get a sense of what
```
[p503_e05] (body) The first thing you will notice when running this is that it takes minutes to complete. That’s a bit puzzling. We predicted about 200,000 tokens a second on our reference hardware during training. The generative loop may add time, but a minute delay is much too slow. What happened? The biggest reason for our slowness, at least on the Jax and TensorFlow backends, is that we are running an uncompiled computation.
[p503_e06] (body) Every time you run `fit()` or `predict()`, Keras compiles the computation that runs on each batch of data. All the `keras.ops` used will be lifted out of Python and heavily optimized by the backend framework. It’s slow for one batch but massively faster for each subsequent call. However, when we directly call the model as we did previously, the backend framework will need to run the forward pass live and unoptimized at each step. The easy solution here is to lean on `predict()`. With `predict()`, Keras will handle compilation for us, but there is one important gotcha to watch out for. When TensorFlow or Jax compiles a function, it will do so for a specific input shape. With a known shape, the backend can optimize for particular hardware, knowing exactly how many individual processor instructions make up a tensor operation. But in our generation function, we call our model with a sequence that changes shape after each prediction. This would trigger recompilation each time we call `predict()`. Instead, we can avoid recompiling the `predict()` function if we pad our input so that our sequence is always the same length. Let’s try that out.
[p503_e07] (listing_caption) Listing 16.10 A compiled generation function for the mini-GPT model
[p503_e08] (code)
```
def compiled_generate(prompt, max_length=64):
    tokens = list(ops.convert_to_numpy(tokenizer(prompt)))
    prompt_length = len(tokens)
    tokens = tokens + [0] * (max_length - prompt_length)
    for i in range(prompt_length, max_length):
        prediction = mini_gpt.predict(np.array([tokens]), verbose=0)
        prediction = prediction[0, i - 1]
        tokens[i] = np.argmax(prediction).item()
    return tokenizer.detokenize(tokens)
```
[p503_e09] (body) Let’s see how fast this new function is:

## Code annotations printed next to listings (callouts, in order)
- Pads tokens
- to the full
- sequence
- length

## Listings
- Listing 16.10: A compiled generation function for the mini-GPT model (code elements p503_e08)

## Chapter outline
Chapter: 16 Text generation (PDF pages 490-531)
  - 16.1 A brief history of sequence generation (p.492)
  - 16.2 Training a mini-GPT (p.494)
      - 16.2.1 Building the model (p.497)
      - 16.2.2 Pretraining the model (p.500)
      - 16.2.3 Generative decoding (p.502)
      - 16.2.4 Sampling strategies (p.504)
  - 16.3 Using a pretrained LLM (p.508)
      - 16.3.1 Text generation with the Gemma model (p.509)
      - 16.3.2 Instruction fine-tuning (p.512)
      - 16.3.3 Low-Rank Adaptation (LoRA) (p.514)
  - 16.4 Going further with LLMs (p.519)
      - 16.4.1 Reinforcement Learning with Human Feedback (RLHF) (p.519)
      - 16.4.2 Multimodal LLMs (p.522)
      - 16.4.3 Retrieval Augmented Generation (RAG) (p.525)
      - 16.4.4 “Reasoning” models (p.526)
  - 16.5 Where are LLMs heading next? (p.528)

## Previous page (PDF 502) text, for continuity
What is a logit?

As we compile our model, you will notice a new value for the loss `SparseCategorical`

`Crossentropy(from_logits=True)`. What is a logit?

The output projection at the end of our transformer model does not contain the usual softmax activation. You can think of this output as a bunch of “unnormalized log probabilities” for each token. If you exponentiate each output value and normalize all values to sum to 1 (this is all the `softmax` function does), you will get a probability value. A common term for an “unnormalized log probability” is a logit, and logits can be easier to work with when generating text, as we will see in the next section. Keras gives you a choice of where to apply the `softmax` function. For classification problems, you can either use a `softmax` as the last activation of the model and output probabilities or move the `softmax` into the loss function and output logits. To do the

latter, you should pass `SparseCategoricalCrossentropy(from_logits=True)` as a

classification loss.

After training, our model can predict the next token in a sequence about 36% of the time on our validation set, though such a metric is just a crude heuristic.

Note that our model is undertrained. Our validation loss will continue to tick down after each epoch, which is unsurprising given that we used a hundred times fewer training steps than GPT-1. Training for longer would be a great idea, but we would need both time and money to pay for compute. Let’s play around with our mini-GPT model.

16.2.3 Generative decoding

To sample some output from our model, we can follow the approach we used to generate Shakespeare or Spanish translations in chapter 15. We feed a prompt of fixed tokens into the model. For each position in the input sequence, the model outputs …

## Next page (PDF 504) text, for continuity
Our generation call went from minutes to less than a second with compilation. That is quite an improvement.

Cached generation

There’s still one more major inefficiency in the generation function we just built. Can you spot it? Each time we call our model, we call it for an entire sequence and then throw away everything but the predictions for a single position. This is wasteful—our sequence only changes by a single token between generation steps. When we did generation with an RNN in chapter 15, we could keep around our RNN state and only compute outputs for a single token at each step. This state vector contained all the information the model needed about the past sequence. Transformers that use a causal attention, like GPT, actually have a similar notion of state. If we think through the entire model we just built, you’ll note that attention is the only place where the model passes information from position to position. The feedforward blocks of a transformer only modify the hidden representation of each token position in isolation. Inside attention, we incorporate information about past tokens through the `key` and `value` vectors. For a given `query` at a position, we compute attention scores by dotting the `query` with all previous `key` vectors and combining all previous `value` vectors. These `key` and `value` vectors never change for past tokens in the sequence—past input is fixed, and the causal mask prevents the Transformer from “looking ahead” to future tokens. So if we cache all `key` and `value` vectors, at each layer of the Transformer, we have the equivalent of an RNN’s state. We can use it to compute Transformer outputs for a single position at a time. Implementing this is a bit clunky, as it involves saving and reusing intermediate arrays from every a …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.505 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.508 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.511 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.507 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.450 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.497 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.451 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.513 — 16 Text generation › 16.3 Using a pretrained LLM

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.92 (printed 68) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › Making TensorFlow functions fast using compilation
When you do this, any call to `dense()` is replaced with a call to a compiled program that implements a more optimized version of the function. The first call to the function will take a bit longer, because TensorFlow will be compiling your code. This only happens once—all subsequent calls to the same function will be fast.

TensorFlow has two compilation modes:

### PDF p.96 (printed 72) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › 
For simplicity, we’ll do batch training instead of mini-batch training: we’ll run each training step (gradient computation and weight update) on the entire data, rather than iterate over the data in small batches. On one hand, this means that each training step will take much longer to run, since we compute the forward pass and the gradients for 2,000 samples at once. On the other hand, each gradient update will be much more effective at reducing the loss on the training data, since it will encompass information from all training samples instead of, say, only 128 random samples. As a result, we will need many fewer steps of training, and we should use a larger learning rate than what we would typically use for mini-batch training (we’ll use `learning_rate = 0.1`, as previously defined).

Listing 3.21 The batch training loop

### PDF p.501 (printed 477) — 16 Text generation › 16.2 Training a mini-GPT › 
Figure 16.2 Warmup makes our updates to model parameters smaller at the beginning of training and can help with stability.

We will train our model using one pass over our 1 billion tokens, split across eight epochs so we can occasionally check our validation set loss and accuracy.

We are training a miniature version of GPT, using 3× fewer parameters than GPT-1 and 100× fewer overall training steps. But despite this being two orders of magnitude cheaper to train than the smallest GPT model, this call to `fit()` will be the most computationally expensive training run in the entire book. If you are running the code as you read, set things off and take a breather!

Listing 16.8 Training the mini-GPT model

### PDF p.242 (printed 218) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › A JAX training step function
When it comes to low-level training code, JAX tends to be the most complex of the three backends because of its fully stateless nature. Statelessness makes JAX highly performant and scalable, making it amenable to compilation and automatic performance optimizations. However, writing stateless code requires you to jump through some hoops.

Since the gradient function is obtained via metaprogramming, you first need to define the function that returns your loss. Further, this function needs to be stateless, so it needs to take as arguments all the variables it’s going to be using, and it needs to return the value of any variable it has updated. Remember those non-trainable weights that can get modified during the forward pass? Those are the variables we need to return. To make it easier to work with the stateless programming paradigm of JAX, Keras models make available a stateless forward pass method: the `stateless_call()` method. It behaves just like `call`, except that

It takes as input the model’s trainable weights and non-trainable weights, in addition to the `inputs` and `training` arguments.

### PDF p.127 (printed 103) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Summary
TensorFlow, PyTorch, and JAX are three popular low-level frameworks for numerical computation and autodifferentiation. They all have their own way of doing things and their own strengths and weaknesses.

Keras is a high-level API for building and training neural networks. It can be used with either TensorFlow, PyTorch, or JAX—just pick the backend you like best.

The central class of Keras is the `Layer`. A layer encapsulates some weights and some computation. Layers are assembled into models.

Before you start training a model, you need to pick an optimizer, a loss, and some metrics, which you specify via the `model.compile()` method.

To train a model, you can use the `fit()` method, which runs mini-batch gradient descent for you. You can also use it to monitor your loss and metrics on validation data, a set of inputs that the model doesn’t see during training.

Once your model is trained, you can use the `model.predict()` method to generate predictions on new inputs.

### PDF p.450 (printed 426) — 15 Language models and the Transformer › 15.1 The language model › 15.1.2 Generating Shakespeare
Now that we have trained a model that can predict the next individual tokens with some accuracy, we would like to use it to extrapolate an entire predicted sequence. We can do this by calling the model in a loop, where the model’s predicted output at one time step becomes the model’s input at the next time step. A model built for this kind of feedback loop is sometimes called an autoregressive model.

To run such a loop, we need to perform a slight surgery on the model we just trained. During training, our model handled only a fixed sequence length of 100 tokens, and the `GRU` cell’s state was handled implicitly when calling the layer. During generation, we would like to predict a single output token at a time and explicitly output the state of the `GRU`’s cell. We need to propagate that state, which contains all information the model has encoded about past input characters, the next time we call the model. Let’s make a model that handles a single input character at a time and allows explicitly passing the RNN state. Because this model will have the same computational structure, with slightly modified inputs and outputs, we can assign weights from one model to another.

Listing 15.

### PDF p.92 (printed 68) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › Making TensorFlow functions fast using compilation
All the TensorFlow code you’ve written so far has been executing “eagerly.” This means operations are executed one after the other in the Python runtime, much like any Python code or NumPy code. Eager execution is great for debugging, but it is typically quite slow. It can often be beneficial to parallelize some computation, or “fuse” operations—replacing two consecutive operations, like `matmul` followed by `relu`, with a single, more efficient operation that does the same thing without materializing the intermediate output.

This can be achieved via compilation. The general idea of compilation is to take certain functions you’ve written in Python, lift them out of Python, automatically rewrite them into a faster and more efficient “compiled program,” and then call that program from the Python runtime. The main benefit of compilation is improved performance. There’s a drawback too: the code you write is no longer the code that gets executed, which can make the debugging experience painful. Only turn on compilation after you’ve already debugged your code in the Python runtime. You can apply compilation to any TensorFlow function by wrapping it in a `tf.function` decorator, like thi

### PDF p.499 (printed 475) — 16 Text generation › 16.2 Training a mini-GPT › 
Let’s build our model. We will stack eight decoder layers into a single “mini” GPT model.

We will also turn on a Keras setting called mixed precision to speed up training. This will allow Keras to run some of the model’s computations much faster by sacrificing some numerical fidelity. For now, this will remain a little mysterious, but a full explanation is waiting in chapter 18.

Listing 16.6 Creating a mini-GPT functional model
