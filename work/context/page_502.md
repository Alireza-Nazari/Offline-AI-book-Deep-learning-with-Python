# Generation bundle for page_502  (PDF page 502, printed page 478)

Page kind: content
Chapter: 16 Text generation
Section: 16.2 Training a mini-GPT
Subsection: 16.2.3 Generative decoding
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_502.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_502.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p502_e03] (heading_subsection) What is a logit?
[p502_e04] (sidebar) As we compile our model, you will notice a new value for the loss `SparseCategorical`
[p502_e05] (body) `Crossentropy(from_logits=True)`. What is a logit?
[p502_e06] (sidebar) The output projection at the end of our transformer model does not contain the usual softmax activation. You can think of this output as a bunch of “unnormalized log probabilities” for each token. If you exponentiate each output value and normalize all values to sum to 1 (this is all the `softmax` function does), you will get a probability value. A common term for an “unnormalized log probability” is a logit, and logits can be easier to work with when generating text, as we will see in the next section. Keras gives you a choice of where to apply the `softmax` function. For classification problems, you can either use a `softmax` as the last activation of the model and output probabilities or move the `softmax` into the loss function and output logits. To do the
[p502_e07] (body) latter, you should pass `SparseCategoricalCrossentropy(from_logits=True)` as a
[p502_e08] (sidebar) classification loss.
[p502_e09] (body) After training, our model can predict the next token in a sequence about 36% of the time on our validation set, though such a metric is just a crude heuristic.
[p502_e10] (body) Note that our model is undertrained. Our validation loss will continue to tick down after each epoch, which is unsurprising given that we used a hundred times fewer training steps than GPT-1. Training for longer would be a great idea, but we would need both time and money to pay for compute. Let’s play around with our mini-GPT model.
[p502_e11] (heading_subsection) 16.2.3 Generative decoding
[p502_e12] (body) To sample some output from our model, we can follow the approach we used to generate Shakespeare or Spanish translations in chapter 15. We feed a prompt of fixed tokens into the model. For each position in the input sequence, the model outputs a probability distribution over the entire vocabulary for the next token. By selecting the most likely next token at the last location, adding it to our sequence, and then repeating this process, we are able to generate a new sequence, one token at a time.
[p502_e13] (listing_caption) Listing 16.9 A simple generation function for the mini-GPT model
[p502_e14] (code)
```
def generate(prompt, max_length=64):
    tokens = list(ops.convert_to_numpy(tokenizer(prompt)))
    prompt_length = len(tokens)
    for _ in range(max_length - prompt_length):
        prediction = mini_gpt(ops.convert_to_numpy([tokens]))
        prediction = ops.convert_to_numpy(prediction[0, -1])
        tokens.append(np.argmax(prediction).item())
    return tokenizer.detokenize(tokens)
```

## Listings
- Listing 16.9: A simple generation function for the mini-GPT model (code elements p502_e14)

## Equation candidates (formulas written inline in the text)
- [p502_e05] `Crossentropy(from_logits=True)`.
- [p502_e07] latter, you should pass `SparseCategoricalCrossentropy(from_logits=True)` as a

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

## Previous page (PDF 501) text, for continuity
Figure 16.2 Warmup makes our updates to model parameters smaller at the beginning of training and can help with stability.

We will train our model using one pass over our 1 billion tokens, split across eight epochs so we can occasionally check our validation set loss and accuracy.

We are training a miniature version of GPT, using 3× fewer parameters than GPT-1 and 100× fewer overall training steps. But despite this being two orders of magnitude cheaper to train than the smallest GPT model, this call to `fit()` will be the most computationally expensive training run in the entire book. If you are running the code as you read, set things off and take a breather!

Listing 16.8 Training the mini-GPT model

## Next page (PDF 503) text, for continuity
Let’s try this out with a text prompt:

The first thing you will notice when running this is that it takes minutes to complete. That’s a bit puzzling. We predicted about 200,000 tokens a second on our reference hardware during training. The generative loop may add time, but a minute delay is much too slow. What happened? The biggest reason for our slowness, at least on the Jax and TensorFlow backends, is that we are running an uncompiled computation.

Every time you run `fit()` or `predict()`, Keras compiles the computation that runs on each batch of data. All the `keras.ops` used will be lifted out of Python and heavily optimized by the backend framework. It’s slow for one batch but massively faster for each subsequent call. However, when we directly call the model as we did previously, the backend framework will need to run the forward pass live and unoptimized at each step. The easy solution here is to lean on `predict()`. With `predict()`, Keras will handle compilation for us, but there is one important gotcha to watch out for. When TensorFlow or Jax compiles a function, it will do so for a specific input shape. With a known shape, the backend can optimize for particular hardware, knowing exactly how many individual processor instructions make up a tensor operation. But in our generation function, we call our model with a sequence that changes shape after each prediction. This would trigger recompilation each time we call `predict()`. Instead, we can avoid recompiling the `predict()` function if we pad our input so that our sequence is always the same length. Let’s try that out.

Listing 16.10 A compiled generation function for the mini-GPT model

Let’s see how fast this new function is:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.505 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.504 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.451 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.511 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.506 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.499 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.450 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.497 — 16 Text generation › 16.2 Training a mini-GPT

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.449 (printed 425) — 15 Language models and the Transformer › 15.1 The language model › 
This model outputs a softmax probability for every possible character in our vocabulary, and we will `compile()` it with a crossentropy loss. Note that our model is still training on a classification problem, it’s just that we will make one classification prediction for every token in our sequence. For our batch of 64 samples with 100 characters each, we will predict 6,400 individual labels. Loss and accuracy metrics reported by Keras during training will be averaged first across each sequence and, second, across each batch.

Let’s go ahead and train our language model.

Listing 15.5 Training a miniature language model

### PDF p.143 (printed 119) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › 
There are two other things you should note about this architecture:

You end the model with a `Dense` layer of size 46. This means for each input sample, the network will output a 46-dimensional vector. Each entry in this vector (each dimension) will encode a different output class.

The last layer uses a `softmax` activation. You saw this pattern in the MNIST example. It means the model will output a probability distribution over the 46 different output classes—for every input sample, the model will produce a 46-dimensional output vector, where `output[i]` is the probability that the sample belongs to class `i`. The 46 scores will sum to 1.

The best loss function to use in this case is `categorical_crossentropy`. It measures the distance between two probability distributions—here, between the probability distribution outputted by the model and the true distribution of the labels. By minimizing the distance between these two distributions, you train the model to output something as close as possible to the true labels.

Like last time, we’ll also monitor accuracy. However, accuracy is a bit of a crude metric in this case: if the model has the correct class as its second choice for

### PDF p.506 (printed 482) — 16 Text generation › 16.2 Training a mini-GPT › 
Our outputs are more diverse, and the model no longer gets stuck in loops. But our sampling is now exploring too much; the output jumps around wildly without any continuity.

You’ll notice we added a parameter called `temperature`. We can use this to sharpen or widen our probability distribution so our sampling explores our distribution less or more. If we pass a low temperature, we will make all logits larger before the `softmax` function, which makes our most likely output even more likely. If we pass a high temperature, our logits will be smaller before the softmax, and our probability distribution will be more spread out. Let’s try this out a few times to see how this affects our sampling:

### PDF p.150 (printed 126) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Wrapping up
Here’s what you should take away from this example:

If you’re trying to classify data points among N classes, your model should end with a `Dense` layer of size N.

In a single-label, multiclass classification problem, your model should end with a `softmax` activation so that it will output a probability distribution over the N output classes.

Categorical crossentropy is almost always the loss function you should use for such problems. It minimizes the distance between the probability distributions output by the model and the true distribution of the targets.

There are two ways to handle labels in multiclass classification: – Encoding the labels via categorical encoding (also known as one-hot encoding) and using `categorical_crossentropy` as a loss function – Encoding the labels as integers and using the `sparse_categorical_` `crossentropy` loss function

If you need to classify data into a large number of categories, you should avoid creating information bottlenecks in your model due to intermediate layers that are too small.

### PDF p.505 (printed 481) — 16 Text generation › 16.2 Training a mini-GPT › 
The Transformer outputs define a categorical distribution where each token has a certain probability of being output at each time step. Instead of just choosing the most likely token, we could sample this distribution directly. `keras.random.categorical()` will pass our predictions through a `softmax` function to get a probability distribution and then randomly sample it. Let’s try it out:

### PDF p.258 (printed 234) — 8 Image classification › 8.1 Introduction to convnets › 
After the last `Conv2D` layer, we end up with an output of shape `(3, 3, 256)`—a 3 × 3 feature map of 256 channels. The next step is to feed this output into a densely connected classifier like those you’re already familiar with: a stack of `Dense` layers. These classifiers process vectors, which are 1D, whereas the current output is a rank-3 tensor. To bridge the gap, we flatten the 3D outputs to 1D with a `GlobalAveragePooling2D` layer before adding the `Dense` layers. This layer will take the average of each 3 × 3 feature map in the tensor of shape `(3,` `3,` `256)`, resulting in an output vector of shape `(256,)`. Finally, we’ll do 10-way classification, so our last layer has 10 outputs and a softmax activation.

Now, let’s train the ConvNet on the MNIST digits. We’ll reuse a lot of the code from the MNIST example in chapter 2. Because we’re doing 10-way classification with a softmax output, we’ll use the categorical crossentropy loss, and because our labels are integers, we’ll use the sparse version, `sparse_categorical_crossentropy`.

Listing 8.3 Training the ConvNet on MNIST images

### PDF p.458 (printed 434) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 
Next, let’s add the decoder—a simple `GRU` layer that takes as its initial state the encoded source sentence. On top of it, we add a `Dense` layer that produces a probability distribution over the Spanish vocabulary for each output step. Here, we want to predict the next tokens based only on what came before, so a `Bidirectional` RNN would break training by making the loss function trivially easy.

Listing 15.12 Building a sequence-to-sequence decoder

### PDF p.504 (printed 480) — 16 Text generation › 16.2 Training a mini-GPT › Cached generation
There’s still one more major inefficiency in the generation function we just built. Can you spot it? Each time we call our model, we call it for an entire sequence and then throw away everything but the predictions for a single position. This is wasteful—our sequence only changes by a single token between generation steps. When we did generation with an RNN in chapter 15, we could keep around our RNN state and only compute outputs for a single token at each step. This state vector contained all the information the model needed about the past sequence. Transformers that use a causal attention, like GPT, actually have a similar notion of state. If we think through the entire model we just built, you’ll note that attention is the only place where the model passes information from position to position. The feedforward blocks of a transformer only modify the hidden representation of each token position in isolation. Inside attention, we incorporate information about past tokens through the `key` and `value` vectors. For a given `query` at a position, we compute attention scores by dotting the `query` with all previous `key` vectors and combining all previous `value` vectors. These `key`
