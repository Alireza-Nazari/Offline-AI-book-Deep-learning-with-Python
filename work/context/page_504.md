# Generation bundle for page_504  (PDF page 504, printed page 480)

Page kind: content
Chapter: 16 Text generation
Section: 16.2 Training a mini-GPT
Subsection: 16.2.4 Sampling strategies
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_504.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_504.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p504_e03] (code)
```
>>> import timeit
>>> tries = 10
>>> timeit.timeit(lambda: compiled_generate(prompt), number=tries) / tries
0.4866470648999893
```
[p504_e04] (body) Our generation call went from minutes to less than a second with compilation. That is quite an improvement.
[p504_e05] (heading_subsection) Cached generation
[p504_e06] (sidebar) There’s still one more major inefficiency in the generation function we just built. Can you spot it? Each time we call our model, we call it for an entire sequence and then throw away everything but the predictions for a single position. This is wasteful—our sequence only changes by a single token between generation steps. When we did generation with an RNN in chapter 15, we could keep around our RNN state and only compute outputs for a single token at each step. This state vector contained all the information the model needed about the past sequence. Transformers that use a causal attention, like GPT, actually have a similar notion of state. If we think through the entire model we just built, you’ll note that attention is the only place where the model passes information from position to position. The feedforward blocks of a transformer only modify the hidden representation of each token position in isolation. Inside attention, we incorporate information about past tokens through the `key` and `value` vectors. For a given `query` at a position, we compute attention scores by dotting the `query` with all previous `key` vectors and combining all previous `value` vectors. These `key` and `value` vectors never change for past tokens in the sequence—past input is fixed, and the causal mask prevents the Transformer from “looking ahead” to future tokens. So if we cache all `key` and `value` vectors, at each layer of the Transformer, we have the equivalent of an RNN’s state. We can use it to compute Transformer outputs for a single position at a time. Implementing this is a bit clunky, as it involves saving and reusing intermediate arrays from every attention layer in the Transformer, but it’s important. Your model inputs can go from being as long as the maximum length of your output to being a single token in length. If you are generating a sequence that is thousands of tokens long, caching can amount to a thousandfold speedup! Any efficient implementation of generative sampling will include `key` and `value` caching.
[p504_e07] (heading_subsection) 16.2.4 Sampling strategies
[p504_e08] (body) Another obvious problem with our generative output is that our model often repeats itself. On our particular training run, the model repeats the group of words “get a sense of what you are doing” over and over.

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

## Previous page (PDF 503) text, for continuity
Let’s try this out with a text prompt:

The first thing you will notice when running this is that it takes minutes to complete. That’s a bit puzzling. We predicted about 200,000 tokens a second on our reference hardware during training. The generative loop may add time, but a minute delay is much too slow. What happened? The biggest reason for our slowness, at least on the Jax and TensorFlow backends, is that we are running an uncompiled computation.

Every time you run `fit()` or `predict()`, Keras compiles the computation that runs on each batch of data. All the `keras.ops` used will be lifted out of Python and heavily optimized by the backend framework. It’s slow for one batch but massively faster for each subsequent call. However, when we directly call the model as we did previously, the backend framework will need to run the forward pass live and unoptimized at each step. The easy solution here is to lean on `predict()`. With `predict()`, Keras will handle compilation for us, but there is one important gotcha to watch out for. When TensorFlow or Jax compiles a function, it will do so for a specific input shape. With a known shape, the backend can optimize for particular hardware, knowing exactly how many individual processor instructions make up a tensor operation. But in our generation function, we call our model with a sequence that changes shape after each prediction. This would trigger recompilation each time we call `predict()`. Instead, we can avoid recompiling the `predict()` function if we pad our input so that our sequence is always the same length. Let’s try that out.

Listing 16.10 A compiled generation function for the mini-GPT model

Let’s see how fast this new function is:

## Next page (PDF 505) text, for continuity
This isn’t so much a bug as it’s a direct consequence of our training objective. Our model is trying to predict the most likely next token in a sequence across about a billion words on many, many topics. If there’s no obvious choice for where a sequence of text should head next, an effective strategy is to guess common words or repeated patterns of words. Unsurprisingly, our model learns to do this during training almost immediately. If you were to stop training our model very early on, it would likely generate the word `"the"` incessantly, as `"the"` is the most common word in the English language. During our generative loop, we have always chosen the most likely predicted token in our model’s output. But our output is not just a single predicted token; it is a probability distribution across all 32,000 tokens in our vocabulary. Using the most likely output at each generation step is called greedy search. It’s the most straightforward approach to using model predictions, but it is hardly the only one. If we instead add some randomness to the process, we can explore the probability distribution learned by the model more broadly. This can keep us from getting stuck in loops of high-probability token sequences. Let’s try this out. We can start by refactoring our generation function so that we can pass a function that maps from a model’s predictions to a choice for the next token. We will call this our sampling strategy:

Now we can write our greedy search as a simple function we pass to `compiled_generate()`:

The Transformer outputs define a categorical distribution where each token has a certain probability of being output at each time step. Instead of just choosing the most likely token, we could sample this distribution directly. `keras.random.categorical()` will pass …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.502 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.450 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.522 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.497 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.498 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.451 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.514 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.472 — 15 Language models and the Transformer › 15.3 The Transformer architecture

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.450 (printed 426) — 15 Language models and the Transformer › 15.1 The language model › 15.1.2 Generating Shakespeare
Now that we have trained a model that can predict the next individual tokens with some accuracy, we would like to use it to extrapolate an entire predicted sequence. We can do this by calling the model in a loop, where the model’s predicted output at one time step becomes the model’s input at the next time step. A model built for this kind of feedback loop is sometimes called an autoregressive model.

To run such a loop, we need to perform a slight surgery on the model we just trained. During training, our model handled only a fixed sequence length of 100 tokens, and the `GRU` cell’s state was handled implicitly when calling the layer. During generation, we would like to predict a single output token at a time and explicitly output the state of the `GRU`’s cell. We need to propagate that state, which contains all information the model has encoded about past input characters, the next time we call the model. Let’s make a model that handles a single input character at a time and allows explicitly passing the RNN state. Because this model will have the same computational structure, with slightly modified inputs and outputs, we can assign weights from one model to another.

Listing 15.

### PDF p.502 (printed 478) — 16 Text generation › 16.2 Training a mini-GPT › 16.2.3 Generative decoding
To sample some output from our model, we can follow the approach we used to generate Shakespeare or Spanish translations in chapter 15. We feed a prompt of fixed tokens into the model. For each position in the input sequence, the model outputs a probability distribution over the entire vocabulary for the next token. By selecting the most likely next token at the last location, adding it to our sequence, and then repeating this process, we are able to generate a new sequence, one token at a time.

Listing 16.9 A simple generation function for the mini-GPT model

### PDF p.498 (printed 474) — 16 Text generation › 16.2 Training a mini-GPT › 
France?”, the learned representation of the word “Where” cannot attend to the words “capital” and “France” in the attention layer. This limits the expressivity of the model but has a massive advantage in terms of simplicity of pretraining. We don’t need to curate datasets with pairs of inputs and outputs; everything can be a single sequence. We can train on any text we can find on the internet at a massive scale.

Let’s copy the `TransformerDecoder` from chapter 15 but remove the cross-attention layer, which allowed the decoder to attend to the encoder sequence. We will also make one minor change, adding dropout after the attention and feedforward blocks. In chapter 15, we only used a single Transformer layer in our encoder and decoder, so we could get away with only using a single dropout layer at the end of our entire model. For our GPT model, we will stack quite a few layers, so adding dropout within each decoder layer is important to prevent overfitting.

Listing 16.4 A Transformer decoder block without cross-attention

### PDF p.451 (printed 427) — 15 Language models and the Transformer › 15.1 The language model › 
To begin generation, we first need to “prime” the internal state of the GRU with our prompt. To do this, we will feed the prompt into the model one token at a time. This will compute the exact RNN state the model would see if this prompt had been encountered during training.

When we feed the very last character of the prompt into the model, our state output will capture information about the entire prompt sequence. We can save the final output prediction to later select the first character of our generated response.

Listing 15.7 Using a fixed prompt to compute a language model’s starting state

### PDF p.461 (printed 437) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
Note that this inference setup, while very simple, is inefficient, since we reprocess the entire source sentence and the entire generated target sentence every time we sample a new word. In a practical application, you’d want to be careful not to recompute any state that has not changed. All we really need to predict a new token in the decoder is the current token and the previous RNN state, which we could cache before each loop iteration. There are many ways this toy model could be improved. We could use a deep stack of recurrent layers for both the encoder and the decoder, we could try other RNN layers like `LSTM`, and so on. Beyond such tweaks, however, the RNN approach to sequence-to-sequence learning has a few fundamental limitations:

The source sequence representation has to be held entirely in the encoder state vector, which significantly limits the size and complexity of the sentences you can translate.

RNNs have trouble dealing with very long sequences since they tend to progressively forget about the past—by the time you’ve reached the 100th token in either sequence, little information remains about the start of the sequence.

Recurrent neural networks dominated sequenc

### PDF p.463 (printed 439) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 15.3.1 Dot-product attention
Let’s revisit our translation RNN and try to add the notion of selective attention. Consider predicting just a single token. After passing the `source` and `target` sequences through our `GRU` layers, we will have a vector representing the target token we are about to predict and a sequence of vectors representing each word in the source text.

With attention, our goal is to give the model a way to score every single vector in our source sequence based on its relevance to the current word we are trying to predict (figure 15.4). If the vector representation of a source token has a high score, we consider it particularly important; if not, we care less about it. For now, let’s assume we have this function `score(target_vector, source_vector)`.

### PDF p.468 (printed 444) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 15.3.2 Transformer encoder block
This is called self-attention, and it is quite powerful. With self-attention, each token can attend to every token in its own sequence, including itself, allowing the model to learn a representation of the word in context.

Consider an example sentence: “The train left the station on time.” Now, consider one word in the sentence: “station.” What kind of station are we talking about? Could it be a radio station? Maybe the International Space Station? With self-attention, the model could learn to give a high attention score to the pair of “station” and “train,” summing the vector used to represent “train” into the representation of the word “station.” Self-attention gives the model an effective way to go from representing a word in a vacuum to representing a word conditioned on all other tokens that appear in the sequence. This sounds a lot like what an RNN is supposed to do. Can we just go ahead and replace our RNN layers with `MultiHeadAttention`? Almost! But not quite; we still need an essential ingredient for any deep neural network—a nonlinear activation function. The `MultiHeadAttention` layer combines linear projections of every element in a source sequence, but that’s it. In 

### PDF p.484 (printed 460) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 15.4.4 Fine-tuning a pretrained Transformer
Before we fine-tune our backbone to predict movie reviews, we need to update it so it outputs a binary classification label. The backbone outputs an entire sequence with shape `(batch_size, sequence_length, 768)`, where each 768-dimensional vector represents an input word in the context of its surrounding words. Before predicting a label, we must condense this sequence to a single vector per sample.

One option would be to do mean pooling or max pooling across the whole sequence, computing an average of all token vectors. What works slightly better is simply using the first token’s representation as the pooled value. This is due to the nature of the attention in our model—the first position in the final encoder layer will be able to attend to all other positions in the sequence and pull information from them. So rather than pooling information with something coarse, like taking an average, attention allows us to pool information contextually across the sequence. Let’s now add a classification head to our backbone. We will also add one final `Dense` projection with a nonlinearity before generating an output prediction.
