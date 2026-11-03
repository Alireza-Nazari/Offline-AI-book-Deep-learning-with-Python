# Generation bundle for page_500  (PDF page 500, printed page 476)

Page kind: content
Chapter: 16 Text generation
Section: 16.2 Training a mini-GPT
Subsection: 16.2.2 Pretraining the model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_500.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_500.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p500_e03] (code)
```
num_heads = 8
num_layers = 8
inputs = keras.Input(shape=(None,), dtype="int32", name="inputs")
embedding = PositionalEmbedding(sequence_length, vocab_size, hidden_dim)
x = embedding(inputs)
x = layers.LayerNormalization()(x)
for i in range(num_layers):
    x = TransformerDecoder(hidden_dim, intermediate_dim, num_heads)(x)
outputs = embedding(x, reverse=True)
mini_gpt = keras.Model(inputs, outputs)
```
[p500_e04] (body) This model has 41 million parameters, which is large for models in this book but quite small compared to most LLMs today, which range from a couple of billion to trillions of parameters.
[p500_e05] (heading_subsection) 16.2.2 Pretraining the model
[p500_e06] (body) Training a large Transformer is famously finicky—the model is sensitive to initializations of parameters and choice of optimizer. When many Transformer layers are stacked, it is easy to suffer from exploding gradients, where parameters update too quickly and our loss function does not converge. A trick that works well is to linearly ease into a full learning rate over a number of warmup steps, so our initial updates to our model parameters are small. This is easy to implement in Keras with a `LearningRateSchedule`.
[p500_e07] (listing_caption) Listing 16.7 Defining a custom learning rate schedule
[p500_e08] (code)
```
class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self):
        self.rate = 2e-4
        self.warmup_steps = 1_000.0
    def __call__(self, step):
        step = ops.cast(step, dtype="float32")
        scale = ops.minimum(step / self.warmup_steps, 1.0)
        return self.rate * scale
```
[p500_e09] (body) We can plot our learning rate over time to make sure it is what we expect (figure 16.2):
[p500_e10] (code)
```
import matplotlib.pyplot as plt
schedule = WarmupSchedule()
x = range(0, 5_000, 100)
y = [ops.convert_to_numpy(schedule(step)) for step in x]
plt.plot(x, y)
plt.xlabel("Train Step")
plt.ylabel("Learning Rate")
plt.show()
```

## Code annotations printed next to listings (callouts, in order)
- Peak learning rate

## Listings
- Listing 16.7: Defining a custom learning rate schedule (code elements p500_e08)

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

## Previous page (PDF 499) text, for continuity
prediction layer because they deal with our vocabulary space. The token embedding weight has shape `(vocab_size, hidden_dim)` to embed every possible token. Our output projection has shape `(hidden_dim, vocab_size)` to make a floating-point prediction for every possible token.

We can actually tie these two weight matrices together. To compute our model’s final predictions, we will multiply our hidden states by the transpose of our token embedding matrix. You can very much think of our final projection as a “reverse embedding.” It maps from hidden space to token space, whereas an embedding maps from token space to hidden space. It turns out that using the same weights for this input and output projection is a good idea. Adding this to our `PositionalEmbedding` is simple; we will just add a `reverse` argument to the `call` method, which computes the projection by the transpose of the token embedding.

Listing 16.5 A positional embedding layer that can reverse a text embedding

Let’s build our model. We will stack eight decoder layers into a single “mini” GPT model.

We will also turn on a Keras setting called mixed precision to speed up training. This will allow Keras to run some of the model’s computations much faster by sacrificing some numerical fidelity. For now, this will remain a little mysterious, but a full explanation is waiting in chapter 18.

Listing 16.6 Creating a mini-GPT functional model

## Next page (PDF 501) text, for continuity
Figure 16.2 Warmup makes our updates to model parameters smaller at the beginning of training and can help with stability.

We will train our model using one pass over our 1 billion tokens, split across eight epochs so we can occasionally check our validation set loss and accuracy.

We are training a miniature version of GPT, using 3× fewer parameters than GPT-1 and 100× fewer overall training steps. But despite this being two orders of magnitude cheaper to train than the smallest GPT model, this call to `fit()` will be the most computationally expensive training run in the entire book. If you are running the code as you read, set things off and take a breather!

Listing 16.8 Training the mini-GPT model

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.431 — 14 Text classification › 14.5 Sequence models
- PDF p.450 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.498 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.476 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.178 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.628 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.515 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.424 — 14 Text classification › 14.4 Set models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.493 (printed 469) — 16 Text generation › 16.1 A brief history of sequence generation › 
In 2018, a year after the “Attention Is All You Need” paper we discussed in the last chapter, a group of researchers at an organization called OpenAI put out a new paper “Improving Language Understanding by Generative Pre-Training.”3 They combined a few ingredients:

Unsupervised pretraining of a language model—essentially training a model to “guess the next token” in a sequence, as we did with our Shakespeare generator in chapter 15

The Transformer architecture

Textual data on various topics via thousands of self-published books

The authors showed that such a pretrained model could be fine-tuned to achieve state-of-the-art performance on a wide array of text classification tasks—from gauging the similarity of two sentences to answering a multiple-choice question. They called the pretrained model GPT, short for Generative Pretrained Transformer.

GPT didn’t come with any modeling or training advancements. What was interesting about the results was that such a general training setup could beat out more involved techniques across a number of tasks. There was no complex text normalization, no need to customize the model architecture or training data per benchmark, just a lot of pre

### PDF p.515 (printed 491) — 16 Text generation › 16.3 Using a pretrained LLM › 
If our `kernel` is shape 2048 × 2048, that is 4,194,304 frozen parameters. But if we keep the `rank` low, say, 8, we will have only 32,768 parameters for the low-rank decomposition. This update will not have the same expressive power as the original kernel; at the narrow middle point, the entire update must be represented as eight floats. But during LLM fine-tuning, you no longer need the expressive power you needed during pretraining (figure 16.4).

The LoRA authors suggest freezing the entire Transformer and adding LoRA weights to only the query and key projections in the attention layer. Let’s try that out. KerasHub models have a built-in method for LoRA training.

Listing 16.13 Enabling LoRA training for a KerasHub model

### PDF p.478 (printed 454) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 15.4 Classification with a pretrained Transformer
After “Attention is all you need,” people started to notice how far Transformer training could scale, especially compared to models that had come before. As we just mentioned, one big plus was that the model is faster to train than RNNs. No more loops during training, which is always good when working with a GPU or TPU.

It is also a very data hungry model architecture. We actually got a little taste of this in the last section. While our RNN translation model plateaued in validation performance after 5 or so epochs, the Transformer model was still improving its validation score after 30 epochs of training. These observations prompted many to try scaling up the Transformer with more data, layers, and parameters—with great results. This caused a distinctive shift in the field toward large pretrained models that can cost millions to train but perform noticeably better on a wide range of problems in the text domain. For our last code example in the text section, we will revisit our IMDb text-classification problem, this time with a pretrained Transformer model.

### PDF p.178 (printed 154) — 5 Fundamentals of machine learning › 5.3 Improving model fit › 
data. Even if nothing about your problem makes sense, you should still be able to train something—if only by memorizing the training data.

When this happens, it’s always a problem with the configuration of the gradient descent process: your choice of optimizer, the distribution of initial values in the weights of your model, your learning rate, or your batch size. All these parameters are interdependent, and as such, it is usually sufficient to tune the learning rate and the batch size while maintaining the rest of the parameters constant. Let’s look at a concrete example: let’s train the MNIST model from chapter 2 with an inappropriately large learning rate, of value 1.

Listing 5.7 Training an MNIST model with an incorrectly high learning rate

### PDF p.528 (printed 504) — 16 Text generation › 16.5 Where are LLMs heading next? › 16.5 Where are LLMs heading next?
Given the trajectory of LLMs discussed at the beginning of this chapter, it may seem obvious where LLMs will be heading. More parameters! Even better performance! In a general sense, that’s probably correct, but our trajectory might not be quite so linear.

If you have a fixed budget for pretraining, say, a million dollars, you can roughly think of it as buying you a fixed amount of compute or floating-point operations (flops). You can either spend those flops on training with more data or training a bigger model. Recent research has pointed out that GPT-3, at 175 billion parameters, was way too big for its computing budget. Training a smaller model on more data would have led to

### PDF p.508 (printed 484) — 16 Text generation › 16.3 Using a pretrained LLM › 16.3 Using a pretrained LLM
Now that we’ve trained a mini-language model from scratch, let’s try using a billion-parameter pretrained model and see what it can do. Given how prohibitively expensive pretraining a Transformer can be, most of the industry has centered around using pretrained models developed by a relatively short list of companies. This is not purely a cost concern but also an environmental one—generative model training is now making up a large percentage of the total data center power consumption of large tech companies.

Meta published some environmental data on Llama 2, an LLM it published in 2023. It’s a good bit smaller than GPT-3, but it needed an estimated 1.3 million kilowatt hours of electricity to train—the daily power usage of about 45,000 American households. If every organization using an LLM ran pretraining themselves, the scale of energy use would be a noticeable percentage of global energy consumption. Let’s play around with a pretrained generative model from Google called Gemma. We will use the third version of the Gemma model, which was released to the public in 2025. To keep the examples in this book accessible, we will use the smallest variation of

### PDF p.514 (printed 490) — 16 Text generation › 16.3 Using a pretrained LLM › 16.3.3 Low-Rank Adaptation (LoRA)
If we ran `fit()` right now on a Colab GPU with 16 GB of device memory, we would quickly trigger an out of memory error. But we’ve already loaded the model and run generation, so why would we run out of memory now?

Our 1-billion-parameter model takes up about 3.7 GB of memory. You can see it in our previous model summary. The `Adam` optimizer we have been using will need to track three extra floating-point numbers for each parameter—the actual gradients, a velocity value, and a momentum value. All told, it comes out to 15 GB just for the weights and optimizer state. We also need a few gigabytes of memory to keep track of intermediate values in the forward pass of the model, but we have none left to spare. Running `fit()` would crash on the first train step. This is a common problem when training LLMs. Because these models have large parameter counts, the throughput of your GPUs and CPUs is a secondary concern to fitting the model on accelerator memory. We’ve seen earlier in this book how we can freeze certain parts of a model during fine-tuning. What we did not mention is that this will save a lot of memory! We do not need to track any optimizer variables for frozen parameters—the

### PDF p.509 (printed 485) — 16 Text generation › 16.3 Using a pretrained LLM › 
Gemma available, which clocks in at almost exactly 1 billion parameters. This “small” model was trained on roughly 2 trillion tokens of pretraining data—2,000 times more tokens than the mini-GPT we just trained!
