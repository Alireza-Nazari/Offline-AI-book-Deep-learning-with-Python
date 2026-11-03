# Generation bundle for page_516  (PDF page 516, printed page 492)

Page kind: content
Chapter: 16 Text generation
Section: 16.3 Using a pretrained LLM
Subsection: 16.3.3 Low-Rank Adaptation (LoRA)
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_516.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_516.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p516_e03] (figure_caption) Figure 16.4 The low-rank kernel decomposition contains far fewer parameters than the kernel itself.
[p516_e04] (heading_subsection) Customizing LoRA training
[p516_e05] (sidebar) The `enable_lora()` method is also available on individual `Dense` layers. We could equivalently write the previous call a little more verbosely by iterating through the layers of the Transformer:
[p516_e06] (code)
```
gemma_lm.backbone.trainable = False
for i in range(gemma_lm.backbone.num_layers):
    layer = gemma_lm.backbone.get_layer(f"decoder_block_{i}")
    layer.attention.key_dense.trainable = True
    layer.attention.key_dense.enable_lora(rank=8)
    layer.attention.query_dense.trainable = True
    layer.attention.query_dense.enable_lora(rank=8)
```
[p516_e07] (sidebar) With this approach, we could add more trainable parameters earlier or later in the model and also add LoRA to the value projection.
[p516_e08] (body) Let’s look at our model summary again:
[p516_e09] (code)
```
>>> gemma_lm.summary()
Preprocessor: "gemma3_causal_lm_preprocessor"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                                 ┃                        Config ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ gemma3_tokenizer (Gemma3Tokenizer)           │           Vocab size: 262,144 │
└──────────────────────────────────────────────┴───────────────────────────────┘
Model: "gemma3_causal_lm"
```

## Code annotations printed next to listings (callouts, in order)
- Sets all layers to
- not be trainable
- Makes key and query
- projections trainable
- and enables LoRA

## Figures
- Figure 16.4: The low-rank kernel decomposition contains far fewer parameters than the kernel itself.  | labels inside the figure: Alpha, (hidden_size, rank), Pretrained weights, (hidden_size, hidden_size), Beta, (rank, hidden_size), Inputs, Outputs

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

## Previous page (PDF 515) text, for continuity
In 2021, researchers at Microsoft proposed a technique called LoRA, short for Low-Rank Adaptation of Large Language Models, specifically to solve this memory issue.4 To explain it, let’s imagine a simple linear projection layer:

The LoRA paper proposes freezing the `kernel` matrix and adding a new “low rank” decomposition of the kernel projection. This decomposition has two new projection matrices, `alpha` and `beta`, which project to and from an inner `rank`. Let’s take a look:

If our `kernel` is shape 2048 × 2048, that is 4,194,304 frozen parameters. But if we keep the `rank` low, say, 8, we will have only 32,768 parameters for the low-rank decomposition. This update will not have the same expressive power as the original kernel; at the narrow middle point, the entire update must be represented as eight floats. But during LLM fine-tuning, you no longer need the expressive power you needed during pretraining (figure 16.4).

The LoRA authors suggest freezing the entire Transformer and adding LoRA weights to only the query and key projections in the attention layer. Let’s try that out. KerasHub models have a built-in method for LoRA training.

Listing 16.13 Enabling LoRA training for a KerasHub model

4 J. Edward Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” arXiv (2021), https://arxiv.org/ abs/2106.09685.

## Next page (PDF 517) text, for continuity
Although our model parameters still occupy 3.7 GB of space, our trainable parameters now use only 5 MB of data—a thousandfold decrease! This can take our optimizer state from many gigabytes to just megabytes on the GPU (figure 16.5).

Figure 16.5 LoRA greatly reduces the memory we need for gradients and optimizer states.

With this optimization in place, we are at last ready to instruction-tune our Gemma model. Let’s give it a go.

Listing 16.14 Fine-tuning a pretrained LLM

After training, we get to 55% accuracy when guessing the next word in our model’s response. That’s a huge jump from the 35% accuracy of our mini-GPT model. This shows the power of a larger model and more extensive pretraining.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.510 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.509 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.514 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.436 — 14 Text classification › 14.5 Sequence models
- PDF p.499 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.424 — 14 Text classification › 14.4 Set models
- PDF p.500 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.483 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.531 (printed 507) — 16 Text generation › 16.5 Where are LLMs heading next? › 
LLM output is unreliable, and all LLMs will occasionally hallucinate factually incorrect information.

LLMs can be fine-tuned to follow instructions in a chat dialog. This type of

fine-tuning is called instruction fine-tuning:

– The simplest form of instruction fine-tuning involves directly training the

model on instruction and response pairs. – More advanced forms of instruction fine-tuning involve reinforcement learning.

The most common resource bottleneck when working with LLMs is accelerator memory.

LoRA is a technique to reduce memory usage by freezing most Transformer parameters and only updating a low-rank decomposition of attention projection weights.

LLMs can input or output data from different modalities if you can figure out how to frame these inputs or outputs as sequences in a sequence prediction problem.

A foundation model is a general term for models of any modality trained using self-supervision for a wide range of downstream tasks.

### PDF p.239 (printed 215) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Training vs. inference
In the low-level training loop examples you’ve seen so far, step 1 (the forward pass) was done via `predictions = model(inputs)`, and step 2 (retrieving the gradients computed by the gradient tape) was done via a backend-specific API, such as

`gradients = tape.gradient(loss, model.weights)` in TensorFlow

`loss.backward()` in PyTorch

`jax.value_and_grad()` in JAX

In the general case, there are actually two subtleties you need to take into account.

Some Keras layers, such as the `Dropout` layer, have different behaviors during training and during inference (when you use them to generate predictions). Such layers expose a `training` Boolean argument in their `call()` method. Calling `dropout(inputs, ` `training=True)` will drop some activation entries, while calling `dropout(inputs, ` `training=False)` does nothing. By extension, Functional models and Sequential models also expose this `training` argument in their `call()` methods. Remember to pass `training=True` when you call a Keras model during the forward pass! Our forward pass thus becomes `predictions = model(inputs, training=True)`. In addition, note that when you retrieve the gradients of the weights of your model, you s

### PDF p.500 (printed 476) — 16 Text generation › 16.2 Training a mini-GPT › 16.2.2 Pretraining the model
Training a large Transformer is famously finicky—the model is sensitive to initializations of parameters and choice of optimizer. When many Transformer layers are stacked, it is easy to suffer from exploding gradients, where parameters update too quickly and our loss function does not converge. A trick that works well is to linearly ease into a full learning rate over a number of warmup steps, so our initial updates to our model parameters are small. This is easy to implement in Keras with a `LearningRateSchedule`.

Listing 16.7 Defining a custom learning rate schedule

### PDF p.514 (printed 490) — 16 Text generation › 16.3 Using a pretrained LLM › 16.3.3 Low-Rank Adaptation (LoRA)
If we ran `fit()` right now on a Colab GPU with 16 GB of device memory, we would quickly trigger an out of memory error. But we’ve already loaded the model and run generation, so why would we run out of memory now?

Our 1-billion-parameter model takes up about 3.7 GB of memory. You can see it in our previous model summary. The `Adam` optimizer we have been using will need to track three extra floating-point numbers for each parameter—the actual gradients, a velocity value, and a momentum value. All told, it comes out to 15 GB just for the weights and optimizer state. We also need a few gigabytes of memory to keep track of intermediate values in the forward pass of the model, but we have none left to spare. Running `fit()` would crash on the first train step. This is a common problem when training LLMs. Because these models have large parameter counts, the throughput of your GPUs and CPUs is a secondary concern to fitting the model on accelerator memory. We’ve seen earlier in this book how we can freeze certain parts of a model during fine-tuning. What we did not mention is that this will save a lot of memory! We do not need to track any optimizer variables for frozen parameters—the

### PDF p.219 (printed 195) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › 
 Trainable params: 256 (1.00 KB)
 Non-trainable params: 0 (0.00 B)
>>> model.add(layers.Dense(10, activation="softmax"))
>>> model.summary()
Model: "sequential_2"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_4 (Dense)                   │ (None, 64)               │           256 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ dense_5 (Dense)                   │ (None, 10)               │           650 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
 Total params: 906 (3.54 KB)
 Trainable params: 906 (3.54 KB)
 Non-trainable params: 0 (0.00 B)

### PDF p.469 (printed 445) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
to a single matrix multiplication! That’s a real problem with the expressiveness of our model.

At some point, all recurrent cells pass the input vector for each token through a dense projection and apply an activation function; we need a plan for something similar. The authors of “Attention is all you need” decided to add this back in the simplest way possible—stacking a feedforward network of two dense layers with an activation in the middle. Attention passes information across the sequence, and the feedforward network updates the representation of individual sequence items. We are ready to start building a Transformer model. Let’s start by replacing the encoder of our translation model. We will use self-attention to pass information along the source sequence of English words. We will also add in two things we learned to be particularly important when building ConvNets back in chapter 9, normalization and residual connections.

Listing 15.14 A Transformer encoder block

### PDF p.638 (printed 614) — index ›  › K
classification with pretrained Transformer

454–461 Kaggle 607 dot-product attention 439 kagglehub package 242, 485 embedding positional information 451 kaleidoscope hypothesis 577 generating Shakespeare 426–428 Keras 60, 190 loss function 462 building models in 192 seq2seq learning 428–437 ecosystem 608 sequence-to-sequence learning with evaluation loops 206–214 Transformer 449 fit() function 222– 225 training Shakespeare language model 422–426 Functional API 195, 197–200 Transformer 444, 447, 462–464 implementing VAEs with 513–518 Transformer architecture 437 low-level usage of metrics 221 last_conv_layer_output 302 mixing and matching different components 204 latent space 509, 568 relationship between frameworks 63 of images 509 Sequential model 192 subclassing Model class 202 of text-to-image models 533–537 summary 230 visualizing 306 training and evaluation loops 214 lateral connections 347 training loops 206–214 Layer API 191 training step functions 216–218 layered representations learning 7 using right tool for job 205 LayerNormalization layer 445 workflows 191 layer_output layer 167 Keras API, recurrent layers in 368–372 layers 596 keras.callbacks.Callback class 210 LayoutM

### PDF p.91 (printed 67) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › 
Here’s an equivalent of the `Dense` layer we saw in chapter 2:
