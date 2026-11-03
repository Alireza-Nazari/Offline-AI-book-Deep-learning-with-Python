# Generation bundle for page_513  (PDF page 513, printed page 489)

Page kind: content
Chapter: 16 Text generation
Section: 16.3 Using a pretrained LLM
Subsection: 16.3.2 Instruction fine-tuning
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_513.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_513.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p513_e03] (code)
```
        features = json.loads(line)
        if features["context"]:
            continue
        data["prompts"].append(PROMPT_TEMPLATE.format(features["instruction"]))
        data["responses"].append(RESPONSE_TEMPLATE.format(features["response"]))
```
[p513_e04] (body) Note that some examples have additional context—textual information related to the instruction. To keep things simple for now, we will discard those examples.
[p513_e05] (body) Let’s take a look at a single element in our dataset:
[p513_e06] (code)
```
>>> data["prompts"][0]
[instruction]
Which is a species of fish? Tope or Rope[end]
[response]
>>> data["responses"][0]
Tope[end]
```
[p513_e07] (body) Our prompt template gives our examples a predictable structure. Although Gemma is not a sequence-to-sequence model like our English-to-Spanish translator, we can still use it in a sequence-to-sequence setting by training on prompts like this and only generating the output after the `"[response]"` marker.
[p513_e08] (body) Let’s make a `tf.data.Dataset` and split some validation data:
[p513_e09] (code)
```
ds = tf.data.Dataset.from_tensor_slices(data).shuffle(2000).batch(2)
val_ds = ds.take(100)
train_ds = ds.skip(100)
```
[p513_e10] (body) The `CausalLM` we loaded from the KerasHub library is a high-level object for end-to-end causal language modeling. It wraps two objects: a `preprocessor` layer, which preprocesses text input, and a `backbone` model, which contains the numerics of the model forward pass.
[p513_e11] (body) Preprocessing is included by default in high-level Keras functions like `fit()` and `predict()`. But let’s run our preprocessing on a single batch so we can better see what it is doing:
[p513_e12] (code)
```
>>> preprocessor = gemma_lm.preprocessor
>>> preprocessor.sequence_length = 512
>>> batch = next(iter(train_ds))
>>> x, y, sample_weight = preprocessor(batch)
>>> x["token_ids"].shape
(2, 512)
>>> x["padding_mask"].shape
```

## Equation candidates (formulas written inline in the text)
- [p513_e09] ds = tf.data.Dataset.from_tensor_slices(data).shuffle(2000).batch(2)
val_ds = ds.take(100)
train_ds = ds.skip(100)

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

## Previous page (PDF 512) text, for continuity
Of course, this is utter nonsense, but the model could not find a more likely way to complete this prompt.

Hallucinations and uncontrollable output are fundamental problems with language models. If there is a silver bullet, we have yet to find it. However, one approach that helps immensely is to further fine-tune a model with examples of the specific types of generative outputs you would like. In the specific case of wanting to build a chatbot that can follow instructions, this type of training is called instruction fine-tuning. Let’s try some instruction fine-tuning with Gemma to make it a lot more useful as a conversation partner.

16.3.2 Instruction fine-tuning

Instruction fine-tuning involves feeding the model input/output pairs—a user instruction followed by a model response. We combine these into a single sequence that becomes new training data for the model. To make it clear during training when an instruction or response ends, we can add special markers like `"[instruction]"` and `"[response]"` directly to the combined sequence. The precise markup will not matter much as long as it is consistent.

We can use the combined sequence as regular training data, with the same “guess the next word” loss we used to pretrain an LLM. By doing further training with examples containing desired responses, we are essentially bending the model’s output in the direction we want. We won’t be learning a latent space for language here; that’s already been done over trillions of tokens of pretraining. We are simply nudging the learned representation a bit to control the tone and content of the output. To begin, we will need a dataset of instruction-response pairs. Training chatbots is a hot topic, so there are many datasets made specifically for this purpose. We will use a dataset …

## Next page (PDF 514) text, for continuity
The preprocessor layer will pad all inputs to a fixed length and compute a padding mask to track which token ID inputs are just padded zeros. The `sample_weight` tensor allows us to only compute a loss value for our response tokens. We don’t really care about the loss for the user prompt; it is fixed, and we definitely don’t want to compute the loss for the zero padding we just added.

If we print a snippet of our token IDs and labels, we can see that this is the regular language model setup, where each label is the next token value:

16.3.3 Low-Rank Adaptation (LoRA)

If we ran `fit()` right now on a Colab GPU with 16 GB of device memory, we would quickly trigger an out of memory error. But we’ve already loaded the model and run generation, so why would we run out of memory now?

Our 1-billion-parameter model takes up about 3.7 GB of memory. You can see it in our previous model summary. The `Adam` optimizer we have been using will need to track three extra floating-point numbers for each parameter—the actual gradients, a velocity value, and a momentum value. All told, it comes out to 15 GB just for the weights and optimizer state. We also need a few gigabytes of memory to keep track of intermediate values in the forward pass of the model, but we have none left to spare. Running `fit()` would crash on the first train step. This is a common problem when training LLMs. Because these models have large parameter counts, the throughput of your GPUs and CPUs is a secondary concern to fitting the model on accelerator memory. We’ve seen earlier in this book how we can freeze certain parts of a model during fine-tuning. What we did not mention is that this will save a lot of memory! We do not need to track any optimizer variables for frozen parameters—they will never update. Thi …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.518 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.497 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.520 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.495 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.430 — 14 Text classification › 14.5 Sequence models
- PDF p.447 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.503 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.524 — 16 Text generation › 16.4 Going further with LLMs

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.509 (printed 485) — 16 Text generation › 16.3 Using a pretrained LLM › Accessing Gemma weights
`CausalLM` is another example of the high-level task API, much like the `ImageClassifier` and `ImageSegmenter` tasks we used earlier in the book. The `CausalLM` task will combine a tokenizer and correctly initialized architecture into a single Keras model. KerasHub will load the Gemma weights into a correctly initialized architecture and load a matching tokenizer for the pretrained weights.

Let’s take a look at the Gemma model summary:

### PDF p.14 (printed xiv) — contents ›  › 
14.5

Sequence models 405

Training a recurrent model 406 ■ Understanding word embeddings 409 ■ Using a word embedding 410 ■ Pretraining a word embedding 414 ■ Using the pretrained embedding for classification 418

15

Language models and the Transformer 421 15.1

The language model 421

Training a Shakespeare language model 422 ■ Generating Shakespeare 426

15.2

Sequence-to-sequence learning 428

English-to-Spanish translation 430 ■ Sequence-to-sequence learning with RNNs 432

15.3

The Transformer architecture 437

Dot-product attention 439 ■ Transformer encoder block 444 Transformer decoder block 446 ■ Sequence-to-sequence learning with a Transformer 448 ■ Embedding positional information 451

15.4

Classification with a pretrained Transformer 454

Pretraining a Transformer encoder 454 ■ Loading a pretrained Transformer 455 ■ Preprocessing IMDb movie reviews 458 Fine-tuning a pretrained Transformer 460

15.5

What makes the Transformer effective? 461 16 Text generation 466 16.1 A brief history of sequence generation 468 16.2 Training a mini-GPT 470

Building the model 473 ■ Pretraining the model 476 Generative decoding 478 ■ Sampling strategies 480

16.3

Using a pretrained LLM

### PDF p.484 (printed 460) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 15.4.4 Fine-tuning a pretrained Transformer
Before we fine-tune our backbone to predict movie reviews, we need to update it so it outputs a binary classification label. The backbone outputs an entire sequence with shape `(batch_size, sequence_length, 768)`, where each 768-dimensional vector represents an input word in the context of its surrounding words. Before predicting a label, we must condense this sequence to a single vector per sample.

One option would be to do mean pooling or max pooling across the whole sequence, computing an average of all token vectors. What works slightly better is simply using the first token’s representation as the pooled value. This is due to the nature of the attention in our model—the first position in the final encoder layer will be able to attend to all other positions in the sequence and pull information from them. So rather than pooling information with something coarse, like taking an average, attention allows us to pool information contextually across the sequence. Let’s now add a classification head to our backbone. We will also add one final `Dense` projection with a nonlinearity before generating an output prediction.

### PDF p.521 (printed 497) — 16 Text generation › 16.4 Going further with LLMs › Choosing a dtype for large models
You might have noticed we passed `dtype="float32"` when creating our Gemma model the first time and `dtype="bfloat16"` now. What is going on? For models like Gemma with more than a billion parameters, the number of bytes used for each floating-point number is an important consideration. When you are training a model, it is often a good idea to use 32 bits (4 bytes) per parameter. 32-bit floats can represent very small values, which can help keep training gradients stable. Here we aren’t doing any training, so we pass `bfloat16`, which uses only 2 bytes per parameter. We don’t need to worry about gradient stability, and we will save many gigabytes of memory by using a lower precision. There’s a detailed discussion of floating-point precision coming up in chapter 18.

Like the earlier Gemma model we fine-tuned ourselves, this instruction-tuned checkpoint comes with a specific template for formatting its input. Again, the exact text does not matter, what is important is that our prompt template matches what was used to tune the model:

### PDF p.524 (printed 500) — 16 Text generation › 16.4 Going further with LLMs › 
Each of our input prompts contains the special token `<start_of_image>`. This is turned into 256 placeholder values in our input sequence, which, in turn, is replaced with the soft tokens representing our image.

Training for a multimodal model like this is quite similar to regular LLM pretraining and fine-tuning. Usually, you would want to first pretrain your image encoder separately, like we first did in Chapter 8 of this book. Then you can simply do the same basic “guess the next word” pretraining and also feed in mixed image and text content combined into a single sequence. Our transformer would not be trained to output image soft tokens; we would simply zero the loss at these image token locations. It might seem almost magical that we can simply add image data to an LLM, but when we consider the power of the sequence model we’re working with, it’s really quite an expected result. We’ve taken a Transformer, recast our image input as sequence data, and done a lot of extra training. The model can preserve the original language model’s ability to ingest and produce text while learning to also embed images in the Transformer’s latent space.

### PDF p.509 (printed 485) — 16 Text generation › 16.3 Using a pretrained LLM › Accessing Gemma weights
As LLMs become increasingly powerful, terms of service like this are becoming more and more common. The Gemma terms of use prohibit using the model for things like generating spam or hate speech.

Listing 16.11 Instantiating a pretrained LLM with KerasHub

### PDF p.528 (printed 504) — 16 Text generation › 16.5 Where are LLMs heading next? › 
In the first attempt, our model was hung up on the superfluous detail that each letter has two pages. In the second attempt, the model gets the problem right. This instruction-tuned Gemma model we are working with has already been tuned on math problems like this; you would not get nearly as good results from the “untuned” Gemma model from the last section.

We could extend this idea to a very simple form of chain-of-thought training: 1 Collect a bunch of basic math and reasoning problems and desired answers. 2 Generate (with some randomness) a number of responses. 3 Find all the responses with a correct answer via string parsing. You can prompt

the model to use a specific text marker for the final answer as we did previously. 4 Run supervised fine-tuning on correct responses, including all the intermediate output. 5 Repeat! The previously described process is a reinforcement learning algorithm. Our answer checking acts as the environment, and the generated outputs are the actions the model uses to learn. As with RLHF, in practice you would use a more complex gradient update step to use information from all responses (even the incorrect ones), but the basic principle is the same. 

### PDF p.450 (printed 426) — 15 Language models and the Transformer › 15.1 The language model › 15.1.2 Generating Shakespeare
Now that we have trained a model that can predict the next individual tokens with some accuracy, we would like to use it to extrapolate an entire predicted sequence. We can do this by calling the model in a loop, where the model’s predicted output at one time step becomes the model’s input at the next time step. A model built for this kind of feedback loop is sometimes called an autoregressive model.

To run such a loop, we need to perform a slight surgery on the model we just trained. During training, our model handled only a fixed sequence length of 100 tokens, and the `GRU` cell’s state was handled implicitly when calling the layer. During generation, we would like to predict a single output token at a time and explicitly output the state of the `GRU`’s cell. We need to propagate that state, which contains all information the model has encoded about past input characters, the next time we call the model. Let’s make a model that handles a single input character at a time and allows explicitly passing the RNN state. Because this model will have the same computational structure, with slightly modified inputs and outputs, we can assign weights from one model to another.

Listing 15.
