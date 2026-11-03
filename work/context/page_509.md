# Generation bundle for page_509  (PDF page 509, printed page 485)

Page kind: content
Chapter: 16 Text generation
Section: 16.3 Using a pretrained LLM
Subsection: 16.3.1 Text generation with the Gemma model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_509.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_509.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p509_e03] (body) Gemma available, which clocks in at almost exactly 1 billion parameters. This “small” model was trained on roughly 2 trillion tokens of pretraining data—2,000 times more tokens than the mini-GPT we just trained!
[p509_e04] (heading_subsection) 16.3.1 Text generation with the Gemma model
[p509_e05] (body) To load this pretrained model, we can use KerasHub, as we have done in previous chapters.
[p509_e06] (heading_subsection) Accessing Gemma weights
[p509_e07] (sidebar) If you are running the code for this chapter yourself, you will need to accept a Terms of Use for the Gemma models before you can download the weights. The model weights are stored on Kaggle, and we can use the `kagglehub` API to log in as we did in chapter 8. Before we do, you will need to do two things: 1 Go to https://www.kaggle.com/models/keras/gemma3, and accept the Gemma Terms of Use at the top of the page. 2 Go to https://www.kaggle.com/settings, and generate a Kaggle API Key (if you did not already do this in chapter 8). With that, we can use our API key to authenticate with Kaggle from our notebook:
[p509_e08] (code)
```
import kagglehub
kagglehub.login()
```
[p509_e09] (sidebar) As LLMs become increasingly powerful, terms of service like this are becoming more and more common. The Gemma terms of use prohibit using the model for things like generating spam or hate speech.
[p509_e10] (listing_caption) Listing 16.11 Instantiating a pretrained LLM with KerasHub
[p509_e11] (code)
```
gemma_lm = keras_hub.models.CausalLM.from_preset(
    "gemma3_1b",
    dtype="float32",
)
```
[p509_e12] (body) `CausalLM` is another example of the high-level task API, much like the `ImageClassifier` and `ImageSegmenter` tasks we used earlier in the book. The `CausalLM` task will combine a tokenizer and correctly initialized architecture into a single Keras model. KerasHub will load the Gemma weights into a correctly initialized architecture and load a matching tokenizer for the pretrained weights.
[p509_e13] (body) Let’s take a look at the Gemma model summary:
[p509_e14] (code)
```
>>> gemma_lm.summary()
Preprocessor: "gemma3_causal_lm_preprocessor"
```

## Listings
- Listing 16.11: Instantiating a pretrained LLM with KerasHub (code elements p509_e11)

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

## Previous page (PDF 508) text, for continuity
zeros out the probability of anything outside the K candidates. You can combine the two, for example, sampling the top five candidates with a temperature of 0.5:

A sampling strategy is an important control when generating text, and there are many more approaches. For example, beam search is a technique that heuristically explores multiple chains of predicted tokens by keeping a fixed number of “beams” (different chains of predicted tokens) to explore at each timestep.

With top-K sampling, our model generates something closer to plausible English text, but there is little apparent utility to such output. This fits with the results of GPT-1. For the initial GPT paper, the generated output was more of a curiosity, and state-of-the-art results were only achieved by fine-tuning classification models. Our mini-GPT is far less trained than GPT-1. To reach the scale of generative LLMs today, we’d need to increase our parameter count by at least 100× and our train step count by at least 1,000×. If we did, we would see the same leaps in quality observed by OpenAI with GPT. And we could do it! The training recipe we used previously is the exact blueprint used by everyone training LLMs today. The only missing pieces are a very large compute budget and some tricks for training across multiple machines that we will cover in chapter 18. For a more practical approach, we will transition to using a pretrained model. This will allow us to explore the behavior of an LLM at today’s scale.

16.3 Using a pretrained LLM

Now that we’ve trained a mini-language model from scratch, let’s try using a billion-parameter pretrained model and see what it can do. Given how prohibitively expensive pretraining a Transformer can be, most of the industry has centered around using pretrained models devel …

## Next page (PDF 510) text, for continuity
Rather than implementing a generation routine ourselves, we can simplify our lives by using the `generate()` function that comes as part of the `CausalLM` class. This `generate()` function can be compiled with different sampling strategies, as we explored in section 16.2:

We can notice a few things right off the bat. First, the output is much more coherent than our mini-GPT model. It would be hard to distinguish this text from much of the training data in the C4 dataset. Second, the output is still not that useful. The model will generate vaguely plausible text, but what you could do with it is unclear.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.517 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.520 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.521 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.480 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer
- PDF p.496 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.513 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.495 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.511 — 16 Text generation › 16.3 Using a pretrained LLM

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.480 (printed 456) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
reviews map to different token indices than they would have during pretraining, we cannot use the learned representations of each token in the model.

A matching model architecture—To use the pretrained model, we need to recreate the math used internally by the model for pretraining exactly.

The pretrained weights—These weights were created by training the model for about a day on 1,024 GPUs and billions of input words.

Recreating the tokenizer and architecture code ourselves would not be too hard. The model internals almost exactly match the `TransformerEncoder` we built previously. However, matching a model implementation is a time-consuming process, and as we have done earlier in this book, we can instead use the KerasHub library to access pretrained model implementations for Keras.

Let’s use KerasHub to load a RoBERTa tokenizer and model. We can use the special constructor `from_preset()` to load a pretrained model’s weights, configuration, and tokenizer assets from disk. We will load RoBERTa’s base model, which is the smallest of the few pretrained checkpoints released with the RoBERTa paper.

Listing 15.20 Loading the RoBERTa pretrained model with KerasHub

### PDF p.521 (printed 497) — 16 Text generation › 16.4 Going further with LLMs › Choosing a dtype for large models
You might have noticed we passed `dtype="float32"` when creating our Gemma model the first time and `dtype="bfloat16"` now. What is going on? For models like Gemma with more than a billion parameters, the number of bytes used for each floating-point number is an important consideration. When you are training a model, it is often a good idea to use 32 bits (4 bytes) per parameter. 32-bit floats can represent very small values, which can help keep training gradients stable. Here we aren’t doing any training, so we pass `bfloat16`, which uses only 2 bytes per parameter. We don’t need to worry about gradient stability, and we will save many gigabytes of memory by using a lower precision. There’s a detailed discussion of floating-point precision coming up in chapter 18.

Like the earlier Gemma model we fine-tuned ourselves, this instruction-tuned checkpoint comes with a specific template for formatting its input. Again, the exact text does not matter, what is important is that our prompt template matches what was used to tune the model:

### PDF p.520 (printed 496) — 16 Text generation › 16.4 Going further with LLMs › 4.1.1. Using a chatbot trained with RLHF
We can make this more concrete by trying a model trained with this form of iterative preference tuning. Since building chatbots is the “killer app” for large Transformer models, it is common practice for companies that release pretrained models like Gemma to release specialized “instruction-tuned” versions, built just for chat. Let’s try loading one now. This will be a 4-billion-parameter model, quadruple the size of the model we just loaded and the largest model we will use in this book.

Listing 16.16 Loading an instruction-tuned Gemma variant

### PDF p.520 (printed 496) — 16 Text generation › 16.4 Going further with LLMs › 4.1.1. Using a chatbot trained with RLHF
Listing 16.16 Loading an instruction-tuned Gemma variant
gemma_lm = keras_hub.models.CausalLM.from_preset(
    "gemma3_instruct_4b",
    dtype="bfloat16",
)

### PDF p.511 (printed 487) — 16 Text generation › 16.3 Using a pretrained LLM › 
As we saw with the mini-GPT example, this is not so much a bug as a consequence of our pretraining objective. The Gemma model was trained with the same “guess the next word” objective we used for mini-GPT, which means it’s effectively a fancy autocomplete for the internet. It will just keep rattling off the most probable word in its single sequence as if your prompt was a snippet of text found in a random document on the web. One way to change our output is to prompt the model with a longer input that makes it obvious which type of output we are looking for. For example, if we prompt the Gemma model with the beginning two sentences of a brownie recipe, we get more helpful output:

### PDF p.517 (printed 493) — 16 Text generation › 16.3 Using a pretrained LLM › 
Although our model parameters still occupy 3.7 GB of space, our trainable parameters now use only 5 MB of data—a thousandfold decrease! This can take our optimizer state from many gigabytes to just megabytes on the GPU (figure 16.5).

Figure 16.5 LoRA greatly reduces the memory we need for gradients and optimizer states.

With this optimization in place, we are at last ready to instruction-tune our Gemma model. Let’s give it a go.

Listing 16.14 Fine-tuning a pretrained LLM

### PDF p.522 (printed 498) — 16 Text generation › 16.4 Going further with LLMs › 16.4.2 Multimodal LLMs
One obvious chatbot extension is the ability to handle new modalities of input. An assistant that can respond to audio input and process images would be far more useful than one that can only operate on text.

Extending a Transformer to different modalities can be done in a conceptually simple way. The Transformer is not a text-specific model; it’s a highly effective model for learning patterns in sequence data. If we can figure out how to coerce other data types into a sequence representation, we can feed this sequence into a Transformer and train with it. In fact, the Gemma model we just loaded does just that. The model comes with a separate 420-million-parameter image encoder that cuts an input image into 256 patches and encodes each patch as a vector with the same dimensionality as Gemma’s hidden transformer dimension. Each image will be embedded as a `(256,` `2560)` sequence. Because 2560 is the hidden dimensionality of the Gemma Transformer model, this image representation can simply be spliced into our text sequence after the token embedding layer. You can think of it like 256 special tokens representing the image, where each `(1,` `2560)` vector is sometimes called a “soft 

### PDF p.513 (printed 489) — 16 Text generation › 16.3 Using a pretrained LLM › 
Our prompt template gives our examples a predictable structure. Although Gemma is not a sequence-to-sequence model like our English-to-Spanish translator, we can still use it in a sequence-to-sequence setting by training on prompts like this and only generating the output after the `"[response]"` marker.

Let’s make a `tf.data.Dataset` and split some validation data:
