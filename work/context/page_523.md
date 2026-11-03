# Generation bundle for page_523  (PDF page 523, printed page 499)

Page kind: content
Chapter: 16 Text generation
Section: 16.4 Going further with LLMs
Subsection: 16.4.2 Multimodal LLMs
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_523.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_523.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p523_e03] (code)
```
image = np.array(keras.utils.load_img(image_path))
plt.axis("off")
plt.imshow(image)
plt.show()
```
[p523_e04] (figure_caption) Figure 16.6 Handling image input by splicing text tokens and soft image tokens together Figure 16.7 A test image for the Gemma model

## Figures
- Figure 16.6: Handling image input by splicing text tokens and soft image tokens together Figure 16.7 A test image for the Gemma model  | labels inside the figure: Image, encoder, Transformer, Decoder block, Decoder block, Decoder block, “hard” tokens, “soft” tokens, ..., What, is, <img>, <img>, <img>, <img>, ?, Text, embedding

## Equation candidates (formulas written inline in the text)
- [p523_e03] image = np.array(keras.utils.load_img(image_path))
plt.axis("off")
plt.imshow(image)
plt.show()

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

## Previous page (PDF 522) text, for continuity
This more capable model refuses to take the bait. This is not the result of a new modeling technique, but rather the result of extensive training on trick questions like this one with responses like the one we just received. In fact, you can see clearly here why removing hallucinations can be a bit like playing whack-a-mole—even though it refused to hallucinate a US president, the model now manages to make up today’s date.

16.4.2 Multimodal LLMs

One obvious chatbot extension is the ability to handle new modalities of input. An assistant that can respond to audio input and process images would be far more useful than one that can only operate on text.

Extending a Transformer to different modalities can be done in a conceptually simple way. The Transformer is not a text-specific model; it’s a highly effective model for learning patterns in sequence data. If we can figure out how to coerce other data types into a sequence representation, we can feed this sequence into a Transformer and train with it. In fact, the Gemma model we just loaded does just that. The model comes with a separate 420-million-parameter image encoder that cuts an input image into 256 patches and encodes each patch as a vector with the same dimensionality as Gemma’s hidden transformer dimension. Each image will be embedded as a `(256,` `2560)` sequence. Because 2560 is the hidden dimensionality of the Gemma Transformer model, this image representation can simply be spliced into our text sequence after the token embedding layer. You can think of it like 256 special tokens representing the image, where each `(1,` `2560)` vector is sometimes called a “soft token” (figure 16.6). Unlike our normal “hard tokens,” where each token ID can only take on a fixed number of possible vectors in our token embeddin …

## Next page (PDF 524) text, for continuity
We can use Gemma to ask some questions about this image:

Each of our input prompts contains the special token `<start_of_image>`. This is turned into 256 placeholder values in our input sequence, which, in turn, is replaced with the soft tokens representing our image.

Training for a multimodal model like this is quite similar to regular LLM pretraining and fine-tuning. Usually, you would want to first pretrain your image encoder separately, like we first did in Chapter 8 of this book. Then you can simply do the same basic “guess the next word” pretraining and also feed in mixed image and text content combined into a single sequence. Our transformer would not be trained to output image soft tokens; we would simply zero the loss at these image token locations. It might seem almost magical that we can simply add image data to an LLM, but when we consider the power of the sequence model we’re working with, it’s really quite an expected result. We’ve taken a Transformer, recast our image input as sequence data, and done a lot of extra training. The model can preserve the original language model’s ability to ingest and produce text while learning to also embed images in the Transformer’s latent space.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.545 — 17 Image generation › 17.2 Diffusion models
- PDF p.555 — 17 Image generation › 17.3 Text-to-image models
- PDF p.533 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.556 — 17 Image generation › 17.3 Text-to-image models
- PDF p.558 — 17 Image generation › 17.3 Text-to-image models
- PDF p.503 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.359 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.345 — 11 Image segmentation › 11.3 Using a pretrained segmentation model

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.555 (printed 531) — 17 Image generation › 17.3 Text-to-image models › 17.3 Text-to-image models
We can use the same basic diffusion process to create a model that maps text input to image output. To do this we need to take a pretrained text encoder (think a transformer encoder like RoBERTa from chapter 15) that can map text to vectors in a continuous embedding space. Then we can train a diffusion model on `(prompt,` `image)` pairs, where each prompt is a short, textual description of the input image.

We can handle the image input in the same way as we did previously, mapping noisy input to a denoised output that progressively approaches our input image. Critically, we can extend this setup by also passing the embedded text prompt to the denoising model. So rather than our denoising model simply taking in a `noisy_images` input, our model will take two inputs: `noisy_images` and `text_embeddings`. This gives a leg up on the flower denoiser we trained previously. Instead of learning to remove noise from an image without any additional information, the model gets to use a textual representation of the final image to help guide the denoising process. After training is when things get a bit more fun. Because we have trained a model that can map pure noise to images conditioned on

### PDF p.530 (printed 506) — 16 Text generation › 16.5 Where are LLMs heading next? › Summary
Large language models, or LLMs, are the combination of a few key ingredients: – The Transformer architecture – A language modeling task (predicting the next token based on past tokens) – A large amount of unlabeled text data

An LLM learns a probability distribution for predicting individual tokens. This can be combined with a sampling strategy to generate a long string of text. There are many popular ways to sample text: – Greedy search takes the most likely predicted token at each generation step. – Random sampling directly samples the predicted categorical distribution over all tokens. – Top-K sampling restricts the categorical distribution to the top set of K candidates.

LLMs use billions of parameters and are trained on trillions of words of text.

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

### PDF p.555 (printed 531) — 17 Image generation › 17.3 Text-to-image models › 17.3 Text-to-image models
Hub with just a couple of lines of code.

Diffusion model

Listing 17.8 Creating a Stable Diffusion text-to-image model

### PDF p.509 (printed 485) — 16 Text generation › 16.3 Using a pretrained LLM › Accessing Gemma weights
As LLMs become increasingly powerful, terms of service like this are becoming more and more common. The Gemma terms of use prohibit using the model for things like generating spam or hate speech.

Listing 16.11 Instantiating a pretrained LLM with KerasHub

### PDF p.545 (printed 521) — 17 Image generation › 17.2 Diffusion models › 
Here’s an example image (figure 17.8):

### PDF p.556 (printed 532) — 17 Image generation › 17.3 Text-to-image models › 
Like the `CausalLM` task we covered last chapter, the `TextToImage` task is a high-level class for performing image generation conditioned on text input. It wraps tokenization and the diffusion process into a high-level generate call.

The Stable Diffusion model actually adds a second “negative prompt” to its model, which can be used to steer the diffusion process away from certain text inputs. There’s nothing magic here. To add a negative prompt, you could simply train a model on triplets: `(image,` `positive_
` `prompt,` `negative_prompt)`, where the positive prompt is a description of the image, and the negative prompt is a series of words that do not describe the image. By feeding the positive and negative text embedding to the denoiser, the denoiser will learn to steer the noise toward images that match the positive prompt and away from images that match the negative prompt (figure 17.14). Let’s try removing

Figure 17.14 Using a negative prompt to steer

the color blue from our input:

the model away from the color blue

### PDF p.533 (printed 509) — 17 Image generation › 17.1 Deep learning for image generation › 17.1.1 Sampling from latent spaces of images
Figure 17.2 Language-guided image generation
