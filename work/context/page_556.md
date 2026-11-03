# Generation bundle for page_556  (PDF page 556, printed page 532)

Page kind: content
Chapter: 17 Image generation
Section: 17.3 Text-to-image models
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_556.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_556.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p556_e03] (code)
```
    dtype="float16",
)
prompt = "A NASA astraunaut riding an origami elephant in New York City"
task.generate(prompt)
```
[p556_e04] (body) Like the `CausalLM` task we covered last chapter, the `TextToImage` task is a high-level class for performing image generation conditioned on text input. It wraps tokenization and the diffusion process into a high-level generate call.
[p556_e05] (body) The Stable Diffusion model actually adds a second “negative prompt” to its model, which can be used to steer the diffusion process away from certain text inputs. There’s nothing magic here. To add a negative prompt, you could simply train a model on triplets: `(image,` `positive_
` `prompt,` `negative_prompt)`, where the positive prompt is a description of the image, and the negative prompt is a series of words that do not describe the image. By feeding the positive and negative text embedding to the denoiser, the denoiser will learn to steer the noise toward images that match the positive prompt and away from images that match the negative prompt (figure 17.14). Let’s try removing
[p556_e06] (figure_caption) Figure 17.14 Using a negative prompt to steer
[p556_e07] (body) the color blue from our input:
[p556_e08] (figure_caption) the model away from the color blue
[p556_e09] (code)
```
task.generate(
    {
        "prompts": prompt,
        "negative_prompts": "blue color",
    }
)
```
[p556_e10] (heading_subsection) Visual artifacts in the Stable Diffusion output
[p556_e11] (sidebar) You will notice plenty of visual artifacts in our Stable Diffusion output if you look closely. Notably, our second elephant has duplicated tusks! Some of this is unavoidable when using diffusion models. Figuring out how to truly draw a human in a space suit sitting on an elephant made of paper would require some understanding of anatomy and physics that our model lacks. The model will

## Code annotations printed next to listings (callouts, in order)
- A trick to keep memory usage
- down. More details in chapter 18.

## Figures
- Figure 17.14: Using a negative prompt to steer  | labels inside the figure: (none - image)
- Figure None: the model away from the color blue  | labels inside the figure: (none - image)

## Equation candidates (formulas written inline in the text)
- [p556_e03]     dtype="float16",
)
prompt = "A NASA astraunaut riding an origami elephant in New York City"
task.generate(prompt)

## Chapter outline
Chapter: 17 Image generation (PDF pages 532-561)
  - 17.1 Deep learning for image generation (p.532)
      - 17.1.1 Sampling from latent spaces of images (p.533)
      - 17.1.2 Variational autoencoders (p.534)
      - 17.1.3 Implementing a VAE with Keras (p.537)
  - 17.2 Diffusion models (p.542)
      - 17.2.1 The Oxford Flowers dataset (p.544)
      - 17.2.2 A U-Net denoising autoencoder (p.545)
      - 17.2.3 The concepts of diffusion time and diffusion schedule (p.547)
      - 17.2.4 The training process (p.549)
      - 17.2.5 The generation process (p.551)
      - 17.2.6 Visualizing results with a custom callback (p.552)
      - 17.2.7 It’s go time! (p.553)
  - 17.3 Text-to-image models (p.555)
      - 17.3.1 Exploring the latent space of a text-to-image model (p.557)

## Previous page (PDF 555) text, for continuity
17.3 Text-to-image models

We can use the same basic diffusion process to create a model that maps text input to image output. To do this we need to take a pretrained text encoder (think a transformer encoder like RoBERTa from chapter 15) that can map text to vectors in a continuous embedding space. Then we can train a diffusion model on `(prompt,` `image)` pairs, where each prompt is a short, textual description of the input image.

We can handle the image input in the same way as we did previously, mapping noisy input to a denoised output that progressively approaches our input image. Critically, we can extend this setup by also passing the embedded text prompt to the denoising model. So rather than our denoising model simply taking in a `noisy_images` input, our model will take two inputs: `noisy_images` and `text_embeddings`. This gives a leg up on the flower denoiser we trained previously. Instead of learning to remove noise from an image without any additional information, the model gets to use a textual representation of the final image to help guide the denoising process. After training is when things get a bit more fun. Because we have trained a model that can map pure noise to images conditioned on a vector representation of some text, we can now pass in pure noise and a never-before-seen prompt and denoise it into an image for our prompt. Let’s try this out. We won’t actually train one of these models from scratch in this book—you have all the ingredients you need, but it’s quite expensive and time consuming to train a text-to-image diffusion model that works well. Instead, we will play with a popular pretrained model in KerasHub called Stable Diffusion (figure 17.13). Stable Diffusion is made by a company named Stability AI that specializes in making open mo …

## Next page (PDF 557) text, for continuity
always try its best to interpolate an output based on its training data, but it doesn’t have any real understanding of the objects it is attempting to represent. However, there is another factor that is easily fixable: we are using the less powerful version of Stable Diffusion 3. The “medium” model we are using is the smallest one released by Stability AI and uses about 3 billion parameters in total. There is a larger 9-billion parameter model available that would produce substantially higher-quality images with fewer visual artifacts. We do not use it simply to keep the code example in this book accessible—9 billion parameters need a lot of RAM!

Like the `generate()` method for text models we used in the last chapter, we have a few additional parameters we can pass to control the generation process. Let’s try passing a variable number of diffusion steps to our model to see the denoising process in action (figure 17.15):

Figure 17.15 Controlling the number of diffusion steps

17.3.1 Exploring the latent space of a text-to-image model

There is probably no better way to see the interpolative nature of deep neural networks than text diffusion models. The text encoder used by our model will learn a smooth, low-dimensional manifold to represent our input prompts. It’s continuous, meaning we have learned a space where we can walk from the text representation of one prompt to another, and each intermediate point will have semantic meaning. We can couple that with our diffusion process to morph between two images by simply describing each end state with a text prompt.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.524 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.558 — 17 Image generation › 17.3 Text-to-image models
- PDF p.559 — 17 Image generation › 17.3 Text-to-image models
- PDF p.503 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.554 — 17 Image generation › 17.2 Diffusion models
- PDF p.544 — 17 Image generation › 17.2 Diffusion models
- PDF p.542 — 17 Image generation › 17.2 Diffusion models
- PDF p.547 — 17 Image generation › 17.2 Diffusion models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.559 (printed 535) — 17 Image generation › 17.3 Text-to-image models › 
representation of the entire token sequence learned by the text encoder. This effectively gives our denoising model more information to work with. The authors do this for both the positive and negative prompts, so we have a total of four tensors here:

The positive prompt’s encoder sequence

The negative prompt’s encoder sequence

The positive prompt’s encoder vector

The negative prompt’s encoder vector

With our `generate()` function decomposed, we can now try walking the latent space between two text prompts. To do so, let’s build a function to interpolate between the text embeddings outputted by the model.

Listing 17.10 A function to interpolate text embeddings

### PDF p.561 (printed 537) — 17 Image generation › 17.3 Text-to-image models › Summary
Image generation with deep learning is done by learning latent spaces that capture statistical information about a dataset of images. By sampling and decoding points from the latent space, you can generate never-before-seen images. There are three major tools to do this: VAEs, diffusion models, and GANs.

VAEs result in highly structured, continuous latent representations. For this reason, they work well for doing all sorts of image editing in latent space: face swapping, turning a frowning face into a smiling face, and so on. They also work nicely for doing latent space–based animations, such as animating a walk along a cross section of the latent space, showing a starting image slowly morphing into different images in a continuous way.

Diffusion models result in very realistic outputs and are the dominant method of image generation today. They work by repeatedly denoising an image, starting from pure noise. They can easily be conditioned on text captions to create text-to-image models.

Stable Diffusion 3 is a state-of-the-art pretrained text-to-image model that you can use to create highly realistic images of your own.

The visual latent space learned by such text-to-image diff

### PDF p.558 (printed 534) — 17 Image generation › 17.3 Text-to-image models › 
Rather than only passing the final, embedded text vector to the denoising model, the Stable Diffusion authors chose to pass both the final output vector and the last

### PDF p.15 (printed xv) — contents ›  › 
17

Image generation 508 17.1

Deep learning for image generation 508

Sampling from latent spaces of images 509 ■ Variational autoencoders 510 ■ Implementing a VAE with Keras 513

17.2

Diffusion models 518

The Oxford Flowers dataset 520 ■ A U-Net denoising autoencoder 521 ■ The concepts of diffusion time and diffusion schedule 523 ■ The training process 525 ■ The generation process 527 ■ Visualizing results with a custom callback 528 It’s go time! 529

17.3

Text-to-image models 531

Exploring the latent space of a text-to-image model 533

18

Best practices for the real world 538 18.1

Getting the most out of your models 539

Hyperparameter optimization 539 ■ Model ensembling 546

18.2

Scaling up model training with multiple devices 548

Multi-GPU training 548 ■ Distributed training in practice 550 TPU training 555

18.3

Speeding up training and inference with lower-precision computation 556

Understanding floating-point precision 556 ■ Float16 inference 558 ■ Mixed-precision training 559 ■ Using loss scaling with mixed precision 559 ■ Beyond mixed precision: float8 training 560 ■ Faster inference with quantization 561

19

The future of AI 564 19.1

The limitations of deep l

### PDF p.532 (printed 508) — 17 Image generation › 17.1 Deep learning for image generation › This chapter covers
Variational autoencoders

Diffusion models

Using a pretrained text-to-image model

Exploring the latent image spaces learned by

text-to-image models

The most popular and successful application of creative AI today is image generation: learning latent visual spaces and sampling from them to create entirely new pictures, interpolated from real ones—pictures of imaginary people, imaginary places, imaginary cats and dogs, and so on.

### PDF p.544 (printed 520) — 17 Image generation › 17.2 Diffusion models › 
A diffusion model is essentially a denoising autoencoder in a loop, capable of turning pure noise into sharp, realistic imagery. You may know this poetic quote from Michelangelo, “Every block of stone has a statue inside it and it is the task of the sculptor to discover it”—well, every square of white noise has an image inside it, and it is the task of the diffusion model to discover it.

Now, let’s build one with Keras.

### PDF p.542 (printed 518) — 17 Image generation › 17.2 Diffusion models › 17.2 Diffusion models
A long-standing application of autoencoders has been denoising: feeding into a model an input that features a small amount of noise—for instance, a low-quality JPEG image— and getting back a cleaned-up version of the same input. This is the one task that autoencoders excel at. In the late 2010s, this idea gave rise to very successful image super-resolution models, capable of taking in low-resolution, potentially noisy images and outputting high-quality, high-resolution versions of them (see figure 17.6). Such models have been shipped as part of every major smartphone camera app for the past few years.

### PDF p.558 (printed 534) — 17 Image generation › 17.3 Text-to-image models › 
Our generation process has three distinct steps:

1 First, we take our prompts, tokenize them, and embed them with our text encoder. 2 Second, we take our text embeddings and pure noise and progressively “denoise”

the noise into an image. This is the same as the flower model we just built. 3 Lastly, we map our model outputs, which are from `[-1, 1]` back to `[0, 255]`, so we can render the image. One thing to note here is that our text embeddings actually contain four separate tensors:
