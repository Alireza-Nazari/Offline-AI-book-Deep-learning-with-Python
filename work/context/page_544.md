# Generation bundle for page_544  (PDF page 544, printed page 520)

Page kind: content
Chapter: 17 Image generation
Section: 17.2 Diffusion models
Subsection: 17.2.1 The Oxford Flowers dataset
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_544.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_544.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p544_e03] (body) higher-resolution version of what you give them. This can potentially lead to funny mishaps. For instance, with some AI-enhanced cameras, you can take a picture of something that looks vaguely moon-like (such as a printout of a severely blurred moon image), and you will get in your camera roll a crisp picture of the moon’s craters. A lot of detail that simply wasn’t present in the printout gets straight-up hallucinated by the camera, because the super-resolution model it uses is overfitted to moon photography images. So, unlike Rick Deckard, definitely don’t use this technique for forensics!
[p544_e04] (body) Early successes in image denoising led researchers to an arresting idea: since you can use an autoencoder to remove a small amount of noise from an image, surely it would be possible to repeat the process multiple times in a loop to remove a large amount of noise. Ultimately, could you denoise an image made out of pure noise? As it turns out, yes, you can. By doing this, you can effectively hallucinate brand new images out of nothing, like in figure 17.7. This is the key insight behind diffusion models, which should more accurately be called reverse diffusion models, since “diffusion” refers to the process of gradually adding noise to an image until it disperses into nothing.
[p544_e05] (figure_caption) Figure 17.7 Reverse diffusion: turning pure noise into an image via repeated denoising
[p544_e06] (body) A diffusion model is essentially a denoising autoencoder in a loop, capable of turning pure noise into sharp, realistic imagery. You may know this poetic quote from Michelangelo, “Every block of stone has a statue inside it and it is the task of the sculptor to discover it”—well, every square of white noise has an image inside it, and it is the task of the diffusion model to discover it.
[p544_e07] (body) Now, let’s build one with Keras.
[p544_e08] (heading_subsection) 17.2.1 The Oxford Flowers dataset
[p544_e09] (body) The dataset we’re going to use is the Oxford Flowers dataset (https://www.robots.ox.ac .uk/~vgg/data/flowers/102/), a collection of 8,189 images of flowers that belong to 102 different species.
[p544_e10] (body) Let’s get the dataset archive and extract it:

## Figures
- Figure 17.7: Reverse diffusion: turning pure noise into an image via repeated denoising  | labels inside the figure: 1, 0.75, 0.5, 0.25, 0

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

## Previous page (PDF 543) text, for continuity
Figure 17.5 Grid of digits decoded from the latent space Figure 17.6 Image super-resolution

Of course, these models aren’t magically recovering lost details hidden in the input, like in the “enhance” scene from Blade Runner (1982). Rather, they’re making educated guesses about what the image should look like—they’re hallucinating a cleaned-up,

## Next page (PDF 545) text, for continuity
`fpath` is now the local path to the extracted directory. The images are contained in the `jpg` subdirectory there. Let’s turn them into an iterable dataset using `image_dataset_from_directory()`.

We need to resize our images to a fixed size, but we don’t want to distort their aspect ratio since this would negatively affect the quality of our generated images, so we use the `crop_to_aspect_ratio` option to extract maximally large undistorted crops of the right size (128 × 128):

Here’s an example image (figure 17.8):

17.2.2 A U-Net denoising autoencoder

The same denoising model gets reused across each iteration of the diffusion denoising process, erasing a little bit of noise each time. To make the job of the model easier,

Figure 17.8 An example image from the

we tell it how much noise it is supposed to

Oxford Flowers dataset

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.547 — 17 Image generation › 17.2 Diffusion models
- PDF p.550 — 17 Image generation › 17.2 Diffusion models
- PDF p.542 — 17 Image generation › 17.2 Diffusion models
- PDF p.555 — 17 Image generation › 17.3 Text-to-image models
- PDF p.551 — 17 Image generation › 17.2 Diffusion models
- PDF p.549 — 17 Image generation › 17.2 Diffusion models
- PDF p.557 — 17 Image generation › 17.3 Text-to-image models
- PDF p.561 — 17 Image generation › 17.3 Text-to-image models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.547 (printed 523) — 17 Image generation › 17.2 Diffusion models › 17.2.3 The concepts of diffusion time and diffusion schedule
The diffusion process is a series of steps in which we apply our denoising autoencoder to erase a small amount of noise from an image, starting with a pure-noise image, and ending with a pure-signal image. The index of the current step in the loop is called the diffusion time (see figure 17.7). In our case, we’ll use a continuous value between 1

### PDF p.542 (printed 518) — 17 Image generation › 17.2 Diffusion models › 17.2 Diffusion models
A long-standing application of autoencoders has been denoising: feeding into a model an input that features a small amount of noise—for instance, a low-quality JPEG image— and getting back a cleaned-up version of the same input. This is the one task that autoencoders excel at. In the late 2010s, this idea gave rise to very successful image super-resolution models, capable of taking in low-resolution, potentially noisy images and outputting high-quality, high-resolution versions of them (see figure 17.6). Such models have been shipped as part of every major smartphone camera app for the past few years.

### PDF p.555 (printed 531) — 17 Image generation › 17.3 Text-to-image models › 17.3 Text-to-image models
We can use the same basic diffusion process to create a model that maps text input to image output. To do this we need to take a pretrained text encoder (think a transformer encoder like RoBERTa from chapter 15) that can map text to vectors in a continuous embedding space. Then we can train a diffusion model on `(prompt,` `image)` pairs, where each prompt is a short, textual description of the input image.

We can handle the image input in the same way as we did previously, mapping noisy input to a denoised output that progressively approaches our input image. Critically, we can extend this setup by also passing the embedded text prompt to the denoising model. So rather than our denoising model simply taking in a `noisy_images` input, our model will take two inputs: `noisy_images` and `text_embeddings`. This gives a leg up on the flower denoiser we trained previously. Instead of learning to remove noise from an image without any additional information, the model gets to use a textual representation of the final image to help guide the denoising process. After training is when things get a bit more fun. Because we have trained a model that can map pure noise to images conditioned on

### PDF p.553 (printed 529) — 17 Image generation › 17.2 Diffusion models › 17.2.7 It’s go time!
It’s finally time to train our diffusion model on the Oxford Flowers dataset. Let’s instantiate the model:

### PDF p.549 (printed 525) — 17 Image generation › 17.2 Diffusion models › 17.2.4 The training process
Let’s create a `DiffusionModel` class to implement the training procedure. It’s going to have our denoising autoencoder as one of its attributes. We’re also going to need a couple more things:

A loss function—We’ll use mean absolute error as our loss, that is to say `mean` `(abs(real_noise_mask - predicted_noise_mask)).`

An image normalization layer—The noise we’ll add to the images will have unit variance and zero mean, so we’d like our images to be normalized as such too, for the value range of the noise to match the value range of the images.

### PDF p.561 (printed 537) — 17 Image generation › 17.3 Text-to-image models › Summary
Image generation with deep learning is done by learning latent spaces that capture statistical information about a dataset of images. By sampling and decoding points from the latent space, you can generate never-before-seen images. There are three major tools to do this: VAEs, diffusion models, and GANs.

VAEs result in highly structured, continuous latent representations. For this reason, they work well for doing all sorts of image editing in latent space: face swapping, turning a frowning face into a smiling face, and so on. They also work nicely for doing latent space–based animations, such as animating a walk along a cross section of the latent space, showing a starting image slowly morphing into different images in a continuous way.

Diffusion models result in very realistic outputs and are the dominant method of image generation today. They work by repeatedly denoising an image, starting from pure noise. They can easily be conditioned on text captions to create text-to-image models.

Stable Diffusion 3 is a state-of-the-art pretrained text-to-image model that you can use to create highly realistic images of your own.

The visual latent space learned by such text-to-image diff

### PDF p.550 (printed 526) — 17 Image generation › 17.2 Diffusion models › 
Next comes the training logic. This is the most important part! Like in the VAE example, we’re going to implement a custom `compute_loss()` method to keep our model backend agnostic. Of course, if you are set on using one specific backend, you could also write a custom `train_step()` with the exact same logic in it, plus the backend-specific logic for gradient computation and weight updates.

Since `compute_loss()` receives as input the output of `call()`, we’re going to put the denoising forward pass in `call()`. Our `call()` takes a batch of clean input images and applies the following steps: 1 Normalizes the images 2 Samples random diffusion times (the denoising model needs to be trained on

the full spectrum of diffusion times) 3 Computes corresponding noise rates and signal rates (using the diffusion schedule) 4 Adds random noise to the clean images (based on the computed noise rates and signal rates) 5 Denoises the images

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
