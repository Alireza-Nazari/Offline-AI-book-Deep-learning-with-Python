# Generation bundle for page_543  (PDF page 543, printed page 519)

Page kind: content
Chapter: 17 Image generation
Section: 17.2 Diffusion models
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_543.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_543.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p543_e03] (figure_caption) Figure 17.5 Grid of digits decoded from the latent space Figure 17.6 Image super-resolution
[p543_e04] (body) Of course, these models aren’t magically recovering lost details hidden in the input, like in the “enhance” scene from Blade Runner (1982). Rather, they’re making educated guesses about what the image should look like—they’re hallucinating a cleaned-up,

## Figures
- Figure 17.5: Grid of digits decoded from the latent space Figure 17.6 Image super-resolution  | labels inside the figure: (none - image)

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

## Previous page (PDF 542) text, for continuity
Listing 17.6 Sampling a grid of points from the 2D latent space and decoding them to images

The grid of sampled digits (see figure 17.5) shows a completely continuous distribution of the different digit classes, with one digit morphing into another as you follow a path through latent space. Specific directions in this space have a meaning: for example, there’s a direction for “four-ness,” “one-ness,” and so on.

In the next section, we’ll cover in detail another major tool for generating images: diffusion models, the architecture behind nearly all commercial image generation services today.

17.2 Diffusion models

A long-standing application of autoencoders has been denoising: feeding into a model an input that features a small amount of noise—for instance, a low-quality JPEG image— and getting back a cleaned-up version of the same input. This is the one task that autoencoders excel at. In the late 2010s, this idea gave rise to very successful image super-resolution models, capable of taking in low-resolution, potentially noisy images and outputting high-quality, high-resolution versions of them (see figure 17.6). Such models have been shipped as part of every major smartphone camera app for the past few years.

## Next page (PDF 544) text, for continuity
higher-resolution version of what you give them. This can potentially lead to funny mishaps. For instance, with some AI-enhanced cameras, you can take a picture of something that looks vaguely moon-like (such as a printout of a severely blurred moon image), and you will get in your camera roll a crisp picture of the moon’s craters. A lot of detail that simply wasn’t present in the printout gets straight-up hallucinated by the camera, because the super-resolution model it uses is overfitted to moon photography images. So, unlike Rick Deckard, definitely don’t use this technique for forensics!

Early successes in image denoising led researchers to an arresting idea: since you can use an autoencoder to remove a small amount of noise from an image, surely it would be possible to repeat the process multiple times in a loop to remove a large amount of noise. Ultimately, could you denoise an image made out of pure noise? As it turns out, yes, you can. By doing this, you can effectively hallucinate brand new images out of nothing, like in figure 17.7. This is the key insight behind diffusion models, which should more accurately be called reverse diffusion models, since “diffusion” refers to the process of gradually adding noise to an image until it disperses into nothing.

Figure 17.7 Reverse diffusion: turning pure noise into an image via repeated denoising

A diffusion model is essentially a denoising autoencoder in a loop, capable of turning pure noise into sharp, realistic imagery. You may know this poetic quote from Michelangelo, “Every block of stone has a statue inside it and it is the task of the sculptor to discover it”—well, every square of white noise has an image inside it, and it is the task of the diffusion model to discover it.

Now, let’s build one with Keras.

 …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.535 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.536 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.533 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.534 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.336 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.545 — 17 Image generation › 17.2 Diffusion models
- PDF p.538 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.561 — 17 Image generation › 17.3 Text-to-image models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.533 (printed 509) — 17 Image generation › 17.1 Deep learning for image generation › 17.1.1 Sampling from latent spaces of images
The key idea of image generation is to develop a low-dimensional latent space of representations (which, like everything else in deep learning, is a vector space) where any point can be mapped to a “valid” image: an image that looks like the real thing. The module capable of realizing this mapping, taking as input a latent point and outputting an image (a grid of pixels), is usually called a generator, or sometimes a decoder. Once such a latent space has been learned, you can sample points from it, and, by mapping them back to image space, generate images that have never been seen before (see figure 17.1)—the in-betweens of the training images.

Figure 17.1 Using a latent vector space to sample new images

Further, text-conditioning makes it possible to map a space of prompts in natural language to the latent space (see figure 17.2), making it possible to do language-guided image generation—generating pictures that correspond to a text description. This category of models is called text-to-image models.

Interpolating between many training images in the latent space enables such models to generate infinite combinations of visual concepts, including many that no one had explicitly c

### PDF p.561 (printed 537) — 17 Image generation › 17.3 Text-to-image models › Summary
Image generation with deep learning is done by learning latent spaces that capture statistical information about a dataset of images. By sampling and decoding points from the latent space, you can generate never-before-seen images. There are three major tools to do this: VAEs, diffusion models, and GANs.

VAEs result in highly structured, continuous latent representations. For this reason, they work well for doing all sorts of image editing in latent space: face swapping, turning a frowning face into a smiling face, and so on. They also work nicely for doing latent space–based animations, such as animating a walk along a cross section of the latent space, showing a starting image slowly morphing into different images in a continuous way.

Diffusion models result in very realistic outputs and are the dominant method of image generation today. They work by repeatedly denoising an image, starting from pure noise. They can easily be conditioned on text captions to create text-to-image models.

Stable Diffusion 3 is a state-of-the-art pretrained text-to-image model that you can use to create highly realistic images of your own.

The visual latent space learned by such text-to-image diff

### PDF p.535 (printed 511) — 17 Image generation › 17.1 Deep learning for image generation › 
latent space of representations, `z_mean` and `z_log_variance`. 2 You randomly sample a point `z` from the latent normal distribution that’s assumed to generate the input image, via `z = z_mean + exp(z_log_variance) * epsilon`, where `epsilon` is a random tensor of small values. 3 A decoder module maps this point in the latent space back to the original input image. Because `epsilon` is random, the process ensures that every point that’s close to the latent location where you encoded `input_img` (`z-mean`) can be decoded to something

### PDF p.534 (printed 510) — 17 Image generation › 17.1 Deep learning for image generation › 
Of course, there are still challenges to overcome. Like with all deep learning models, the latent space doesn’t encode a consistent model of the physical world, so you might occasionally see hands with extra fingers, incoherent lighting, or garbled objects. The coherence of generated images is still an area of active research. In the case of figure 17.2, despite having seen tens of thousands of images of people riding bikes, the model doesn’t understand in a human sense what it means to ride a bike—concepts like pedaling, steering, or maintaining upright balance. That’s why your bike-riding horse is unlikely to get depicted pedaling with its hind legs in a believable manner, the way a human artist would draw it.

There’s a range of different strategies for learning such latent spaces of image representations, each with its own characteristics. The most common types of image generation models are

Diffusion models

Variational autoencoders (VAEs)

Generative adversarial networks (GANs)

While previous editions of this book covered GANs, they have gradually fallen out of fashion in recent years and have been all but replaced by diffusion models. In this edition, we’ll cover both VAEs

### PDF p.557 (printed 533) — 17 Image generation › 17.3 Text-to-image models › 17.3.1 Exploring the latent space of a text-to-image model
There is probably no better way to see the interpolative nature of deep neural networks than text diffusion models. The text encoder used by our model will learn a smooth, low-dimensional manifold to represent our input prompts. It’s continuous, meaning we have learned a space where we can walk from the text representation of one prompt to another, and each intermediate point will have semantic meaning. We can couple that with our diffusion process to morph between two images by simply describing each end state with a text prompt.

### PDF p.536 (printed 512) — 17 Image generation › 17.1 Deep learning for image generation › 
and `z_log_variance`

Figure 17.4 A VAE maps an image to two vectors, `z_mean` and `z_log_variance`, which define a probability distribution over the latent space, used to sample a latent point to decode.

similar to `input_img`, thus forcing the latent space to be continuously meaningful. Any two close points in the latent space will decode to highly similar images. Continuity, combined with the low dimensionality of the latent space, forces every direction in the latent space to encode a meaningful axis of variation of the data, making the latent space very structured and thus highly suitable to manipulation via concept vectors.

The parameters of a VAE are trained via two loss functions: a reconstruction loss that forces the decoded samples to match the initial inputs, and a regularization loss that helps learn well-rounded latent distributions and reduces overfitting to the training data. Schematically, the process looks like this:

### PDF p.555 (printed 531) — 17 Image generation › 17.3 Text-to-image models › 17.3 Text-to-image models
We can use the same basic diffusion process to create a model that maps text input to image output. To do this we need to take a pretrained text encoder (think a transformer encoder like RoBERTa from chapter 15) that can map text to vectors in a continuous embedding space. Then we can train a diffusion model on `(prompt,` `image)` pairs, where each prompt is a short, textual description of the input image.

We can handle the image input in the same way as we did previously, mapping noisy input to a denoised output that progressively approaches our input image. Critically, we can extend this setup by also passing the embedded text prompt to the denoising model. So rather than our denoising model simply taking in a `noisy_images` input, our model will take two inputs: `noisy_images` and `text_embeddings`. This gives a leg up on the flower denoiser we trained previously. Instead of learning to remove noise from an image without any additional information, the model gets to use a textual representation of the final image to help guide the denoising process. After training is when things get a bit more fun. Because we have trained a model that can map pure noise to images conditioned on

### PDF p.555 (printed 531) — 17 Image generation › 17.3 Text-to-image models › 17.3 Text-to-image models
Hub with just a couple of lines of code.

Diffusion model

Listing 17.8 Creating a Stable Diffusion text-to-image model
