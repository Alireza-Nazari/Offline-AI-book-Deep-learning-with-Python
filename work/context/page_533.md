# Generation bundle for page_533  (PDF page 533, printed page 509)

Page kind: content
Chapter: 17 Image generation
Section: 17.1 Deep learning for image generation
Subsection: 17.1.1 Sampling from latent spaces of images
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_533.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_533.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p533_e03] (body) latent spaces of sound or music using similar models—but in practice, the most interesting results so far have been obtained with pictures, and that’s what we focus on here.
[p533_e04] (heading_subsection) 17.1.1 Sampling from latent spaces of images
[p533_e05] (body) The key idea of image generation is to develop a low-dimensional latent space of representations (which, like everything else in deep learning, is a vector space) where any point can be mapped to a “valid” image: an image that looks like the real thing. The module capable of realizing this mapping, taking as input a latent point and outputting an image (a grid of pixels), is usually called a generator, or sometimes a decoder. Once such a latent space has been learned, you can sample points from it, and, by mapping them back to image space, generate images that have never been seen before (see figure 17.1)—the in-betweens of the training images.
[p533_e06] (figure_caption) Figure 17.1 Using a latent vector space to sample new images
[p533_e07] (body) Further, text-conditioning makes it possible to map a space of prompts in natural language to the latent space (see figure 17.2), making it possible to do language-guided image generation—generating pictures that correspond to a text description. This category of models is called text-to-image models.
[p533_e08] (body) Interpolating between many training images in the latent space enables such models to generate infinite combinations of visual concepts, including many that no one had explicitly come up with before. A horse riding a bike on the moon? You got it. This makes image generation a powerful brush for creative-minded people to play with.
[p533_e09] (code)
```
"Horse
riding bike
on the
moon"
```
[p533_e10] (figure_caption) Figure 17.2 Language-guided image generation

## Figures
- Figure 17.1: Using a latent vector space to sample new images  | labels inside the figure: Latent image space, (a vector), Decoder / generator, generated image
- Figure 17.2: Language-guided image generation  | labels inside the figure: generated image, Latent text/image, space, Text encoder, Image decoder

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

## Previous page (PDF 532) text, for continuity
Image generation

This chapter covers

Variational autoencoders

Diffusion models

Using a pretrained text-to-image model

Exploring the latent image spaces learned by

text-to-image models

The most popular and successful application of creative AI today is image generation: learning latent visual spaces and sampling from them to create entirely new pictures, interpolated from real ones—pictures of imaginary people, imaginary places, imaginary cats and dogs, and so on.

17.1 Deep learning for image generation

In this section and the next, we’ll review some high-level concepts pertaining to image generation, alongside implementation details relative to two of the main techniques in this domain: variational autoencoders (VAEs) and diffusion models. Do note that the techniques we present here aren’t specific to images—you could develop

## Next page (PDF 534) text, for continuity
Of course, there are still challenges to overcome. Like with all deep learning models, the latent space doesn’t encode a consistent model of the physical world, so you might occasionally see hands with extra fingers, incoherent lighting, or garbled objects. The coherence of generated images is still an area of active research. In the case of figure 17.2, despite having seen tens of thousands of images of people riding bikes, the model doesn’t understand in a human sense what it means to ride a bike—concepts like pedaling, steering, or maintaining upright balance. That’s why your bike-riding horse is unlikely to get depicted pedaling with its hind legs in a believable manner, the way a human artist would draw it.

There’s a range of different strategies for learning such latent spaces of image representations, each with its own characteristics. The most common types of image generation models are

Diffusion models

Variational autoencoders (VAEs)

Generative adversarial networks (GANs)

While previous editions of this book covered GANs, they have gradually fallen out of fashion in recent years and have been all but replaced by diffusion models. In this edition, we’ll cover both VAEs and diffusion models and we will skip GANs. In the models we’ll build ourselves, we’ll focus on unconditioned image generation—sampling images from a latent space without text conditioning. However, you will also learn how to use a pretrained text-to-image model and how to explore its latent space.

17.1.2 Variational autoencoders

VAEs, simultaneously discovered by Kingma and Welling in December 20131 and Rezende, Mohamed, and Wierstra in January 2014,2 are a kind of generative model that’s especially appropriate for the task of image editing via concept vectors. They’re a kind of autoencode …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.561 — 17 Image generation › 17.3 Text-to-image models
- PDF p.542 — 17 Image generation › 17.2 Diffusion models
- PDF p.539 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.557 — 17 Image generation › 17.3 Text-to-image models
- PDF p.538 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.535 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.15 — contents › 
- PDF p.491 — 16 Text generation › 

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.561 (printed 537) — 17 Image generation › 17.3 Text-to-image models › Summary
Image generation with deep learning is done by learning latent spaces that capture statistical information about a dataset of images. By sampling and decoding points from the latent space, you can generate never-before-seen images. There are three major tools to do this: VAEs, diffusion models, and GANs.

VAEs result in highly structured, continuous latent representations. For this reason, they work well for doing all sorts of image editing in latent space: face swapping, turning a frowning face into a smiling face, and so on. They also work nicely for doing latent space–based animations, such as animating a walk along a cross section of the latent space, showing a starting image slowly morphing into different images in a continuous way.

Diffusion models result in very realistic outputs and are the dominant method of image generation today. They work by repeatedly denoising an image, starting from pure noise. They can easily be conditioned on text captions to create text-to-image models.

Stable Diffusion 3 is a state-of-the-art pretrained text-to-image model that you can use to create highly realistic images of your own.

The visual latent space learned by such text-to-image diff

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

### PDF p.542 (printed 518) — 17 Image generation › 17.2 Diffusion models › 
Listing 17.6 Sampling a grid of points from the 2D latent space and decoding them to images

### PDF p.557 (printed 533) — 17 Image generation › 17.3 Text-to-image models › 17.3.1 Exploring the latent space of a text-to-image model
There is probably no better way to see the interpolative nature of deep neural networks than text diffusion models. The text encoder used by our model will learn a smooth, low-dimensional manifold to represent our input prompts. It’s continuous, meaning we have learned a space where we can walk from the text representation of one prompt to another, and each intermediate point will have semantic meaning. We can couple that with our diffusion process to morph between two images by simply describing each end state with a text prompt.

### PDF p.537 (printed 513) — 17 Image generation › 17.1 Deep learning for image generation › 17.1.3 Implementing a VAE with Keras
We’re going to be implementing a VAE that can generate MNIST digits. It’s going to have three parts:

An encoder network that turns a real image into a mean and a variance in the latent space

A sampling layer that takes such a mean and variance and uses them to sample a random point from the latent space

A decoder network that turns points from the latent space back into images

The following listing shows the encoder network you’ll use, mapping images to the parameters of a probability distribution over the latent space. It’s a simple ConvNet that maps the input image `x` to two vectors, `z_mean` and `z_log_var`. One important detail is that we use strides for downsampling feature maps, instead of max pooling. The last time we did this was in the image segmentation example of chapter 11. Recall that, in general, strides are preferable to max pooling for any model that cares about information location—that is, where stuff is in the image—and this one does, since it will have to produce an image encoding that can be used to reconstruct a valid image.

Listing 17.1 VAE encoder network

### PDF p.535 (printed 511) — 17 Image generation › 17.1 Deep learning for image generation › 
latent space of representations, `z_mean` and `z_log_variance`. 2 You randomly sample a point `z` from the latent normal distribution that’s assumed to generate the input image, via `z = z_mean + exp(z_log_variance) * epsilon`, where `epsilon` is a random tensor of small values. 3 A decoder module maps this point in the latent space back to the original input image. Because `epsilon` is random, the process ensures that every point that’s close to the latent location where you encoded `input_img` (`z-mean`) can be decoded to something

### PDF p.535 (printed 511) — 17 Image generation › 17.1 Deep learning for image generation › 
the original image, via a decoder module (see figure 17.3). It’s then trained by using as target data the same images as the input images, meaning the autoencoder learns to reconstruct the original inputs. By imposing various constraints on the code (the output of the encoder), you can get the autoencoder to learn more or less interesting latent representations of the data. Most commonly, you’ll constrain the code to be low dimensional and sparse (mostly zeros), in which case the encoder acts as a way to compress the input data into fewer bits of information.

Figure 17.3 An autoencoder: mapping an input `x` to a compressed representation and then decoding it back as `x'`

In practice, such classical autoencoders don’t lead to particularly useful or nicely structured latent spaces. They’re not much good at compression either. For these reasons, they have largely fallen out of fashion. VAEs, however, augment autoencoders with a little bit of statistical magic that forces them to learn continuous, highly structured latent spaces. They have turned out to be a powerful tool for image generation.

A VAE, instead of compressing its input image into a fixed code in the latent space, turns

### PDF p.539 (printed 515) — 17 Image generation › 17.1 Deep learning for image generation › 
Listing 17.3 VAE decoder network, mapping latent space points to images
