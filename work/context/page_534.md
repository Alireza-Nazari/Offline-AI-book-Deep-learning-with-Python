# Generation bundle for page_534  (PDF page 534, printed page 510)

Page kind: content
Chapter: 17 Image generation
Section: 17.1 Deep learning for image generation
Subsection: 17.1.2 Variational autoencoders
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_534.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_534.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p534_e03] (body) Of course, there are still challenges to overcome. Like with all deep learning models, the latent space doesn’t encode a consistent model of the physical world, so you might occasionally see hands with extra fingers, incoherent lighting, or garbled objects. The coherence of generated images is still an area of active research. In the case of figure 17.2, despite having seen tens of thousands of images of people riding bikes, the model doesn’t understand in a human sense what it means to ride a bike—concepts like pedaling, steering, or maintaining upright balance. That’s why your bike-riding horse is unlikely to get depicted pedaling with its hind legs in a believable manner, the way a human artist would draw it.
[p534_e04] (body) There’s a range of different strategies for learning such latent spaces of image representations, each with its own characteristics. The most common types of image generation models are
[p534_e05] (bullet) Diffusion models
[p534_e06] (bullet) Variational autoencoders (VAEs)
[p534_e07] (bullet) Generative adversarial networks (GANs)
[p534_e08] (body) While previous editions of this book covered GANs, they have gradually fallen out of fashion in recent years and have been all but replaced by diffusion models. In this edition, we’ll cover both VAEs and diffusion models and we will skip GANs. In the models we’ll build ourselves, we’ll focus on unconditioned image generation—sampling images from a latent space without text conditioning. However, you will also learn how to use a pretrained text-to-image model and how to explore its latent space.
[p534_e09] (heading_subsection) 17.1.2 Variational autoencoders
[p534_e10] (body) VAEs, simultaneously discovered by Kingma and Welling in December 20131 and Rezende, Mohamed, and Wierstra in January 2014,2 are a kind of generative model that’s especially appropriate for the task of image editing via concept vectors. They’re a kind of autoencoder—a type of network that aims to encode an input to a low-dimensional latent space and then decode it back—that mixes ideas from deep learning with Bayesian inference.
[p534_e11] (body) VAEs have been around for over a decade, but they remain relevant to this day and continue to be used in recent research. While VAEs will never be the first choice for generating high-fidelity images—where diffusion models excel—they remain an important tool in the deep learning toolbox, particularly when interpretability, control over the latent space, and data reconstruction capabilities are crucial. It’s also your first contact with the concept of the autoencoder, which is useful to know about. VAEs beautifully illustrate the core idea behind this class of models. A classical image autoencoder takes an image, maps it to a latent vector space via an encoder module, and then decodes it back to an output with the same dimensions as
[p534_e12] (footnote) 1 Diederik P. Kingma and Max Welling, “Auto-Encoding Variational Bayes,” arXiv (2013), https://arxiv.org/ abs/1312.6114. 2 Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra, “Stochastic Backpropagation and Approximate Inference in Deep Generative Models,” arXiv (2014), https://arxiv.org/abs/1401.4082.

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

## Previous page (PDF 533) text, for continuity
latent spaces of sound or music using similar models—but in practice, the most interesting results so far have been obtained with pictures, and that’s what we focus on here.

17.1.1 Sampling from latent spaces of images

The key idea of image generation is to develop a low-dimensional latent space of representations (which, like everything else in deep learning, is a vector space) where any point can be mapped to a “valid” image: an image that looks like the real thing. The module capable of realizing this mapping, taking as input a latent point and outputting an image (a grid of pixels), is usually called a generator, or sometimes a decoder. Once such a latent space has been learned, you can sample points from it, and, by mapping them back to image space, generate images that have never been seen before (see figure 17.1)—the in-betweens of the training images.

Figure 17.1 Using a latent vector space to sample new images

Further, text-conditioning makes it possible to map a space of prompts in natural language to the latent space (see figure 17.2), making it possible to do language-guided image generation—generating pictures that correspond to a text description. This category of models is called text-to-image models.

Interpolating between many training images in the latent space enables such models to generate infinite combinations of visual concepts, including many that no one had explicitly come up with before. A horse riding a bike on the moon? You got it. This makes image generation a powerful brush for creative-minded people to play with.

Figure 17.2 Language-guided image generation

## Next page (PDF 535) text, for continuity
the original image, via a decoder module (see figure 17.3). It’s then trained by using as target data the same images as the input images, meaning the autoencoder learns to reconstruct the original inputs. By imposing various constraints on the code (the output of the encoder), you can get the autoencoder to learn more or less interesting latent representations of the data. Most commonly, you’ll constrain the code to be low dimensional and sparse (mostly zeros), in which case the encoder acts as a way to compress the input data into fewer bits of information.

Figure 17.3 An autoencoder: mapping an input `x` to a compressed representation and then decoding it back as `x'`

In practice, such classical autoencoders don’t lead to particularly useful or nicely structured latent spaces. They’re not much good at compression either. For these reasons, they have largely fallen out of fashion. VAEs, however, augment autoencoders with a little bit of statistical magic that forces them to learn continuous, highly structured latent spaces. They have turned out to be a powerful tool for image generation.

A VAE, instead of compressing its input image into a fixed code in the latent space, turns the image into the parameters of a statistical distribution: a mean and a variance. Essentially, this means we’re assuming the input image has been generated by a statistical process, and that the randomness of this process should be taken into account during encoding and decoding. The VAE then uses the mean and variance parameters to randomly sample one element of the distribution, and decodes that element back to the original input (see figure 17.4). The stochasticity of this process improves robustness and forces the latent space to encode meaningful representations everywhere: every point …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.561 — 17 Image generation › 17.3 Text-to-image models
- PDF p.532 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.536 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.557 — 17 Image generation › 17.3 Text-to-image models
- PDF p.15 — contents › 
- PDF p.542 — 17 Image generation › 17.2 Diffusion models
- PDF p.537 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.555 — 17 Image generation › 17.3 Text-to-image models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.561 (printed 537) — 17 Image generation › 17.3 Text-to-image models › Summary
Image generation with deep learning is done by learning latent spaces that capture statistical information about a dataset of images. By sampling and decoding points from the latent space, you can generate never-before-seen images. There are three major tools to do this: VAEs, diffusion models, and GANs.

VAEs result in highly structured, continuous latent representations. For this reason, they work well for doing all sorts of image editing in latent space: face swapping, turning a frowning face into a smiling face, and so on. They also work nicely for doing latent space–based animations, such as animating a walk along a cross section of the latent space, showing a starting image slowly morphing into different images in a continuous way.

Diffusion models result in very realistic outputs and are the dominant method of image generation today. They work by repeatedly denoising an image, starting from pure noise. They can easily be conditioned on text captions to create text-to-image models.

Stable Diffusion 3 is a state-of-the-art pretrained text-to-image model that you can use to create highly realistic images of your own.

The visual latent space learned by such text-to-image diff

### PDF p.532 (printed 508) — 17 Image generation › 17.1 Deep learning for image generation › 17.1 Deep learning for image generation
In this section and the next, we’ll review some high-level concepts pertaining to image generation, alongside implementation details relative to two of the main techniques in this domain: variational autoencoders (VAEs) and diffusion models. Do note that the techniques we present here aren’t specific to images—you could develop

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

### PDF p.557 (printed 533) — 17 Image generation › 17.3 Text-to-image models › 17.3.1 Exploring the latent space of a text-to-image model
There is probably no better way to see the interpolative nature of deep neural networks than text diffusion models. The text encoder used by our model will learn a smooth, low-dimensional manifold to represent our input prompts. It’s continuous, meaning we have learned a space where we can walk from the text representation of one prompt to another, and each intermediate point will have semantic meaning. We can couple that with our diffusion process to morph between two images by simply describing each end state with a text prompt.

### PDF p.537 (printed 513) — 17 Image generation › 17.1 Deep learning for image generation › 17.1.3 Implementing a VAE with Keras
We’re going to be implementing a VAE that can generate MNIST digits. It’s going to have three parts:

An encoder network that turns a real image into a mean and a variance in the latent space

A sampling layer that takes such a mean and variance and uses them to sample a random point from the latent space

A decoder network that turns points from the latent space back into images

The following listing shows the encoder network you’ll use, mapping images to the parameters of a probability distribution over the latent space. It’s a simple ConvNet that maps the input image `x` to two vectors, `z_mean` and `z_log_var`. One important detail is that we use strides for downsampling feature maps, instead of max pooling. The last time we did this was in the image segmentation example of chapter 11. Recall that, in general, strides are preferable to max pooling for any model that cares about information location—that is, where stuff is in the image—and this one does, since it will have to produce an image encoding that can be used to reconstruct a valid image.

Listing 17.1 VAE encoder network

### PDF p.555 (printed 531) — 17 Image generation › 17.3 Text-to-image models › 17.3 Text-to-image models
We can use the same basic diffusion process to create a model that maps text input to image output. To do this we need to take a pretrained text encoder (think a transformer encoder like RoBERTa from chapter 15) that can map text to vectors in a continuous embedding space. Then we can train a diffusion model on `(prompt,` `image)` pairs, where each prompt is a short, textual description of the input image.

We can handle the image input in the same way as we did previously, mapping noisy input to a denoised output that progressively approaches our input image. Critically, we can extend this setup by also passing the embedded text prompt to the denoising model. So rather than our denoising model simply taking in a `noisy_images` input, our model will take two inputs: `noisy_images` and `text_embeddings`. This gives a leg up on the flower denoiser we trained previously. Instead of learning to remove noise from an image without any additional information, the model gets to use a textual representation of the final image to help guide the denoising process. After training is when things get a bit more fun. Because we have trained a model that can map pure noise to images conditioned on

### PDF p.536 (printed 512) — 17 Image generation › 17.1 Deep learning for image generation › 
and `z_log_variance`

Figure 17.4 A VAE maps an image to two vectors, `z_mean` and `z_log_variance`, which define a probability distribution over the latent space, used to sample a latent point to decode.

similar to `input_img`, thus forcing the latent space to be continuously meaningful. Any two close points in the latent space will decode to highly similar images. Continuity, combined with the low dimensionality of the latent space, forces every direction in the latent space to encode a meaningful axis of variation of the data, making the latent space very structured and thus highly suitable to manipulation via concept vectors.

The parameters of a VAE are trained via two loss functions: a reconstruction loss that forces the decoded samples to match the initial inputs, and a regularization loss that helps learn well-rounded latent distributions and reduces overfitting to the training data. Schematically, the process looks like this:
