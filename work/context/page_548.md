# Generation bundle for page_548  (PDF page 548, printed page 524)

Page kind: content
Chapter: 17 Image generation
Section: 17.2 Diffusion models
Subsection: 17.2.3 The concepts of diffusion time and diffusion schedule
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_548.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_548.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p548_e03] (body) and 0 for this index—a value of 1 indicates the start of the process, where the amount of noise is maximal and the amount of signal is minimal, and a value of 0 indicates the end of the process, where the image is almost all signal and no noise.
[p548_e04] (body) The relationship between the current diffusion time and the amount of noise and signal present in the image is called the diffusion schedule. In our experiment, we’re going to use a cosine schedule to smoothly transition from a high signal rate (low noise) at the beginning to a low signal rate (high noise) at the end of the diffusion process.
[p548_e05] (listing_caption) Listing 17.7 The diffusion schedule
[p548_e06] (code)
```
def diffusion_schedule(
    diffusion_times,
    min_signal_rate=0.02,
    max_signal_rate=0.95,
):
    start_angle = ops.cast(ops.arccos(max_signal_rate), "float32")
    end_angle = ops.cast(ops.arccos(min_signal_rate), "float32")
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
    signal_rates = ops.cos(diffusion_angles)
    noise_rates = ops.sin(diffusion_angles)
    return noise_rates, signal_rates
```
[p548_e07] (body) This `diffusion_schedule()` function takes as input a `diffusion_times` tensor, which represents the progression of the diffusion process and returns the corresponding `noise_rates` and `signal_rates` tensors. These rates will be used to guide the denoising process. The logic behind using a cosine schedule is to maintain the relationship `noise_rates ** 2 + signal_rates ** 2 == 1` (see figure 17.10).
[p548_e08] (figure_caption) Figure 17.10 Cosine relationship between noise rates and signal rates

## Listings
- Listing 17.7: The diffusion schedule (code elements p548_e06)

## Figures
- Figure 17.10: Cosine relationship between noise rates and signal rates  | labels inside the figure: Signal rates, Noise rates

## Equation candidates (formulas written inline in the text)
- [p548_e07] The logic behind using a cosine schedule is to maintain the relationship `noise_rates ** 2 + signal_rates ** 2 == 1` (see figure 17.10).

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

## Previous page (PDF 547) text, for continuity
You would instantiate the model with something like `get_model(image_size=128, ` `widths=[32,` `64,` `96,` `128],` `block_depth=2)`. The `widths` argument is a list containing the `Conv2D` layer sizes for each successive downsampling or upsampling stage. We typically want the layers to get bigger as we downsample the inputs (going from 32 to 128 units here) and then get smaller as as upsample (from 128 back to 32 here).

17.2.3 The concepts of diffusion time and diffusion schedule

The diffusion process is a series of steps in which we apply our denoising autoencoder to erase a small amount of noise from an image, starting with a pure-noise image, and ending with a pure-signal image. The index of the current step in the loop is called the diffusion time (see figure 17.7). In our case, we’ll use a continuous value between 1

## Next page (PDF 549) text, for continuity
Let’s plot how this function maps diffusion times (between 0 and 1) to specific noise rates and signal rates (see figure 17.11):

Figure 17.11 Our cosine diffusion schedule

17.2.4 The training process

Let’s create a `DiffusionModel` class to implement the training procedure. It’s going to have our denoising autoencoder as one of its attributes. We’re also going to need a couple more things:

A loss function—We’ll use mean absolute error as our loss, that is to say `mean` `(abs(real_noise_mask - predicted_noise_mask)).`

An image normalization layer—The noise we’ll add to the images will have unit variance and zero mean, so we’d like our images to be normalized as such too, for the value range of the noise to match the value range of the images.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.551 — 17 Image generation › 17.2 Diffusion models
- PDF p.550 — 17 Image generation › 17.2 Diffusion models
- PDF p.552 — 17 Image generation › 17.2 Diffusion models
- PDF p.544 — 17 Image generation › 17.2 Diffusion models
- PDF p.557 — 17 Image generation › 17.3 Text-to-image models
- PDF p.555 — 17 Image generation › 17.3 Text-to-image models
- PDF p.542 — 17 Image generation › 17.2 Diffusion models
- PDF p.545 — 17 Image generation › 17.2 Diffusion models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.550 (printed 526) — 17 Image generation › 17.2 Diffusion models › 
Next comes the training logic. This is the most important part! Like in the VAE example, we’re going to implement a custom `compute_loss()` method to keep our model backend agnostic. Of course, if you are set on using one specific backend, you could also write a custom `train_step()` with the exact same logic in it, plus the backend-specific logic for gradient computation and weight updates.

Since `compute_loss()` receives as input the output of `call()`, we’re going to put the denoising forward pass in `call()`. Our `call()` takes a batch of clean input images and applies the following steps: 1 Normalizes the images 2 Samples random diffusion times (the denoising model needs to be trained on

the full spectrum of diffusion times) 3 Computes corresponding noise rates and signal rates (using the diffusion schedule) 4 Adds random noise to the clean images (based on the computed noise rates and signal rates) 5 Denoises the images

### PDF p.545 (printed 521) — 17 Image generation › 17.2 Diffusion models › 17.2.2 A U-Net denoising autoencoder
The same denoising model gets reused across each iteration of the diffusion denoising process, erasing a little bit of noise each time. To make the job of the model easier,

Figure 17.8 An example image from the

we tell it how much noise it is supposed to

Oxford Flowers dataset

### PDF p.544 (printed 520) — 17 Image generation › 17.2 Diffusion models › 
A diffusion model is essentially a denoising autoencoder in a loop, capable of turning pure noise into sharp, realistic imagery. You may know this poetic quote from Michelangelo, “Every block of stone has a statue inside it and it is the task of the sculptor to discover it”—well, every square of white noise has an image inside it, and it is the task of the diffusion model to discover it.

Now, let’s build one with Keras.

### PDF p.555 (printed 531) — 17 Image generation › 17.3 Text-to-image models › 17.3 Text-to-image models
We can use the same basic diffusion process to create a model that maps text input to image output. To do this we need to take a pretrained text encoder (think a transformer encoder like RoBERTa from chapter 15) that can map text to vectors in a continuous embedding space. Then we can train a diffusion model on `(prompt,` `image)` pairs, where each prompt is a short, textual description of the input image.

We can handle the image input in the same way as we did previously, mapping noisy input to a denoised output that progressively approaches our input image. Critically, we can extend this setup by also passing the embedded text prompt to the denoising model. So rather than our denoising model simply taking in a `noisy_images` input, our model will take two inputs: `noisy_images` and `text_embeddings`. This gives a leg up on the flower denoiser we trained previously. Instead of learning to remove noise from an image without any additional information, the model gets to use a textual representation of the final image to help guide the denoising process. After training is when things get a bit more fun. Because we have trained a model that can map pure noise to images conditioned on

### PDF p.551 (printed 527) — 17 Image generation › 17.2 Diffusion models › 17.2.5 The generation process
    def generate(self, num_images, diffusion_steps):
        noisy_images = keras.random.normal(
            (num_images, self.image_size, self.image_size, 3),
            seed=self.seed_generator,
        )
        step_size = 1.0 / diffusion_steps
        for step in range(diffusion_steps):
            diffusion_times = ops.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = diffusion_schedule(diffusion_times)

### PDF p.557 (printed 533) — 17 Image generation › 17.3 Text-to-image models › 
Figure 17.15 Controlling the number of diffusion steps

### PDF p.544 (printed 520) — 17 Image generation › 17.2 Diffusion models › 
higher-resolution version of what you give them. This can potentially lead to funny mishaps. For instance, with some AI-enhanced cameras, you can take a picture of something that looks vaguely moon-like (such as a printout of a severely blurred moon image), and you will get in your camera roll a crisp picture of the moon’s craters. A lot of detail that simply wasn’t present in the printout gets straight-up hallucinated by the camera, because the super-resolution model it uses is overfitted to moon photography images. So, unlike Rick Deckard, definitely don’t use this technique for forensics!

Early successes in image denoising led researchers to an arresting idea: since you can use an autoencoder to remove a small amount of noise from an image, surely it would be possible to repeat the process multiple times in a loop to remove a large amount of noise. Ultimately, could you denoise an image made out of pure noise? As it turns out, yes, you can. By doing this, you can effectively hallucinate brand new images out of nothing, like in figure 17.7. This is the key insight behind diffusion models, which should more accurately be called reverse diffusion models, since “diffusion” refers t

### PDF p.555 (printed 531) — 17 Image generation › 17.3 Text-to-image models › 17.3 Text-to-image models
Hub with just a couple of lines of code.

Diffusion model

Listing 17.8 Creating a Stable Diffusion text-to-image model
