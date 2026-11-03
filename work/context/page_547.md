# Generation bundle for page_547  (PDF page 547, printed page 523)

Page kind: content
Chapter: 17 Image generation
Section: 17.2 Diffusion models
Subsection: 17.2.3 The concepts of diffusion time and diffusion schedule
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_547.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_547.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p547_e03] (code)
```
    x = layers.Conv2D(width, 3, padding="same", activation="swish")(x)
    x = layers.Conv2D(width, 3, padding="same")(x)
    x = x + residual
    return x
def get_model(image_size, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_rates = keras.Input(shape=(1, 1, 1))
    x = layers.Conv2D(widths[0], 1)(noisy_images)
    n = layers.UpSampling2D(image_size, interpolation="nearest")(noise_rates)
    x = layers.Concatenate()([x, n])
    skips = []
    for width in widths[:-1]:
        for _ in range(block_depth):
            x = residual_block(x, width)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
    for _ in range(block_depth):
        x = residual_block(x, widths[-1])
    for width in reversed(widths[:-1]):
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = residual_block(x, width)
    pred_noise_masks = layers.Conv2D(3, 1, kernel_initializer="zeros")(x)
    return keras.Model([noisy_images, noise_rates], pred_noise_masks)
```
[p547_e04] (body) You would instantiate the model with something like `get_model(image_size=128, ` `widths=[32,` `64,` `96,` `128],` `block_depth=2)`. The `widths` argument is a list containing the `Conv2D` layer sizes for each successive downsampling or upsampling stage. We typically want the layers to get bigger as we downsample the inputs (going from 32 to 128 units here) and then get smaller as as upsample (from 128 back to 32 here).
[p547_e05] (heading_subsection) 17.2.3 The concepts of diffusion time and diffusion schedule
[p547_e06] (body) The diffusion process is a series of steps in which we apply our denoising autoencoder to erase a small amount of noise from an image, starting with a pure-noise image, and ending with a pure-signal image. The index of the current step in the loop is called the diffusion time (see figure 17.7). In our case, we’ll use a continuous value between 1

## Code annotations printed next to listings (callouts, in order)
- Dowsampling stage
- Upsampling stage
- Middle stage
- Creates the functional model
- We set the kernel initializer for the last layer to “zeros,”
- making the model predict only zeros after initialization (that
- is, our default assumption before training is “no noise”).

## Equation candidates (formulas written inline in the text)
- [p547_e04] You would instantiate the model with something like `get_model(image_size=128, ` `widths=[32,` `64,` `96,` `128],` `block_depth=2)`.

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

## Previous page (PDF 546) text, for continuity
extract for a given input image—that’s the `noise_rates` input. Rather than outputting a denoised image, we make our model output a predicted noise mask, which we can subtract from the input to denoise it.

For our denoising model, we’re going to use a U-Net—a kind of ConvNet originally developed for image segmentation. It looks like figure 17.9.

Figure 17.9 Our U-Net-style denoising autoencoder architecture

This architecture features three stages:

1 A downsampling stage, made of several blocks of convolution layers, where the

inputs get downsampled from their original 128 × 128 size down to a much smaller size (in our case, 16 × 16). 2 A middle stage, where the feature map has a constant size. 3 An upsampling stage, where the feature map get upsampled back to 128 × 128. There is a 1:1 mapping between the blocks of the downsampling and upsampling stages: each upsampling block is the inverse of a downsampling block. Importantly, the model features concatenative residual connections going from each downsampling block to the corresponding upsampling block. These connections help avoid loss of image detail information across the successive downsampling and upsampling operations. Let’s assemble the model using the Functional API:

## Next page (PDF 548) text, for continuity
and 0 for this index—a value of 1 indicates the start of the process, where the amount of noise is maximal and the amount of signal is minimal, and a value of 0 indicates the end of the process, where the image is almost all signal and no noise.

The relationship between the current diffusion time and the amount of noise and signal present in the image is called the diffusion schedule. In our experiment, we’re going to use a cosine schedule to smoothly transition from a high signal rate (low noise) at the beginning to a low signal rate (high noise) at the end of the diffusion process.

Listing 17.7 The diffusion schedule

This `diffusion_schedule()` function takes as input a `diffusion_times` tensor, which represents the progression of the diffusion process and returns the corresponding `noise_rates` and `signal_rates` tensors. These rates will be used to guide the denoising process. The logic behind using a cosine schedule is to maintain the relationship `noise_rates ** 2 + signal_rates ** 2 == 1` (see figure 17.10).

Figure 17.10 Cosine relationship between noise rates and signal rates

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.550 — 17 Image generation › 17.2 Diffusion models
- PDF p.551 — 17 Image generation › 17.2 Diffusion models
- PDF p.553 — 17 Image generation › 17.2 Diffusion models
- PDF p.544 — 17 Image generation › 17.2 Diffusion models
- PDF p.549 — 17 Image generation › 17.2 Diffusion models
- PDF p.555 — 17 Image generation › 17.3 Text-to-image models
- PDF p.257 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.552 — 17 Image generation › 17.2 Diffusion models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.544 (printed 520) — 17 Image generation › 17.2 Diffusion models › 
A diffusion model is essentially a denoising autoencoder in a loop, capable of turning pure noise into sharp, realistic imagery. You may know this poetic quote from Michelangelo, “Every block of stone has a statue inside it and it is the task of the sculptor to discover it”—well, every square of white noise has an image inside it, and it is the task of the diffusion model to discover it.

Now, let’s build one with Keras.

### PDF p.550 (printed 526) — 17 Image generation › 17.2 Diffusion models › 
Next comes the training logic. This is the most important part! Like in the VAE example, we’re going to implement a custom `compute_loss()` method to keep our model backend agnostic. Of course, if you are set on using one specific backend, you could also write a custom `train_step()` with the exact same logic in it, plus the backend-specific logic for gradient computation and weight updates.

Since `compute_loss()` receives as input the output of `call()`, we’re going to put the denoising forward pass in `call()`. Our `call()` takes a batch of clean input images and applies the following steps: 1 Normalizes the images 2 Samples random diffusion times (the denoising model needs to be trained on

the full spectrum of diffusion times) 3 Computes corresponding noise rates and signal rates (using the diffusion schedule) 4 Adds random noise to the clean images (based on the computed noise rates and signal rates) 5 Denoises the images

### PDF p.545 (printed 521) — 17 Image generation › 17.2 Diffusion models › 17.2.2 A U-Net denoising autoencoder
The same denoising model gets reused across each iteration of the diffusion denoising process, erasing a little bit of noise each time. To make the job of the model easier,

Figure 17.8 An example image from the

we tell it how much noise it is supposed to

Oxford Flowers dataset

### PDF p.555 (printed 531) — 17 Image generation › 17.3 Text-to-image models › 17.3 Text-to-image models
We can use the same basic diffusion process to create a model that maps text input to image output. To do this we need to take a pretrained text encoder (think a transformer encoder like RoBERTa from chapter 15) that can map text to vectors in a continuous embedding space. Then we can train a diffusion model on `(prompt,` `image)` pairs, where each prompt is a short, textual description of the input image.

We can handle the image input in the same way as we did previously, mapping noisy input to a denoised output that progressively approaches our input image. Critically, we can extend this setup by also passing the embedded text prompt to the denoising model. So rather than our denoising model simply taking in a `noisy_images` input, our model will take two inputs: `noisy_images` and `text_embeddings`. This gives a leg up on the flower denoiser we trained previously. Instead of learning to remove noise from an image without any additional information, the model gets to use a textual representation of the final image to help guide the denoising process. After training is when things get a bit more fun. Because we have trained a model that can map pure noise to images conditioned on

### PDF p.553 (printed 529) — 17 Image generation › 17.2 Diffusion models › 17.2.7 It’s go time!
model = DiffusionModel(image_size, widths=[32, 64, 96, 128], block_depth=2)
model.normalizer.adapt(dataset)

### PDF p.549 (printed 525) — 17 Image generation › 17.2 Diffusion models › 
Let’s plot how this function maps diffusion times (between 0 and 1) to specific noise rates and signal rates (see figure 17.11):

### PDF p.549 (printed 525) — 17 Image generation › 17.2 Diffusion models › 
diffusion_times = ops.arange(0.0, 1.0, 0.01)
noise_rates, signal_rates = diffusion_schedule(diffusion_times)
diffusion_times = ops.convert_to_numpy(diffusion_times)
noise_rates = ops.convert_to_numpy(noise_rates)
signal_rates = ops.convert_to_numpy(signal_rates)
plt.plot(diffusion_times, noise_rates, label="Noise rate")
plt.plot(diffusion_times, signal_rates, label="Signal rate")
plt.xlabel("Diffusion time")
plt.legend()

### PDF p.544 (printed 520) — 17 Image generation › 17.2 Diffusion models › 
higher-resolution version of what you give them. This can potentially lead to funny mishaps. For instance, with some AI-enhanced cameras, you can take a picture of something that looks vaguely moon-like (such as a printout of a severely blurred moon image), and you will get in your camera roll a crisp picture of the moon’s craters. A lot of detail that simply wasn’t present in the printout gets straight-up hallucinated by the camera, because the super-resolution model it uses is overfitted to moon photography images. So, unlike Rick Deckard, definitely don’t use this technique for forensics!

Early successes in image denoising led researchers to an arresting idea: since you can use an autoencoder to remove a small amount of noise from an image, surely it would be possible to repeat the process multiple times in a loop to remove a large amount of noise. Ultimately, could you denoise an image made out of pure noise? As it turns out, yes, you can. By doing this, you can effectively hallucinate brand new images out of nothing, like in figure 17.7. This is the key insight behind diffusion models, which should more accurately be called reverse diffusion models, since “diffusion” refers t
