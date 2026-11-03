# Generation bundle for page_551  (PDF page 551, printed page 527)

Page kind: content
Chapter: 17 Image generation
Section: 17.2 Diffusion models
Subsection: 17.2.5 The generation process
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_551.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_551.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p551_e03] (body) It returns
[p551_e04] (bullet) The predicted denoised images
[p551_e05] (bullet) The predicted noise masks
[p551_e06] (bullet) The actual noise masks it applied
[p551_e07] (body) These last two quantities are then used in `compute_loss()` to compute the loss of the model on the noise mask prediction task:
[p551_e08] (code)
```
    def call(self, images):
        images = self.normalizer(images)
        noise_masks = keras.random.normal(
            (batch_size, self.image_size, self.image_size, 3),
            seed=self.seed_generator,
        )
        diffusion_times = keras.random.uniform(
            (batch_size, 1, 1, 1),
            minval=0.0,
            maxval=1.0,
            seed=self.seed_generator,
        )
        noise_rates, signal_rates = diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noise_masks
        pred_images, pred_noise_masks = self.denoise(
            noisy_images, noise_rates, signal_rates
        )
        return pred_images, pred_noise_masks, noise_masks
    def compute_loss(self, x, y, y_pred, sample_weight=None, training=True):
        _, pred_noise_masks, noise_masks = y_pred
        return self.loss(noise_masks, pred_noise_masks)
```
[p551_e09] (heading_subsection) 17.2.5 The generation process
[p551_e10] (body) Finally, let’s implement the image generation process. We start from pure random noise, and we repeatedly apply the `denoise()` method until we get high-signal, low-noise images.
[p551_e11] (code)
```
    def generate(self, num_images, diffusion_steps):
        noisy_images = keras.random.normal(
            (num_images, self.image_size, self.image_size, 3),
            seed=self.seed_generator,
        )
        step_size = 1.0 / diffusion_steps
        for step in range(diffusion_steps):
            diffusion_times = ops.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = diffusion_schedule(diffusion_times)
```

## Code annotations printed next to listings (callouts, in order)
- Samples random
- noise masks
- Samples random
- diffusion times
- Adds noise to the images
- Denoises them
- Computes appropriate
- noise rates and signal rates
- Starts from pure noise

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

## Previous page (PDF 550) text, for continuity
Let’s start by writing the model constructor:

The first method we’re going to need is the denoising method. It simply calls the denoising model to retrieve a predicted noise mask, and it uses it to reconstruct a denoised image:

Next comes the training logic. This is the most important part! Like in the VAE example, we’re going to implement a custom `compute_loss()` method to keep our model backend agnostic. Of course, if you are set on using one specific backend, you could also write a custom `train_step()` with the exact same logic in it, plus the backend-specific logic for gradient computation and weight updates.

Since `compute_loss()` receives as input the output of `call()`, we’re going to put the denoising forward pass in `call()`. Our `call()` takes a batch of clean input images and applies the following steps: 1 Normalizes the images 2 Samples random diffusion times (the denoising model needs to be trained on

the full spectrum of diffusion times) 3 Computes corresponding noise rates and signal rates (using the diffusion schedule) 4 Adds random noise to the clean images (based on the computed noise rates and signal rates) 5 Denoises the images

## Next page (PDF 552) text, for continuity
17.2.6 Visualizing results with a custom callback

We don’t have a proper metric to judge the quality of our generated images, so you’re going to want to visualize the generated images yourself over the course of training to judge if your model is getting somewhere. An easy way to do this is with a custom callback. The following callback uses the `generate()` method at the end of each epoch to display a 3 × 6 grid of generated images:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.549 — 17 Image generation › 17.2 Diffusion models
- PDF p.547 — 17 Image generation › 17.2 Diffusion models
- PDF p.538 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.544 — 17 Image generation › 17.2 Diffusion models
- PDF p.558 — 17 Image generation › 17.3 Text-to-image models
- PDF p.548 — 17 Image generation › 17.2 Diffusion models
- PDF p.540 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.557 — 17 Image generation › 17.3 Text-to-image models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.558 (printed 534) — 17 Image generation › 17.3 Text-to-image models › 
Our generation process has three distinct steps:

1 First, we take our prompts, tokenize them, and embed them with our text encoder. 2 Second, we take our text embeddings and pure noise and progressively “denoise”

the noise into an image. This is the same as the flower model we just built. 3 Lastly, we map our model outputs, which are from `[-1, 1]` back to `[0, 255]`, so we can render the image. One thing to note here is that our text embeddings actually contain four separate tensors:

### PDF p.549 (printed 525) — 17 Image generation › 17.2 Diffusion models › 17.2.4 The training process
Let’s create a `DiffusionModel` class to implement the training procedure. It’s going to have our denoising autoencoder as one of its attributes. We’re also going to need a couple more things:

A loss function—We’ll use mean absolute error as our loss, that is to say `mean` `(abs(real_noise_mask - predicted_noise_mask)).`

An image normalization layer—The noise we’ll add to the images will have unit variance and zero mean, so we’d like our images to be normalized as such too, for the value range of the noise to match the value range of the images.

### PDF p.545 (printed 521) — 17 Image generation › 17.2 Diffusion models › 17.2.2 A U-Net denoising autoencoder
The same denoising model gets reused across each iteration of the diffusion denoising process, erasing a little bit of noise each time. To make the job of the model easier,

Figure 17.8 An example image from the

we tell it how much noise it is supposed to

Oxford Flowers dataset

### PDF p.547 (printed 523) — 17 Image generation › 17.2 Diffusion models › 17.2.3 The concepts of diffusion time and diffusion schedule
The diffusion process is a series of steps in which we apply our denoising autoencoder to erase a small amount of noise from an image, starting with a pure-noise image, and ending with a pure-signal image. The index of the current step in the loop is called the diffusion time (see figure 17.7). In our case, we’ll use a continuous value between 1

### PDF p.555 (printed 531) — 17 Image generation › 17.3 Text-to-image models › 17.3 Text-to-image models
We can use the same basic diffusion process to create a model that maps text input to image output. To do this we need to take a pretrained text encoder (think a transformer encoder like RoBERTa from chapter 15) that can map text to vectors in a continuous embedding space. Then we can train a diffusion model on `(prompt,` `image)` pairs, where each prompt is a short, textual description of the input image.

We can handle the image input in the same way as we did previously, mapping noisy input to a denoised output that progressively approaches our input image. Critically, we can extend this setup by also passing the embedded text prompt to the denoising model. So rather than our denoising model simply taking in a `noisy_images` input, our model will take two inputs: `noisy_images` and `text_embeddings`. This gives a leg up on the flower denoiser we trained previously. Instead of learning to remove noise from an image without any additional information, the model gets to use a textual representation of the final image to help guide the denoising process. After training is when things get a bit more fun. Because we have trained a model that can map pure noise to images conditioned on

### PDF p.544 (printed 520) — 17 Image generation › 17.2 Diffusion models › 
A diffusion model is essentially a denoising autoencoder in a loop, capable of turning pure noise into sharp, realistic imagery. You may know this poetic quote from Michelangelo, “Every block of stone has a statue inside it and it is the task of the sculptor to discover it”—well, every square of white noise has an image inside it, and it is the task of the diffusion model to discover it.

Now, let’s build one with Keras.

### PDF p.548 (printed 524) — 17 Image generation › 17.2 Diffusion models › 
and 0 for this index—a value of 1 indicates the start of the process, where the amount of noise is maximal and the amount of signal is minimal, and a value of 0 indicates the end of the process, where the image is almost all signal and no noise.

The relationship between the current diffusion time and the amount of noise and signal present in the image is called the diffusion schedule. In our experiment, we’re going to use a cosine schedule to smoothly transition from a high signal rate (low noise) at the beginning to a low signal rate (high noise) at the end of the diffusion process.

Listing 17.7 The diffusion schedule

### PDF p.549 (printed 525) — 17 Image generation › 17.2 Diffusion models › 
Let’s plot how this function maps diffusion times (between 0 and 1) to specific noise rates and signal rates (see figure 17.11):
