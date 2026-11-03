# Generation bundle for page_552  (PDF page 552, printed page 528)

Page kind: content
Chapter: 17 Image generation
Section: 17.2 Diffusion models
Subsection: 17.2.6 Visualizing results with a custom callback
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_552.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_552.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p552_e03] (code)
```
            pred_images, pred_noises = self.denoise(
                noisy_images, noise_rates, signal_rates
            )
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = diffusion_schedule(
                next_diffusion_times
            )
            noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
        images = (
            self.normalizer.mean + pred_images * self.normalizer.variance**0.5
        )
        return ops.clip(images, 0.0, 255.0)
```
[p552_e04] (heading_subsection) 17.2.6 Visualizing results with a custom callback
[p552_e05] (body) We don’t have a proper metric to judge the quality of our generated images, so you’re going to want to visualize the generated images yourself over the course of training to judge if your model is getting somewhere. An easy way to do this is with a custom callback. The following callback uses the `generate()` method at the end of each epoch to display a 3 × 6 grid of generated images:
[p552_e06] (code)
```
class VisualizationCallback(keras.callbacks.Callback):
    def __init__(self, diffusion_steps=20, num_rows=3, num_cols=6):
        self.diffusion_steps = diffusion_steps
        self.num_rows = num_rows
        self.num_cols = num_cols
    def on_epoch_end(self, epoch=None, logs=None):
        generated_images = self.model.generate(
            num_images=self.num_rows * self.num_cols,
            diffusion_steps=self.diffusion_steps,
        )
        plt.figure(figsize=(self.num_cols * 2.0, self.num_rows * 2.0))
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                i = row * self.num_cols + col
                plt.subplot(self.num_rows, self.num_cols, i + 1)
                img = ops.convert_to_numpy(generated_images[i]).astype("uint8")
                plt.imshow(img)
                plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()
```

## Code annotations printed next to listings (callouts, in order)
- Calls denoising model
- Denormalizes images so their
- values fit between 0 and 255
- Prepares noisy images for the next iteration

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

## Previous page (PDF 551) text, for continuity
It returns

The predicted denoised images

The predicted noise masks

The actual noise masks it applied

These last two quantities are then used in `compute_loss()` to compute the loss of the model on the noise mask prediction task:

17.2.5 The generation process

Finally, let’s implement the image generation process. We start from pure random noise, and we repeatedly apply the `denoise()` method until we get high-signal, low-noise images.

## Next page (PDF 553) text, for continuity
17.2.7 It’s go time!

It’s finally time to train our diffusion model on the Oxford Flowers dataset. Let’s instantiate the model:

We’re going to use `AdamW` as our optimizer, with a few neat options enabled to help stabilize training and improve the quality of the generated images:

Learning rate decay—We gradually reduce the learning rate during training, via an `InverseTimeDecay` schedule.

Exponential moving average of model weights—Also known as Polyak averaging. This technique maintains a running average of the model’s weights during training. Every 100 batches, we overwrite the model’s weights with this averaged set of weights. This helps stabilize the model’s representations in scenarios where the loss landscape is noisy.

The code is

Let’s fit the model. We’ll use our `VisualizationCallback` callback to plot examples of generated images after each epoch, and we’ll save the model’s weights with the `ModelCheckpoint` callback:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.550 — 17 Image generation › 17.2 Diffusion models
- PDF p.549 — 17 Image generation › 17.2 Diffusion models
- PDF p.545 — 17 Image generation › 17.2 Diffusion models
- PDF p.557 — 17 Image generation › 17.3 Text-to-image models
- PDF p.547 — 17 Image generation › 17.2 Diffusion models
- PDF p.558 — 17 Image generation › 17.3 Text-to-image models
- PDF p.235 — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops
- PDF p.540 — 17 Image generation › 17.1 Deep learning for image generation

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.235 (printed 211) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › 
Listing 7.20 Creating a custom callback by subclassing the `Callback` class
from matplotlib import pyplot as plt
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.per_batch_losses = []
    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get("loss"))
    def on_epoch_end(self, epoch, logs):
        plt.clf()
        plt.plot(
            range(len(self.per_batch_losses)),
            self.per_batch_losses,
            label="Training loss for each batch",
        )
        plt.xlabel(f"Batch (epoch {epoch})")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"plot_at_epoch_{epoch}", dpi=300)
        self.per_batch_losses = []

### PDF p.236 (printed 212) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › 
We get plots that look like figure 7.5.

Figure 7.5 The output of our custom history– plotting callback

### PDF p.235 (printed 211) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › 
These methods are all called with a `logs` argument, which is a dictionary containing information about the previous batch, epoch, or training run: training and validation metrics, and so on. The `on_epoch_*` and `on_batch_*` methods also take the epoch or batch index as first argument (an integer).

Here’s a simple example callback that saves a list of per-batch loss values during training and plots these values at the end of each epoch.

Listing 7.20 Creating a custom callback by subclassing the `Callback` class

### PDF p.550 (printed 526) — 17 Image generation › 17.2 Diffusion models › 
Next comes the training logic. This is the most important part! Like in the VAE example, we’re going to implement a custom `compute_loss()` method to keep our model backend agnostic. Of course, if you are set on using one specific backend, you could also write a custom `train_step()` with the exact same logic in it, plus the backend-specific logic for gradient computation and weight updates.

Since `compute_loss()` receives as input the output of `call()`, we’re going to put the denoising forward pass in `call()`. Our `call()` takes a batch of clean input images and applies the following steps: 1 Normalizes the images 2 Samples random diffusion times (the denoising model needs to be trained on

the full spectrum of diffusion times) 3 Computes corresponding noise rates and signal rates (using the diffusion schedule) 4 Adds random noise to the clean images (based on the computed noise rates and signal rates) 5 Denoises the images

### PDF p.233 (printed 209) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › The EarlyStopping and ModelCheckpoint callbacks
When you’re training a model, there are many things you can’t predict at the start. In particular, you can’t tell how many epochs will be needed to get to an optimal validation loss. Our examples so far have adopted the strategy of training for enough epochs that you begin overfitting, using the first run to figure out the optimal number of epochs, and then finally launching a new training run from scratch using this optimal number. Of course, this approach is wasteful. A much better way to handle this is to stop training when you measure that the validation loss is no longer improving. This can be achieved using the `EarlyStopping` callback.

The `EarlyStopping` callback interrupts training once a target metric being monitored has stopped improving for a fixed number of epochs. For instance, this callback allows you to interrupt training as soon as you start overfitting, thus avoiding having to retrain your model for a smaller number of epochs. This callback is typically used in combination with `ModelCheckpoint`, which lets you continually save the model during training (and, optionally, save only the current best model so far: the version of the model that achieved the best perf

### PDF p.557 (printed 533) — 17 Image generation › 17.3 Text-to-image models › 
always try its best to interpolate an output based on its training data, but it doesn’t have any real understanding of the objects it is attempting to represent. However, there is another factor that is easily fixable: we are using the less powerful version of Stable Diffusion 3. The “medium” model we are using is the smallest one released by Stability AI and uses about 3 billion parameters in total. There is a larger 9-billion parameter model available that would produce substantially higher-quality images with fewer visual artifacts. We do not use it simply to keep the code example in this book accessible—9 billion parameters need a lot of RAM!

Like the `generate()` method for text models we used in the last chapter, we have a few additional parameters we can pass to control the generation process. Let’s try passing a variable number of diffusion steps to our model to see the denoising process in action (figure 17.15):

### PDF p.561 (printed 537) — 17 Image generation › 17.3 Text-to-image models › Summary
Image generation with deep learning is done by learning latent spaces that capture statistical information about a dataset of images. By sampling and decoding points from the latent space, you can generate never-before-seen images. There are three major tools to do this: VAEs, diffusion models, and GANs.

VAEs result in highly structured, continuous latent representations. For this reason, they work well for doing all sorts of image editing in latent space: face swapping, turning a frowning face into a smiling face, and so on. They also work nicely for doing latent space–based animations, such as animating a walk along a cross section of the latent space, showing a starting image slowly morphing into different images in a continuous way.

Diffusion models result in very realistic outputs and are the dominant method of image generation today. They work by repeatedly denoising an image, starting from pure noise. They can easily be conditioned on text captions to create text-to-image models.

Stable Diffusion 3 is a state-of-the-art pretrained text-to-image model that you can use to create highly realistic images of your own.

The visual latent space learned by such text-to-image diff

### PDF p.533 (printed 509) — 17 Image generation › 17.1 Deep learning for image generation › 17.1.1 Sampling from latent spaces of images
The key idea of image generation is to develop a low-dimensional latent space of representations (which, like everything else in deep learning, is a vector space) where any point can be mapped to a “valid” image: an image that looks like the real thing. The module capable of realizing this mapping, taking as input a latent point and outputting an image (a grid of pixels), is usually called a generator, or sometimes a decoder. Once such a latent space has been learned, you can sample points from it, and, by mapping them back to image space, generate images that have never been seen before (see figure 17.1)—the in-betweens of the training images.

Figure 17.1 Using a latent vector space to sample new images

Further, text-conditioning makes it possible to map a space of prompts in natural language to the latent space (see figure 17.2), making it possible to do language-guided image generation—generating pictures that correspond to a text description. This category of models is called text-to-image models.

Interpolating between many training images in the latent space enables such models to generate infinite combinations of visual concepts, including many that no one had explicitly c
