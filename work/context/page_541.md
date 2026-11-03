# Generation bundle for page_541  (PDF page 541, printed page 517)

Page kind: content
Chapter: 17 Image generation
Section: 17.1 Deep learning for image generation
Subsection: 17.1.3 Implementing a VAE with Keras
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_541.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_541.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p541_e03] (code)
```
        original = x
        z_mean, z_log_var = y_pred
        reconstruction = self.decoder(self.sampler(z_mean, z_log_var))
        reconstruction_loss = ops.mean(
            ops.sum(
                keras.losses.binary_crossentropy(x, reconstruction), axis=(1, 2)
            )
        )
        kl_loss = -0.5 * (
            1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var)
        )
        total_loss = reconstruction_loss + ops.mean(kl_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return total_loss
```
[p541_e04] (body) Finally, you’re ready to instantiate and train the model on MNIST digits. Because `compute_loss()` already takes care of the loss, you don’t specify an external loss at compile time (`loss=None`), which, in turn, means you won’t pass target data during training (as you can see, you only pass `x_train` to the model in `fit`).
[p541_e05] (listing_caption) Listing 17.5 Training the VAE
[p541_e06] (code)
```
import numpy as np
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(mnist_digits, epochs=30, batch_size=128)
```
[p541_e07] (body) Once the model is trained, you can use the `decoder` network to turn arbitrary latent space vectors into images.

## Code annotations printed next to listings (callouts, in order)
- Argument
- This is our
- y_pred is
- reconstructed
- Argument x is
- the output
- image.
- the model’s input.
- of call().
- Updates the state of
- our loss-tracking metrics
- We sum the reconstruction loss over the
- spatial dimensions (axes 1 and 2) and take
- Adds the regularization term
- its mean over the batch dimension.
- (Kullback–Leibler divergence)
- We train on all MNIST digits, so we
- concatenate the training and test samples.
- We don’t pass a
- loss argument in
- compile(), since the
- loss is already part
- We don’t pass targets in fit(), since
- of the train_step().
- train_step() doesn’t expect any.

## Listings
- Listing 17.5: Training the VAE (code elements p541_e06)

## Equation candidates (formulas written inline in the text)
- [p541_e04] Because `compute_loss()` already takes care of the loss, you don’t specify an external loss at compile time (`loss=None`), which, in turn, means you won’t pass target data during training (as you can see, you only pass `x_train` to the model in `fit`).

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

## Previous page (PDF 540) text, for continuity
Now, let’s create the VAE model itself. This is your first example of a model that isn’t doing supervised learning (an autencoder is an example of self-supervised learning because it uses its inputs as targets). Whenever you depart from classic supervised learning, it’s common to subclass the `Model` class and implement a custom `train_` `step()` to specify the new training logic, a workflow you’ve learned about in chapter 7. We could easily do that here, but a downside of this technique is that the `train_step()` contents must be backend specific—you’d use `GradientTape` with TensorFlow, you’d use `loss.backward()` with PyTorch, and so on. A simpler way to customize your training logic is to just implement the `compute_loss()` method instead and keep the default `train_step()`. `compute_loss()` is the key bit of differentiable logic called by the built-in `train_step()`. Since it doesn’t involve direct manipulation of gradients, it’s easy to keep it backend agnostic.

Its signature is as follows:

where `x` is the model’s input; `y` is the model’s target (in our case, it is `None` since the dataset we use only has inputs, no targets); and `y_pred` is the output of `call()`—the model’s predictions. In any supervised training workflow, you’d compute the loss based on `y` and `y_pred`. In our case, since `y` is `None` and `y_pred` contains the latent parameters, we’ll compute the loss using `x` (the original input) and the `reconstruction` derived from `y_pred`.

The method must return a scalar, the loss value to be minimized. You can also use `compute_loss()` to update the state of your metrics, which is something we’ll want to do in our case. Now, let’s write our VAE with a custom `compute_loss()` method. It works with all backends with no code changes!

Listing 17.4 VA …

## Next page (PDF 542) text, for continuity
Listing 17.6 Sampling a grid of points from the 2D latent space and decoding them to images

The grid of sampled digits (see figure 17.5) shows a completely continuous distribution of the different digit classes, with one digit morphing into another as you follow a path through latent space. Specific directions in this space have a meaning: for example, there’s a direction for “four-ness,” “one-ness,” and so on.

In the next section, we’ll cover in detail another major tool for generating images: diffusion models, the architecture behind nearly all commercial image generation services today.

17.2 Diffusion models

A long-standing application of autoencoders has been denoising: feeding into a model an input that features a small amount of noise—for instance, a low-quality JPEG image— and getting back a cleaned-up version of the same input. This is the one task that autoencoders excel at. In the late 2010s, this idea gave rise to very successful image super-resolution models, capable of taking in low-resolution, potentially noisy images and outputting high-quality, high-resolution versions of them (see figure 17.6). Such models have been shipped as part of every major smartphone camera app for the past few years.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.537 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.538 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.536 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.539 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.535 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.550 — 17 Image generation › 17.2 Diffusion models
- PDF p.258 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.557 — 17 Image generation › 17.3 Text-to-image models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.537 (printed 513) — 17 Image generation › 17.1 Deep learning for image generation › 17.1.3 Implementing a VAE with Keras
We’re going to be implementing a VAE that can generate MNIST digits. It’s going to have three parts:

An encoder network that turns a real image into a mean and a variance in the latent space

A sampling layer that takes such a mean and variance and uses them to sample a random point from the latent space

A decoder network that turns points from the latent space back into images

The following listing shows the encoder network you’ll use, mapping images to the parameters of a probability distribution over the latent space. It’s a simple ConvNet that maps the input image `x` to two vectors, `z_mean` and `z_log_var`. One important detail is that we use strides for downsampling feature maps, instead of max pooling. The last time we did this was in the image segmentation example of chapter 11. Recall that, in general, strides are preferable to max pooling for any model that cares about information location—that is, where stuff is in the image—and this one does, since it will have to produce an image encoding that can be used to reconstruct a valid image.

Listing 17.1 VAE encoder network

### PDF p.536 (printed 512) — 17 Image generation › 17.1 Deep learning for image generation › 
You can then train the model using the reconstruction loss and the regularization loss. For the regularization loss, we typically use an expression (the Kullback–Leibler divergence) meant to nudge the distribution of the encoder output toward a well-rounded normal distribution centered around 0. This provides the encoder with a sensible assumption about the structure of the latent space it’s modeling.

Now let’s see what implementing a VAE looks like in practice!

### PDF p.539 (printed 515) — 17 Image generation › 17.1 Deep learning for image generation › 
Listing 17.3 VAE decoder network, mapping latent space points to images
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(
    x
)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(
    x
)
decoder_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

### PDF p.536 (printed 512) — 17 Image generation › 17.1 Deep learning for image generation › 
and `z_log_variance`

Figure 17.4 A VAE maps an image to two vectors, `z_mean` and `z_log_variance`, which define a probability distribution over the latent space, used to sample a latent point to decode.

similar to `input_img`, thus forcing the latent space to be continuously meaningful. Any two close points in the latent space will decode to highly similar images. Continuity, combined with the low dimensionality of the latent space, forces every direction in the latent space to encode a meaningful axis of variation of the data, making the latent space very structured and thus highly suitable to manipulation via concept vectors.

The parameters of a VAE are trained via two loss functions: a reconstruction loss that forces the decoded samples to match the initial inputs, and a regularization loss that helps learn well-rounded latent distributions and reduces overfitting to the training data. Schematically, the process looks like this:

### PDF p.535 (printed 511) — 17 Image generation › 17.1 Deep learning for image generation › 
the original image, via a decoder module (see figure 17.3). It’s then trained by using as target data the same images as the input images, meaning the autoencoder learns to reconstruct the original inputs. By imposing various constraints on the code (the output of the encoder), you can get the autoencoder to learn more or less interesting latent representations of the data. Most commonly, you’ll constrain the code to be low dimensional and sparse (mostly zeros), in which case the encoder acts as a way to compress the input data into fewer bits of information.

Figure 17.3 An autoencoder: mapping an input `x` to a compressed representation and then decoding it back as `x'`

In practice, such classical autoencoders don’t lead to particularly useful or nicely structured latent spaces. They’re not much good at compression either. For these reasons, they have largely fallen out of fashion. VAEs, however, augment autoencoders with a little bit of statistical magic that forces them to learn continuous, highly structured latent spaces. They have turned out to be a powerful tool for image generation.

A VAE, instead of compressing its input image into a fixed code in the latent space, turns

### PDF p.550 (printed 526) — 17 Image generation › 17.2 Diffusion models › 
Next comes the training logic. This is the most important part! Like in the VAE example, we’re going to implement a custom `compute_loss()` method to keep our model backend agnostic. Of course, if you are set on using one specific backend, you could also write a custom `train_step()` with the exact same logic in it, plus the backend-specific logic for gradient computation and weight updates.

Since `compute_loss()` receives as input the output of `call()`, we’re going to put the denoising forward pass in `call()`. Our `call()` takes a batch of clean input images and applies the following steps: 1 Normalizes the images 2 Samples random diffusion times (the denoising model needs to be trained on

the full spectrum of diffusion times) 3 Computes corresponding noise rates and signal rates (using the diffusion schedule) 4 Adds random noise to the clean images (based on the computed noise rates and signal rates) 5 Denoises the images

### PDF p.537 (printed 513) — 17 Image generation › 17.1 Deep learning for image generation › 17.1.3 Implementing a VAE with Keras
Its summary looks like this:

### PDF p.41 (printed 17) — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network › A first look at a neural network
Let’s look at a concrete example of a neural network that uses the machine learning library Keras to learn to classify handwritten digits. We will use Keras extensively throughout this book. It’s a simple, high-level library that will allow us to stay focused on the concepts we would like to cover.

Unless you already have experience with Keras or similar libraries, you won’t understand everything about this first example right away. That’s fine. In a few sections, we’ll review each element in the example and explain it in detail. So don’t worry if some steps seem arbitrary or look like magic to you! We’ve got to start somewhere. The problem we’re trying to solve here is to classify grayscale images of handwritten digits (28 × 28 pixels) into their 10 categories (0 through 9). We’ll use the MNIST dataset, a classic in the machine learning community, which has been around almost as long as the field itself and has been intensively studied. It’s a set of 60,000 training images, plus 10,000 test images, assembled by the National Institute of Standards and Technology (the NIST in MNIST) in the 1980s. You can think of “solving” MNIST as the “Hello World” of deep learning—it’s what you d
