# Generation bundle for page_546  (PDF page 546, printed page 522)

Page kind: content
Chapter: 17 Image generation
Section: 17.2 Diffusion models
Subsection: 17.2.2 A U-Net denoising autoencoder
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_546.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_546.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p546_e03] (body) extract for a given input image—that’s the `noise_rates` input. Rather than outputting a denoised image, we make our model output a predicted noise mask, which we can subtract from the input to denoise it.
[p546_e04] (body) For our denoising model, we’re going to use a U-Net—a kind of ConvNet originally developed for image segmentation. It looks like figure 17.9.
[p546_e05] (figure_caption) Figure 17.9 Our U-Net-style denoising autoencoder architecture
[p546_e06] (body) This architecture features three stages:
[p546_e07] (body) 1 A downsampling stage, made of several blocks of convolution layers, where the
[p546_e08] (body) inputs get downsampled from their original 128 × 128 size down to a much smaller size (in our case, 16 × 16). 2 A middle stage, where the feature map has a constant size. 3 An upsampling stage, where the feature map get upsampled back to 128 × 128. There is a 1:1 mapping between the blocks of the downsampling and upsampling stages: each upsampling block is the inverse of a downsampling block. Importantly, the model features concatenative residual connections going from each downsampling block to the corresponding upsampling block. These connections help avoid loss of image detail information across the successive downsampling and upsampling operations. Let’s assemble the model using the Functional API:
[p546_e09] (code)
```
def residual_block(x, width):
    input_width = x.shape[3]
    if input_width == width:
        residual = x
    else:
        residual = layers.Conv2D(width, 1)(x)
    x = layers.BatchNormalization(center=False, scale=False)(x)
```

## Code annotations printed next to listings (callouts, in order)
- Utility function to apply a block of
- layers with a residual connection

## Figures
- Figure 17.9: Our U-Net-style denoising autoencoder architecture  | labels inside the figure: Concat, Noise rate, 128 × 128, 128 × 128, 16 × 16, feature map, 16 × 16, feature map, Concat, Concat, Concat

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

## Previous page (PDF 545) text, for continuity
`fpath` is now the local path to the extracted directory. The images are contained in the `jpg` subdirectory there. Let’s turn them into an iterable dataset using `image_dataset_from_directory()`.

We need to resize our images to a fixed size, but we don’t want to distort their aspect ratio since this would negatively affect the quality of our generated images, so we use the `crop_to_aspect_ratio` option to extract maximally large undistorted crops of the right size (128 × 128):

Here’s an example image (figure 17.8):

17.2.2 A U-Net denoising autoencoder

The same denoising model gets reused across each iteration of the diffusion denoising process, erasing a little bit of noise each time. To make the job of the model easier,

Figure 17.8 An example image from the

we tell it how much noise it is supposed to

Oxford Flowers dataset

## Next page (PDF 547) text, for continuity
You would instantiate the model with something like `get_model(image_size=128, ` `widths=[32,` `64,` `96,` `128],` `block_depth=2)`. The `widths` argument is a list containing the `Conv2D` layer sizes for each successive downsampling or upsampling stage. We typically want the layers to get bigger as we downsample the inputs (going from 32 to 128 units here) and then get smaller as as upsample (from 128 back to 32 here).

17.2.3 The concepts of diffusion time and diffusion schedule

The diffusion process is a series of steps in which we apply our denoising autoencoder to erase a small amount of noise from an image, starting with a pure-noise image, and ending with a pure-signal image. The index of the current step in the loop is called the diffusion time (see figure 17.7). In our case, we’ll use a continuous value between 1

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.297 — 9 ConvNet architecture patterns › 9.2 Residual connections
- PDF p.339 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.550 — 17 Image generation › 17.2 Diffusion models
- PDF p.265 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.260 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.542 — 17 Image generation › 17.2 Diffusion models
- PDF p.305 — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model
- PDF p.298 — 9 ConvNet architecture patterns › 9.2 Residual connections

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.644 (printed 620) — index ›  › U
attribute 262

underfitting 137–142 parameters 39

ambiguous features 139 weights 215 noisy training data 138 training rare features and spurious correlations 140–142 diffusion models 525 U-Net denoising autoencoder 522 inference vs. 215 unigrams 403 lower-precision computation 556–562 universal workflow of machine learning training arguments 215, 218 deploying model 183–189 training data 147 developing model 179–183 training loops 10, 39, 206–214 update_state() method 207, 222 callbacks 208, 210 upsampling stage 522 monitoring and visualization with use_casual_mask 448 TensorBoard 212 using fit() with custom training loops 222–225

### PDF p.555 (printed 531) — 17 Image generation › 17.3 Text-to-image models › 17.3 Text-to-image models
We can use the same basic diffusion process to create a model that maps text input to image output. To do this we need to take a pretrained text encoder (think a transformer encoder like RoBERTa from chapter 15) that can map text to vectors in a continuous embedding space. Then we can train a diffusion model on `(prompt,` `image)` pairs, where each prompt is a short, textual description of the input image.

We can handle the image input in the same way as we did previously, mapping noisy input to a denoised output that progressively approaches our input image. Critically, we can extend this setup by also passing the embedded text prompt to the denoising model. So rather than our denoising model simply taking in a `noisy_images` input, our model will take two inputs: `noisy_images` and `text_embeddings`. This gives a leg up on the flower denoiser we trained previously. Instead of learning to remove noise from an image without any additional information, the model gets to use a textual representation of the final image to help guide the denoising process. After training is when things get a bit more fun. Because we have trained a model that can map pure noise to images conditioned on

### PDF p.550 (printed 526) — 17 Image generation › 17.2 Diffusion models › 
The first method we’re going to need is the denoising method. It simply calls the denoising model to retrieve a predicted noise mask, and it uses it to reconstruct a denoised image:

### PDF p.544 (printed 520) — 17 Image generation › 17.2 Diffusion models › 
A diffusion model is essentially a denoising autoencoder in a loop, capable of turning pure noise into sharp, realistic imagery. You may know this poetic quote from Michelangelo, “Every block of stone has a statue inside it and it is the task of the sculptor to discover it”—well, every square of white noise has an image inside it, and it is the task of the diffusion model to discover it.

Now, let’s build one with Keras.

### PDF p.542 (printed 518) — 17 Image generation › 17.2 Diffusion models › 17.2 Diffusion models
A long-standing application of autoencoders has been denoising: feeding into a model an input that features a small amount of noise—for instance, a low-quality JPEG image— and getting back a cleaned-up version of the same input. This is the one task that autoencoders excel at. In the late 2010s, this idea gave rise to very successful image super-resolution models, capable of taking in low-resolution, potentially noisy images and outputting high-quality, high-resolution versions of them (see figure 17.6). Such models have been shipped as part of every major smartphone camera app for the past few years.

### PDF p.265 (printed 241) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
the initial input. The high-level patterns learned by the ConvNet will still be very small with regard to the initial input, which may not be enough to learn to classify digits (try recognizing a digit by only looking at it through windows that are 7 × 7 pixels!). We need the features from the last convolution layer to contain information about the totality of the input.

The final feature map has dimensions 22 × 22. That’s huge—when you take the average of each 22 × 22 feature map, you are going to be destroying a lot of information compared to when your feature maps were only 3 × 3.

In short, the reason to use downsampling is to reduce the size of the feature maps, making the information they contain increasingly less spatially distributed and increasingly contained in the channels, while also inducing spatial-filter hierarchies by making successive convolution layers “look” at increasingly large windows (in terms of the fraction of the original input image they cover).

### PDF p.549 (printed 525) — 17 Image generation › 17.2 Diffusion models › 17.2.4 The training process
Let’s create a `DiffusionModel` class to implement the training procedure. It’s going to have our denoising autoencoder as one of its attributes. We’re also going to need a couple more things:

A loss function—We’ll use mean absolute error as our loss, that is to say `mean` `(abs(real_noise_mask - predicted_noise_mask)).`

An image normalization layer—The noise we’ll add to the images will have unit variance and zero mean, so we’d like our images to be normalized as such too, for the value range of the noise to match the value range of the images.

### PDF p.297 (printed 273) — 9 ConvNet architecture patterns › 9.2 Residual connections › 
Note that adding the input back to the output of a block implies that the output should have the same shape as the input. This is not the case if your block includes convolutional layers with an increased number of filters or a max pooling layer. In such cases, use a 1 × 1 `Conv2D` layer with no activation to linearly project the residual to the desired output shape. You’d typically use `padding="same"` in the convolution layers in your target block to avoid spatial downsampling due to padding, and you’d use strides in the residual projection to match any downsampling caused by a max pooling layer.

Listing 9.2 The target block changing the number of output filters
