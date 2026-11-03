# Generation bundle for page_553  (PDF page 553, printed page 529)

Page kind: content
Chapter: 17 Image generation
Section: 17.2 Diffusion models
Subsection: 17.2.7 It’s go time!
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_553.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_553.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p553_e03] (heading_subsection) 17.2.7 It’s go time!
[p553_e04] (body) It’s finally time to train our diffusion model on the Oxford Flowers dataset. Let’s instantiate the model:
[p553_e05] (code)
```
model = DiffusionModel(image_size, widths=[32, 64, 96, 128], block_depth=2)
model.normalizer.adapt(dataset)
```
[p553_e06] (body) We’re going to use `AdamW` as our optimizer, with a few neat options enabled to help stabilize training and improve the quality of the generated images:
[p553_e07] (bullet) Learning rate decay—We gradually reduce the learning rate during training, via an `InverseTimeDecay` schedule.
[p553_e08] (bullet) Exponential moving average of model weights—Also known as Polyak averaging. This technique maintains a running average of the model’s weights during training. Every 100 batches, we overwrite the model’s weights with this averaged set of weights. This helps stabilize the model’s representations in scenarios where the loss landscape is noisy.
[p553_e09] (body) The code is
[p553_e10] (code)
```
model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=1e-3,
            decay_steps=1000,
            decay_rate=0.1,
        ),
        use_ema=True,
        ema_overwrite_frequency=100,
    ),
)
```
[p553_e11] (body) Let’s fit the model. We’ll use our `VisualizationCallback` callback to plot examples of generated images after each epoch, and we’ll save the model’s weights with the `ModelCheckpoint` callback:
[p553_e12] (code)
```
model.fit(
    dataset,
    epochs=100,
```

## Code annotations printed next to listings (callouts, in order)
- Computes the mean and
- variance necessary to perform
- normalization—don’t forget it!
- Configures the learning
- rate decay schedule
- Turns on Polyak
- averaging
- Configures how often to overwrite
- the model’s weights with their
- exponential moving average

## Equation candidates (formulas written inline in the text)
- [p553_e05] model = DiffusionModel(image_size, widths=[32, 64, 96, 128], block_depth=2)
model.normalizer.adapt(dataset)
- [p553_e12] model.fit(
    dataset,
    epochs=100,

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

## Previous page (PDF 552) text, for continuity
17.2.6 Visualizing results with a custom callback

We don’t have a proper metric to judge the quality of our generated images, so you’re going to want to visualize the generated images yourself over the course of training to judge if your model is getting somewhere. An easy way to do this is with a custom callback. The following callback uses the `generate()` method at the end of each epoch to display a 3 × 6 grid of generated images:

## Next page (PDF 554) text, for continuity
If you’re running on Colab, you might run into the error, “Buffered data was truncated after reaching the output size limit.” This happens because the logs of `fit()` include images, which take up a lot of space, whereas the allowed output for a single notebook cell is limited. To get around the problem, you can simply chain five `model.fit(..., epochs=20)` calls, in five successive cells. This is equivalent to a single `fit(..., epochs=100)` call.

After 100 epochs (which takes about 90 minutes on a T4, the free Colab GPU), we get pretty generative flowers like these (see figure 17.12).

Figure 17.12 Examples of generated flowers

You can keep training for even longer and get increasingly realistic results.

So that’s how image generation with diffusion works! Now, the next step to unlock their potential is to add text conditioning, which would result in a text-to-image model, capable of producing images that match a given text caption.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.274 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.544 — 17 Image generation › 17.2 Diffusion models
- PDF p.550 — 17 Image generation › 17.2 Diffusion models
- PDF p.547 — 17 Image generation › 17.2 Diffusion models
- PDF p.76 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.545 — 17 Image generation › 17.2 Diffusion models
- PDF p.79 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.541 — 17 Image generation › 17.1 Deep learning for image generation

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

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

### PDF p.644 (printed 620) — index ›  › U
attribute 262

underfitting 137–142 parameters 39

ambiguous features 139 weights 215 noisy training data 138 training rare features and spurious correlations 140–142 diffusion models 525 U-Net denoising autoencoder 522 inference vs. 215 unigrams 403 lower-precision computation 556–562 universal workflow of machine learning training arguments 215, 218 deploying model 183–189 training data 147 developing model 179–183 training loops 10, 39, 206–214 update_state() method 207, 222 callbacks 208, 210 upsampling stage 522 monitoring and visualization with use_casual_mask 448 TensorBoard 212 using fit() with custom training loops 222–225

### PDF p.76 (printed 52) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › 
Now you understand what happens when you call `fit`: the model will start to iterate on the training data in mini-batches of 128 samples, 5 times over (each iteration over all the training data is called an epoch). For each batch, the model will compute the gradient of the loss with regard to the weights (using the Backpropagation algorithm, which derives from the chain rule in calculus) and move the weights in the direction that will reduce the value of the loss for this batch.

After these 5 epochs, the model will have performed 2,345 gradient updates (469 per epoch), and the loss of the model will be sufficiently low that the model will be capable of classifying handwritten digits with high accuracy.

### PDF p.635 (printed 611) — index ›  › D
as REST API 185 data augmentation 242, 252 in browser 187 layers 252 on device 186 DataParallel depth of model 7 constructor 551 depthwise separable convolution layers 272–280, distribution instance 556

602 data parallelism 548, 549 depthwise separable convolutions 278–280 data preprocessing 247 derivatives 41 fitting model 250 of tensor operations 42–43 TensorFlow Dataset objects 248 detokenize() function 471 dataset.cache() method 556 DeviceMesh API 552 Dataset object 248, 356, 555 differentiable 598 datasets, downloading segmentation dataset 311 DiffusionModel class 525 decoders 464, 509 diffusion models 501, 509, 518–530 block 447 diffusion time and diffusion schedule 524 network 517 generation process 527 deep learning 1 Oxford Flowers dataset 520

### PDF p.555 (printed 531) — 17 Image generation › 17.3 Text-to-image models › 17.3 Text-to-image models
We can use the same basic diffusion process to create a model that maps text input to image output. To do this we need to take a pretrained text encoder (think a transformer encoder like RoBERTa from chapter 15) that can map text to vectors in a continuous embedding space. Then we can train a diffusion model on `(prompt,` `image)` pairs, where each prompt is a short, textual description of the input image.

We can handle the image input in the same way as we did previously, mapping noisy input to a denoised output that progressively approaches our input image. Critically, we can extend this setup by also passing the embedded text prompt to the denoising model. So rather than our denoising model simply taking in a `noisy_images` input, our model will take two inputs: `noisy_images` and `text_embeddings`. This gives a leg up on the flower denoiser we trained previously. Instead of learning to remove noise from an image without any additional information, the model gets to use a textual representation of the final image to help guide the denoising process. After training is when things get a bit more fun. Because we have trained a model that can map pure noise to images conditioned on

### PDF p.545 (printed 521) — 17 Image generation › 17.2 Diffusion models › 17.2.2 A U-Net denoising autoencoder
The same denoising model gets reused across each iteration of the diffusion denoising process, erasing a little bit of noise each time. To make the job of the model easier,

Figure 17.8 An example image from the

we tell it how much noise it is supposed to

Oxford Flowers dataset

### PDF p.550 (printed 526) — 17 Image generation › 17.2 Diffusion models › 
Next comes the training logic. This is the most important part! Like in the VAE example, we’re going to implement a custom `compute_loss()` method to keep our model backend agnostic. Of course, if you are set on using one specific backend, you could also write a custom `train_step()` with the exact same logic in it, plus the backend-specific logic for gradient computation and weight updates.

Since `compute_loss()` receives as input the output of `call()`, we’re going to put the denoising forward pass in `call()`. Our `call()` takes a batch of clean input images and applies the following steps: 1 Normalizes the images 2 Samples random diffusion times (the denoising model needs to be trained on

the full spectrum of diffusion times) 3 Computes corresponding noise rates and signal rates (using the diffusion schedule) 4 Adds random noise to the clean images (based on the computed noise rates and signal rates) 5 Denoises the images

### PDF p.79 (printed 55) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › Running one training step
The most difficult part of the process is the “training step”: updating the weights of the model after running it on one batch of data. We need to

Compute the predictions of the model for the images in the batch

Compute the loss value for these predictions given the actual labels

Compute the gradient of the loss with regard to the model’s weights

Move the weights by a small amount in the direction opposite to the gradient

Listing 2.9 A single step of training
