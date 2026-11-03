# Generation bundle for page_277  (PDF page 277, printed page 253)

Page kind: content
Chapter: 8 Image classification
Section: 8.2 Training a convnet from scratch on a small dataset
Subsection: 8.2.5 Using data augmentation
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_277.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_277.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p277_e03] (code)
```
def data_augmentation(images, targets):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images, targets
augmented_train_dataset = train_dataset.map(
    data_augmentation, num_parallel_calls=8
)
augmented_train_dataset = augmented_train_dataset.prefetch(tf.data.AUTOTUNE)
```
[p277_e04] (body) These are just a few of the layers available (for more, see the Keras documentation). Let’s quickly go over this code:
[p277_e05] (bullet) `RandomFlip("horizontal")` will apply horizontal flipping to a random 50% of the images that go through it.
[p277_e06] (bullet) `RandomRotation(0.1)` will rotate the input images by a random value in the range [–10%, +10%] (these are fractions of a full circle—in degrees the range would be [–36 degrees, +36 degrees]).
[p277_e07] (bullet) `RandomZoom(0.2)` will zoom in or out of the image by a random factor in the range [–20%, +20%].
[p277_e08] (body) Let’s look at the augmented images (see figure 8.10).
[p277_e09] (figure_caption) Figure 8.10 Generating variations of a very good boy via random data augmentation

## Code annotations printed next to listings (callouts, in order)
- Creates a function that
- applies them sequentially
- Maps this function into the dataset
- Enables prefetching of batches on GPU
- memory; important for best performance

## Figures
- Figure 8.10: Generating variations of a very good boy via random data augmentation  | labels inside the figure: (none - image)

## Chapter outline
Chapter: 8 Image classification (PDF pages 255-291)
  - 8.1 Introduction to convnets (p.256)
      - 8.1.1 The convolution operation (p.258)
      - 8.1.2 The max-pooling operation (p.263)
  - 8.2 Training a convnet from scratch on a small dataset (p.265)
      - 8.2.1 The relevance of deep learning for small-data problems (p.266)
      - 8.2.2 Downloading the data (p.266)
      - 8.2.3 Building your model (p.269)
      - 8.2.4 Data preprocessing (p.271)
      - 8.2.5 Using data augmentation (p.276)
  - 8.3 Using a pretrained model (p.280)
      - 8.3.1 Feature extraction with a pretrained model (p.280)
      - 8.3.2 Fine-tuning a pretrained model (p.288)

## Previous page (PDF 276) text, for continuity
Listing 8.17 Evaluating the model on the test set

We get a test accuracy of 78.6% (due to the randomness of neural network initializations, you may get numbers within 1 percentage point of that).

Because you have relatively few training samples (2,000), overfitting will be your number-one concern. You already know about a number of techniques that can help mitigate overfitting, such as dropout and weight decay (L2 regularization). We’re now going to work with a new one, specific to computer vision and used almost universally when processing images with deep learning models: data augmentation.

8.2.5

Using data augmentation

Overfitting is caused by having too few samples to learn from, rendering you unable to train a model that can generalize to new data. Given infinite data, your model would be exposed to every possible aspect of the data distribution at hand: you would never overfit. Data augmentation takes the approach of generating more training data from existing training samples, by augmenting the samples via a number of random transformations that yield believable-looking images. The goal is that at training time, your model will never see the exact same picture twice. This helps expose the model to more aspects of the data and generalize better.

In Keras, this can be done via data augmentation layers. Such layers could be added in one of two ways:

At the start of the model—Inside the model. In our case, the layers would come right before the `Rescaling` layer.

Inside the data pipeline—Outside the model. In our case, we’d apply them to our `Dataset` via a `map()` call.

The main difference between these two options is that data augmentation done inside the model would be running on the GPU, just like the rest of the model. Meanwhile, data augmentation done  …

## Next page (PDF 278) text, for continuity
Listing 8.19 Displaying some randomly augmented training images

If you train a new model using this data augmentation configuration, the model will never see the same input twice. But the inputs it sees are still heavily intercorrelated, because they come from a small number of original images—you can’t produce new information; you can only remix existing information. As such, this may not be enough to completely get rid of overfitting. To further fight overfitting, you’ll also add a `Dropout` layer to your model, right before the densely connected classifier.

Listing 8.20 Defining a new ConvNet that includes dropout

Let’s train the model using data augmentation and dropout. Because we expect overfitting to occur much later during training, we will train for twice as many epochs— 100. Note that we evaluate on images that aren’t augmented—data augmentation is usually only performed at training time, as it is a regularization technique.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.283 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.279 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.366 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.282 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.269 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.339 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.284 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.271 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.265 (printed 241) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › Training a ConvNet from scratch on a small dataset
Having to train an image-classification model using very little data is a common situation, which you’ll likely encounter in practice if you ever do computer vision in a professional context. A “few” samples can mean anywhere from a few hundred to a few tens of thousands of images. As a practical example, we’ll focus on classifying images as dogs or cats. We’ll work with a dataset containing 5,000 pictures of cats and dogs (2,500 cats, 2,500 dogs), taken from the original Kaggle dataset. We’ll use 2,000 pictures for training, 1,000 for validation, and 2,000 for testing.

In this section, we’ll review one basic strategy to tackle this problem: training a new model from scratch using what little data we have. We’ll start by naively training a small ConvNet on the 2,000 training samples, without any regularization, to set a baseline for what can be achieved. This will get us to a classification accuracy of about 80%. At that

### PDF p.94 (printed 70) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › 
Figure 3.1 Our synthetic data: two classes of random points in the 2D plane

### PDF p.282 (printed 258) — 8 Image classification › 8.3 Using a pretrained model › 
You’ll note a couple of things. First, KerasHub uses the term backbone to refer to the underlying feature extractor network without the classification head (it’s a little easier to type than “convolutional base”). It also uses a special constructor called `from_` `preset()` that will download the configuration and weights for the Xception model.

What’s that “41” in the name of the model we are using? Pretrained ConvNets are by convention often named by how “deep” they are. In this case, the 41 means that our Xception model has 41 trainable layers (conv and dense layers) stacked on top of each other. It’s the “deepest” model we’ve used so far in the book by a good margin. There’s one more missing piece we need before we can use this model. Every pretrained ConvNet will do some rescaling and resizing of images before pretraining. It’s important to make sure our input images match; otherwise, our model will need to relearn how to extract features from images with a totally different input range. Rather than keep track of which pretrained models use a `[0, 1]` input range for pixel values and which use a `[-1, 1]` range, we can use a KerasHub layer called `ImageConverter` that will re

### PDF p.317 (printed 293) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 
To use this model, we can simply call it on some input data, but we should be careful to apply our model-specific image preprocessing so that our images are scaled to the same range as the Xception pretraining data.

Listing 10.10 Using the feature extractor

### PDF p.537 (printed 513) — 17 Image generation › 17.1 Deep learning for image generation › 17.1.3 Implementing a VAE with Keras
We’re going to be implementing a VAE that can generate MNIST digits. It’s going to have three parts:

An encoder network that turns a real image into a mean and a variance in the latent space

A sampling layer that takes such a mean and variance and uses them to sample a random point from the latent space

A decoder network that turns points from the latent space back into images

The following listing shows the encoder network you’ll use, mapping images to the parameters of a probability distribution over the latent space. It’s a simple ConvNet that maps the input image `x` to two vectors, `z_mean` and `z_log_var`. One important detail is that we use strides for downsampling feature maps, instead of max pooling. The last time we did this was in the image segmentation example of chapter 11. Recall that, in general, strides are preferable to max pooling for any model that cares about information location—that is, where stuff is in the image—and this one does, since it will have to produce an image encoding that can be used to reconstruct a valid image.

Listing 17.1 VAE encoder network

### PDF p.279 (printed 255) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
Listing 8.21 Training the regularized ConvNet on augmented images
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch_with_augmentation.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]
history = model.fit(
    augmented_train_dataset,
    epochs=100,
    validation_data=validation_dataset,
    callbacks=callbacks,
)

### PDF p.12 (printed xii) — contents ›  › 
8.2

Training a ConvNet from scratch on a small dataset 241

The relevance of deep learning for small-data problems 242 Downloading the data 242 ■ Building your model 245 Data preprocessing 247 ■ Using data augmentation 252

8.3

Using a pretrained model 256

Feature extraction with a pretrained model 256 ■ Fine-tuning a pretrained model 264

9

ConvNet architecture patterns 268 9.1

Modularity, hierarchy, and reuse 269 9.2 Residual connections 272 9.3 Batch normalization 276 9.4 Depthwise separable convolutions 278 9.5 Putting it together: A mini Xception-like model 280 9.6 Beyond convolution: Vision Transformers 282 10 Interpreting what ConvNets learn 284 10.1 Visualizing intermediate activations 285 10.2 Visualizing ConvNet filters 291

Gradient ascent in TensorFlow 294 ■ Gradient ascent in PyTorch 295 ■ Gradient ascent in JAX 295 ■ The filter visualization loop 296

10.3

Visualizing heatmaps of class activation 299

Getting the gradient of the top class: TensorFlow version 302 Getting the gradient of the top class: PyTorch version 302 Getting the gradient of the top class: JAX version 303 Displaying the class activation heatmap 304

10.4 Visualizing the latent space of a ConvN

### PDF p.283 (printed 259) — 8 Image classification › 8.3 Using a pretrained model › 
At this point, there are two ways you could proceed:

Running the convolutional base over your dataset, recording its output to a NumPy array on disk, and then using this data as input to a standalone, densely connected classifier similar to those you saw in chapters 4 and 5. This solution is fast and cheap to run, because it only requires running the convolutional base once for every input image, and the convolutional base is by far the most expensive part of the pipeline. But for the same reason, this technique won’t allow you to use data augmentation.

Extending the model you have (`conv_base`) by adding `Dense` layers on top and running the whole thing end to end on the input data. This will allow you to use data augmentation because every input image goes through the convolutional base every time it’s seen by the model. But for the same reason, this technique is far more expensive than the first.

We’ll cover both techniques. Let’s walk through the code required to set up the first one: recording the output of `conv_base` on your data and using these outputs as inputs to a new model.
