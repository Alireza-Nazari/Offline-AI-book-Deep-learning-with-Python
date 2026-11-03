# Generation bundle for page_278  (PDF page 278, printed page 254)

Page kind: content
Chapter: 8 Image classification
Section: 8.2 Training a convnet from scratch on a small dataset
Subsection: 8.2.5 Using data augmentation
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_278.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_278.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p278_e03] (listing_caption) Listing 8.19 Displaying some randomly augmented training images
[p278_e04] (code)
```
plt.figure(figsize=(10, 10))
for image_batch, _ in train_dataset.take(1):
    image = image_batch[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image, _ = data_augmentation(image, None)
        augmented_image = keras.ops.convert_to_numpy(augmented_image)
        plt.imshow(augmented_image.astype("uint8"))
        plt.axis("off")
```
[p278_e05] (body) If you train a new model using this data augmentation configuration, the model will never see the same input twice. But the inputs it sees are still heavily intercorrelated, because they come from a small number of original images—you can’t produce new information; you can only remix existing information. As such, this may not be enough to completely get rid of overfitting. To further fight overfitting, you’ll also add a `Dropout` layer to your model, right before the densely connected classifier.
[p278_e06] (listing_caption) Listing 8.20 Defining a new ConvNet that includes dropout
[p278_e07] (code)
```
inputs = keras.Input(shape=(180, 180, 3))
x = layers.Rescaling(1.0 / 255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=512, kernel_size=3, activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)
```
[p278_e08] (body) Let’s train the model using data augmentation and dropout. Because we expect overfitting to occur much later during training, we will train for twice as many epochs— 100. Note that we evaluate on images that aren’t augmented—data augmentation is usually only performed at training time, as it is a regularization technique.

## Code annotations printed next to listings (callouts, in order)
- You can use take(N) to only sample
- N batches from the dataset. This is
- equivalent to inserting a break in
- the loop after the Nth batch.
- Displays the first image in the output batch.
- For each of the nine iterations, this is a
- different augmentation of the same image.

## Listings
- Listing 8.19: Displaying some randomly augmented training images (code elements p278_e04)
- Listing 8.20: Defining a new ConvNet that includes dropout (code elements p278_e07)

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

## Previous page (PDF 277) text, for continuity
These are just a few of the layers available (for more, see the Keras documentation). Let’s quickly go over this code:

`RandomFlip("horizontal")` will apply horizontal flipping to a random 50% of the images that go through it.

`RandomRotation(0.1)` will rotate the input images by a random value in the range [–10%, +10%] (these are fractions of a full circle—in degrees the range would be [–36 degrees, +36 degrees]).

`RandomZoom(0.2)` will zoom in or out of the image by a random factor in the range [–20%, +20%].

Let’s look at the augmented images (see figure 8.10).

Figure 8.10 Generating variations of a very good boy via random data augmentation

## Next page (PDF 279) text, for continuity
Listing 8.21 Training the regularized ConvNet on augmented images

Let’s plot the results again; see figure 8.11. Thanks to data augmentation and dropout, we start overfitting much later, around epochs 60–70 (compared to epoch 10 for the original model). The validation accuracy ends up peaking above 85%—a big improvement over our first try.

Figure 8.11 Training and validation metrics with data augmentation

Let’s check the test accuracy.

Listing 8.22 Evaluating the model on the test set

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.284 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.264 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.270 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.286 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.256 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.276 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.192 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.625 — 20 Conclusions › 20.1 Key concepts in review

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.276 (printed 252) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › Using data augmentation
Overfitting is caused by having too few samples to learn from, rendering you unable to train a model that can generalize to new data. Given infinite data, your model would be exposed to every possible aspect of the data distribution at hand: you would never overfit. Data augmentation takes the approach of generating more training data from existing training samples, by augmenting the samples via a number of random transformations that yield believable-looking images. The goal is that at training time, your model will never see the exact same picture twice. This helps expose the model to more aspects of the data and generalize better.

In Keras, this can be done via data augmentation layers. Such layers could be added in one of two ways:

At the start of the model—Inside the model. In our case, the layers would come right before the `Rescaling` layer.

Inside the data pipeline—Outside the model. In our case, we’d apply them to our `Dataset` via a `map()` call.

The main difference between these two options is that data augmentation done inside the model would be running on the GPU, just like the rest of the model. Meanwhile, data augmentation done in the data pipeline would be runni

### PDF p.291 (printed 267) — 8 Image classification › 8.3 Using a pretrained model › Summary
ConvNets excel at computer vision tasks. It’s possible to train one from scratch, even on a very small dataset, with decent results.

ConvNets work by learning a hierarchy of modular patterns and concepts to represent the visual world.

On a small dataset, overfitting will be the main issue. Data augmentation is a powerful way to fight overfitting when you’re working with image data.

It’s easy to reuse an existing ConvNet on a new dataset via feature extraction. This is a valuable technique for working with small image datasets.

As a complement to feature extraction, you can use fine-tuning, which adapts to a new problem some of the representations previously learned by an existing model. This pushes performance a bit further.

### PDF p.276 (printed 252) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
We get a test accuracy of 78.6% (due to the randomness of neural network initializations, you may get numbers within 1 percentage point of that).

Because you have relatively few training samples (2,000), overfitting will be your number-one concern. You already know about a number of techniques that can help mitigate overfitting, such as dropout and weight decay (L2 regularization). We’re now going to work with a new one, specific to computer vision and used almost universally when processing images with deep learning models: data augmentation.

### PDF p.287 (printed 263) — 8 Image classification › 8.3 Using a pretrained model › 
With this setup, only the weights from the two `Dense` layers that you added will be trained. That’s a total of four weight tensors: two per layer (the main weight matrix and the bias vector). Note that for these changes to take effect, you must first compile the model. If you ever modify weight trainability after compilation, you should then recompile the model, or these changes will be ignored.

Let’s train our model. We’ll reuse our augmented dataset `augmented_train_dataset`. Thanks to data augmentation, it will take much longer for the model to start overfitting, so we can train for more epochs—let’s do 30:

### PDF p.266 (printed 242) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
point, the main issue will be overfitting. Then we’ll introduce data augmentation, a powerful technique for mitigating overfitting in computer vision. By using data augmentation, we’ll improve the model to reach a test accuracy of about 84%.

In the next section, we’ll review two more essential techniques for applying deep learning to small datasets: feature extraction with a pretrained model and fine-tuning a pretrained model (which will get us to a final accuracy of 98.5%). Together, these three strategies—training a small model from scratch, doing feature extraction using a pretrained model, and fine-tuning a pretrained model—will constitute your future toolbox for tackling the problem of performing image classification with small datasets.

### PDF p.397 (printed 373) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
curves, it’s evident that the model is quickly overfitting, despite only having very few units: the training and validation losses start to diverge considerably after a few epochs. You’re already familiar with a classic technique for fighting this phenomenon: dropout, which randomly zeros out input units of a layer to break happenstance correlations in the training data that the layer is exposed to. But how to correctly apply dropout in recurrent networks isn’t a trivial question.

It has long been known that applying dropout before a recurrent layer hinders learning rather than helping with regularization. In 2015, Yarin Gal, as part of his PhD thesis on Bayesian deep learning,5 determined the proper way to use dropout with a recurrent network: the same dropout mask (the same pattern of dropped units) should be applied at every timestep, instead of a dropout mask that varies randomly from timestep to timestep. What’s more, to regularize the representations formed by the recurrent gates of layers such as `GRU` and `LSTM`, a temporally constant dropout mask should be applied to the inner recurrent activations of the layer (a recurrent dropout mask). Using the same dropout mask at ev

### PDF p.285 (printed 261) — 8 Image classification › 8.3 Using a pretrained model › 
Figure 8.13 Training and validation metrics for plain feature extraction

You reach a validation accuracy of slightly over 98%—much better than you achieved in the previous section with the small model trained from scratch. This is a bit of an unfair comparison, however, because ImageNet contains many dog and cat instances, which means that our pretrained model already has the exact knowledge required for the task at hand. This won’t always be the case when you use pretrained features.

However, the plots also indicate that you’re overfitting almost from the start—despite using dropout with a fairly large rate. That’s because this technique doesn’t use data augmentation, which is essential for preventing overfitting with small image datasets. Let’s check the test accuracy:

### PDF p.283 (printed 259) — 8 Image classification › 8.3 Using a pretrained model › 
At this point, there are two ways you could proceed:

Running the convolutional base over your dataset, recording its output to a NumPy array on disk, and then using this data as input to a standalone, densely connected classifier similar to those you saw in chapters 4 and 5. This solution is fast and cheap to run, because it only requires running the convolutional base once for every input image, and the convolutional base is by far the most expensive part of the pipeline. But for the same reason, this technique won’t allow you to use data augmentation.

Extending the model you have (`conv_base`) by adding `Dense` layers on top and running the whole thing end to end on the input data. This will allow you to use data augmentation because every input image goes through the convolutional base every time it’s seen by the model. But for the same reason, this technique is far more expensive than the first.

We’ll cover both techniques. Let’s walk through the code required to set up the first one: recording the output of `conv_base` on your data and using these outputs as inputs to a new model.
