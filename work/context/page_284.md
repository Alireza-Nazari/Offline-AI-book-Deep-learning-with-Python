# Generation bundle for page_284  (PDF page 284, printed page 260)

Page kind: content
Chapter: 8 Image classification
Section: 8.3 Using a pretrained model
Subsection: 8.3.1 Feature extraction with a pretrained model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_284.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_284.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p284_e03] (body) Importantly, `predict()` only expects images, not labels, but our current dataset yields batches that contain both images and their labels.
[p284_e04] (body) The extracted features are currently of shape `(samples, 6, 6, 2048)`:
[p284_e05] (code)
```
>>> train_features.shape
(2000, 6, 6, 2048)
```
[p284_e06] (body) At this point, you can define your densely connected classifier (note the use of dropout for regularization) and train it on the data and labels that you just recorded.
[p284_e07] (listing_caption) Listing 8.26 Defining and training the densely connected classifier
[p284_e08] (code)
```
inputs = keras.Input(shape=(6, 6, 2048))
x = layers.GlobalAveragePooling2D()(inputs)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="feature_extraction.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]
history = model.fit(
    train_features,
    train_labels,
    epochs=10,
    validation_data=(val_features, val_labels),
    callbacks=callbacks,
)
```
[p284_e09] (body) Training is very fast because you only have to deal with two `Dense` layers—an epoch takes less than 1 second even on CPU.
[p284_e10] (body) Let’s look at the loss and accuracy curves during training (see figure 8.13).
[p284_e11] (listing_caption) Listing 8.27 Plotting the results
[p284_e12] (code)
```
import matplotlib.pyplot as plt
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
```

## Code annotations printed next to listings (callouts, in order)
- Averages spatial dimensions
- to flatten the feature map

## Listings
- Listing 8.26: Defining and training the densely connected classifier (code elements p284_e08)
- Listing 8.27: Plotting the results (code elements p284_e12)

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

## Previous page (PDF 283) text, for continuity
Listing 8.24 Instantiating the preprocessing paired with the Xception model

At this point, there are two ways you could proceed:

Running the convolutional base over your dataset, recording its output to a NumPy array on disk, and then using this data as input to a standalone, densely connected classifier similar to those you saw in chapters 4 and 5. This solution is fast and cheap to run, because it only requires running the convolutional base once for every input image, and the convolutional base is by far the most expensive part of the pipeline. But for the same reason, this technique won’t allow you to use data augmentation.

Extending the model you have (`conv_base`) by adding `Dense` layers on top and running the whole thing end to end on the input data. This will allow you to use data augmentation because every input image goes through the convolutional base every time it’s seen by the model. But for the same reason, this technique is far more expensive than the first.

We’ll cover both techniques. Let’s walk through the code required to set up the first one: recording the output of `conv_base` on your data and using these outputs as inputs to a new model.

Fast feature extraction without data augmentation

We’ll start by extracting features as NumPy arrays, by calling the `predict()` method of the `conv_base` model on our training, validation, and testing datasets. Let’s iterate over our datasets to extract the pretrained model’s features.

Listing 8.25 Extracting the image features and corresponding labels

## Next page (PDF 285) text, for continuity
Figure 8.13 Training and validation metrics for plain feature extraction

You reach a validation accuracy of slightly over 98%—much better than you achieved in the previous section with the small model trained from scratch. This is a bit of an unfair comparison, however, because ImageNet contains many dog and cat instances, which means that our pretrained model already has the exact knowledge required for the task at hand. This won’t always be the case when you use pretrained features.

However, the plots also indicate that you’re overfitting almost from the start—despite using dropout with a fairly large rate. That’s because this technique doesn’t use data augmentation, which is essential for preventing overfitting with small image datasets. Let’s check the test accuracy:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.278 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.287 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.274 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.625 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.275 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.279 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.286 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.258 — 8 Image classification › 8.1 Introduction to convnets

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.280 (printed 256) — 8 Image classification › 8.3 Using a pretrained model › Feature extraction with a pretrained model
Feature extraction consists of using the representations learned by a previously trained model to extract interesting features from new samples. These features are then run through a new classifier, which is trained from scratch.

As you saw previously, ConvNets used for image classification comprise two parts: they start with a series of pooling and convolution layers, and they end with a densely connected classifier. The first part is called the convolutional base or backbone of the

### PDF p.282 (printed 258) — 8 Image classification › 8.3 Using a pretrained model › 
only the first few layers of the model to do feature extraction, rather than using the entire convolutional base.

In this case, because the ImageNet class set contains multiple dog and cat classes, it’s likely to be beneficial to reuse the information contained in the densely connected layers of the original model. But we’ll choose not to, so we can cover the more general case where the class set of the new problem doesn’t overlap the class set of the original model. Let’s put this in practice by using the convolutional base of our pretrained model to extract interesting features from cat and dog images and then train a dogs-versus-cats classifier on top of these features. We will use the KerasHub library to create all pretrained models used in this book. KerasHub contains Keras implementations of popular pretrained model architectures paired with pretrained weights that can be downloaded to your machine. It contains a number of ConvNets like Xception, ResNet, EfficientNet and MobileNet, as well as larger, generative models we will use in the later chapters of this book. Let’s try using it to instantiate the Xception model trained on the ImageNet dataset.

NOTE KerasHub comes as a

### PDF p.281 (printed 257) — 8 Image classification › 8.3 Using a pretrained model › 
model. In the case of ConvNets, feature extraction consists of taking the convolutional base of a previously trained network, running the new data through it, and training a new classifier on top of the output (see figure 8.12).

Figure 8.12 Swapping classifiers while keeping the same convolutional base

Why only reuse the convolutional base? Could you reuse the densely connected classifier as well? In general, doing so should be avoided. The reason is that the representations learned by the convolutional base are likely to be more generic and therefore more reusable: the feature maps of a ConvNet are presence maps of generic concepts over a picture, which is likely to be useful regardless of the computer vision problem at hand. But the representations learned by the classifier will necessarily be specific to the set of classes on which the model was trained—they will only contain information about the presence probability of this or that class in the entire picture. Additionally, representations found in densely connected layers no longer contain any information about where objects are located in the input image: these layers get rid of the notion of space, whereas the object loca

### PDF p.286 (printed 262) — 8 Image classification › 8.3 Using a pretrained model › Feature extraction together with data augmentation
Now, let’s review the second technique we mentioned for doing feature extraction, which is much slower and more expensive but allows you to use data augmentation during training: creating a model that chains the `conv_base` with a new dense classifier and training it end to end on the inputs.

To do this, we will first freeze the convolutional base. Freezing a layer or set of layers means preventing their weights from being updated during training. Here, if you don’t do this, then the representations that were previously learned by the convolutional base will be modified during training. Because the `Dense` layers on top are randomly initialized, very large weight updates would be propagated through the network, effectively destroying the representations previously learned. In Keras, you freeze a layer or model by setting its `trainable` attribute to `False`.

Listing 8.28 Creating the frozen convolutional base

### PDF p.258 (printed 234) — 8 Image classification › 8.1 Introduction to convnets › 
After the last `Conv2D` layer, we end up with an output of shape `(3, 3, 256)`—a 3 × 3 feature map of 256 channels. The next step is to feed this output into a densely connected classifier like those you’re already familiar with: a stack of `Dense` layers. These classifiers process vectors, which are 1D, whereas the current output is a rank-3 tensor. To bridge the gap, we flatten the 3D outputs to 1D with a `GlobalAveragePooling2D` layer before adding the `Dense` layers. This layer will take the average of each 3 × 3 feature map in the tensor of shape `(3,` `3,` `256)`, resulting in an output vector of shape `(256,)`. Finally, we’ll do 10-way classification, so our last layer has 10 outputs and a softmax activation.

Now, let’s train the ConvNet on the MNIST digits. We’ll reuse a lot of the code from the MNIST example in chapter 2. Because we’re doing 10-way classification with a softmax output, we’ll use the categorical crossentropy loss, and because our labels are integers, we’ll use the sparse version, `sparse_categorical_crossentropy`.

Listing 8.3 Training the ConvNet on MNIST images

### PDF p.280 (printed 256) — 8 Image classification › 8.3 Using a pretrained model › Using a pretrained model
A common and highly effective approach to deep learning on small image datasets is to use a pretrained model. A pretrained model is a model that was previously trained on a large dataset, typically on a large-scale image-classification task. If this original dataset is large enough and general enough, then the spatial hierarchy of features learned by the pretrained model can effectively act as a generic model of the visual world, and hence its features can prove useful for many different computer vision problems, even though these new problems may involve completely different classes than those of the original task. For instance, you might train a model on ImageNet (where classes are mostly animals and everyday objects) and then repurpose this trained model for something as remote as identifying furniture items in images. Such portability of learned features across different problems is a key advantage of deep learning compared to many older, shallow learning approaches, and it makes deep learning very effective for small-data problems.

In this case, let’s consider a large ConvNet trained on the ImageNet dataset (1.4 million labeled images and 1,000 different classes). ImageNet co

### PDF p.625 (printed 601) — 20 Conclusions › 20.1 Key concepts in review › 
assume no specific structure in the input features: they’re called densely connected because the units of a `Dense` layer are connected to every other unit. The layer attempts to map relationships between any two input features; this is unlike a 2D convolution layer, for instance, which only looks at local relationships.

Densely connected networks are most commonly used for categorical data (for example, where the input features are lists of attributes), such as the Boston Housing Price dataset used in chapter 4. They’re also used as the final classification or regression stage of most networks. For instance, the ConvNets covered in chapter 8 typically end with one or two `Dense` layers, and so do the recurrent networks in chapter 13. Remember, to perform binary classification, end your stack of layers with a `Dense` layer with a single unit and a `sigmoid` activation, and use `binary_crossentropy` as the loss. Your targets should be either 0 or 1:

### PDF p.274 (printed 250) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › Fitting the model
Let’s look at the output of one of these `Dataset` objects: it yields batches of 180 × 180 RGB images (shape `(32, 180, 180, 3)`) and integer labels (shape `(32,)`). There are 32 samples in each batch (the batch size).

Listing 8.14 Displaying the shapes yielded by the `Dataset`
