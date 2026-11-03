# Generation bundle for page_283  (PDF page 283, printed page 259)

Page kind: content
Chapter: 8 Image classification
Section: 8.3 Using a pretrained model
Subsection: 8.3.1 Feature extraction with a pretrained model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_283.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_283.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p283_e03] (listing_caption) Listing 8.24 Instantiating the preprocessing paired with the Xception model
[p283_e04] (code)
```
preprocessor = keras_hub.layers.ImageConverter.from_preset(
    "xception_41_imagenet",
    image_size=(180, 180),
)
```
[p283_e05] (body) At this point, there are two ways you could proceed:
[p283_e06] (bullet) Running the convolutional base over your dataset, recording its output to a NumPy array on disk, and then using this data as input to a standalone, densely connected classifier similar to those you saw in chapters 4 and 5. This solution is fast and cheap to run, because it only requires running the convolutional base once for every input image, and the convolutional base is by far the most expensive part of the pipeline. But for the same reason, this technique won’t allow you to use data augmentation.
[p283_e07] (bullet) Extending the model you have (`conv_base`) by adding `Dense` layers on top and running the whole thing end to end on the input data. This will allow you to use data augmentation because every input image goes through the convolutional base every time it’s seen by the model. But for the same reason, this technique is far more expensive than the first.
[p283_e08] (body) We’ll cover both techniques. Let’s walk through the code required to set up the first one: recording the output of `conv_base` on your data and using these outputs as inputs to a new model.
[p283_e09] (heading_subsubsection) Fast feature extraction without data augmentation
[p283_e10] (body) We’ll start by extracting features as NumPy arrays, by calling the `predict()` method of the `conv_base` model on our training, validation, and testing datasets. Let’s iterate over our datasets to extract the pretrained model’s features.
[p283_e11] (listing_caption) Listing 8.25 Extracting the image features and corresponding labels
[p283_e12] (code)
```
def get_features_and_labels(dataset):
    all_features = []
    all_labels = []
    for images, labels in dataset:
        preprocessed_images = preprocessor(images)
        features = conv_base.predict(preprocessed_images, verbose=0)
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)
train_features, train_labels = get_features_and_labels(train_dataset)
val_features, val_labels = get_features_and_labels(validation_dataset)
test_features, test_labels = get_features_and_labels(test_dataset)
```

## Listings
- Listing 8.24: Instantiating the preprocessing paired with the Xception model (code elements p283_e04)
- Listing 8.25: Extracting the image features and corresponding labels (code elements p283_e12)

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

## Previous page (PDF 282) text, for continuity
only the first few layers of the model to do feature extraction, rather than using the entire convolutional base.

In this case, because the ImageNet class set contains multiple dog and cat classes, it’s likely to be beneficial to reuse the information contained in the densely connected layers of the original model. But we’ll choose not to, so we can cover the more general case where the class set of the new problem doesn’t overlap the class set of the original model. Let’s put this in practice by using the convolutional base of our pretrained model to extract interesting features from cat and dog images and then train a dogs-versus-cats classifier on top of these features. We will use the KerasHub library to create all pretrained models used in this book. KerasHub contains Keras implementations of popular pretrained model architectures paired with pretrained weights that can be downloaded to your machine. It contains a number of ConvNets like Xception, ResNet, EfficientNet and MobileNet, as well as larger, generative models we will use in the later chapters of this book. Let’s try using it to instantiate the Xception model trained on the ImageNet dataset.

NOTE KerasHub comes as a separate package from Keras. This package is preinstalled in Colab and Kaggle notebooks, but if you want to use it outside these environments you can install it yourself with `pip install keras-hub`.

Listing 8.23 Instantiating the Xception convolutional base

You’ll note a couple of things. First, KerasHub uses the term backbone to refer to the underlying feature extractor network without the classification head (it’s a little easier to type than “convolutional base”). It also uses a special constructor called `from_` `preset()` that will download the configuration and weights for the Xcepti …

## Next page (PDF 284) text, for continuity
Importantly, `predict()` only expects images, not labels, but our current dataset yields batches that contain both images and their labels.

The extracted features are currently of shape `(samples, 6, 6, 2048)`:

At this point, you can define your densely connected classifier (note the use of dropout for regularization) and train it on the data and labels that you just recorded.

Listing 8.26 Defining and training the densely connected classifier

Training is very fast because you only have to deal with two `Dense` layers—an epoch takes less than 1 second even on CPU.

Let’s look at the loss and accuracy curves during training (see figure 8.13).

Listing 8.27 Plotting the results

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.286 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.280 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.272 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.271 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.276 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.285 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.316 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.317 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.286 (printed 262) — 8 Image classification › 8.3 Using a pretrained model › Feature extraction together with data augmentation
Now, let’s review the second technique we mentioned for doing feature extraction, which is much slower and more expensive but allows you to use data augmentation during training: creating a model that chains the `conv_base` with a new dense classifier and training it end to end on the inputs.

To do this, we will first freeze the convolutional base. Freezing a layer or set of layers means preventing their weights from being updated during training. Here, if you don’t do this, then the representations that were previously learned by the convolutional base will be modified during training. Because the `Dense` layers on top are randomly initialized, very large weight updates would be propagated through the network, effectively destroying the representations previously learned. In Keras, you freeze a layer or model by setting its `trainable` attribute to `False`.

Listing 8.28 Creating the frozen convolutional base

### PDF p.280 (printed 256) — 8 Image classification › 8.3 Using a pretrained model › Feature extraction with a pretrained model
Feature extraction consists of using the representations learned by a previously trained model to extract interesting features from new samples. These features are then run through a new classifier, which is trained from scratch.

As you saw previously, ConvNets used for image classification comprise two parts: they start with a series of pooling and convolution layers, and they end with a densely connected classifier. The first part is called the convolutional base or backbone of the

### PDF p.286 (printed 262) — 8 Image classification › 8.3 Using a pretrained model › Feature extraction together with data augmentation
Listing 8.28 Creating the frozen convolutional base
import keras_hub
conv_base = keras_hub.models.Backbone.from_preset(
    "xception_41_imagenet",
    trainable=False,
)

### PDF p.266 (printed 242) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
point, the main issue will be overfitting. Then we’ll introduce data augmentation, a powerful technique for mitigating overfitting in computer vision. By using data augmentation, we’ll improve the model to reach a test accuracy of about 84%.

In the next section, we’ll review two more essential techniques for applying deep learning to small datasets: feature extraction with a pretrained model and fine-tuning a pretrained model (which will get us to a final accuracy of 98.5%). Together, these three strategies—training a small model from scratch, doing feature extraction using a pretrained model, and fine-tuning a pretrained model—will constitute your future toolbox for tackling the problem of performing image classification with small datasets.

### PDF p.280 (printed 256) — 8 Image classification › 8.3 Using a pretrained model › Using a pretrained model
A common and highly effective approach to deep learning on small image datasets is to use a pretrained model. A pretrained model is a model that was previously trained on a large dataset, typically on a large-scale image-classification task. If this original dataset is large enough and general enough, then the spatial hierarchy of features learned by the pretrained model can effectively act as a generic model of the visual world, and hence its features can prove useful for many different computer vision problems, even though these new problems may involve completely different classes than those of the original task. For instance, you might train a model on ImageNet (where classes are mostly animals and everyday objects) and then repurpose this trained model for something as remote as identifying furniture items in images. Such portability of learned features across different problems is a key advantage of deep learning compared to many older, shallow learning approaches, and it makes deep learning very effective for small-data problems.

In this case, let’s consider a large ConvNet trained on the ImageNet dataset (1.4 million labeled images and 1,000 different classes). ImageNet co

### PDF p.281 (printed 257) — 8 Image classification › 8.3 Using a pretrained model › 
model. In the case of ConvNets, feature extraction consists of taking the convolutional base of a previously trained network, running the new data through it, and training a new classifier on top of the output (see figure 8.12).

Figure 8.12 Swapping classifiers while keeping the same convolutional base

Why only reuse the convolutional base? Could you reuse the densely connected classifier as well? In general, doing so should be avoided. The reason is that the representations learned by the convolutional base are likely to be more generic and therefore more reusable: the feature maps of a ConvNet are presence maps of generic concepts over a picture, which is likely to be useful regardless of the computer vision problem at hand. But the representations learned by the classifier will necessarily be specific to the set of classes on which the model was trained—they will only contain information about the presence probability of this or that class in the entire picture. Additionally, representations found in densely connected layers no longer contain any information about where objects are located in the input image: these layers get rid of the notion of space, whereas the object loca

### PDF p.317 (printed 293) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 
To use this model, we can simply call it on some input data, but we should be careful to apply our model-specific image preprocessing so that our images are scaled to the same range as the Xception pretraining data.

Listing 10.10 Using the feature extractor

### PDF p.276 (printed 252) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › Using data augmentation
Overfitting is caused by having too few samples to learn from, rendering you unable to train a model that can generalize to new data. Given infinite data, your model would be exposed to every possible aspect of the data distribution at hand: you would never overfit. Data augmentation takes the approach of generating more training data from existing training samples, by augmenting the samples via a number of random transformations that yield believable-looking images. The goal is that at training time, your model will never see the exact same picture twice. This helps expose the model to more aspects of the data and generalize better.

In Keras, this can be done via data augmentation layers. Such layers could be added in one of two ways:

At the start of the model—Inside the model. In our case, the layers would come right before the `Rescaling` layer.

Inside the data pipeline—Outside the model. In our case, we’d apply them to our `Dataset` via a `map()` call.

The main difference between these two options is that data augmentation done inside the model would be running on the GPU, just like the rest of the model. Meanwhile, data augmentation done in the data pipeline would be runni
