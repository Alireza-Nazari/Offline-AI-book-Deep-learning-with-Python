# Generation bundle for page_286  (PDF page 286, printed page 262)

Page kind: content
Chapter: 8 Image classification
Section: 8.3 Using a pretrained model
Subsection: 8.3.1 Feature extraction with a pretrained model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_286.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_286.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p286_e03] (body) We get test accuracy of 98.1%—a very nice improvement over training a model from scratch!
[p286_e04] (heading_subsubsection) Feature extraction together with data augmentation
[p286_e05] (body) Now, let’s review the second technique we mentioned for doing feature extraction, which is much slower and more expensive but allows you to use data augmentation during training: creating a model that chains the `conv_base` with a new dense classifier and training it end to end on the inputs.
[p286_e06] (body) To do this, we will first freeze the convolutional base. Freezing a layer or set of layers means preventing their weights from being updated during training. Here, if you don’t do this, then the representations that were previously learned by the convolutional base will be modified during training. Because the `Dense` layers on top are randomly initialized, very large weight updates would be propagated through the network, effectively destroying the representations previously learned. In Keras, you freeze a layer or model by setting its `trainable` attribute to `False`.
[p286_e07] (listing_caption) Listing 8.28 Creating the frozen convolutional base
[p286_e08] (code)
```
import keras_hub
conv_base = keras_hub.models.Backbone.from_preset(
    "xception_41_imagenet",
    trainable=False,
)
```
[p286_e09] (body) Setting `trainable` to `False` empties the list of trainable weights of the layer or model.
[p286_e10] (listing_caption) Listing 8.29 Printing the list of trainable weights before and after freezing
[p286_e11] (code)
```
>>> conv_base.trainable = True
>>> len(conv_base.trainable_weights)
26
>>> conv_base.trainable = False
>>> len(conv_base.trainable_weights)
0
```
[p286_e12] (body) Now, we can just create a new model that chains together our frozen convolutional base and a dense classifier, like this:
[p286_e13] (code)
```
inputs = keras.Input(shape=(180, 180, 3))
x = preprocessor(inputs)
x = conv_base(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
```

## Code annotations printed next to listings (callouts, in order)
- The number of trainable weights
- before freezing the conv base
- The number of trainable weights
- after freezing the conv base

## Listings
- Listing 8.28: Creating the frozen convolutional base (code elements p286_e08)
- Listing 8.29: Printing the list of trainable weights before and after freezing (code elements p286_e11)

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

## Previous page (PDF 285) text, for continuity
Figure 8.13 Training and validation metrics for plain feature extraction

You reach a validation accuracy of slightly over 98%—much better than you achieved in the previous section with the small model trained from scratch. This is a bit of an unfair comparison, however, because ImageNet contains many dog and cat instances, which means that our pretrained model already has the exact knowledge required for the task at hand. This won’t always be the case when you use pretrained features.

However, the plots also indicate that you’re overfitting almost from the start—despite using dropout with a fairly large rate. That’s because this technique doesn’t use data augmentation, which is essential for preventing overfitting with small image datasets. Let’s check the test accuracy:

## Next page (PDF 287) text, for continuity
With this setup, only the weights from the two `Dense` layers that you added will be trained. That’s a total of four weight tensors: two per layer (the main weight matrix and the bias vector). Note that for these changes to take effect, you must first compile the model. If you ever modify weight trainability after compilation, you should then recompile the model, or these changes will be ignored.

Let’s train our model. We’ll reuse our augmented dataset `augmented_train_dataset`. Thanks to data augmentation, it will take much longer for the model to start overfitting, so we can train for more epochs—let’s do 30:

NOTE This technique is expensive enough that you should only attempt it if you have access to a GPU (such as the free GPU available in Colab)—it’s intractable on CPU. If you can’t run your code on GPU, then the previous technique is the way to go.

Let’s plot the results again (see figure 8.14). This model reaches a validation accuracy of 98.2%.

Let’s check the test accuracy.

Listing 8.30 Evaluating the model on the test set

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.283 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.288 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.280 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.282 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.284 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.289 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.258 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.278 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.288 (printed 264) — 8 Image classification › 8.3 Using a pretrained model › Fine-tuning a pretrained model
Another widely used technique for model reuse, complementary to feature extraction, is fine-tuning (see figure 8.15). Fine-tuning consists of unfreezing the frozen model base used for feature extraction and jointly training both the newly added part of the model (in this case, the fully connected classifier) and the base model. This is called fine-tuning because it slightly adjusts the more abstract representations of the model being reused to make them more relevant for the problem at hand.

We stated earlier that it’s necessary to freeze the pretrained convolution base first to be able to train a randomly initialized classifier on top. For the same reason, it’s only possible to fine-tune the convolutional base once the classifier on top has already been trained. If the classifier isn’t already trained, then the error signal propagating through the network during training will be too large, and the representations previously learned by the layers being fine-tuned will be destroyed. Thus, the steps for fine-tuning a network are as follows: 1 Add your custom network on top of an already trained base network. 2 Freeze the base network. 3 Train the part you added. 4 Unfreeze the base 

### PDF p.282 (printed 258) — 8 Image classification › 8.3 Using a pretrained model › 
only the first few layers of the model to do feature extraction, rather than using the entire convolutional base.

In this case, because the ImageNet class set contains multiple dog and cat classes, it’s likely to be beneficial to reuse the information contained in the densely connected layers of the original model. But we’ll choose not to, so we can cover the more general case where the class set of the new problem doesn’t overlap the class set of the original model. Let’s put this in practice by using the convolutional base of our pretrained model to extract interesting features from cat and dog images and then train a dogs-versus-cats classifier on top of these features. We will use the KerasHub library to create all pretrained models used in this book. KerasHub contains Keras implementations of popular pretrained model architectures paired with pretrained weights that can be downloaded to your machine. It contains a number of ConvNets like Xception, ResNet, EfficientNet and MobileNet, as well as larger, generative models we will use in the later chapters of this book. Let’s try using it to instantiate the Xception model trained on the ImageNet dataset.

NOTE KerasHub comes as a

### PDF p.280 (printed 256) — 8 Image classification › 8.3 Using a pretrained model › Feature extraction with a pretrained model
Feature extraction consists of using the representations learned by a previously trained model to extract interesting features from new samples. These features are then run through a new classifier, which is trained from scratch.

As you saw previously, ConvNets used for image classification comprise two parts: they start with a series of pooling and convolution layers, and they end with a densely connected classifier. The first part is called the convolutional base or backbone of the

### PDF p.283 (printed 259) — 8 Image classification › 8.3 Using a pretrained model › 
At this point, there are two ways you could proceed:

Running the convolutional base over your dataset, recording its output to a NumPy array on disk, and then using this data as input to a standalone, densely connected classifier similar to those you saw in chapters 4 and 5. This solution is fast and cheap to run, because it only requires running the convolutional base once for every input image, and the convolutional base is by far the most expensive part of the pipeline. But for the same reason, this technique won’t allow you to use data augmentation.

Extending the model you have (`conv_base`) by adding `Dense` layers on top and running the whole thing end to end on the input data. This will allow you to use data augmentation because every input image goes through the convolutional base every time it’s seen by the model. But for the same reason, this technique is far more expensive than the first.

We’ll cover both techniques. Let’s walk through the code required to set up the first one: recording the output of `conv_base` on your data and using these outputs as inputs to a new model.

### PDF p.282 (printed 258) — 8 Image classification › 8.3 Using a pretrained model › 
You’ll note a couple of things. First, KerasHub uses the term backbone to refer to the underlying feature extractor network without the classification head (it’s a little easier to type than “convolutional base”). It also uses a special constructor called `from_` `preset()` that will download the configuration and weights for the Xception model.

What’s that “41” in the name of the model we are using? Pretrained ConvNets are by convention often named by how “deep” they are. In this case, the 41 means that our Xception model has 41 trainable layers (conv and dense layers) stacked on top of each other. It’s the “deepest” model we’ve used so far in the book by a good margin. There’s one more missing piece we need before we can use this model. Every pretrained ConvNet will do some rescaling and resizing of images before pretraining. It’s important to make sure our input images match; otherwise, our model will need to relearn how to extract features from images with a totally different input range. Rather than keep track of which pretrained models use a `[0, 1]` input range for pixel values and which use a `[-1, 1]` range, we can use a KerasHub layer called `ImageConverter` that will re

### PDF p.266 (printed 242) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
point, the main issue will be overfitting. Then we’ll introduce data augmentation, a powerful technique for mitigating overfitting in computer vision. By using data augmentation, we’ll improve the model to reach a test accuracy of about 84%.

In the next section, we’ll review two more essential techniques for applying deep learning to small datasets: feature extraction with a pretrained model and fine-tuning a pretrained model (which will get us to a final accuracy of 98.5%). Together, these three strategies—training a small model from scratch, doing feature extraction using a pretrained model, and fine-tuning a pretrained model—will constitute your future toolbox for tackling the problem of performing image classification with small datasets.

### PDF p.283 (printed 259) — 8 Image classification › 8.3 Using a pretrained model › Fast feature extraction without data augmentation
We’ll start by extracting features as NumPy arrays, by calling the `predict()` method of the `conv_base` model on our training, validation, and testing datasets. Let’s iterate over our datasets to extract the pretrained model’s features.

Listing 8.25 Extracting the image features and corresponding labels

### PDF p.225 (printed 201) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › 
This enables you to do feature extraction: creating models that reuse intermediate features from another model.

Let’s say you want to add another output to the model we previously defined—you want to also predict an estimate of how long a given issue ticket will take to resolve, a kind of difficulty rating. You could do this via a classification layer over three categories—”quick,” “medium,” and “difficult.” You don’t need to recreate and retrain a model from scratch! You can just start from the intermediate features of your previous model, since you have access to them.

Listing 7.13 Creating a new model by reusing intermediate layer outputs
