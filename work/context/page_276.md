# Generation bundle for page_276  (PDF page 276, printed page 252)

Page kind: content
Chapter: 8 Image classification
Section: 8.2 Training a convnet from scratch on a small dataset
Subsection: 8.2.5 Using data augmentation
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_276.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_276.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p276_e03] (listing_caption) Listing 8.17 Evaluating the model on the test set
[p276_e04] (code)
```
test_model = keras.models.load_model("convnet_from_scratch.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")
```
[p276_e05] (body) We get a test accuracy of 78.6% (due to the randomness of neural network initializations, you may get numbers within 1 percentage point of that).
[p276_e06] (body) Because you have relatively few training samples (2,000), overfitting will be your number-one concern. You already know about a number of techniques that can help mitigate overfitting, such as dropout and weight decay (L2 regularization). We’re now going to work with a new one, specific to computer vision and used almost universally when processing images with deep learning models: data augmentation.
[p276_e07] (heading_subsection) 8.2.5
[p276_e08] (heading_subsection) Using data augmentation
[p276_e09] (body) Overfitting is caused by having too few samples to learn from, rendering you unable to train a model that can generalize to new data. Given infinite data, your model would be exposed to every possible aspect of the data distribution at hand: you would never overfit. Data augmentation takes the approach of generating more training data from existing training samples, by augmenting the samples via a number of random transformations that yield believable-looking images. The goal is that at training time, your model will never see the exact same picture twice. This helps expose the model to more aspects of the data and generalize better.
[p276_e10] (body) In Keras, this can be done via data augmentation layers. Such layers could be added in one of two ways:
[p276_e11] (bullet) At the start of the model—Inside the model. In our case, the layers would come right before the `Rescaling` layer.
[p276_e12] (bullet) Inside the data pipeline—Outside the model. In our case, we’d apply them to our `Dataset` via a `map()` call.
[p276_e13] (body) The main difference between these two options is that data augmentation done inside the model would be running on the GPU, just like the rest of the model. Meanwhile, data augmentation done in the data pipeline would be running on the CPU, typically in a parallel way on multiple CPU cores. Sometimes, there can be performance benefits to doing the former, but the latter is usually the better option. So let’s go with that!
[p276_e14] (listing_caption) Listing 8.18 Defining a data augmentation stage
[p276_e15] (code)
```
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
]
```

## Code annotations printed next to listings (callouts, in order)
- Defines the transformations
- to apply as a list

## Listings
- Listing 8.17: Evaluating the model on the test set (code elements p276_e04)
- Listing 8.18: Defining a data augmentation stage (code elements p276_e15)

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

## Previous page (PDF 275) text, for continuity
Listing 8.16 Displaying curves of loss and accuracy during training

Figure 8.9 Training and validation metrics for a simple ConvNet

These plots are characteristic of overfitting. The training accuracy increases linearly over time, until it reaches nearly 100%, whereas the validation accuracy peaks around 80%. The validation loss reaches its minimum after only 10 epochs and then stalls, whereas the training loss keeps decreasing linearly as training proceeds.

Let’s check the test accuracy. We’ll reload the model from its saved file to evaluate it as it was before it started overfitting.

## Next page (PDF 277) text, for continuity
These are just a few of the layers available (for more, see the Keras documentation). Let’s quickly go over this code:

`RandomFlip("horizontal")` will apply horizontal flipping to a random 50% of the images that go through it.

`RandomRotation(0.1)` will rotate the input images by a random value in the range [–10%, +10%] (these are fractions of a full circle—in degrees the range would be [–36 degrees, +36 degrees]).

`RandomZoom(0.2)` will zoom in or out of the image by a random factor in the range [–20%, +20%].

Let’s look at the augmented images (see figure 8.10).

Figure 8.10 Generating variations of a very good boy via random data augmentation

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.279 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.278 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.287 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.288 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.283 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.285 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.266 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.286 — 8 Image classification › 8.3 Using a pretrained model

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.266 (printed 242) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
point, the main issue will be overfitting. Then we’ll introduce data augmentation, a powerful technique for mitigating overfitting in computer vision. By using data augmentation, we’ll improve the model to reach a test accuracy of about 84%.

In the next section, we’ll review two more essential techniques for applying deep learning to small datasets: feature extraction with a pretrained model and fine-tuning a pretrained model (which will get us to a final accuracy of 98.5%). Together, these three strategies—training a small model from scratch, doing feature extraction using a pretrained model, and fine-tuning a pretrained model—will constitute your future toolbox for tackling the problem of performing image classification with small datasets.

### PDF p.278 (printed 254) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
Let’s train the model using data augmentation and dropout. Because we expect overfitting to occur much later during training, we will train for twice as many epochs— 100. Note that we evaluate on images that aren’t augmented—data augmentation is usually only performed at training time, as it is a regularization technique.

### PDF p.278 (printed 254) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
If you train a new model using this data augmentation configuration, the model will never see the same input twice. But the inputs it sees are still heavily intercorrelated, because they come from a small number of original images—you can’t produce new information; you can only remix existing information. As such, this may not be enough to completely get rid of overfitting. To further fight overfitting, you’ll also add a `Dropout` layer to your model, right before the densely connected classifier.

Listing 8.20 Defining a new ConvNet that includes dropout

### PDF p.279 (printed 255) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
Let’s plot the results again; see figure 8.11. Thanks to data augmentation and dropout, we start overfitting much later, around epochs 60–70 (compared to epoch 10 for the original model). The validation accuracy ends up peaking above 85%—a big improvement over our first try.

Figure 8.11 Training and validation metrics with data augmentation

Let’s check the test accuracy.

Listing 8.22 Evaluating the model on the test set

### PDF p.291 (printed 267) — 8 Image classification › 8.3 Using a pretrained model › Summary
ConvNets excel at computer vision tasks. It’s possible to train one from scratch, even on a very small dataset, with decent results.

ConvNets work by learning a hierarchy of modular patterns and concepts to represent the visual world.

On a small dataset, overfitting will be the main issue. Data augmentation is a powerful way to fight overfitting when you’re working with image data.

It’s easy to reuse an existing ConvNet on a new dataset via feature extraction. This is a valuable technique for working with small image datasets.

As a complement to feature extraction, you can use fine-tuning, which adapts to a new problem some of the representations previously learned by an existing model. This pushes performance a bit further.

### PDF p.283 (printed 259) — 8 Image classification › 8.3 Using a pretrained model › 
At this point, there are two ways you could proceed:

Running the convolutional base over your dataset, recording its output to a NumPy array on disk, and then using this data as input to a standalone, densely connected classifier similar to those you saw in chapters 4 and 5. This solution is fast and cheap to run, because it only requires running the convolutional base once for every input image, and the convolutional base is by far the most expensive part of the pipeline. But for the same reason, this technique won’t allow you to use data augmentation.

Extending the model you have (`conv_base`) by adding `Dense` layers on top and running the whole thing end to end on the input data. This will allow you to use data augmentation because every input image goes through the convolutional base every time it’s seen by the model. But for the same reason, this technique is far more expensive than the first.

We’ll cover both techniques. Let’s walk through the code required to set up the first one: recording the output of `conv_base` on your data and using these outputs as inputs to a new model.

### PDF p.288 (printed 264) — 8 Image classification › 8.3 Using a pretrained model › 
Figure 8.14 Training and validation metrics for feature extraction with data augmentation

We get a test accuracy of 98.4%. This is not an improvement over the previous model, which is a bit disappointing. This could be a sign that our data augmentation configuration does not exactly match the distribution of the test data. Let’s see if we can do better with our latest attempt.

### PDF p.285 (printed 261) — 8 Image classification › 8.3 Using a pretrained model › 
Figure 8.13 Training and validation metrics for plain feature extraction

You reach a validation accuracy of slightly over 98%—much better than you achieved in the previous section with the small model trained from scratch. This is a bit of an unfair comparison, however, because ImageNet contains many dog and cat instances, which means that our pretrained model already has the exact knowledge required for the task at hand. This won’t always be the case when you use pretrained features.

However, the plots also indicate that you’re overfitting almost from the start—despite using dropout with a fairly large rate. That’s because this technique doesn’t use data augmentation, which is essential for preventing overfitting with small image datasets. Let’s check the test accuracy:
