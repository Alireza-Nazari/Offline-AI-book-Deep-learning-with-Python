# Generation bundle for page_300  (PDF page 300, printed page 276)

Page kind: content
Chapter: 9 ConvNet architecture patterns
Section: 9.3 Batch normalization
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_300.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_300.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p300_e03] (code)
```
├──────────────────────┼────────────────────┼────────────┼─────────────────────┤
│ conv2d_11 (Conv2D)   │ (None, 8, 8, 64)   │      2,112 │ add_2[0][0]         │
├──────────────────────┼────────────────────┼────────────┼─────────────────────┤
│ add_3 (Add)          │ (None, 8, 8, 64)   │          0 │ max_pooling2d_2[0]… │
│                      │                    │            │ conv2d_11[0][0]     │
├──────────────────────┼────────────────────┼────────────┼─────────────────────┤
│ conv2d_12 (Conv2D)   │ (None, 8, 8, 128)  │     73,856 │ add_3[0][0]         │
├──────────────────────┼────────────────────┼────────────┼─────────────────────┤
│ conv2d_13 (Conv2D)   │ (None, 8, 8, 128)  │    147,584 │ conv2d_12[0][0]     │
├──────────────────────┼────────────────────┼────────────┼─────────────────────┤
│ conv2d_14 (Conv2D)   │ (None, 8, 8, 128)  │      8,320 │ add_3[0][0]         │
├──────────────────────┼────────────────────┼────────────┼─────────────────────┤
│ add_4 (Add)          │ (None, 8, 8, 128)  │          0 │ conv2d_13[0][0],    │
│                      │                    │            │ conv2d_14[0][0]     │
├──────────────────────┼────────────────────┼────────────┼─────────────────────┤
│ global_average_pool… │ (None, 128)        │          0 │ add_4[0][0]         │
│ (GlobalAveragePooli… │                    │            │                     │
├──────────────────────┼────────────────────┼────────────┼─────────────────────┤
│ dense (Dense)        │ (None, 1)          │        129 │ global_average_poo… │
└──────────────────────┴────────────────────┴────────────┴─────────────────────┘
 Total params: 297,697 (1.14 MB)
 Trainable params: 297,697 (1.14 MB)
 Non-trainable params: 0 (0.00 B)
```
[p300_e04] (body) With residual connections, you can build networks of arbitrary depth, without having to worry about vanishing gradients. Now, let’s move on to the next essential ConvNet architecture pattern: batch normalization.
[p300_e05] (heading_section) 9.3
[p300_e06] (heading_section) Batch normalization
[p300_e07] (body) Normalization in machine learning is a broad category of methods that seek to make different samples seen by a machine learning model more similar to each other, which helps the model learn and generalize well to new data. The most common form of data normalization is one you’ve seen several times in this book already: centering the data on zero by subtracting the mean from the data and giving the data a unit standard deviation by dividing the data by its standard deviation. In effect, this makes the assumption that the data follows a normal (or Gaussian) distribution and makes sure this distribution is centered and scaled to unit variance:
[p300_e08] (code)
```
normalized_data = (data - np.mean(data, axis=...)) / np.std(data, axis=...)
```
[p300_e09] (body) Previous examples you saw in this book normalized data before feeding it into models. But data normalization may be a concern after every transformation performed by the network: even if the data entering a `Dense` or `Conv2D` network has a 0 mean and unit variance, there’s no reason to expect a priori that this will be the case for the data coming out. Could normalizing intermediate activations help?

## Equation candidates (formulas written inline in the text)
- [p300_e08] normalized_data = (data - np.mean(data, axis=...)) / np.std(data, axis=...)

## Chapter outline
Chapter: 9 ConvNet architecture patterns (PDF pages 292-307)
  - 9.1 Modularity, hierarchy, and reuse (p.293)
  - 9.2 Residual connections (p.296)
  - 9.3 Batch normalization (p.300)
  - 9.4 Depthwise separable convolutions (p.302)
  - 9.5 Putting it together: A mini Xception-like model (p.304)
  - 9.6 Beyond convolution: Vision Transformers (p.306)

## Previous page (PDF 299) text, for continuity
Let’s take a look at the model summary:

## Next page (PDF 301) text, for continuity
Batch normalization does just that. It’s a type of layer (`BatchNormalization` in Keras) introduced in 2015 by Ioffe and Szegedy;2 it can adaptively normalize data even as the mean and variance change over time during training. During training, it uses the mean and variance of the current batch of data to normalize samples, and during inference (when a big enough batch of representative data may not be available), it uses an exponential moving average of the batchwise mean and variance of the data seen during training. Although Ioffe and Szegedy’s original paper suggested that batch normalization operates by “reducing internal covariate shift,” no one really knows for sure why batch normalization helps. There are various hypotheses but no certitudes. You’ll find that this is true of many things in deep learning—deep learning is not an exact science but a set of ever-changing, empirically derived engineering best practices, woven together by unreliable narratives. You will sometimes feel like the book you have in hand tells you how to do something but doesn’t quite satisfactorily say why it works: that’s because we know the how but we don’t know the why. Whenever a reliable explanation is available, we make sure to mention it. Batch normalization isn’t one of those cases. In practice, the main effect of batch normalization appears to be that it helps with gradient propagation—much like residual connections—and thus allows for deeper networks. Some very deep networks can only be trained if they include multiple `Batch` `Normalization` layers. For instance, batch normalization is used liberally in many of the advanced ConvNet architectures that come packaged with Keras, such as ResNet50, EfficientNet, and Xception. The `BatchNormalization` layer can be used after any layer …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.257 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.310 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.302 — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions
- PDF p.627 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.264 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.305 — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model
- PDF p.538 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.152 — 4 Classification and regression › 4.3 Predicting house prices: A regression example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.302 (printed 278) — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions › 
The intuitive reason why is that batch normalization will center your inputs on zero, while your ReLU activation uses zero as a pivot for keeping or dropping activated channels: doing normalization before the activation maximizes the utilization of the ReLU. That said, this ordering best practice is not exactly critical, so if you do convolution-activation-batch normalization, your model will still train, and you won’t necessarily see worse results.

NOTE Batch normalization has many quirks. One of the main ones relates to fine-tuning: when fine-tuning a model that includes `BatchNormalization` layers, I recommend leaving these layers frozen (set their `trainable` attribute to `False`). Otherwise, they will keep updating their internal mean and variance, which can interfere with the very small updates applied to the surrounding `Conv2D` layers.

Now, let’s take a look at the last architecture pattern in our series: depthwise separable convolutions.

### PDF p.627 (printed 603) — 20 Conclusions › 20.1 Key concepts in review › 
When building a very deep ConvNet, it’s common to add batch normalization layers as well as residual connections—two architecture patterns that help gradient information flow smoothly through the network.

### PDF p.204 (printed 180) — 6 The universal workflow of machine learning › 6.2 Developing a model › 
converging. To make learning easier for your network, your data should have the following characteristics:

Take small values—Typically, most values should be in the 0–1 range.

Be homogeneous—That is, all features should take values in roughly the same range.

Additionally, the following stricter normalization practice is common and can help, although it isn’t always necessary (for example, you didn’t do this in the digit-classification example):

Normalize each feature independently to have a mean of 0.

Normalize each feature independently to have a standard deviation of 1.

This is easy to do with NumPy arrays:

### PDF p.305 (printed 281) — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model › 
Deep and narrow is better than broad and shallow.

Introducing residual connections around blocks of layers helps you train deeper networks.

It can be beneficial to introduce batch normalization layers after your convolution layers.

It can be beneficial to replace `Conv2D` layers with `SeparableConv2D` layers, which are more parameter efficient.

Let’s bring all of these ideas together into a single model. Its architecture resembles a smaller version of Xception. We’ll apply it to the dogs-versus-cats task from last chapter. For data loading and model training, simply reuse the exact same setup as what we used in chapter 8, section 8.2—but replace the model definition with the following ConvNet:

### PDF p.289 (printed 265) — 8 Image classification › 8.3 Using a pretrained model › 
Note that you should not unfreeze “batch normalization” layers (`BatchNormalization`). Batch normalization and its effect on fine-tuning is explained in the next chapter.

You already completed the first three steps when doing feature extraction. Let’s proceed with step 4: you’ll unfreeze your `conv_base`.

### PDF p.152 (printed 128) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Preparing the data
It would be problematic to feed into a neural network values that all take wildly different ranges. The model might be able to automatically adapt to such heterogeneous data, but it would definitely make learning more difficult. A widespread best practice to deal with such data is to do feature-wise normalization: for each feature in the input data (a column in the input data matrix), you subtract the mean of the feature and divide by the standard deviation, so that the feature is centered around 0 and has a unit standard deviation. This is easily done in NumPy.

Listing 4.25 Normalizing the data

### PDF p.469 (printed 445) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
You’ll note that the normalization layers we’re using here aren’t `BatchNormalization` layers like those we’ve used in image models. That’s because `BatchNormalization` doesn’t work well for sequence data. Instead, we’re using the `LayerNormalization` layer, which normalizes each sequence independently from other sequences in the batch— like this, in NumPy-like pseudocode:

### PDF p.203 (printed 179) — 6 The universal workflow of machine learning › 6.2 Developing a model › Value normalization
In the MNIST digit-classification example from chapter 2, you started from image data encoded as integers in the 0–255 range, encoding grayscale values. Before you fed this data into your network, you had to cast it to `float32` and divide by 255 so you’d end up with floating-point values in the 0–1 range. Similarly, when predicting house prices, you started from features that took a variety of ranges—some features had small floating-point values, others had fairly large integer values. Before you fed this data into your network, you had to normalize each feature independently so that it had a standard deviation of 1 and a mean of 0.

In general, it isn’t safe to feed into a neural network data that takes relatively large values (for example, multi-digit integers, which are much larger than the initial values taken by the weights of a network) or data that is heterogeneous (for example, data where one feature is in the range 0–1, and another is in the range 100–200). Doing so can trigger large gradient updates that will prevent the network from
