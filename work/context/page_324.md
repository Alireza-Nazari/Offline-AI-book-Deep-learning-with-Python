# Generation bundle for page_324  (PDF page 324, printed page 300)

Page kind: content
Chapter: 10 Interpreting what ConvNets learn
Section: 10.3 Visualizing heatmaps of class activation
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_324.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_324.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p324_e03] (figure_caption) Figure 10.7 Test picture of African elephants
[p324_e04] (body) So far, we have only used KerasHub to instantiate a pretrained feature extractor network using the backbone class. For Grad-CAM, we need the entire Xception model including the classification head—recall that Xception was trained on the ImageNet dataset with ~1 million labeled images belonging to 1,000 different classes.
[p324_e05] (body) KerasHub provides a high-level task API for common end-to-end workflows like image classification, text classification, image generation, and so on. A task wraps preprocessing, a feature extraction network, and a task-specific head into a single class that is easy to use. Let’s try it out:
[p324_e06] (code)
```
>>> model = keras_hub.models.ImageClassifier.from_preset(
...    "xception_41_imagenet",
...    activation="softmax",
... )
>>> preds = model.predict(img_array)
>>> preds.shape
(1, 1000)
>>> keras_hub.utils.decode_imagenet_predictions(preds)
[[("African_elephant", 0.90331),
  ("tusker", 0.05487),
  ("Indian_elephant", 0.01637),
```

## Code annotations printed next to listings (callouts, in order)
- We can configure the final activation of
- the classifier. Here, we use a softmax
- activation so our outputs are probabilities.
- ImageNet has
- 1,000 classes, so
- each prediction
- from our classifier
- has 1,000 entries.

## Figures
- Figure 10.7: Test picture of African elephants  | labels inside the figure: (none - image)

## Chapter outline
Chapter: 10 Interpreting what ConvNets learn (PDF pages 308-331)
  - 10.1 Visualizing intermediate activations (p.309)
  - 10.2 Visualizing ConvNet filters (p.315)
      - 10.2.1 Gradient ascent in TensorFlow (p.318)
      - 10.2.2 Gradient ascent in PyTorch (p.319)
      - 10.2.3 Gradient ascent in JAX (p.319)
      - 10.2.4 The filter visualization loop (p.320)
  - 10.3 Visualizing heatmaps of class activation (p.323)
      - 10.3.1 Getting the gradient of the top class: TensorFlow version (p.326)
      - 10.3.2 Getting the gradient of the top class: PyTorch version (p.326)
      - 10.3.3 Getting the gradient of the top class: JAX version (p.327)
      - 10.3.4 Displaying the class activation heatmap (p.328)
  - 10.4 Visualizing the latent space of a ConvNet (p.330)

## Previous page (PDF 323) text, for continuity
The filters in higher layers begin to resemble textures found in natural images: feathers, eyes, leaves, and so on.

10.3 Visualizing heatmaps of class activation

Here’s one last visualization technique—one that is useful for understanding which parts of a given image led a ConvNet to its final classification decision. This is helpful for “debugging” the decision process of a ConvNet, particularly in the case of a classification mistake (a problem domain called model interpretability). It can also allow you to locate specific objects in an image.

This general category of techniques is called class activation map (CAM) visualization, and it consists of producing heatmaps of class activation over input images. A class activation heatmap is a 2D grid of scores associated with a specific output class, computed for every location in any input image, indicating how important each location is with respect to the class under consideration. For instance, given an image fed into a dogs-versus-cats ConvNet, CAM visualization would allow you to generate a heatmap for the class “cat,” indicating how cat-like different parts of the image are, and also a heatmap for the class “dog,” indicating how dog-like parts of the image are. The specific implementation we’ll use is the one described in Selvaraju et al.1 Grad-CAM consists of taking the output feature map of a convolution layer, given an input image, and weighting every channel in that feature map by the gradient of the class with respect to the channel. Intuitively, one way to understand this trick is that you’re weighting a spatial map of “how intensely the input image activates different channels” by “how important each channel is with regard to the class,” resulting in a spatial map of “how intensely the input image activates …

## Next page (PDF 325) text, for continuity
The top five classes predicted for this image are as follows:

African elephant (with 90% probability)

Tusker (with 5% probability)

Indian elephant (with 2% probability)

Triceratops and Mexican hairless dog with less than 0.1% probability

The network has recognized the image as containing an undetermined quantity of African elephants. The entry in the prediction vector that was maximally activated is the one corresponding to the “African elephant” class, at index 386:

To visualize which parts of the image are the most African elephant–like, let’s set up the Grad-CAM process.

You will note that we didn’t need to preprocess our image before calling the task model. That’s because the KerasHub `ImageClassifier` is preprocessing inputs for us as part of `predict()`. Let’s preprocess the image ourselves so we can use the preprocessed inputs directly:

First, we create a model that maps the input image to the activations of the last convolutional layer.

Listing 10.18 Returning the last convolutional output

Second, we create a model that maps the activations of the last convolutional layer to the final class predictions.

Listing 10.19 Going from the last convolutional output to final predictions

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.282 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.280 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.317 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.316 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.310 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.372 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.330 — 10 Interpreting what ConvNets learn › 10.4 Visualizing the latent space of a ConvNet
- PDF p.283 — 8 Image classification › 8.3 Using a pretrained model

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.282 (printed 258) — 8 Image classification › 8.3 Using a pretrained model › 
You’ll note a couple of things. First, KerasHub uses the term backbone to refer to the underlying feature extractor network without the classification head (it’s a little easier to type than “convolutional base”). It also uses a special constructor called `from_` `preset()` that will download the configuration and weights for the Xception model.

What’s that “41” in the name of the model we are using? Pretrained ConvNets are by convention often named by how “deep” they are. In this case, the 41 means that our Xception model has 41 trainable layers (conv and dense layers) stacked on top of each other. It’s the “deepest” model we’ve used so far in the book by a good margin. There’s one more missing piece we need before we can use this model. Every pretrained ConvNet will do some rescaling and resizing of images before pretraining. It’s important to make sure our input images match; otherwise, our model will need to relearn how to extract features from images with a totally different input range. Rather than keep track of which pretrained models use a `[0, 1]` input range for pixel values and which use a `[-1, 1]` range, we can use a KerasHub layer called `ImageConverter` that will re

### PDF p.282 (printed 258) — 8 Image classification › 8.3 Using a pretrained model › 
only the first few layers of the model to do feature extraction, rather than using the entire convolutional base.

In this case, because the ImageNet class set contains multiple dog and cat classes, it’s likely to be beneficial to reuse the information contained in the densely connected layers of the original model. But we’ll choose not to, so we can cover the more general case where the class set of the new problem doesn’t overlap the class set of the original model. Let’s put this in practice by using the convolutional base of our pretrained model to extract interesting features from cat and dog images and then train a dogs-versus-cats classifier on top of these features. We will use the KerasHub library to create all pretrained models used in this book. KerasHub contains Keras implementations of popular pretrained model architectures paired with pretrained weights that can be downloaded to your machine. It contains a number of ConvNets like Xception, ResNet, EfficientNet and MobileNet, as well as larger, generative models we will use in the later chapters of this book. Let’s try using it to instantiate the Xception model trained on the ImageNet dataset.

NOTE KerasHub comes as a

### PDF p.280 (printed 256) — 8 Image classification › 8.3 Using a pretrained model › Using a pretrained model
A common and highly effective approach to deep learning on small image datasets is to use a pretrained model. A pretrained model is a model that was previously trained on a large dataset, typically on a large-scale image-classification task. If this original dataset is large enough and general enough, then the spatial hierarchy of features learned by the pretrained model can effectively act as a generic model of the visual world, and hence its features can prove useful for many different computer vision problems, even though these new problems may involve completely different classes than those of the original task. For instance, you might train a model on ImageNet (where classes are mostly animals and everyday objects) and then repurpose this trained model for something as remote as identifying furniture items in images. Such portability of learned features across different problems is a key advantage of deep learning compared to many older, shallow learning approaches, and it makes deep learning very effective for small-data problems.

In this case, let’s consider a large ConvNet trained on the ImageNet dataset (1.4 million labeled images and 1,000 different classes). ImageNet co

### PDF p.513 (printed 489) — 16 Text generation › 16.3 Using a pretrained LLM › 
The `CausalLM` we loaded from the KerasHub library is a high-level object for end-to-end causal language modeling. It wraps two objects: a `preprocessor` layer, which preprocesses text input, and a `backbone` model, which contains the numerics of the model forward pass.

Preprocessing is included by default in high-level Keras functions like `fit()` and `predict()`. But let’s run our preprocessing on a single batch so we can better see what it is doing:

### PDF p.316 (printed 292) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 
gradient descent loop (the first one was in chapter 2). We will show it for TensorFlow, PyTorch, and Jax.

First, let’s instantiate the Xception model trained on the ImageNet dataset. We can once again use the KerasHub library, exactly as we did in chapter 8.

Listing 10.7 Instantiating the Xception convolutional base

### PDF p.343 (printed 319) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3.1 Downloading the Segment Anything Model
First, let’s instantiate SAM and download its weights. Once again, we can use the KerasHub package to use this pretrained model without needing to implement it ourselves from scratch.

Remember the `ImageClassifier` task we used in the previous chapter? We can use another KerasHub task `ImageSegmenter` for wrapping pretrained image segmentation models into a high-level model with standard inputs and outputs. Here, we’ll use the `sam_huge_sa1b` pretrained model, where `sam` stands for the model, `huge` refers to the number of parameters in the model, and `sa1b` stands for the SA-1B dataset released along with the model, with 1 billion annotated masks. Let’s download it now:

### PDF p.372 (printed 348) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 
Next, let’s download the model and make a prediction. As we did in the previous chapter, we can use the high-level task API in KerasHub to create an `ObjectDetector` and use it—preprocessing included.

Listing 12.14 Creating the ResNet model

### PDF p.317 (printed 293) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 
To use this model, we can simply call it on some input data, but we should be careful to apply our model-specific image preprocessing so that our images are scaled to the same range as the Xception pretraining data.

Listing 10.10 Using the feature extractor
