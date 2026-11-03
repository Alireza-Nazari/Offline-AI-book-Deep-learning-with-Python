# Generation bundle for page_325  (PDF page 325, printed page 301)

Page kind: content
Chapter: 10 Interpreting what ConvNets learn
Section: 10.3 Visualizing heatmaps of class activation
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_325.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_325.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p325_e03] (code)
```
  ("triceratops", 0.00029),
  ("Mexican_hairless", 0.00018)]]
```
[p325_e04] (body) The top five classes predicted for this image are as follows:
[p325_e05] (bullet) African elephant (with 90% probability)
[p325_e06] (bullet) Tusker (with 5% probability)
[p325_e07] (bullet) Indian elephant (with 2% probability)
[p325_e08] (bullet) Triceratops and Mexican hairless dog with less than 0.1% probability
[p325_e09] (body) The network has recognized the image as containing an undetermined quantity of African elephants. The entry in the prediction vector that was maximally activated is the one corresponding to the “African elephant” class, at index 386:
[p325_e10] (code)
```
>>> np.argmax(preds[0])
386
```
[p325_e11] (body) To visualize which parts of the image are the most African elephant–like, let’s set up the Grad-CAM process.
[p325_e12] (body) You will note that we didn’t need to preprocess our image before calling the task model. That’s because the KerasHub `ImageClassifier` is preprocessing inputs for us as part of `predict()`. Let’s preprocess the image ourselves so we can use the preprocessed inputs directly:
[p325_e13] (code)
```
img_array = model.preprocessor(img_array)
```
[p325_e14] (body) First, we create a model that maps the input image to the activations of the last convolutional layer.
[p325_e15] (listing_caption) Listing 10.18 Returning the last convolutional output
[p325_e16] (code)
```
last_conv_layer_name = "block14_sepconv2_act"
last_conv_layer = model.backbone.get_layer(last_conv_layer_name)
last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
```
[p325_e17] (body) Second, we create a model that maps the activations of the last convolutional layer to the final class predictions.
[p325_e18] (listing_caption) Listing 10.19 Going from the last convolutional output to final predictions
[p325_e19] (code)
```
classifier_input = last_conv_layer.output
x = classifier_input
```

## Code annotations printed next to listings (callouts, in order)
- KerasHub tasks like ImageClassifier
- have a preprocessor layer.

## Listings
- Listing 10.18: Returning the last convolutional output (code elements p325_e16)
- Listing 10.19: Going from the last convolutional output to final predictions (code elements p325_e19)

## Equation candidates (formulas written inline in the text)
- [p325_e13] img_array = model.preprocessor(img_array)

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

## Previous page (PDF 324) text, for continuity
Figure 10.7 Test picture of African elephants

So far, we have only used KerasHub to instantiate a pretrained feature extractor network using the backbone class. For Grad-CAM, we need the entire Xception model including the classification head—recall that Xception was trained on the ImageNet dataset with ~1 million labeled images belonging to 1,000 different classes.

KerasHub provides a high-level task API for common end-to-end workflows like image classification, text classification, image generation, and so on. A task wraps preprocessing, a feature extraction network, and a task-specific head into a single class that is easy to use. Let’s try it out:

## Next page (PDF 326) text, for continuity
Then, we compute the gradient of the top predicted class for our input image with respect to the activations of the last convolution layer. Once again, having to compute gradients means we have to use backend APIs.

10.3.1 Getting the gradient of the top class: TensorFlow version

Let’s start with the TensorFlow version, once again using `GradientTape`.

Listing 10.20 Computing the top class gradients with TensorFlow

10.3.2 Getting the gradient of the top class: PyTorch version

Next, here’s the PyTorch version, using `.backward()` and `.grad`.

Listing 10.21 Computing the top class gradients with PyTorch

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.323 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.330 — 10 Interpreting what ConvNets learn › 10.4 Visualizing the latent space of a ConvNet
- PDF p.312 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.316 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.258 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.309 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.310 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.143 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.323 (printed 299) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 10.3 Visualizing heatmaps of class activation
Here’s one last visualization technique—one that is useful for understanding which parts of a given image led a ConvNet to its final classification decision. This is helpful for “debugging” the decision process of a ConvNet, particularly in the case of a classification mistake (a problem domain called model interpretability). It can also allow you to locate specific objects in an image.

This general category of techniques is called class activation map (CAM) visualization, and it consists of producing heatmaps of class activation over input images. A class activation heatmap is a 2D grid of scores associated with a specific output class, computed for every location in any input image, indicating how important each location is with respect to the class under consideration. For instance, given an image fed into a dogs-versus-cats ConvNet, CAM visualization would allow you to generate a heatmap for the class “cat,” indicating how cat-like different parts of the image are, and also a heatmap for the class “dog,” indicating how dog-like parts of the image are. The specific implementation we’ll use is the one described in Selvaraju et al.1 Grad-CAM consists of taking the output feature 

### PDF p.329 (printed 305) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 
Finally, let’s generate an image that superimposes the original image on the heatmap you just obtained (see figure 10.9).

Figure 10.9 African elephant class activation heatmap over the test picture

Listing 10.25 Superimposing the heatmap with the original picture

### PDF p.330 (printed 306) — 10 Interpreting what ConvNets learn › 10.4 Visualizing the latent space of a ConvNet › 
This visualization technique answers two important questions:

Why did the network think this image contained an African elephant?

Where is the African elephant located in the picture?

In particular, it’s interesting to note that the ears of the elephant calf are strongly activated: this is probably how the network can tell the difference between African and Indian elephants.

### PDF p.323 (printed 299) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 10.3 Visualizing heatmaps of class activation
Listing 10.17 Preprocessing an input image for Xception
img_path = keras.utils.get_file(
    fname="elephant.jpg",
    origin="https://img-datasets.s3.amazonaws.com/elephant.jpg",
)
img = keras.utils.load_img(img_path)
img_array = np.expand_dims(img, axis=0)

### PDF p.282 (printed 258) — 8 Image classification › 8.3 Using a pretrained model › 
only the first few layers of the model to do feature extraction, rather than using the entire convolutional base.

In this case, because the ImageNet class set contains multiple dog and cat classes, it’s likely to be beneficial to reuse the information contained in the densely connected layers of the original model. But we’ll choose not to, so we can cover the more general case where the class set of the new problem doesn’t overlap the class set of the original model. Let’s put this in practice by using the convolutional base of our pretrained model to extract interesting features from cat and dog images and then train a dogs-versus-cats classifier on top of these features. We will use the KerasHub library to create all pretrained models used in this book. KerasHub contains Keras implementations of popular pretrained model architectures paired with pretrained weights that can be downloaded to your machine. It contains a number of ConvNets like Xception, ResNet, EfficientNet and MobileNet, as well as larger, generative models we will use in the later chapters of this book. Let’s try using it to instantiate the Xception model trained on the ImageNet dataset.

NOTE KerasHub comes as a

### PDF p.363 (printed 339) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Our backbone outputs have shape `(batch_size, 14, 14, 2048)`. That is 401,408 output floats per image, a bit too many to feed into our dense layers. We downscale the feature maps with a strided conv layer to `(batch_size, 6, 6, 512)` with a more manageable 18,432 floats per image.

Next, we can add our two densely connected layers. We flatten the entire feature map, pass it through a `Dense` with a `relu` activation and then pass it through a final `Dense` with our exact number of output predictions—5 for the bounding box and confidence and 91 for each object class at each grid location. Finally, we reshape the outputs back to a 6 × 6 grid and split our box and class predictions. As usual for our classification outputs, we apply a softmax. The box outputs will need more special consideration; we will cover this later. Looking good! Note that because we flatten the entire feature map through the classification layer, every grid detector can use the entire image’s features; there’s no locality constraint. This is by design—large objects will not stay contained to a single grid cell.

### PDF p.311 (printed 287) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 
Figure 10.1 The test cat picture

To extract the feature maps you want to look at, you’ll create a Keras model that takes batches of images as input and outputs the activations of all convolution and pooling layers.

Listing 10.3 Instantiating a model that returns layer activations

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
