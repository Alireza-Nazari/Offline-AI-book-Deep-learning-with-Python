# Generation bundle for page_311  (PDF page 311, printed page 287)

Page kind: content
Chapter: 10 Interpreting what ConvNets learn
Section: 10.1 Visualizing intermediate activations
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_311.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_311.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p311_e03] (body) Let’s display the picture (see figure 10.1).
[p311_e04] (listing_caption) Listing 10.2 Displaying the test picture
[p311_e05] (code)
```
import matplotlib.pyplot as plt
plt.axis("off")
plt.imshow(img_tensor[0].astype("uint8"))
plt.show()
```
[p311_e06] (figure_caption) Figure 10.1 The test cat picture
[p311_e07] (body) To extract the feature maps you want to look at, you’ll create a Keras model that takes batches of images as input and outputs the activations of all convolution and pooling layers.
[p311_e08] (listing_caption) Listing 10.3 Instantiating a model that returns layer activations
[p311_e09] (code)
```
from keras import layers
layer_outputs = []
layer_names = []
for layer in model.layers:
    if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D)):
        layer_outputs.append(layer.output)
        layer_names.append(layer.name)
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
```

## Code annotations printed next to listings (callouts, in order)
- Extracts the outputs of all
- Conv2D and MaxPooling2D
- layers and put them in a list
- Creates a model that will return
- these outputs, given the model input
- Saves the layer names for later

## Listings
- Listing 10.2: Displaying the test picture (code elements p311_e05)
- Listing 10.3: Instantiating a model that returns layer activations (code elements p311_e09)

## Figures
- Figure 10.1: The test cat picture  | labels inside the figure: (none - image)

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

## Previous page (PDF 310) text, for continuity
Next, you’ll get an input image—a picture of a cat, not part of the images the network was trained on.

Listing 10.1 Preprocessing a single image

## Next page (PDF 312) text, for continuity
When fed an image input, this model returns the values of the layer activations in the original model, as a list. This is the first time you’ve encountered a multi-output model in this book in practice since you learned about them in chapter 7: until now, the models you’ve seen have had exactly one input and one output. This one has one input and nine outputs—one output per layer activation.

Listing 10.4 Using the model to compute layer activations

For instance, this is the activation of the first convolution layer for the cat image input:

It’s a 178 × 178 feature map with 32 channels. Let’s try plotting the sixth channel of the activation of the first layer of the original model (see figure 10.2).

Listing 10.5 Visualizing the sixth channel

Figure 10.2 Sixth channel of the activation of the first layer on the test cat picture

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.316 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.325 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.256 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.309 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.224 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.313 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.269 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.258 — 8 Image classification › 8.1 Introduction to convnets

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.309 (printed 285) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 10.1 Visualizing intermediate activations
Visualizing intermediate activations consists of displaying the values returned by various convolution and pooling layers in a model, given a certain input (the output of a layer is often called its activation, the output of the activation function). This gives a view into how an input is decomposed into the different filters learned by the network. You want to visualize feature maps with three dimensions: width, height, and depth (channels). Each channel encodes relatively independent features, so the proper way to visualize these feature maps is by independently plotting the contents of every channel as a 2D image. Let’s start by loading the model that you saved in section 8.2:

### PDF p.325 (printed 301) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 
First, we create a model that maps the input image to the activations of the last convolutional layer.

Listing 10.18 Returning the last convolutional output

### PDF p.309 (printed 285) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 10.1 Visualizing intermediate activations
>>> import keras
>>> model = keras.models.load_model(
...     "convnet_from_scratch_with_augmentation.keras"
... )
>>> model.summary()
Model: "functional_3"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_3 (InputLayer)        │ (None, 180, 180, 3)      │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ rescaling_1 (Rescaling)           │ (None, 180, 180, 3)      │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ conv2d_11 (Conv2D)                │ (None, 178, 178, 32)     │           896 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤

### PDF p.314 (printed 290) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 
Figure 10.3 Every channel of every layer activation on the test cat picture

There are a few things to note here:

The first layer acts as a collection of various edge detectors. At that stage, the activations retain almost all of the information present in the initial picture.

As you go higher, the activations become increasingly abstract and less visually interpretable. They begin to encode higher-level concepts such as “cat ear” and “cat eye.” Higher representations carry increasingly less information about the visual contents of the image and increasingly more information related to the class of the image.

The sparsity of the activations increases with the depth of the layer: in the first layer, all filters are activated by the input image, but in the following layers, more and more filters are blank. This means the pattern encoded by the filter isn’t found in the input image.

### PDF p.325 (printed 301) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 
Second, we create a model that maps the activations of the last convolutional layer to the final class predictions.

Listing 10.19 Going from the last convolutional output to final predictions

### PDF p.225 (printed 201) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › 
This enables you to do feature extraction: creating models that reuse intermediate features from another model.

Let’s say you want to add another output to the model we previously defined—you want to also predict an estimate of how long a given issue ticket will take to resolve, a kind of difficulty rating. You could do this via a classification layer over three categories—”quick,” “medium,” and “difficult.” You don’t need to recreate and retrain a model from scratch! You can just start from the intermediate features of your previous model, since you have access to them.

Listing 7.13 Creating a new model by reusing intermediate layer outputs

### PDF p.269 (printed 245) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › Building your model
Listing 8.7 Instantiating a small ConvNet for dogs vs. cats classification
import keras
from keras import layers
inputs = keras.Input(shape=(180, 180, 3))
x = layers.Rescaling(1.0 / 255)(inputs)

### PDF p.638 (printed 614) — index ›  › 
intelligent agents 571

keras.callbacks module 209 intermediate activations, visualizing 285–291 keras.callbacks.TensorBoard callback 213 intermediate ConvNet outputs 285 keras.datasets 172 intermediate meaning, defined 462–463 KerasHub library 258, 300 interpolation 145, 463 keras_hub.utils.coco_id_to_name(id) utility 334 interpretability, defined 284 keras.metrics.Mean metric 221 intuition, defined 586 keras.metrics.Metric class 207 InverseTimeDecay schedule 529 keras.metrics module 207 IoU (Intersection over Union) 316, 343 keras.ops.image.resize() operation 322 KerasTuner 540–546
