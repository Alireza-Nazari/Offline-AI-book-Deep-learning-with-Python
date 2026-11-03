# Generation bundle for page_317  (PDF page 317, printed page 293)

Page kind: content
Chapter: 10 Interpreting what ConvNets learn
Section: 10.2 Visualizing ConvNet filters
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_317.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_317.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p317_e03] (body) To use this model, we can simply call it on some input data, but we should be careful to apply our model-specific image preprocessing so that our images are scaled to the same range as the Xception pretraining data.
[p317_e04] (listing_caption) Listing 10.10 Using the feature extractor
[p317_e05] (code)
```
activation = feature_extractor(preprocessor(img_tensor))
```
[p317_e06] (body) Let’s use our feature extractor model to define a function that returns a scalar value quantifying how much a given input image “activates” a given filter in the layer. This is the loss function that we’ll maximize during the gradient ascent process:
[p317_e07] (code)
```
from keras import ops
def compute_loss(image, filter_index):
    activation = feature_extractor(image)
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return ops.mean(filter_activation)
```
[p317_e08] (heading_subsection) The difference between model.predict(x) and model(x)
[p317_e09] (sidebar) In the previous chapter, we used `predict(x)` for feature extraction. Here, we’re using `model(x)`. What gives? Both `y = model.predict(x)` and `y = model(x)` (where `x` is an array of input data) mean “run the model on `x` and retrieve the output `y`.” Yet, they aren’t exactly the same thing. `predict()` loops over the data in batches (in fact, you can specify the batch size via `predict(x, batch_size=64)`) and extracts the NumPy value of the outputs. It’s schematically equivalent to
[p317_e10] (code)
```
def predict(x):
    y_batches = []
    for x_batch in get_batches(x):
        y_batch = model(x).numpy()
        y_batches.append(y_batch)
    return np.concatenate(y_batches)
```
[p317_e11] (sidebar) This means that `predict()` calls can scale to very large arrays. Meanwhile, `model(x)` happens in-memory and doesn’t scale. On the other hand, `predict()` is not differentiable: TensorFlow, PyTorch, and JAX cannot backpropagate through it.

## Code annotations printed next to listings (callouts, in order)
- The loss function takes an image
- tensor and the index of the filter
- we consider (an integer).
- We avoid border artifacts by
- Returns the mean of the
- only involving nonborder
- activation values for the filter
- pixels in the loss: we discard
- the first 2 pixels along the
- sides of the activation.

## Listings
- Listing 10.10: Using the feature extractor (code elements p317_e05)

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

## Previous page (PDF 316) text, for continuity
gradient descent loop (the first one was in chapter 2). We will show it for TensorFlow, PyTorch, and Jax.

First, let’s instantiate the Xception model trained on the ImageNet dataset. We can once again use the KerasHub library, exactly as we did in chapter 8.

Listing 10.7 Instantiating the Xception convolutional base

We’re interested in the convolutional layers of the model—the `Conv2D` and `Separable` `Conv2D` layers. We’ll need to know their names so we can retrieve their outputs. Let’s print their names, in order of depth.

Listing 10.8 Printing the names of all convolutional layers in Xception

You’ll notice that the `SeparableConv2D` layers here are all named something like `block6_sepconv1`, `block7_sepconv2`, etc.—Xception is structured into blocks, each containing several convolutional layers.

Now let’s create a second model that returns the output of a specific layer—a “feature extractor” model. Because our model is a Functional API model, it is inspectable: you can query the `output` of one of its layers and reuse it in a new model. No need to copy the entire Xception code.

Listing 10.9 A feature extractor model returning a specific output

## Next page (PDF 318) text, for continuity
(continued)

You should use `model(x)` when you need to retrieve the gradients of the model call. And you should use `predict()` if you just need the output value. In other words, always use `predict()`, unless you’re in the middle of writing a low-level gradient descent loop (as we are now).

A non-obvious trick to help the gradient-ascent process go smoothly is to normalize the gradient tensor by dividing it by its L2 norm (the square root of the sum of the squares of the values in the tensor). This ensures that the magnitude of the updates done to the input image is always within the same range.

Let’s set up the gradient ascent step function. Anything that involves gradients requires calling backend-level APIs, such as `GradientTape` in TensorFlow, `.backward()` in PyTorch, and `jax.grad()` in JAX. Let’s line up all the code snippets for each of the three backends, starting with TensorFlow.

10.2.1 Gradient ascent in TensorFlow

For TensorFlow, we can just open a `GradientTape` scope and compute the loss inside of it to retrieve the gradients we need. We’ll use a `@tf.function` decorator to speed up computation:

Listing 10.11 Loss maximization via stochastic gradient ascent: TensorFlow

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.283 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.324 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.319 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.315 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.323 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.320 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.325 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.312 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.315 (printed 291) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 10.2 Visualizing ConvNet filters
Another easy way to inspect the filters learned by ConvNets is to display the visual pattern that each filter is meant to respond to. This can be done with gradient ascent in input space, applying gradient descent to the value of the input image of a ConvNet so as to maximize the response of a specific filter, starting from a blank input image. The resulting input image will be one that the chosen filter is maximally responsive to.

Let’s try this with the filters of the Xception model. The process is simple: we’ll build a loss function that maximizes the value of a given filter in a given convolution layer, and then we’ll use stochastic gradient descent to adjust the values of the input image so as to maximize this activation value. This will be your second example of a low-level

### PDF p.283 (printed 259) — 8 Image classification › 8.3 Using a pretrained model › Fast feature extraction without data augmentation
We’ll start by extracting features as NumPy arrays, by calling the `predict()` method of the `conv_base` model on our training, validation, and testing datasets. Let’s iterate over our datasets to extract the pretrained model’s features.

Listing 8.25 Extracting the image features and corresponding labels

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

### PDF p.282 (printed 258) — 8 Image classification › 8.3 Using a pretrained model › 
You’ll note a couple of things. First, KerasHub uses the term backbone to refer to the underlying feature extractor network without the classification head (it’s a little easier to type than “convolutional base”). It also uses a special constructor called `from_` `preset()` that will download the configuration and weights for the Xception model.

What’s that “41” in the name of the model we are using? Pretrained ConvNets are by convention often named by how “deep” they are. In this case, the 41 means that our Xception model has 41 trainable layers (conv and dense layers) stacked on top of each other. It’s the “deepest” model we’ve used so far in the book by a good margin. There’s one more missing piece we need before we can use this model. Every pretrained ConvNet will do some rescaling and resizing of images before pretraining. It’s important to make sure our input images match; otherwise, our model will need to relearn how to extract features from images with a totally different input range. Rather than keep track of which pretrained models use a `[0, 1]` input range for pixel values and which use a `[-1, 1]` range, we can use a KerasHub layer called `ImageConverter` that will re

### PDF p.126 (printed 102) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Inference: Using a model after training
For instance, if we use `predict()` on some of our validation data with the linear model we trained earlier, we get scalar scores that correspond to the model’s prediction for each input sample:

### PDF p.95 (printed 71) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › 
Because our linear classifier operates on 2D inputs, `W` is really just two scalar coefficients: `W` `=` `[[w1],` `[w2]]`. Meanwhile, `b` is a single scalar coefficient. As such, for given input point `[x,` `y]`, its prediction value is `prediction` `=` `[[w1],` `[w2]]` `•` `[x,` `y]` `+` `b` `=` `w1` `*` `x + w2` `*` `y` `+` `b`.

Here’s our loss function.

Listing 3.19 The mean squared error loss function

### PDF p.126 (printed 102) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Inference: Using a model after training
However, this will process all inputs in `new_inputs` at once, which may not be feasible if you’re looking at a lot of data (in particular, it may require more memory than your GPU has).

A better way to do inference is to use the `predict()` method. It will iterate over the data in small batches and return a NumPy array of predictions. And unlike `__call__()`, it can also process TensorFlow `Dataset` objects:

### PDF p.319 (printed 295) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 10.2.2 Gradient ascent in PyTorch
In the case of PyTorch, we use `loss.backward()` and `image.grad` to obtain the gradients of the loss with respect to the input image, like this.

Listing 10.12 Loss maximization via stochastic gradient ascent: PyTorch
