# Generation bundle for page_328  (PDF page 328, printed page 304)

Page kind: content
Chapter: 10 Interpreting what ConvNets learn
Section: 10.3 Visualizing heatmaps of class activation
Subsection: 10.3.4 Displaying the class activation heatmap
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_328.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_328.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p328_e03] (heading_subsection) 10.3.4 Displaying the class activation heatmap
[p328_e04] (body) Now, we apply pooling and importance weighting to the gradient tensor to obtain our heatmap of class activation.
[p328_e05] (listing_caption) Listing 10.23 Gradient pooling and channel importance weighting
[p328_e06] (code)
```
pooled_grads = np.mean(grads, axis=(0, 1, 2))
last_conv_layer_output = last_conv_layer_output[0].copy()
for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:, :, i] *= pooled_grads[i]
heatmap = np.mean(last_conv_layer_output, axis=-1)
```
[p328_e07] (body) For visualization purposes, you’ll also normalize the heatmap between 0 and 1. The result is shown in figure 10.8.
[p328_e08] (listing_caption) Listing 10.24 Heatmap post-processing
[p328_e09] (code)
```
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
```
[p328_e10] (figure_caption) Figure 10.8 Standalone class activation heatmap

## Code annotations printed next to listings (callouts, in order)
- This is a vector where each entry is the mean intensity of the
- gradient for a given channel. It quantifies the importance of
- each channel with regard to the top predicted class.
- The channel-wise mean of the resulting feature
- map is our heatmap of class activation.
- Multiplies each channel in the output of the last
- convolutional layer by how important this channel is

## Listings
- Listing 10.23: Gradient pooling and channel importance weighting (code elements p328_e06)
- Listing 10.24: Heatmap post-processing (code elements p328_e09)

## Figures
- Figure 10.8: Standalone class activation heatmap  | labels inside the figure: (none - image)

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

## Previous page (PDF 327) text, for continuity
10.3.3 Getting the gradient of the top class: JAX version

Finally, let’s do JAX. We define a separate loss computation function that takes the final layer’s output and returns the activation channel corresponding to the top predicted class. We use this activation value as our loss, allowing us to compute the gradient.

Listing 10.22 Computing the top class gradients with Jax

## Next page (PDF 329) text, for continuity
Finally, let’s generate an image that superimposes the original image on the heatmap you just obtained (see figure 10.9).

Figure 10.9 African elephant class activation heatmap over the test picture

Listing 10.25 Superimposing the heatmap with the original picture

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.323 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.326 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.312 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.313 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.309 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.12 — contents › 
- PDF p.264 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.311 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.323 (printed 299) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 10.3 Visualizing heatmaps of class activation
Here’s one last visualization technique—one that is useful for understanding which parts of a given image led a ConvNet to its final classification decision. This is helpful for “debugging” the decision process of a ConvNet, particularly in the case of a classification mistake (a problem domain called model interpretability). It can also allow you to locate specific objects in an image.

This general category of techniques is called class activation map (CAM) visualization, and it consists of producing heatmaps of class activation over input images. A class activation heatmap is a 2D grid of scores associated with a specific output class, computed for every location in any input image, indicating how important each location is with respect to the class under consideration. For instance, given an image fed into a dogs-versus-cats ConvNet, CAM visualization would allow you to generate a heatmap for the class “cat,” indicating how cat-like different parts of the image are, and also a heatmap for the class “dog,” indicating how dog-like parts of the image are. The specific implementation we’ll use is the one described in Selvaraju et al.1 Grad-CAM consists of taking the output feature 

### PDF p.323 (printed 299) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 10.3 Visualizing heatmaps of class activation
1 Ramprasaath R. Selvaraju, et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,” arxiv (2019), https://arxiv.org/abs/1610.02391.

### PDF p.643 (printed 619) — index ›  › T
TextVectorization layer 399, 402, 403, 405, 415, tanh activation 116

424, 430, 431 target 105, 214 tf.data 342 target boxes, prompting model with 327 tf.data API 248 target leaking 178 tf.data.Dataset class 248 target points, prompting model with 323–327 tf.data.Dataset instance 556 tar shell utility 311 tf.data pipeline 432, 473 task API 300 tf.GradientTape() function 302 TensorBoard 212 tf module 302 TensorFlow 60 theory of mind 570 class activation heatmap 302 timeseries_dataset_from_array() function 356, 357 customizing fit() with 223 timeseries forecasting 351 Dataset objects 248 recurrent neural networks 364–378 gradient ascent in 294 tasks 352 relationship between frameworks 63 temperature forecasting example 352–364 training step functions 216 tokenization 387 train_step() metrics handling with 226 tokens 384 tensorflow module 302 Top-K sampling 482

### PDF p.309 (printed 285) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 10.1 Visualizing intermediate activations
Visualizing intermediate activations consists of displaying the values returned by various convolution and pooling layers in a model, given a certain input (the output of a layer is often called its activation, the output of the activation function). This gives a view into how an input is decomposed into the different filters learned by the network. You want to visualize feature maps with three dimensions: width, height, and depth (channels). Each channel encodes relatively independent features, so the proper way to visualize these feature maps is by independently plotting the contents of every channel as a 2D image. Let’s start by loading the model that you saved in section 8.2:

### PDF p.326 (printed 302) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 10.3.1 Getting the gradient of the top class: TensorFlow version
Let’s start with the TensorFlow version, once again using `GradientTape`.

Listing 10.20 Computing the top class gradients with TensorFlow

### PDF p.326 (printed 302) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 10.3.2 Getting the gradient of the top class: PyTorch version
Listing 10.21 Computing the top class gradients with PyTorch
def get_top_class_gradients(img_array):
    last_conv_layer_output = last_conv_layer_model(img_array)

### PDF p.97 (printed 73) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › 
Figure 3.2 Our model’s predictions on the training inputs: pretty similar to the training targets

Recall that the prediction value for a given point `[x, y]` is simply `prediction == [[w1], ` `[w2]] • [x, y] + b == w1 * x + w2 * y + b`. Thus, class “0” is defined as `w1 * x + w2 * ` `y + b < 0.5` and class “1” is defined as `w1 * x + w2 * y + b > 0.5`. You’ll notice that what you’re looking at is really the equation of a line in the 2D plane: `w1 * x + w2 * y + b ` `= 0.5`. Class 1 is above the line; class 0 is below the line. You may be used to seeing line equations in the format `y = a * x + b`; in the same format, our line becomes `y = - w1 ` `/ w2 * x + (0.5 - b) / w2`.

Let’s plot this line, as shown in figure 3.3:

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
