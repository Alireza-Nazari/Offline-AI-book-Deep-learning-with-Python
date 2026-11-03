# Generation bundle for page_340  (PDF page 340, printed page 316)

Page kind: content
Chapter: 11 Image segmentation
Section: 11.2 Training a segmentation model from scratch
Subsection: 11.2.2 Building and training the segmentation model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_340.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_340.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p340_e03] (body) simply apply the corresponding sequence of `Conv2DTranspose` layers followed by a final `Conv2D` layer to produce outputs of shape `(200, 200, 3)`.
[p340_e04] (body) To evaluate the model, we’ll use a metric named Intersection over Union (IoU). It’s a measure of the match between the ground truth segmentation masks and the predicted masks. It can be computed separately for each class or averaged over multiple classes. Here’s how it works: 1 Compute the intersection between the masks, the area where the prediction and
[p340_e05] (body) ground truth overlap. 2 Compute the union of the masks, the total area covered by both masks combined. This is the whole space we’re interested in—the target object and any extra bits your model might have included by mistake. 3 Divide the intersection area by the union area to get the IoU. It’s a number between 0 and 1, where 1 denotes a perfect match, and 0 denotes a complete miss. We can simply use a built-in Keras metric rather than building this ourselves:
[p340_e06] (code)
```
foreground_iou = keras.metrics.IoU(
    num_classes=3,
    target_class_ids=(0,),
    name="foreground_iou",
    sparse_y_true=True,
    sparse_y_pred=False,
)
```
[p340_e07] (body) We can now compile and fit our model:
[p340_e08] (code)
```
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[foreground_iou],
)
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "oxford_segmentation.keras",
        save_best_only=True,
    ),
]
history = model.fit(
    train_input_imgs,
    train_targets,
    epochs=50,
    callbacks=callbacks,
```

## Code annotations printed next to listings (callouts, in order)
- Specifies the total number of classes
- Specifies the class to compute
- IoU for (0 = foreground)
- Our targets are sparse
- (integer class IDs).
- But our model’s predictions
- are a dense softmax!

## Equation candidates (formulas written inline in the text)
- [p340_e06] foreground_iou = keras.metrics.IoU(
    num_classes=3,
    target_class_ids=(0,),
    name="foreground_iou",
    sparse_y_true=True,
    sparse_y_pred=False,
)

## Chapter outline
Chapter: 11 Image segmentation (PDF pages 332-352)
  - 11.1 Computer vision tasks (p.332)
      - 11.1.1 Types of image segmentation (p.334)
  - 11.2 Training a segmentation model from scratch (p.335)
      - 11.2.1 Downloading a segmentation dataset (p.335)
      - 11.2.2 Building and training the segmentation model (p.338)
  - 11.3 Using a pretrained segmentation model (p.342)
      - 11.3.1 Downloading the Segment Anything Model (p.343)
      - 11.3.2 How Segment Anything works (p.343)
      - 11.3.3 Preparing a test image (p.345)
      - 11.3.4 Prompting the model with a target point (p.347)
      - 11.3.5 Prompting the model with a target box (p.351)

## Previous page (PDF 339) text, for continuity
The first half of the model closely resembles the kind of ConvNet you’d use for image classification: a stack of `Conv2D` layers, with gradually increasing filter sizes. We downsample our images three times by a factor of two each—ending up with activations of size `(25, 25, 256)`. The purpose of this first half is to encode the images into smaller feature maps, where each spatial location (or “pixel”) contains information about a large spatial chunk of the original image. You can understand it as a kind of compression.

One important difference between the first half of this model and the classification models you’ve seen before is the way we do downsampling: in the classification ConvNets from chapter 8, we used `MaxPooling2D` layers to downsample feature maps. Here, we downsample by adding strides to every other convolution layer (if you don’t remember the details of how convolution strides work, see chapter 8, section 8.1.1). We do this because, in the case of image segmentation, we care a lot about the spatial location of information in the image since we need to produce per-pixel target masks as output of the model. When you do 2 × 2 max pooling, you are completely destroying location information within each pooling window: you return one scalar value per window, with zero knowledge of which of the four locations in the windows the value came from. So, while max pooling layers perform well for classification tasks, they would hurt us quite a bit for a segmentation task. Meanwhile, strided convolutions do a better job at downsampling feature maps while retaining location information. Throughout this book, you’ll notice that we tend to use strides instead of max pooling in any model that cares about feature location, such as the generative models in chapter 17. The  …

## Next page (PDF 341) text, for continuity
Let’s display our training and validation loss (see figure 11.5):

Figure 11.5 Displaying training and validation loss curves

You can see that we start overfitting midway, around epoch 25. Let’s reload our best-performing model according to validation loss and demonstrate how to use it to predict a segmentation mask (see figure 11.6):

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.348 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.287 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.367 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.284 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.352 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.335 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.290 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.274 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.367 (printed 343) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
box is. To create a smooth estimate of how good a box prediction is, the authors propose using the Intersection over Union (IoU) metric we saw last chapter. If a grid cell is empty, the predicted confidence at the location should be zero. However, if a grid cell contains an object, we can use the IoU score between the current box prediction and the actual box as the target confidence value. This way, as the model becomes better at predicting box locations, the IoU score and the learned confidence values will go up.

This calls for a custom loss function. We can start be defining a utility to compute IoU scores for target and predicted boxes.

Listing 12.10 Computing IoU for two boxes

### PDF p.352 (printed 328) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › Summary
Image segmentation is one of the main categories of computer vision tasks. It consists of computing segmentation masks that describe the contents of an image at the pixel level.

To build your own segmentation model, use a stack of strided `Conv2D` layers to “compress” the input image into a smaller feature map, followed by a stack of corresponding `Conv2DTranspose` layers to “expand” the feature map into a segmentation mask the same size as the input image.

You can also use a pretrained segmentation model. Segment Anything, included in KerasHub, is a powerful model that supports image prompting, text prompting, point prompting, and box prompting.

### PDF p.348 (printed 324) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
The return value `outputs` has a `"masks"` field which contains four 256 × 256 candidate masks for the target object, ranked by decreasing match quality. The quality scores of the masks are available under the `"iou_pred"` field as part of the model’s output:

### PDF p.335 (printed 311) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 11.2.1 Downloading a segmentation dataset
We’ll work with the Oxford-IIIT Pets dataset (https://www.robots.ox.ac.uk/~vgg/data/ pets/), which contains 7,390 pictures of various breeds of cats and dogs, together with foreground-background segmentation masks for each picture. A segmentation mask is the image segmentation equivalent of a label: it’s an image the same size as the input image, with a single color channel where each integer value corresponds to the class of the corresponding pixel in the input image. In our case, the pixels of our segmentation masks can take one of three integer values:

1 (foreground)

2 (background)

3 (contour)

Let’s start by downloading and uncompressing our dataset, using the `wget` and `tar` shell utilities:

### PDF p.338 (printed 314) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 11.2.2 Building and training the segmentation model
import keras
from keras.layers import Rescaling, Conv2D, Conv2DTranspose
def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    x = Rescaling(1.0 / 255)(inputs)
    x = Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = Conv2D(64, 3, activation="relu", padding="same")(x)
    x = Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
    x = Conv2D(128, 3, activation="relu", padding="same")(x)
    x = Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = Conv2D(256, 3, activation="relu", padding="same")(x)
    x = Conv2DTranspose(256, 3, activation="relu", padding="same")(x)
    x = Conv2DTranspose(256, 3, strides=2, activation="relu", padding="same")(x)
    x = Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    x = Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(x)
    x = Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)

### PDF p.343 (printed 319) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3.1 Downloading the Segment Anything Model
First, let’s instantiate SAM and download its weights. Once again, we can use the KerasHub package to use this pretrained model without needing to implement it ourselves from scratch.

Remember the `ImageClassifier` task we used in the previous chapter? We can use another KerasHub task `ImageSegmenter` for wrapping pretrained image segmentation models into a high-level model with standard inputs and outputs. Here, we’ll use the `sam_huge_sa1b` pretrained model, where `sam` stands for the model, `huge` refers to the number of parameters in the model, and `sa1b` stands for the SA-1B dataset released along with the model, with 1 billion annotated masks. Let’s download it now:

### PDF p.338 (printed 314) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 11.2.2 Building and training the segmentation model
Now, it’s time to define our model:

### PDF p.344 (printed 320) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
dataset of images, which was used to train an initial model. This model was used to help drive a semiautomated stage of data collection, where images were first segmented by SAM and improved by human correction and further annotation.

Figure 11.7 An example image from the SA-1B dataset

The model is trained on `(image, prompt, mask)` triples. `image` and `prompt` are the model inputs. The image can be any input image, and the prompt can take a couple of forms:

A point inside the object to mask

A box around the object to mask

Given the `image` and `prompt` input, the model is expected to produce an accurate predicted mask for the object indicated by the prompt, which is compared with a ground truth `mask` label.

The model consists of a few separate components. An image encoder, similar to the Xception model we used in previous chapters, will take an input image and output a much smaller image embedding. This is something we already know how to build. Next, we add a prompt encoder, which is responsible for mapping prompts in any of the previously mentioned forms to an embedded vector, and a mask decoder, which takes in both the image embedding and prompt embedding and outputs a 
