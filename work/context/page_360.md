# Generation bundle for page_360  (PDF page 360, printed page 336)

Page kind: content
Chapter: 12 Object detection
Section: 12.2 Training a YOLO model from scratch
Subsection: 12.2.2 Creating a YOLO model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_360.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_360.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p360_e03] (body) While it would be fun to train on all 18 gigabytes of our input data, we want to keep the examples in this book easily runnable on modest hardware. If we limit ourselves to only images with four or fewer boxes, we will make our training problem easier and a halve the data size. Let’s do this and shuffle our data—the images are grouped by object type, which would be terrible for training:
[p360_e04] (code)
```
import random
metadata = list(filter(lambda x: len(x["boxes"]) <= 4, metadata))
random.shuffle(metadata)
```
[p360_e05] (body) That’s it for data loading! Let’s start creating our YOLO model.
[p360_e06] (heading_subsection) 12.2.2 Creating a YOLO model
[p360_e07] (body) As mentioned previously, the YOLO model is a single stage detector. Rather than first attempting to identify all candidate objects in a scene and then classifying the object regions, YOLO will propose bounding boxes and object labels in one go.
[p360_e08] (body) Our model will divide an image up into a grid and predict two separate outputs at each grid location—a bounding box and a class label. In the original paper by Redmon et al.,3 the model actually predicted multiple boxes per grid location, but we keep things simple and just predict one box in each grid square. Most images will not have objects evenly distributed across a grid, and to account for this, the model will output a confidence score along with each box, as shown in figure 12.4. We’d like this confidence to be high when an object is detected at a location, and zero when there’s no object. Most grid locations will have no object and should report a near-zero confidence. Like many models in computer vision, the YOLO model uses a ConvNet backbone to obtain interesting high-level features for an input image, a concept we first explored in chapter 8. In their paper, the authors created their own backbone model and pretrained it with ImageNet for classification. Rather than do this ourselves, we can instead use KerasHub to load a pretrained backbone. Instead of using the Xception backbone we’ve used so far in this book, we will switch to ResNet, a family of models we first mentioned in chapter 9. The structure is quite similar to Xception, but ResNet uses strides instead of pooling layers to downsample the image. As we mentioned in chapter 11, strided convolutions are better when we care about the spatial location of the input. Let’s load up our pretrained model and matching preprocessing (to rescale the image). We will resize our images to 448 × 448; image input size is quite important for the object detection task.
[p360_e09] (footnote) 3 Redmon et al., “You Only Look Once: Unified, Real-Time Object Detection,” CoRR (2015), https://arxiv.org/ abs/1506.02640.

## Chapter outline
Chapter: 12 Object detection (PDF pages 353-374)
  - 12.1 Single-stage vs. two-stage object detectors (p.354)
      - 12.1.1 Two-stage R-CNN detectors (p.354)
      - 12.1.2 Single-stage detectors (p.356)
  - 12.2 Training a YOLO model from scratch (p.356)
      - 12.2.1 Downloading the COCO dataset (p.356)
      - 12.2.2 Creating a YOLO model (p.360)
      - 12.2.3 Readying the COCO data for the YOLO model (p.363)
      - 12.2.4 Training the YOLO model (p.366)
  - 12.3 Using a pretrained RetinaNet detector (p.370)

## Previous page (PDF 359) text, for continuity
Let’s use our new visualization to look at the sample image2 we were inspecting earlier (see figure 12.3):

Figure 12.3 YOLO outputs a bounding box prediction and class label for each image region.

2 Image from the COCO 2017 dataset, https://cocodataset.org/. Image from Flickr, http://farm8.staticflickr.com /7250/7520201840_3e01349e3f_z.jpg, CC BY 2.0 https://creativecommons.org/licenses/by/2.0/.

## Next page (PDF 361) text, for continuity
Figure 12.4 YOLO outputs as visualized in the first YOLO paper

Listing 12.5 Loading the ResNet model

Next, we can turn our backbone into a detection model by adding new layers for outputting box and class predictions. The setup proposed in the YOLO paper is quite simple. Take the output of a ConvNet backbone and feed it through two densely connected layers with an activation in the middle. Then, split the output. The first five numbers will be used for bounding box prediction (four for the box and one for the box confidence). The rest will be used for the class probability map shown in figure 12.4—a classification prediction for each grid location over all possible 91 labels.

Let’s write this out.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.363 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.370 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.364 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.366 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.372 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.374 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.357 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.356 — 12 Object detection › 12.2 Training a YOLO model from scratch

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.370 (printed 346) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 
Figure 12.7 Every bounding box predicted by the YOLO model

Our model learns very low-confidence values because it has not yet learned to consistently locate objects in a scene. To further improve the model, we should try a number of things:

Train for more epochs

Use the whole COCO dataset

Data augmentation (e.g., translating and rotating input images and boxes)

Improve our class probability map for overlapping boxes

Predict multiple boxes per grid location using a bigger output grid

All of these would positively affect the model performance and get us closer to the original YOLO training recipe. However, this example is really just to get a feel for object detection training—training an accurate COCO detection model from scratch would take a large amount of compute and time. Instead, to get a sense of a better-performing detection model, let’s try using a pretrained object detection model called RetinaNet.

### PDF p.364 (printed 340) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Let’s rework our training data so it conforms to this new grid structure. We can create two arrays as long as our dataset with our grid:

The first will contain our class probability map. We will mark all grid cells that intersect with a bounding box with the correct label. To keep our code simple, we won’t worry about overlapping boxes.

The second will contain the boxes themselves. We will translate all boxes to the grid and label the correct grid cell with the coordinates for the box. The confidence for an actual box in our label data will always be one, and the confidence for all other locations will be zero.

Listing 12.7 Creating the YOLO targets

### PDF p.363 (printed 339) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2.3 Readying the COCO data for the YOLO model
Our model is relatively simple, but we still need to preprocess our inputs to align them with the prediction grid. Each grid detector will be responsible for detecting any boxes whose center falls inside the grid box. Our model will output five floats for the box `(x, y, w, h, confidence)`. The `x` and `y` will represent the object’s center relative to the bounds of the grid cell (from 0 to 1). The `w` and `h` will represent the object’s size relative to the image size.

We already have the right `w` and `h` values in our training data. However, we need to translate our `x` and `y` values to and from the grid. Let’s define two utilities:

### PDF p.356 (printed 332) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2 Training a YOLO model from scratch
Overall, building an object detector can be a bit of an undertaking—not that there’s anything theoretically complex about it. There’s just a lot of code needed to handle manipulating bounding boxes and predicted output. To keep things simple, we will recreate the very first YOLO model from 2015. There are 12 YOLO versions as of this writing, but the original is a bit simpler to work with.

### PDF p.366 (printed 342) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2.4 Training the YOLO model
We have our model and our training data ready, but there’s one last element we need before we can actually run `fit()`: the loss function. Our model outputs predicted boxes and predicted grid labels. We saw in chapter 7 how we can define multiple losses for each output—Keras will simply sum the losses together during training. We can handle the classification loss with `sparse_categorical_crossentropy` as usual.

The box loss, however, needs some special consideration. The basic loss proposed by the YOLO authors is fairly simple. They use the sum-squared error of the difference between the target box parameters and the predicted ones. We will only compute this error for grid cells with actual boxes in the labeled data. The tricky part of the loss is the box confidence output. The authors wanted the confidence output to reflect not just the presence of an object, but also how good the predicted

### PDF p.374 (printed 350) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › Summary
Object detection identifies and locates objects within an image using bounding boxes. It’s basically a weaker version of image segmentation, but one that can be run much more efficiently.

There are two primary approaches to object detection: – Region-based Convolutional Neural Networks (R-CNNs), which are two-stage models that first propose regions of interest and then classify them with a ConvNet. – Single-stage detectors (like RetinaNet and YOLO), which perform both tasks in a single step. Single-stage detectors are generally faster and more efficient, making them suitable for real-time applications (e.g., self-driving cars).

YOLO computes two separate outputs simultaneously during training—possible bounding boxes and a class probability map: – Each candidate bounding box is paired with a confidence score, which is trained to target the Intersection over Union of the predicted box and the ground truth box. – The class probability map classifies different regions of an image as belonging to different objects.

RetinaNet builds on this idea by using a feature pyramid network (FPN), which combines features from multiple ConvNet layers to create feature maps at different scales, al

### PDF p.372 (printed 348) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 
We have four different model outputs: bounding boxes, `confidences`, labels, and the total number of detections. This is overall quite similar to our YOLO model. The model can predict a total of 100 objects for each input model.

Let’s try displaying the prediction with our box-drawing utilities (figure 12.9).

Listing 12.15 Running inference with RetinaNet

### PDF p.13 (printed xiii) — contents ›  › 
11.3

Using a pretrained segmentation model 318

Downloading the Segment Anything Model 319 ■ How Segment Anything works 319 ■ Preparing a test image 321 ■ Prompting the model with a target point 323 ■ Prompting the model with a target box 327

12

Object detection 329 12.1

Single-stage vs. two-stage object detectors 330

Two-stage R-CNN detectors 330 ■ Single-stage detectors 332

12.2

Training a YOLO model from scratch 332

Downloading the COCO dataset 332 ■ Creating a YOLO model 336 ■ Readying the COCO data for the YOLO model 339 Training the YOLO model 342

12.3

Using a pretrained RetinaNet detector 346 13 Timeseries forecasting 351 13.1 Different kinds of timeseries tasks 351 13.2 A temperature forecasting example 352

Preparing the data 356 ■ A commonsense, non-machine-learning baseline 359 ■ Let’s try a basic machine learning model 360 Let’s try a 1D convolutional model 362

13.3

Recurrent neural networks 364

Understanding recurrent neural networks 365 ■ A recurrent layer in Keras 368 ■ Getting the most out of recurrent neural networks 372 ■ Using recurrent dropout to fight overfitting 372 Stacking recurrent layers 375 ■ Using bidirectional RNNs 377

13.4

Going even fur
