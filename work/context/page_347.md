# Generation bundle for page_347  (PDF page 347, printed page 323)

Page kind: content
Chapter: 11 Image segmentation
Section: 11.3 Using a pretrained segmentation model
Subsection: 11.3.4 Prompting the model with a target point
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_347.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_347.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p347_e03] (code)
```
import matplotlib.pyplot as plt
from keras import ops
def show_image(image, ax):
    ax.imshow(ops.convert_to_numpy(image).astype("uint8"))
def show_mask(mask, ax):
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w, _ = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
def show_points(points, ax):
    x, y = points[:, 0], points[:, 1]
    ax.scatter(x, y, c="green", marker="*", s=375, ec="white", lw=1.25)
def show_box(box, ax):
    box = box.reshape(-1)
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, ec="red", fc="none", lw=2))
```
[p347_e04] (heading_subsection) 11.3.4 Prompting the model with a target point
[p347_e05] (body) To use SAM, you need to prompt it. This means we need one of the following:
[p347_e06] (bullet) Point prompts—Select a point in an image and let the model segment the object that the point belongs to.
[p347_e07] (bullet) Box prompts—Draw an approximate box around an object (it does not need to be particularly precise) and let the model segment the object in the box.
[p347_e08] (body) Let’s start with a point prompt. Points are labeled, with 1 indicating the foreground (the object you want to segment) and 0 indicating the background (everything around the object). In ambiguous cases, to improve your results, you could pass multiple labeled points, instead of a single point, to refine your definition of what should be included (points labeled 1) and what should be excluded (points labeled 0).
[p347_e09] (body) We try a single foreground point (see figure 11.10). Here’s a test point:
[p347_e10] (code)
```
import numpy as np
input_point = np.array([[580, 450]])
input_label = np.array([1])
plt.figure(figsize=(10, 10))
show_image(image, plt.gca())
show_points(input_point, plt.gca())
plt.show()
```

## Code annotations printed next to listings (callouts, in order)
- Coordinates of our point
- 1 means foreground, and
- 0 means background.
- “gca” means “get current
- axis”—the current figure.

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

## Previous page (PDF 346) text, for continuity
Figure 11.9 Our test image

SAM expects inputs that are 1024 × 1024. However, forcibly resizing arbitrary images to 1024 × 1024 would distort their aspect ratio—for instance, our image isn’t square. It’s better to first resize the image so that its longest side becomes 1,024 pixels long and then pad the remaining pixels with a filler value, such as 0. We can achieve this with the `pad_to_aspect_ratio` argument in the `keras.ops.image.resize()` operation, like this:

Next, let’s define a few utilities that will come in handy when using the model. We’re going to need to

Display images.

Display segmentation masks overlaid on an image.

Highlight specific points on an image.

Display boxes overlaid on an image.

All our utilities take a Matplotlib `axis` object (noted `ax`) so that they can all write to the same figure:

## Next page (PDF 348) text, for continuity
Figure 11.10 A prompt point, landing on a peach

Let’s prompt SAM with it:

The return value `outputs` has a `"masks"` field which contains four 256 × 256 candidate masks for the target object, ranked by decreasing match quality. The quality scores of the masks are available under the `"iou_pred"` field as part of the model’s output:

Let’s overlay the first mask on the image (see figure 11.11):

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.349 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.351 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.345 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.352 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.359 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.373 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.344 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.358 — 12 Object detection › 12.2 Training a YOLO model from scratch

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.351 (printed 327) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3.5 Prompting the model with a target box
Besides providing one or more target points, you can also provide boxes approximating the location of the object to segment. These boxes should be passed via the coordinates of their top-left and bottom-right corners. Here’s a box around the mango (see figure 11.14):

### PDF p.351 (printed 327) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3.5 Prompting the model with a target box
Figure 11.14 Box prompt around the mango

Let’s prompt SAM with it (see figure 11.15):

### PDF p.345 (printed 321) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
Figure 11.8 The Segment Anything high-level architecture overview

All of these subcomponents are trained simultaneously by forming batches of new `(image,` `prompt,` `mask)` triples to train on from the SA-1B image and mask data. The process here is actually quite simple. For a given input image, choose a random mask in the input. Next, randomly choose whether to create a box prompt or a point prompt. To create a point prompt, choose a random pixel inside the mask label. To create a box prompt, draw a box around all points inside the mask label. We can repeat this process indefinitely, sampling a number of `(image,` `prompt,` `mask)` triples from each image input.

### PDF p.344 (printed 320) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
dataset of images, which was used to train an initial model. This model was used to help drive a semiautomated stage of data collection, where images were first segmented by SAM and improved by human correction and further annotation.

Figure 11.7 An example image from the SA-1B dataset

The model is trained on `(image, prompt, mask)` triples. `image` and `prompt` are the model inputs. The image can be any input image, and the prompt can take a couple of forms:

A point inside the object to mask

A box around the object to mask

Given the `image` and `prompt` input, the model is expected to produce an accurate predicted mask for the object indicated by the prompt, which is compared with a ground truth `mask` label.

The model consists of a few separate components. An image encoder, similar to the Xception model we used in previous chapters, will take an input image and output a much smaller image embedding. This is something we already know how to build. Next, we add a prompt encoder, which is responsible for mapping prompts in any of the previously mentioned forms to an embedded vector, and a mask decoder, which takes in both the image embedding and prompt embedding and outputs a 

### PDF p.352 (printed 328) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › Summary
Image segmentation is one of the main categories of computer vision tasks. It consists of computing segmentation masks that describe the contents of an image at the pixel level.

To build your own segmentation model, use a stack of strided `Conv2D` layers to “compress” the input image into a smaller feature map, followed by a stack of corresponding `Conv2DTranspose` layers to “expand” the feature map into a segmentation mask the same size as the input image.

You can also use a pretrained segmentation model. Segment Anything, included in KerasHub, is a powerful model that supports image prompting, text prompting, point prompting, and box prompting.

### PDF p.354 (printed 330) — 12 Object detection › 12.1 Single-stage vs. two-stage object detectors › 
Figure 12.1 Object detectors draw boxes around objects in an image and label them.

You might be thinking, if I have a segmentation mask for an object instance, I can already compute the coordinates of the smallest box that contains the mask. So couldn’t we just use image segmentation all the time? Do we need object detection models at all?

Indeed, segmentation is a strict superset of detection. It returns all the information that could be returned by a detection model—plus a lot more. This increased wealth of information has a significant computational cost: a good object detection model will typically run much faster than an image segmentation model. It also has a data labeling cost: to train a segmentation model, you need to collect pixel-precise masks, which are much more time-consuming to produce than the mere bounding boxes required by object detection models. As a result, you will always want to use an object detection model if you have no need for pixel-level information—for instance, if all you want is to count objects in an image.

### PDF p.342 (printed 318) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3 Using a pretrained segmentation model
In the image classification example from chapter 8, you saw how using a pretrained model could significantly boost your accuracy—especially when you only have a few samples to train on. Image segmentation is no different.

The Segment Anything Model ,1 or SAM for short, is a powerful pretrained segmentation model you can use for, well, almost anything. It was developed by Meta AI and released in April 2023. It was trained on 11 million images and their segmentation masks, covering over 1 billion object instances. This massive amount of training data

1 Kirillov et al., “Segment Anything,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, arXiv

(2023), https://arxiv.org/abs/2304.02643.

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
