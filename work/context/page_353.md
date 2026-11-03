# Generation bundle for page_353  (PDF page 353, printed page 329)

Page kind: chapter_opener
Chapter: 12 Object detection
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_353.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_353.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p353_e01] (chapter_number) 12
[p353_e02] (chapter_title) Object detection
[p353_e03] (chapter_covers) This chapter covers
[p353_e04] (bullet) Understanding the object detection problem
[p353_e05] (bullet) Two-stage and single-stage object detectors
[p353_e06] (bullet) Training a simple single-stage detector from
[p353_e07] (bullet) scratch
[p353_e08] (bullet) Using a pretrained object detector
[p353_e09] (body) Object detection is all about drawing boxes (called bounding boxes) around objects of interest in a picture (see figure 12.1). This enables you to know not just which objects are in a picture, but also where they are. Some of its most common applications are
[p353_e10] (bullet) Counting—Find out how many instances of an object are in an image.
[p353_e11] (bullet) Tracking—Track how objects move in a scene over time by performing object detection on every frame of a movie.
[p353_e12] (bullet) Cropping—Identify the area of an image that contains an object of interest to crop it and send a higher-resolution version of the image patch to a classifier or an Optical Character Recognition (OCR) model.

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

## Previous page (PDF 352) text, for continuity
Figure 11.15 Segmented mango

SAM can be a powerful tool to quickly create large datasets of images annotated with segmentation masks.

Summary

Image segmentation is one of the main categories of computer vision tasks. It consists of computing segmentation masks that describe the contents of an image at the pixel level.

To build your own segmentation model, use a stack of strided `Conv2D` layers to “compress” the input image into a smaller feature map, followed by a stack of corresponding `Conv2DTranspose` layers to “expand” the feature map into a segmentation mask the same size as the input image.

You can also use a pretrained segmentation model. Segment Anything, included in KerasHub, is a powerful model that supports image prompting, text prompting, point prompting, and box prompting.

## Next page (PDF 354) text, for continuity
Figure 12.1 Object detectors draw boxes around objects in an image and label them.

You might be thinking, if I have a segmentation mask for an object instance, I can already compute the coordinates of the smallest box that contains the mask. So couldn’t we just use image segmentation all the time? Do we need object detection models at all?

Indeed, segmentation is a strict superset of detection. It returns all the information that could be returned by a detection model—plus a lot more. This increased wealth of information has a significant computational cost: a good object detection model will typically run much faster than an image segmentation model. It also has a data labeling cost: to train a segmentation model, you need to collect pixel-precise masks, which are much more time-consuming to produce than the mere bounding boxes required by object detection models. As a result, you will always want to use an object detection model if you have no need for pixel-level information—for instance, if all you want is to count objects in an image.

12.1 Single-stage vs. two-stage object detectors

There are two broad categories of object detection architectures:

Two-stage detectors, which first extract region proposals, known as Region-based Convolutional Neural Networks (R-CNN) models

Single-stage detectors, such as RetinaNet or the You Only Look Once family of models

Here’s how they work.

12.1.1 Two-stage R-CNN detectors

A region-based ConvNet, or R-CNN model, is a two-stage model. The first stage takes an image and produces a few thousand partially overlapping bounding boxes around

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.374 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.370 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.360 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.356 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.355 — 12 Object detection › 12.1 Single-stage vs. two-stage object detectors
- PDF p.373 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.333 — 11 Image segmentation › 11.1 Computer vision tasks
- PDF p.363 — 12 Object detection › 12.2 Training a YOLO model from scratch

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.374 (printed 350) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › Summary
Object detection identifies and locates objects within an image using bounding boxes. It’s basically a weaker version of image segmentation, but one that can be run much more efficiently.

There are two primary approaches to object detection: – Region-based Convolutional Neural Networks (R-CNNs), which are two-stage models that first propose regions of interest and then classify them with a ConvNet. – Single-stage detectors (like RetinaNet and YOLO), which perform both tasks in a single step. Single-stage detectors are generally faster and more efficient, making them suitable for real-time applications (e.g., self-driving cars).

YOLO computes two separate outputs simultaneously during training—possible bounding boxes and a class probability map: – Each candidate bounding box is paired with a confidence score, which is trained to target the Intersection over Union of the predicted box and the ground truth box. – The class probability map classifies different regions of an image as belonging to different objects.

RetinaNet builds on this idea by using a feature pyramid network (FPN), which combines features from multiple ConvNet layers to create feature maps at different scales, al

### PDF p.356 (printed 332) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.1.2 Single-stage detectors
Around 2015, researchers and practitioners began experimenting with using a single deep learning model to jointly predict bounding box coordinates together with their labels, an architecture called a single-stage detector. The main families of single-stage detectors are RetinaNet, Single Shot MultiBox Detectors (SSD), and the You Only Look Once family, abbreviated as YOLO. Yes, like the meme. That’s on purpose.

Single-stage detectors, especially recent YOLO iterations, boast significantly faster speeds and greater efficiency than their two-stage counterparts, albeit with a minor potential tradeoff in accuracy. Nowadays, YOLO is arguably the most popular object detection model out there, especially when it comes to real-time applications. There’s usually a new version of it that comes out every year—interestingly, each new version tends to be developed by a separate organization. In the next section, we will build a simplified YOLO model from scratch.

### PDF p.360 (printed 336) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2.2 Creating a YOLO model
As mentioned previously, the YOLO model is a single stage detector. Rather than first attempting to identify all candidate objects in a scene and then classifying the object regions, YOLO will propose bounding boxes and object labels in one go.

Our model will divide an image up into a grid and predict two separate outputs at each grid location—a bounding box and a class label. In the original paper by Redmon et al.,3 the model actually predicted multiple boxes per grid location, but we keep things simple and just predict one box in each grid square. Most images will not have objects evenly distributed across a grid, and to account for this, the model will output a confidence score along with each box, as shown in figure 12.4. We’d like this confidence to be high when an object is detected at a location, and zero when there’s no object. Most grid locations will have no object and should report a near-zero confidence. Like many models in computer vision, the YOLO model uses a ConvNet backbone to obtain interesting high-level features for an input image, a concept we first explored in chapter 8. In their paper, the authors created their own backbone model and pretrained it with Image

### PDF p.370 (printed 346) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 
Figure 12.7 Every bounding box predicted by the YOLO model

Our model learns very low-confidence values because it has not yet learned to consistently locate objects in a scene. To further improve the model, we should try a number of things:

Train for more epochs

Use the whole COCO dataset

Data augmentation (e.g., translating and rotating input images and boxes)

Improve our class probability map for overlapping boxes

Predict multiple boxes per grid location using a bigger output grid

All of these would positively affect the model performance and get us closer to the original YOLO training recipe. However, this example is really just to get a feel for object detection training—training an accurate COCO detection model from scratch would take a large amount of compute and time. Instead, to get a sense of a better-performing detection model, let’s try using a pretrained object detection model called RetinaNet.

### PDF p.355 (printed 331) — 12 Object detection › 12.1 Single-stage vs. two-stage object detectors › 
areas that look object-like. These boxes are called region proposals. This stage isn’t very smart, so at that point we aren’t quite sure whether the proposed regions do contain objects and, if so, what objects they contain.

That’s the job of the second stage—a ConvNet that looks at each region proposal and classifies it into a number of predetermined classes, just like the models you’ve seen in chapter 9 (see figure 12.2). Region proposals that have a low score across all classes considered are discarded. We are then left with a much smaller set of boxes, each with a high class-presence score for one particular class. Finally, bounding boxes around each object are further refined to eliminate duplicates and make each bounding box as precise as possible.

Figure 12.2 An R-CNN first extracts region proposals and then classifies the proposals with a ConvNet (a CNN).

In early R-CNN versions, the first stage was a heuristic model called Selective Search that used some definition of spatial consistency to identify object-like areas. Heuristic is a term you’ll hear quite a lot in machine learning—it simply means “a bundle of hard-coded rules someone made up.” It’s usually used in opposi

### PDF p.370 (printed 346) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 12.3 Using a pretrained RetinaNet detector
RetinaNet is also a single-stage object detector and operates on the same basic principles as the YOLO model. The biggest conceptual difference between our model and

### PDF p.333 (printed 309) — 11 Image segmentation › 11.1 Computer vision tasks › 
Image classification, where the goal is to assign one or more labels to an image. It may be either single-label classification (meaning categories are mutually exclusive) or multilabel classification (tagging all categories that an image belongs to, as shown in figure 11.1). For example, when you search for a keyword on the Google Photos app, behind the scenes you’re querying a very large multilabel classification model—one with over 20,000 different classes, trained on millions of images.

Image segmentation, where the goal is to “segment” or “partition” an image into different areas, with each area usually representing a category (as shown in figure 11.1). For instance, when Zoom or Google Meet displays a custom background behind you in a video call, it’s using an image segmentation model to distinguish your face from what’s behind it, with pixel-level precision.

Object detection, where the goal is to draw rectangles (called bounding boxes) around objects of interest in an image and associate each rectangle with a class. A self-driving car could use an object detection model to monitor cars, pedestrians, and signs in view of its cameras, for instance.

Figure 11.1 The three main

### PDF p.373 (printed 349) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 
Figure 12.9 Predictions on a test image from the RetinaNet model

The RetinaNet model is able to generalize to a pointillist painting with ease, despite no training on this style of input! This is actually one of the advantages of single-stage object detectors. Paintings and photographs are very different at a pixel level but share a similar structure at a high level. Two-stage detectors like R-CNNs, in contrast, are forced to classify small patches of an input image in isolation, which is extra difficult when small patches of pixels look very different than training data. Single-stage detectors can draw on features from the entire input and are more robust to novel test-time inputs.

With that, you have reached the end of the computer vision section of this book! We have trained image classifiers, segmenters, and object detectors from scratch. We’ve developed a good intuition for how ConvNets work, the first major success of the deep learning era. We aren’t quite done with images yet; you will see them again in chapter 17 when we start generating image output.
