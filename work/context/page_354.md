# Generation bundle for page_354  (PDF page 354, printed page 330)

Page kind: content
Chapter: 12 Object detection
Section: 12.1 Single-stage vs. two-stage object detectors
Subsection: 12.1.1 Two-stage R-CNN detectors
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_354.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_354.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p354_e03] (figure_caption) Figure 12.1 Object detectors draw boxes around objects in an image and label them.
[p354_e04] (body) You might be thinking, if I have a segmentation mask for an object instance, I can already compute the coordinates of the smallest box that contains the mask. So couldn’t we just use image segmentation all the time? Do we need object detection models at all?
[p354_e05] (body) Indeed, segmentation is a strict superset of detection. It returns all the information that could be returned by a detection model—plus a lot more. This increased wealth of information has a significant computational cost: a good object detection model will typically run much faster than an image segmentation model. It also has a data labeling cost: to train a segmentation model, you need to collect pixel-precise masks, which are much more time-consuming to produce than the mere bounding boxes required by object detection models. As a result, you will always want to use an object detection model if you have no need for pixel-level information—for instance, if all you want is to count objects in an image.
[p354_e06] (heading_section) 12.1 Single-stage vs. two-stage object detectors
[p354_e07] (body) There are two broad categories of object detection architectures:
[p354_e08] (bullet) Two-stage detectors, which first extract region proposals, known as Region-based Convolutional Neural Networks (R-CNN) models
[p354_e09] (bullet) Single-stage detectors, such as RetinaNet or the You Only Look Once family of models
[p354_e10] (body) Here’s how they work.
[p354_e11] (heading_subsection) 12.1.1 Two-stage R-CNN detectors
[p354_e12] (body) A region-based ConvNet, or R-CNN model, is a two-stage model. The first stage takes an image and produces a few thousand partially overlapping bounding boxes around

## Figures
- Figure 12.1: Object detectors draw boxes around objects in an image and label them.  | labels inside the figure: (none - image)

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

## Previous page (PDF 353) text, for continuity
Object detection

This chapter covers

Understanding the object detection problem

Two-stage and single-stage object detectors

Training a simple single-stage detector from

scratch

Using a pretrained object detector

Object detection is all about drawing boxes (called bounding boxes) around objects of interest in a picture (see figure 12.1). This enables you to know not just which objects are in a picture, but also where they are. Some of its most common applications are

Counting—Find out how many instances of an object are in an image.

Tracking—Track how objects move in a scene over time by performing object detection on every frame of a movie.

Cropping—Identify the area of an image that contains an object of interest to crop it and send a higher-resolution version of the image patch to a classifier or an Optical Character Recognition (OCR) model.

## Next page (PDF 355) text, for continuity
areas that look object-like. These boxes are called region proposals. This stage isn’t very smart, so at that point we aren’t quite sure whether the proposed regions do contain objects and, if so, what objects they contain.

That’s the job of the second stage—a ConvNet that looks at each region proposal and classifies it into a number of predetermined classes, just like the models you’ve seen in chapter 9 (see figure 12.2). Region proposals that have a low score across all classes considered are discarded. We are then left with a much smaller set of boxes, each with a high class-presence score for one particular class. Finally, bounding boxes around each object are further refined to eliminate duplicates and make each bounding box as precise as possible.

Figure 12.2 An R-CNN first extracts region proposals and then classifies the proposals with a ConvNet (a CNN).

In early R-CNN versions, the first stage was a heuristic model called Selective Search that used some definition of spatial consistency to identify object-like areas. Heuristic is a term you’ll hear quite a lot in machine learning—it simply means “a bundle of hard-coded rules someone made up.” It’s usually used in opposition to learned models (where the rules are automatically derived) or theory-derived models. In later versions of R-CNN, such as Faster-R-CNN, the box generation stage became a deep learning model, called a Region Proposal Network.

The two-stage approach of R-CNN works very well in practice, but it’s quite computationally expensive, most notably because it requires you to classify thousands of patches—for every single image you process. That makes it unsuitable for most real-time applications and for embedded systems. My take is that, in practical applications, you generally don’t ever need a …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.374 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.373 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.370 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.356 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.360 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.13 — contents › 
- PDF p.334 — 11 Image segmentation › 11.1 Computer vision tasks
- PDF p.332 — 11 Image segmentation › 11.1 Computer vision tasks

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.374 (printed 350) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › Summary
Object detection identifies and locates objects within an image using bounding boxes. It’s basically a weaker version of image segmentation, but one that can be run much more efficiently.

There are two primary approaches to object detection: – Region-based Convolutional Neural Networks (R-CNNs), which are two-stage models that first propose regions of interest and then classify them with a ConvNet. – Single-stage detectors (like RetinaNet and YOLO), which perform both tasks in a single step. Single-stage detectors are generally faster and more efficient, making them suitable for real-time applications (e.g., self-driving cars).

YOLO computes two separate outputs simultaneously during training—possible bounding boxes and a class probability map: – Each candidate bounding box is paired with a confidence score, which is trained to target the Intersection over Union of the predicted box and the ground truth box. – The class probability map classifies different regions of an image as belonging to different objects.

RetinaNet builds on this idea by using a feature pyramid network (FPN), which combines features from multiple ConvNet layers to create feature maps at different scales, al

### PDF p.356 (printed 332) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.1.2 Single-stage detectors
Around 2015, researchers and practitioners began experimenting with using a single deep learning model to jointly predict bounding box coordinates together with their labels, an architecture called a single-stage detector. The main families of single-stage detectors are RetinaNet, Single Shot MultiBox Detectors (SSD), and the You Only Look Once family, abbreviated as YOLO. Yes, like the meme. That’s on purpose.

Single-stage detectors, especially recent YOLO iterations, boast significantly faster speeds and greater efficiency than their two-stage counterparts, albeit with a minor potential tradeoff in accuracy. Nowadays, YOLO is arguably the most popular object detection model out there, especially when it comes to real-time applications. There’s usually a new version of it that comes out every year—interestingly, each new version tends to be developed by a separate organization. In the next section, we will build a simplified YOLO model from scratch.

### PDF p.373 (printed 349) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 
Figure 12.9 Predictions on a test image from the RetinaNet model

The RetinaNet model is able to generalize to a pointillist painting with ease, despite no training on this style of input! This is actually one of the advantages of single-stage object detectors. Paintings and photographs are very different at a pixel level but share a similar structure at a high level. Two-stage detectors like R-CNNs, in contrast, are forced to classify small patches of an input image in isolation, which is extra difficult when small patches of pixels look very different than training data. Single-stage detectors can draw on features from the entire input and are more robust to novel test-time inputs.

With that, you have reached the end of the computer vision section of this book! We have trained image classifiers, segmenters, and object detectors from scratch. We’ve developed a good intuition for how ConvNets work, the first major success of the deep learning era. We aren’t quite done with images yet; you will see them again in chapter 17 when we start generating image output.

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

### PDF p.360 (printed 336) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2.2 Creating a YOLO model
As mentioned previously, the YOLO model is a single stage detector. Rather than first attempting to identify all candidate objects in a scene and then classifying the object regions, YOLO will propose bounding boxes and object labels in one go.

Our model will divide an image up into a grid and predict two separate outputs at each grid location—a bounding box and a class label. In the original paper by Redmon et al.,3 the model actually predicted multiple boxes per grid location, but we keep things simple and just predict one box in each grid square. Most images will not have objects evenly distributed across a grid, and to account for this, the model will output a confidence score along with each box, as shown in figure 12.4. We’d like this confidence to be high when an object is detected at a location, and zero when there’s no object. Most grid locations will have no object and should report a near-zero confidence. Like many models in computer vision, the YOLO model uses a ConvNet backbone to obtain interesting high-level features for an input image, a concept we first explored in chapter 8. In their paper, the authors created their own backbone model and pretrained it with Image

### PDF p.370 (printed 346) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 12.3 Using a pretrained RetinaNet detector
RetinaNet is also a single-stage object detector and operates on the same basic principles as the YOLO model. The biggest conceptual difference between our model and

### PDF p.333 (printed 309) — 11 Image segmentation › 11.1 Computer vision tasks › 
Image classification, where the goal is to assign one or more labels to an image. It may be either single-label classification (meaning categories are mutually exclusive) or multilabel classification (tagging all categories that an image belongs to, as shown in figure 11.1). For example, when you search for a keyword on the Google Photos app, behind the scenes you’re querying a very large multilabel classification model—one with over 20,000 different classes, trained on millions of images.

Image segmentation, where the goal is to “segment” or “partition” an image into different areas, with each area usually representing a category (as shown in figure 11.1). For instance, when Zoom or Google Meet displays a custom background behind you in a video call, it’s using an image segmentation model to distinguish your face from what’s behind it, with pixel-level precision.

Object detection, where the goal is to draw rectangles (called bounding boxes) around objects of interest in an image and associate each rectangle with a class. A self-driving car could use an object detection model to monitor cars, pedestrians, and signs in view of its cameras, for instance.

Figure 11.1 The three main

### PDF p.332 (printed 308) — 11 Image segmentation › 11.1 Computer vision tasks › This chapter covers
The different branches of computer vision: image

classification, image segmentation, and object

detection

Building a segmentation model from scratch

Using the pretrained Segment Anything Model

Chapter 8 gave you a first introduction to deep learning for computer vision via a simple use case: binary image classification. But there’s more to computer vision than image classification! This chapter dives deeper into another essential computer vision application—image segmentation.
