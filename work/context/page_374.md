# Generation bundle for page_374  (PDF page 374, printed page 350)

Page kind: content
Chapter: 12 Object detection
Section: 12.3 Using a pretrained RetinaNet detector
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_374.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_374.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p374_e03] (heading_section) Summary
[p374_e04] (bullet) Object detection identifies and locates objects within an image using bounding boxes. It’s basically a weaker version of image segmentation, but one that can be run much more efficiently.
[p374_e05] (bullet) There are two primary approaches to object detection: – Region-based Convolutional Neural Networks (R-CNNs), which are two-stage models that first propose regions of interest and then classify them with a ConvNet. – Single-stage detectors (like RetinaNet and YOLO), which perform both tasks in a single step. Single-stage detectors are generally faster and more efficient, making them suitable for real-time applications (e.g., self-driving cars).
[p374_e06] (bullet) YOLO computes two separate outputs simultaneously during training—possible bounding boxes and a class probability map: – Each candidate bounding box is paired with a confidence score, which is trained to target the Intersection over Union of the predicted box and the ground truth box. – The class probability map classifies different regions of an image as belonging to different objects.
[p374_e07] (bullet) RetinaNet builds on this idea by using a feature pyramid network (FPN), which combines features from multiple ConvNet layers to create feature maps at different scales, allowing it to more accurately detect objects of different sizes.

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

## Previous page (PDF 373) text, for continuity
Figure 12.9 Predictions on a test image from the RetinaNet model

The RetinaNet model is able to generalize to a pointillist painting with ease, despite no training on this style of input! This is actually one of the advantages of single-stage object detectors. Paintings and photographs are very different at a pixel level but share a similar structure at a high level. Two-stage detectors like R-CNNs, in contrast, are forced to classify small patches of an input image in isolation, which is extra difficult when small patches of pixels look very different than training data. Single-stage detectors can draw on features from the entire input and are more robust to novel test-time inputs.

With that, you have reached the end of the computer vision section of this book! We have trained image classifiers, segmenters, and object detectors from scratch. We’ve developed a good intuition for how ConvNets work, the first major success of the deep learning era. We aren’t quite done with images yet; you will see them again in chapter 17 when we start generating image output.

## Next page (PDF 375) text, for continuity
Timeseries forecasting

This chapter covers

An overview of machine learning for timeseries

Understanding recurrent neural networks (RNNs)

Applying RNNs to a temperature forecasting

example

This chapter tackles timeseries, where temporal order is everything. We’ll focus on the most common and valuable timeseries task: forecasting. Using the recent past to predict the near future is a powerful capability, whether you’re trying to anticipate energy demand, manage inventory, or simply forecast the weather.

13.1 Different kinds of timeseries tasks

A timeseries can be any data obtained via measurements at regular intervals, like the daily price of a stock, the hourly electricity consumption of a city, or the weekly sales of a store. Timeseries are everywhere, whether we’re looking at natural phenomena (like seismic activity, the evolution of fish populations in a river, or the weather at a location) or human activity patterns (like visitors to a website, a country’s GDP, or credit card transactions). Unlike the types of data you’ve encountered so far,

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.354 — 12 Object detection › 12.1 Single-stage vs. two-stage object detectors
- PDF p.353 — 12 Object detection › 
- PDF p.355 — 12 Object detection › 12.1 Single-stage vs. two-stage object detectors
- PDF p.370 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.360 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.371 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.356 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.361 — 12 Object detection › 12.2 Training a YOLO model from scratch

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.354 (printed 330) — 12 Object detection › 12.1 Single-stage vs. two-stage object detectors › 12.1 Single-stage vs. two-stage object detectors
There are two broad categories of object detection architectures:

Two-stage detectors, which first extract region proposals, known as Region-based Convolutional Neural Networks (R-CNN) models

Single-stage detectors, such as RetinaNet or the You Only Look Once family of models

Here’s how they work.

### PDF p.360 (printed 336) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2.2 Creating a YOLO model
As mentioned previously, the YOLO model is a single stage detector. Rather than first attempting to identify all candidate objects in a scene and then classifying the object regions, YOLO will propose bounding boxes and object labels in one go.

Our model will divide an image up into a grid and predict two separate outputs at each grid location—a bounding box and a class label. In the original paper by Redmon et al.,3 the model actually predicted multiple boxes per grid location, but we keep things simple and just predict one box in each grid square. Most images will not have objects evenly distributed across a grid, and to account for this, the model will output a confidence score along with each box, as shown in figure 12.4. We’d like this confidence to be high when an object is detected at a location, and zero when there’s no object. Most grid locations will have no object and should report a near-zero confidence. Like many models in computer vision, the YOLO model uses a ConvNet backbone to obtain interesting high-level features for an input image, a concept we first explored in chapter 8. In their paper, the authors created their own backbone model and pretrained it with Image

### PDF p.371 (printed 347) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 
RetinaNet is that RetinaNet uses its underlying ConvNet differently to better handle both small and large objects simultaneously.

In our YOLO model, we simply took the final outputs of our ConvNet and used them to build our object detector. These output features map to large areas on our input image—as a result, they are not very effective at finding small objects in the scene. One option to solve this scale issue would be to directly use the output of earlier layers in our ConvNet. This would extract high-resolution features that map to small localized areas of our input image. However, the output of these early layers is not very semantically interesting. It might map to different types of simple features like edges and curves, but only later in the ConvNet layers do we start building latent representations for entire objects. The solution used by RetinaNet is called a feature pyramid network. The final features from the ConvNet base model are upsampled with progressive `Conv2DTranspose` layers, just like we saw in the previous chapter. But critically, we also include lateral connections where we sum these upsampled feature maps with the feature maps of the same size from the or

### PDF p.356 (printed 332) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.1.2 Single-stage detectors
Around 2015, researchers and practitioners began experimenting with using a single deep learning model to jointly predict bounding box coordinates together with their labels, an architecture called a single-stage detector. The main families of single-stage detectors are RetinaNet, Single Shot MultiBox Detectors (SSD), and the You Only Look Once family, abbreviated as YOLO. Yes, like the meme. That’s on purpose.

Single-stage detectors, especially recent YOLO iterations, boast significantly faster speeds and greater efficiency than their two-stage counterparts, albeit with a minor potential tradeoff in accuracy. Nowadays, YOLO is arguably the most popular object detection model out there, especially when it comes to real-time applications. There’s usually a new version of it that comes out every year—interestingly, each new version tends to be developed by a separate organization. In the next section, we will build a simplified YOLO model from scratch.

### PDF p.353 (printed 329) — 12 Object detection ›  › This chapter covers
Understanding the object detection problem

Two-stage and single-stage object detectors

Training a simple single-stage detector from

scratch

Using a pretrained object detector

Object detection is all about drawing boxes (called bounding boxes) around objects of interest in a picture (see figure 12.1). This enables you to know not just which objects are in a picture, but also where they are. Some of its most common applications are

Counting—Find out how many instances of an object are in an image.

Tracking—Track how objects move in a scene over time by performing object detection on every frame of a movie.

Cropping—Identify the area of an image that contains an object of interest to crop it and send a higher-resolution version of the image patch to a classifier or an Optical Character Recognition (OCR) model.

### PDF p.370 (printed 346) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 12.3 Using a pretrained RetinaNet detector
RetinaNet is also a single-stage object detector and operates on the same basic principles as the YOLO model. The biggest conceptual difference between our model and

### PDF p.354 (printed 330) — 12 Object detection › 12.1 Single-stage vs. two-stage object detectors › 12.1.1 Two-stage R-CNN detectors
A region-based ConvNet, or R-CNN model, is a two-stage model. The first stage takes an image and produces a few thousand partially overlapping bounding boxes around

### PDF p.370 (printed 346) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 
Figure 12.7 Every bounding box predicted by the YOLO model

Our model learns very low-confidence values because it has not yet learned to consistently locate objects in a scene. To further improve the model, we should try a number of things:

Train for more epochs

Use the whole COCO dataset

Data augmentation (e.g., translating and rotating input images and boxes)

Improve our class probability map for overlapping boxes

Predict multiple boxes per grid location using a bigger output grid

All of these would positively affect the model performance and get us closer to the original YOLO training recipe. However, this example is really just to get a feel for object detection training—training an accurate COCO detection model from scratch would take a large amount of compute and time. Instead, to get a sense of a better-performing detection model, let’s try using a pretrained object detection model called RetinaNet.
