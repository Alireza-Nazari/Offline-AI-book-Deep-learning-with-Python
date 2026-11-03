# Generation bundle for page_334  (PDF page 334, printed page 310)

Page kind: content
Chapter: 11 Image segmentation
Section: 11.1 Computer vision tasks
Subsection: 11.1.1 Types of image segmentation
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_334.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_334.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p334_e03] (body) Deep learning for computer vision also encompasses a number of somewhat more niche tasks besides these three, such as image similarity scoring (estimating how visually similar two images are), keypoint detection (pinpointing attributes of interest in an image, such as facial features), pose estimation, 3D mesh estimation, depth estimation, and so on. But to start with, image classification, image segmentation, and object detection form the foundation that every machine learning engineer should be familiar with. Almost all computer vision applications boil down to one of these three.
[p334_e04] (body) You’ve seen image classification in action in chapter 8. Next, let’s dive into image segmentation. It’s a very useful and very versatile technique, and you can straightforwardly approach it with what you’ve already learned so far. Then, in the next chapter, you’ll learn about object detection in detail.
[p334_e05] (heading_subsection) 11.1.1 Types of image segmentation
[p334_e06] (body) Image segmentation with deep learning is about using a model to assign a class to each pixel in an image, thus segmenting the image into different zones (such as “background” and “foreground” or “road,” “car,” and “sidewalk”). This general category of techniques can be used to power a considerable variety of valuable applications in image and video editing, autonomous driving, robotics, medical imaging, and so on.
[p334_e07] (body) There are three different flavors of image segmentation that you should know about:
[p334_e08] (bullet) Semantic segmentation, where each pixel is independently classified into a semantic category, like “cat.” If there are two cats in the image, the corresponding pixels are all mapped to the same generic “cat” category (see figure 11.2).
[p334_e09] (bullet) Instance segmentation, which seeks to parse out individual object instances. In an image with two cats in it, instance segmentation would distinguish between pixels belonging to “cat 1” and pixels belonging to “cat 2” (see figure 11.2).
[p334_e10] (bullet) Panoptic segmentation, which combines semantic segmentation and instance segmentation by assigning to each pixel in an image both a semantic label (like “cat”) and an instance label (like “cat 2”). This is the most informative of all three segmentation types.
[p334_e11] (figure_caption) Figure 11.2 Semantic segmentation vs. instance segmentation

## Figures
- Figure 11.2: Semantic segmentation vs. instance segmentation  | labels inside the figure: (none - image)

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

## Previous page (PDF 333) text, for continuity
Image classification, where the goal is to assign one or more labels to an image. It may be either single-label classification (meaning categories are mutually exclusive) or multilabel classification (tagging all categories that an image belongs to, as shown in figure 11.1). For example, when you search for a keyword on the Google Photos app, behind the scenes you’re querying a very large multilabel classification model—one with over 20,000 different classes, trained on millions of images.

Image segmentation, where the goal is to “segment” or “partition” an image into different areas, with each area usually representing a category (as shown in figure 11.1). For instance, when Zoom or Google Meet displays a custom background behind you in a video call, it’s using an image segmentation model to distinguish your face from what’s behind it, with pixel-level precision.

Object detection, where the goal is to draw rectangles (called bounding boxes) around objects of interest in an image and associate each rectangle with a class. A self-driving car could use an object detection model to monitor cars, pedestrians, and signs in view of its cameras, for instance.

Figure 11.1 The three main computer vision tasks: classification, segmentation, and detection

## Next page (PDF 335) text, for continuity
To get more familiar with segmentation, let’s get started with training a small segmentation model from scratch on your own data.

11.2 Training a segmentation model from scratch

In this first example, we’ll focus on semantic segmentation. We’ll be looking once again at images of cats and dogs, and this time we’ll be learning to tell apart the main subject and its background.

11.2.1 Downloading a segmentation dataset

We’ll work with the Oxford-IIIT Pets dataset (https://www.robots.ox.ac.uk/~vgg/data/ pets/), which contains 7,390 pictures of various breeds of cats and dogs, together with foreground-background segmentation masks for each picture. A segmentation mask is the image segmentation equivalent of a label: it’s an image the same size as the input image, with a single color channel where each integer value corresponds to the class of the corresponding pixel in the input image. In our case, the pixels of our segmentation masks can take one of three integer values:

1 (foreground)

2 (background)

3 (contour)

Let’s start by downloading and uncompressing our dataset, using the `wget` and `tar` shell utilities:

The input pictures are stored as JPG files in the `images/` folder (such as `images/` `Abyssinian_1.jpg`), and the corresponding segmentation mask is stored as a PNG file with the same name in the `annotations/trimaps/` folder (such as `annotations/` `trimaps/Abyssinian_1.png`).

Let’s prepare the list of input file paths, as well as the list of the corresponding mask file paths:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.332 — 11 Image segmentation › 11.1 Computer vision tasks
- PDF p.352 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.354 — 12 Object detection › 12.1 Single-stage vs. two-stage object detectors
- PDF p.342 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.345 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.343 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.355 — 12 Object detection › 12.1 Single-stage vs. two-stage object detectors
- PDF p.374 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.332 (printed 308) — 11 Image segmentation › 11.1 Computer vision tasks › This chapter covers
The different branches of computer vision: image

classification, image segmentation, and object

detection

Building a segmentation model from scratch

Using the pretrained Segment Anything Model

Chapter 8 gave you a first introduction to deep learning for computer vision via a simple use case: binary image classification. But there’s more to computer vision than image classification! This chapter dives deeper into another essential computer vision application—image segmentation.

### PDF p.332 (printed 308) — 11 Image segmentation › 11.1 Computer vision tasks › 11.1 Computer vision tasks
So far, we’ve focused on image classification models: an image goes in, a label comes out. “This image likely contains a cat; this other one likely contains a dog.” But image classification is only one of several possible applications of deep learning in computer vision. In general, there are three essential computer vision tasks you need to know about:

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

### PDF p.374 (printed 350) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › Summary
Object detection identifies and locates objects within an image using bounding boxes. It’s basically a weaker version of image segmentation, but one that can be run much more efficiently.

There are two primary approaches to object detection: – Region-based Convolutional Neural Networks (R-CNNs), which are two-stage models that first propose regions of interest and then classify them with a ConvNet. – Single-stage detectors (like RetinaNet and YOLO), which perform both tasks in a single step. Single-stage detectors are generally faster and more efficient, making them suitable for real-time applications (e.g., self-driving cars).

YOLO computes two separate outputs simultaneously during training—possible bounding boxes and a class probability map: – Each candidate bounding box is paired with a confidence score, which is trained to target the Intersection over Union of the predicted box and the ground truth box. – The class probability map classifies different regions of an image as belonging to different objects.

RetinaNet builds on this idea by using a feature pyramid network (FPN), which combines features from multiple ConvNet layers to create feature maps at different scales, al

### PDF p.7 (printed vii) — brief contents ›  › brief contents
1

■ What is deep learning? 1

2

■ The mathematical building blocks of neural networks 16

3

■ Introduction to TensorFlow, PyTorch, JAX, and Keras 60

4

■ Classification and regression 104

5

■ Fundamentals of machine learning 136

6

■ The universal workflow of machine learning 171

7

■ A deep dive on Keras 190

8

■ Image classification 231

9

■ ConvNet architecture patterns 268

10

■ Interpreting what ConvNets learn 284

11

■ Image segmentation 308

12

■ Object detection 329

13

■ Timeseries forecasting 351

14

■ Text classification 381

15

■ Language models and the Transformer 421

16

■ Text generation 466

17

■ Image generation 508

18

■ Best practices for the real world 538

19

■ The future of AI 564

20

■ Conclusions 595

v

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
