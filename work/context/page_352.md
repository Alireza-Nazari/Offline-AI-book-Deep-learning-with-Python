# Generation bundle for page_352  (PDF page 352, printed page 328)

Page kind: content
Chapter: 11 Image segmentation
Section: 11.3 Using a pretrained segmentation model
Subsection: 11.3.5 Prompting the model with a target box
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_352.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_352.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p352_e03] (code)
```
        "images": ops.expand_dims(image, axis=0),
        "boxes": ops.expand_dims(input_box, axis=(0, 1)),
    }
)
mask = get_mask(outputs, 0)
plt.figure(figsize=(10, 10))
show_image(image, plt.gca())
show_mask(mask, plt.gca())
show_box(input_box, plt.gca())
plt.show()
```
[p352_e04] (figure_caption) Figure 11.15 Segmented mango
[p352_e05] (body) SAM can be a powerful tool to quickly create large datasets of images annotated with segmentation masks.
[p352_e06] (heading_section) Summary
[p352_e07] (bullet) Image segmentation is one of the main categories of computer vision tasks. It consists of computing segmentation masks that describe the contents of an image at the pixel level.
[p352_e08] (bullet) To build your own segmentation model, use a stack of strided `Conv2D` layers to “compress” the input image into a smaller feature map, followed by a stack of corresponding `Conv2DTranspose` layers to “expand” the feature map into a segmentation mask the same size as the input image.
[p352_e09] (bullet) You can also use a pretrained segmentation model. Segment Anything, included in KerasHub, is a powerful model that supports image prompting, text prompting, point prompting, and box prompting.

## Figures
- Figure 11.15: Segmented mango  | labels inside the figure: 0, 200, 400, 600, 800, 1000, 0, 200, 400, 600, 800, 1000

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

## Previous page (PDF 351) text, for continuity
As you can see here, an alternative segmentation found by the model includes both bananas.

11.3.5 Prompting the model with a target box

Besides providing one or more target points, you can also provide boxes approximating the location of the object to segment. These boxes should be passed via the coordinates of their top-left and bottom-right corners. Here’s a box around the mango (see figure 11.14):

Figure 11.14 Box prompt around the mango

Let’s prompt SAM with it (see figure 11.15):

## Next page (PDF 353) text, for continuity
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

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.349 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.342 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.350 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.335 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.347 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.348 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.343 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.345 — 11 Image segmentation › 11.3 Using a pretrained segmentation model

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.342 (printed 318) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3 Using a pretrained segmentation model
In the image classification example from chapter 8, you saw how using a pretrained model could significantly boost your accuracy—especially when you only have a few samples to train on. Image segmentation is no different.

The Segment Anything Model ,1 or SAM for short, is a powerful pretrained segmentation model you can use for, well, almost anything. It was developed by Meta AI and released in April 2023. It was trained on 11 million images and their segmentation masks, covering over 1 billion object instances. This massive amount of training data

1 Kirillov et al., “Segment Anything,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, arXiv

(2023), https://arxiv.org/abs/2304.02643.

### PDF p.343 (printed 319) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3.2 How Segment Anything works
Before we try running some segmentation with the model, let’s talk a little more about how SAM works. Much of the capability of the model comes from the scale of the pretraining dataset. Meta developed the SA-1B dataset along with the model, where the partially trained model was used to assist with the data labeling process. That is, the dataset and model were developed together in a feedback loop of sorts.

The goal with the SA-1B dataset is to create fully segmented images, where every object in an image is given a unique segmentation mask. See figure 11.7 as an example. Each image in the dataset has ~100 masks on average, and some images have over 500 individually masked objects. This was done through a pipeline of increasingly automated data collection. At first, human experts manually segmented a small example

### PDF p.332 (printed 308) — 11 Image segmentation › 11.1 Computer vision tasks › This chapter covers
The different branches of computer vision: image

classification, image segmentation, and object

detection

Building a segmentation model from scratch

Using the pretrained Segment Anything Model

Chapter 8 gave you a first introduction to deep learning for computer vision via a simple use case: binary image classification. But there’s more to computer vision than image classification! This chapter dives deeper into another essential computer vision application—image segmentation.

### PDF p.343 (printed 319) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3.1 Downloading the Segment Anything Model
First, let’s instantiate SAM and download its weights. Once again, we can use the KerasHub package to use this pretrained model without needing to implement it ourselves from scratch.

Remember the `ImageClassifier` task we used in the previous chapter? We can use another KerasHub task `ImageSegmenter` for wrapping pretrained image segmentation models into a high-level model with standard inputs and outputs. Here, we’ll use the `sam_huge_sa1b` pretrained model, where `sam` stands for the model, `huge` refers to the number of parameters in the model, and `sa1b` stands for the SA-1B dataset released along with the model, with 1 billion annotated masks. Let’s download it now:

### PDF p.335 (printed 311) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 11.2.1 Downloading a segmentation dataset
We’ll work with the Oxford-IIIT Pets dataset (https://www.robots.ox.ac.uk/~vgg/data/ pets/), which contains 7,390 pictures of various breeds of cats and dogs, together with foreground-background segmentation masks for each picture. A segmentation mask is the image segmentation equivalent of a label: it’s an image the same size as the input image, with a single color channel where each integer value corresponds to the class of the corresponding pixel in the input image. In our case, the pixels of our segmentation masks can take one of three integer values:

1 (foreground)

2 (background)

3 (contour)

Let’s start by downloading and uncompressing our dataset, using the `wget` and `tar` shell utilities:

### PDF p.354 (printed 330) — 12 Object detection › 12.1 Single-stage vs. two-stage object detectors › 
Figure 12.1 Object detectors draw boxes around objects in an image and label them.

You might be thinking, if I have a segmentation mask for an object instance, I can already compute the coordinates of the smallest box that contains the mask. So couldn’t we just use image segmentation all the time? Do we need object detection models at all?

Indeed, segmentation is a strict superset of detection. It returns all the information that could be returned by a detection model—plus a lot more. This increased wealth of information has a significant computational cost: a good object detection model will typically run much faster than an image segmentation model. It also has a data labeling cost: to train a segmentation model, you need to collect pixel-precise masks, which are much more time-consuming to produce than the mere bounding boxes required by object detection models. As a result, you will always want to use an object detection model if you have no need for pixel-level information—for instance, if all you want is to count objects in an image.

### PDF p.334 (printed 310) — 11 Image segmentation › 11.1 Computer vision tasks › 11.1.1 Types of image segmentation
Image segmentation with deep learning is about using a model to assign a class to each pixel in an image, thus segmenting the image into different zones (such as “background” and “foreground” or “road,” “car,” and “sidewalk”). This general category of techniques can be used to power a considerable variety of valuable applications in image and video editing, autonomous driving, robotics, medical imaging, and so on.

There are three different flavors of image segmentation that you should know about:

Semantic segmentation, where each pixel is independently classified into a semantic category, like “cat.” If there are two cats in the image, the corresponding pixels are all mapped to the same generic “cat” category (see figure 11.2).

Instance segmentation, which seeks to parse out individual object instances. In an image with two cats in it, instance segmentation would distinguish between pixels belonging to “cat 1” and pixels belonging to “cat 2” (see figure 11.2).

Panoptic segmentation, which combines semantic segmentation and instance segmentation by assigning to each pixel in an image both a semantic label (like “cat”) and an instance label (like “cat 2”). This is the most inform

### PDF p.345 (printed 321) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
Figure 11.8 The Segment Anything high-level architecture overview

All of these subcomponents are trained simultaneously by forming batches of new `(image,` `prompt,` `mask)` triples to train on from the SA-1B image and mask data. The process here is actually quite simple. For a given input image, choose a random mask in the input. Next, randomly choose whether to create a box prompt or a point prompt. To create a point prompt, choose a random pixel inside the mask label. To create a box prompt, draw a box around all points inside the mask label. We can repeat this process indefinitely, sampling a number of `(image,` `prompt,` `mask)` triples from each image input.
