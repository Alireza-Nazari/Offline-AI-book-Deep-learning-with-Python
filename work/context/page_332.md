# Generation bundle for page_332  (PDF page 332, printed page 308)

Page kind: chapter_opener
Chapter: 11 Image segmentation
Section: 11.1 Computer vision tasks
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_332.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_332.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p332_e01] (chapter_number) 11
[p332_e02] (chapter_title) Image segmentation
[p332_e03] (chapter_covers) This chapter covers
[p332_e04] (bullet) The different branches of computer vision: image
[p332_e05] (bullet) classification, image segmentation, and object
[p332_e06] (bullet) detection
[p332_e07] (bullet) Building a segmentation model from scratch
[p332_e08] (bullet) Using the pretrained Segment Anything Model
[p332_e09] (body) Chapter 8 gave you a first introduction to deep learning for computer vision via a simple use case: binary image classification. But there’s more to computer vision than image classification! This chapter dives deeper into another essential computer vision application—image segmentation.
[p332_e10] (heading_section) 11.1 Computer vision tasks
[p332_e11] (body) So far, we’ve focused on image classification models: an image goes in, a label comes out. “This image likely contains a cat; this other one likely contains a dog.” But image classification is only one of several possible applications of deep learning in computer vision. In general, there are three essential computer vision tasks you need to know about:

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

## Previous page (PDF 331) text, for continuity
This task is not as impossible as it may sound. You see, while the points you’re working with may be, say, 2048-dimensional, the curve they represent has a much lower intrinsic dimensionality. That’s precisely why we refer to it as a manifold—a manifold is a lower-dimensional curve inside a higher-dimensional space. For instance, while planet Earth is a three-dimensional object, its surface is actually 2D, and so it possible to project it on a 2D plane. That’s what world maps are all about. In fact, world maps illustrate both the feasibility and the challenges of lower-dimensional projections of higher-dimensional manifolds. All projections distort reality. The widely used Mercator map projection distorts distances, in particular by making areas near the equator look smaller than they actually are, and inversely by making areas near the poles stretch towards infinity. And that’s even though the surface you’re projecting is in fact 2D—whereas deep learning latent manifolds may be much higher-dimensional than that. There’s no projection that won’t destroy at least some information.

Summary

ConvNets process images by applying a set of learned filters. Filters from earlier layers detect edges and basic textures, while filters from later layers detect increasingly abstract concepts.

You can visualize both the pattern that a filter detects and a filter’s response map across an image.

You can use the Grad-CAM technique to visualize what area(s) in an image were responsible for a classifier’s decision.

Together, these techniques make ConvNets highly interpretable.

## Next page (PDF 333) text, for continuity
Image classification, where the goal is to assign one or more labels to an image. It may be either single-label classification (meaning categories are mutually exclusive) or multilabel classification (tagging all categories that an image belongs to, as shown in figure 11.1). For example, when you search for a keyword on the Google Photos app, behind the scenes you’re querying a very large multilabel classification model—one with over 20,000 different classes, trained on millions of images.

Image segmentation, where the goal is to “segment” or “partition” an image into different areas, with each area usually representing a category (as shown in figure 11.1). For instance, when Zoom or Google Meet displays a custom background behind you in a video call, it’s using an image segmentation model to distinguish your face from what’s behind it, with pixel-level precision.

Object detection, where the goal is to draw rectangles (called bounding boxes) around objects of interest in an image and associate each rectangle with a class. A self-driving car could use an object detection model to monitor cars, pedestrians, and signs in view of its cameras, for instance.

Figure 11.1 The three main computer vision tasks: classification, segmentation, and detection

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.334 — 11 Image segmentation › 11.1 Computer vision tasks
- PDF p.352 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.342 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.280 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.335 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.354 — 12 Object detection › 12.1 Single-stage vs. two-stage object detectors
- PDF p.256 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.255 — 8 Image classification › 

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.334 (printed 310) — 11 Image segmentation › 11.1 Computer vision tasks › 
Deep learning for computer vision also encompasses a number of somewhat more niche tasks besides these three, such as image similarity scoring (estimating how visually similar two images are), keypoint detection (pinpointing attributes of interest in an image, such as facial features), pose estimation, 3D mesh estimation, depth estimation, and so on. But to start with, image classification, image segmentation, and object detection form the foundation that every machine learning engineer should be familiar with. Almost all computer vision applications boil down to one of these three.

You’ve seen image classification in action in chapter 8. Next, let’s dive into image segmentation. It’s a very useful and very versatile technique, and you can straightforwardly approach it with what you’ve already learned so far. Then, in the next chapter, you’ll learn about object detection in detail.

### PDF p.352 (printed 328) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › Summary
Image segmentation is one of the main categories of computer vision tasks. It consists of computing segmentation masks that describe the contents of an image at the pixel level.

To build your own segmentation model, use a stack of strided `Conv2D` layers to “compress” the input image into a smaller feature map, followed by a stack of corresponding `Conv2DTranspose` layers to “expand” the feature map into a segmentation mask the same size as the input image.

You can also use a pretrained segmentation model. Segment Anything, included in KerasHub, is a powerful model that supports image prompting, text prompting, point prompting, and box prompting.

### PDF p.342 (printed 318) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3 Using a pretrained segmentation model
In the image classification example from chapter 8, you saw how using a pretrained model could significantly boost your accuracy—especially when you only have a few samples to train on. Image segmentation is no different.

The Segment Anything Model ,1 or SAM for short, is a powerful pretrained segmentation model you can use for, well, almost anything. It was developed by Meta AI and released in April 2023. It was trained on 11 million images and their segmentation masks, covering over 1 billion object instances. This massive amount of training data

1 Kirillov et al., “Segment Anything,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, arXiv

(2023), https://arxiv.org/abs/2304.02643.

### PDF p.334 (printed 310) — 11 Image segmentation › 11.1 Computer vision tasks › 11.1.1 Types of image segmentation
Image segmentation with deep learning is about using a model to assign a class to each pixel in an image, thus segmenting the image into different zones (such as “background” and “foreground” or “road,” “car,” and “sidewalk”). This general category of techniques can be used to power a considerable variety of valuable applications in image and video editing, autonomous driving, robotics, medical imaging, and so on.

There are three different flavors of image segmentation that you should know about:

Semantic segmentation, where each pixel is independently classified into a semantic category, like “cat.” If there are two cats in the image, the corresponding pixels are all mapped to the same generic “cat” category (see figure 11.2).

Instance segmentation, which seeks to parse out individual object instances. In an image with two cats in it, instance segmentation would distinguish between pixels belonging to “cat 1” and pixels belonging to “cat 2” (see figure 11.2).

Panoptic segmentation, which combines semantic segmentation and instance segmentation by assigning to each pixel in an image both a semantic label (like “cat”) and an instance label (like “cat 2”). This is the most inform

### PDF p.256 (printed 232) — 8 Image classification › 8.1 Introduction to convnets › 
community had spent many years investing in methods other than neural networks, and it wasn’t quite ready to give up on them just because there was a new kid on the block. In 2013 and 2014, deep learning still faced intense skepticism from many senior computer vision researchers. It was only in 2016 that it finally became dominant. One author remembers exhorting an ex-professor, in February 2014, to pivot to deep learning. “It’s the next big thing!” he would say. “Well, maybe it’s just a fad,” the professor would reply. By 2016, his entire lab was doing deep learning. There’s no stopping an idea whose time has come.

Today, you’re constantly interacting with deep learning–based vision models—via Google Photos, Google image search, the camera on your phone, YouTube, OCR software, and many more. These models are also at the heart of cutting-edge research in autonomous driving, robotics, AI-assisted medical diagnosis, autonomous retail checkout systems, and even autonomous farming. This chapter introduces convolutional neural networks, also known as ConvNets or CNNs, the type of deep-learning model that is used by most computer vision applications. You’ll learn to apply ConvNets to im

### PDF p.255 (printed 231) — 8 Image classification ›  › This chapter covers
Understanding convolutional neural networks

(ConvNets)

Using data augmentation to mitigate overfitting

Using a pretrained ConvNet for feature extraction

Fine-tuning a pretrained ConvNet

Computer vision was the first big success story of deep learning. It led to the initial rise of deep learning between 2011 and 2015. A type of deep learning called convolutional neural networks started getting remarkably good results on image classification competitions around that time, first with Dan Ciresan winning two niche competitions (the ICDAR 2011 Chinese character recognition competition and the IJCNN 2011 German traffic signs recognition competition) and then, more notably, in fall 2012, with Hinton’s group winning the high-profile ImageNet large-scale visual recognition challenge. Many more promising results quickly started bubbling up in other computer vision tasks.

Interestingly, these early successes weren’t quite enough to make deep learning mainstream at the time—it took a few years. The computer vision research

### PDF p.280 (printed 256) — 8 Image classification › 8.3 Using a pretrained model › Using a pretrained model
A common and highly effective approach to deep learning on small image datasets is to use a pretrained model. A pretrained model is a model that was previously trained on a large dataset, typically on a large-scale image-classification task. If this original dataset is large enough and general enough, then the spatial hierarchy of features learned by the pretrained model can effectively act as a generic model of the visual world, and hence its features can prove useful for many different computer vision problems, even though these new problems may involve completely different classes than those of the original task. For instance, you might train a model on ImageNet (where classes are mostly animals and everyday objects) and then repurpose this trained model for something as remote as identifying furniture items in images. Such portability of learned features across different problems is a key advantage of deep learning compared to many older, shallow learning approaches, and it makes deep learning very effective for small-data problems.

In this case, let’s consider a large ConvNet trained on the ImageNet dataset (1.4 million labeled images and 1,000 different classes). ImageNet co

### PDF p.354 (printed 330) — 12 Object detection › 12.1 Single-stage vs. two-stage object detectors › 
Figure 12.1 Object detectors draw boxes around objects in an image and label them.

You might be thinking, if I have a segmentation mask for an object instance, I can already compute the coordinates of the smallest box that contains the mask. So couldn’t we just use image segmentation all the time? Do we need object detection models at all?

Indeed, segmentation is a strict superset of detection. It returns all the information that could be returned by a detection model—plus a lot more. This increased wealth of information has a significant computational cost: a good object detection model will typically run much faster than an image segmentation model. It also has a data labeling cost: to train a segmentation model, you need to collect pixel-precise masks, which are much more time-consuming to produce than the mere bounding boxes required by object detection models. As a result, you will always want to use an object detection model if you have no need for pixel-level information—for instance, if all you want is to count objects in an image.
