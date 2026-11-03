# Generation bundle for page_342  (PDF page 342, printed page 318)

Page kind: content
Chapter: 11 Image segmentation
Section: 11.3 Using a pretrained segmentation model
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_342.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_342.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p342_e03] (code)
```
mask = model.predict(np.expand_dims(test_image, 0))[0]
def display_mask(pred):
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    plt.axis("off")
    plt.imshow(mask)
display_mask(mask)
```
[p342_e04] (figure_caption) Figure 11.6 A test image and its predicted segmentation mask
[p342_e05] (body) There are a couple of small artifacts in our predicted mask, caused by geometric shapes in the foreground and background. Nevertheless, our model appears to work nicely.
[p342_e06] (heading_section) 11.3 Using a pretrained segmentation model
[p342_e07] (body) In the image classification example from chapter 8, you saw how using a pretrained model could significantly boost your accuracy—especially when you only have a few samples to train on. Image segmentation is no different.
[p342_e08] (body) The Segment Anything Model ,1 or SAM for short, is a powerful pretrained segmentation model you can use for, well, almost anything. It was developed by Meta AI and released in April 2023. It was trained on 11 million images and their segmentation masks, covering over 1 billion object instances. This massive amount of training data
[p342_e09] (body) 1 Kirillov et al., “Segment Anything,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, arXiv
[p342_e10] (footnote) (2023), https://arxiv.org/abs/2304.02643.

## Code annotations printed next to listings (callouts, in order)
- Utility to display a
- model’s prediction

## Figures
- Figure 11.6: A test image and its predicted segmentation mask  | labels inside the figure: (none - image)

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

## Previous page (PDF 341) text, for continuity
Let’s display our training and validation loss (see figure 11.5):

Figure 11.5 Displaying training and validation loss curves

You can see that we start overfitting midway, around epoch 25. Let’s reload our best-performing model according to validation loss and demonstrate how to use it to predict a segmentation mask (see figure 11.6):

## Next page (PDF 343) text, for continuity
provides the model with built-in knowledge of virtually any object that appears in natural images.

The main innovation of SAM is that it’s not limited to a predefined set of object classes. You can use it for segmenting new objects simply by providing an example of what you’re looking for. You don’t even need to fine-tune the model first. Let’s see how that works.

11.3.1 Downloading the Segment Anything Model

First, let’s instantiate SAM and download its weights. Once again, we can use the KerasHub package to use this pretrained model without needing to implement it ourselves from scratch.

Remember the `ImageClassifier` task we used in the previous chapter? We can use another KerasHub task `ImageSegmenter` for wrapping pretrained image segmentation models into a high-level model with standard inputs and outputs. Here, we’ll use the `sam_huge_sa1b` pretrained model, where `sam` stands for the model, `huge` refers to the number of parameters in the model, and `sa1b` stands for the SA-1B dataset released along with the model, with 1 billion annotated masks. Let’s download it now:

One thing we can note off the bat is that our model is, indeed, huge:

At 641 million parameters, SAM is the largest model we have used so far in this book. The trend of pretrained models getting larger and larger and using more and more data will be discussed in more detail in chapter 16.

11.3.2 How Segment Anything works

Before we try running some segmentation with the model, let’s talk a little more about how SAM works. Much of the capability of the model comes from the scale of the pretraining dataset. Meta developed the SA-1B dataset along with the model, where the partially trained model was used to assist with the data labeling process. That is, the dataset and model were developed t …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.352 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.348 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.345 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.335 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.336 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.350 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.344 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.349 — 11 Image segmentation › 11.3 Using a pretrained segmentation model

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.352 (printed 328) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › Summary
Image segmentation is one of the main categories of computer vision tasks. It consists of computing segmentation masks that describe the contents of an image at the pixel level.

To build your own segmentation model, use a stack of strided `Conv2D` layers to “compress” the input image into a smaller feature map, followed by a stack of corresponding `Conv2DTranspose` layers to “expand” the feature map into a segmentation mask the same size as the input image.

You can also use a pretrained segmentation model. Segment Anything, included in KerasHub, is a powerful model that supports image prompting, text prompting, point prompting, and box prompting.

### PDF p.332 (printed 308) — 11 Image segmentation › 11.1 Computer vision tasks › This chapter covers
The different branches of computer vision: image

classification, image segmentation, and object

detection

Building a segmentation model from scratch

Using the pretrained Segment Anything Model

Chapter 8 gave you a first introduction to deep learning for computer vision via a simple use case: binary image classification. But there’s more to computer vision than image classification! This chapter dives deeper into another essential computer vision application—image segmentation.

### PDF p.354 (printed 330) — 12 Object detection › 12.1 Single-stage vs. two-stage object detectors › 
Figure 12.1 Object detectors draw boxes around objects in an image and label them.

You might be thinking, if I have a segmentation mask for an object instance, I can already compute the coordinates of the smallest box that contains the mask. So couldn’t we just use image segmentation all the time? Do we need object detection models at all?

Indeed, segmentation is a strict superset of detection. It returns all the information that could be returned by a detection model—plus a lot more. This increased wealth of information has a significant computational cost: a good object detection model will typically run much faster than an image segmentation model. It also has a data labeling cost: to train a segmentation model, you need to collect pixel-precise masks, which are much more time-consuming to produce than the mere bounding boxes required by object detection models. As a result, you will always want to use an object detection model if you have no need for pixel-level information—for instance, if all you want is to count objects in an image.

### PDF p.344 (printed 320) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
dataset of images, which was used to train an initial model. This model was used to help drive a semiautomated stage of data collection, where images were first segmented by SAM and improved by human correction and further annotation.

Figure 11.7 An example image from the SA-1B dataset

The model is trained on `(image, prompt, mask)` triples. `image` and `prompt` are the model inputs. The image can be any input image, and the prompt can take a couple of forms:

A point inside the object to mask

A box around the object to mask

Given the `image` and `prompt` input, the model is expected to produce an accurate predicted mask for the object indicated by the prompt, which is compared with a ground truth `mask` label.

The model consists of a few separate components. An image encoder, similar to the Xception model we used in previous chapters, will take an input image and output a much smaller image embedding. This is something we already know how to build. Next, we add a prompt encoder, which is responsible for mapping prompts in any of the previously mentioned forms to an embedded vector, and a mask decoder, which takes in both the image embedding and prompt embedding and outputs a 

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

### PDF p.335 (printed 311) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 11.2.1 Downloading a segmentation dataset
We’ll work with the Oxford-IIIT Pets dataset (https://www.robots.ox.ac.uk/~vgg/data/ pets/), which contains 7,390 pictures of various breeds of cats and dogs, together with foreground-background segmentation masks for each picture. A segmentation mask is the image segmentation equivalent of a label: it’s an image the same size as the input image, with a single color channel where each integer value corresponds to the class of the corresponding pixel in the input image. In our case, the pixels of our segmentation masks can take one of three integer values:

1 (foreground)

2 (background)

3 (contour)

Let’s start by downloading and uncompressing our dataset, using the `wget` and `tar` shell utilities:

### PDF p.280 (printed 256) — 8 Image classification › 8.3 Using a pretrained model › Using a pretrained model
A common and highly effective approach to deep learning on small image datasets is to use a pretrained model. A pretrained model is a model that was previously trained on a large dataset, typically on a large-scale image-classification task. If this original dataset is large enough and general enough, then the spatial hierarchy of features learned by the pretrained model can effectively act as a generic model of the visual world, and hence its features can prove useful for many different computer vision problems, even though these new problems may involve completely different classes than those of the original task. For instance, you might train a model on ImageNet (where classes are mostly animals and everyday objects) and then repurpose this trained model for something as remote as identifying furniture items in images. Such portability of learned features across different problems is a key advantage of deep learning compared to many older, shallow learning approaches, and it makes deep learning very effective for small-data problems.

In this case, let’s consider a large ConvNet trained on the ImageNet dataset (1.4 million labeled images and 1,000 different classes). ImageNet co

### PDF p.352 (printed 328) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
Figure 11.15 Segmented mango

SAM can be a powerful tool to quickly create large datasets of images annotated with segmentation masks.
