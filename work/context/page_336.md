# Generation bundle for page_336  (PDF page 336, printed page 312)

Page kind: content
Chapter: 11 Image segmentation
Section: 11.2 Training a segmentation model from scratch
Subsection: 11.2.1 Downloading a segmentation dataset
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_336.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_336.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p336_e03] (body) Now, what does one of these inputs and its mask look like? Let’s take a quick look (see figure 11.3).
[p336_e04] (code)
```
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array, array_to_img
plt.axis("off")
plt.imshow(load_img(input_img_paths[9]))
```
[p336_e05] (figure_caption) Figure 11.3 An example image
[p336_e06] (body) Let’s look at its target mask as well (see figure 11.4):
[p336_e07] (code)
```
def display_target(target_array):
    normalized_array = (target_array.astype("uint8") - 1) * 127
    plt.axis("off")
    plt.imshow(normalized_array[:, :, 0])
img = img_to_array(load_img(target_paths[9], color_mode="grayscale"))
display_target(img)
```

## Code annotations printed next to listings (callouts, in order)
- Displays input
- image number 9
- The original labels are 1, 2, and 3. We subtract 1 so that the
- labels range from 0 to 2, and then we multiply by 127 so that
- the labels become 0 (black), 127 (gray), 254 (near-white).
- We use color_mode='grayscale' so that the image
- we load is treated as having a single color channel.

## Figures
- Figure 11.3: An example image  | labels inside the figure: (none - image)

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

## Previous page (PDF 335) text, for continuity
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

## Next page (PDF 337) text, for continuity
Figure 11.4 The corresponding target mask

Next, let’s load our inputs and targets into two NumPy arrays. Since the dataset is very small, we can load everything into memory:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.345 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.341 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.348 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.342 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.545 — 17 Image generation › 17.2 Diffusion models
- PDF p.347 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.350 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.311 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.344 (printed 320) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
dataset of images, which was used to train an initial model. This model was used to help drive a semiautomated stage of data collection, where images were first segmented by SAM and improved by human correction and further annotation.

Figure 11.7 An example image from the SA-1B dataset

The model is trained on `(image, prompt, mask)` triples. `image` and `prompt` are the model inputs. The image can be any input image, and the prompt can take a couple of forms:

A point inside the object to mask

A box around the object to mask

Given the `image` and `prompt` input, the model is expected to produce an accurate predicted mask for the object indicated by the prompt, which is compared with a ground truth `mask` label.

The model consists of a few separate components. An image encoder, similar to the Xception model we used in previous chapters, will take an input image and output a much smaller image embedding. This is something we already know how to build. Next, we add a prompt encoder, which is responsible for mapping prompts in any of the previously mentioned forms to an embedded vector, and a mask decoder, which takes in both the image embedding and prompt embedding and outputs a 

### PDF p.341 (printed 317) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 
Figure 11.5 Displaying training and validation loss curves

You can see that we start overfitting midway, around epoch 25. Let’s reload our best-performing model according to validation loss and demonstrate how to use it to predict a segmentation mask (see figure 11.6):

### PDF p.342 (printed 318) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
Figure 11.6 A test image and its predicted segmentation mask

There are a couple of small artifacts in our predicted mask, caused by geometric shapes in the foreground and background. Nevertheless, our model appears to work nicely.

### PDF p.343 (printed 319) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3.2 How Segment Anything works
Before we try running some segmentation with the model, let’s talk a little more about how SAM works. Much of the capability of the model comes from the scale of the pretraining dataset. Meta developed the SA-1B dataset along with the model, where the partially trained model was used to assist with the data labeling process. That is, the dataset and model were developed together in a feedback loop of sorts.

The goal with the SA-1B dataset is to create fully segmented images, where every object in an image is given a unique segmentation mask. See figure 11.7 as an example. Each image in the dataset has ~100 masks on average, and some images have over 500 individually masked objects. This was done through a pipeline of increasingly automated data collection. At first, human experts manually segmented a small example

### PDF p.352 (printed 328) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › Summary
Image segmentation is one of the main categories of computer vision tasks. It consists of computing segmentation masks that describe the contents of an image at the pixel level.

To build your own segmentation model, use a stack of strided `Conv2D` layers to “compress” the input image into a smaller feature map, followed by a stack of corresponding `Conv2DTranspose` layers to “expand” the feature map into a segmentation mask the same size as the input image.

You can also use a pretrained segmentation model. Segment Anything, included in KerasHub, is a powerful model that supports image prompting, text prompting, point prompting, and box prompting.

### PDF p.354 (printed 330) — 12 Object detection › 12.1 Single-stage vs. two-stage object detectors › 
Figure 12.1 Object detectors draw boxes around objects in an image and label them.

You might be thinking, if I have a segmentation mask for an object instance, I can already compute the coordinates of the smallest box that contains the mask. So couldn’t we just use image segmentation all the time? Do we need object detection models at all?

Indeed, segmentation is a strict superset of detection. It returns all the information that could be returned by a detection model—plus a lot more. This increased wealth of information has a significant computational cost: a good object detection model will typically run much faster than an image segmentation model. It also has a data labeling cost: to train a segmentation model, you need to collect pixel-precise masks, which are much more time-consuming to produce than the mere bounding boxes required by object detection models. As a result, you will always want to use an object detection model if you have no need for pixel-level information—for instance, if all you want is to count objects in an image.

### PDF p.350 (printed 326) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
Figure 11.12 Segmented banana

Now, what about the other mask candidates? Those can come in handy for ambiguous prompts. Let’s try to plot the other three masks (see figure 11.13):

### PDF p.348 (printed 324) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
Let’s overlay the first mask on the image (see figure 11.11):
