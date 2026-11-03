# Generation bundle for page_337  (PDF page 337, printed page 313)

Page kind: content
Chapter: 11 Image segmentation
Section: 11.2 Training a segmentation model from scratch
Subsection: 11.2.1 Downloading a segmentation dataset
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_337.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_337.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p337_e03] (figure_caption) Figure 11.4 The corresponding target mask
[p337_e04] (body) Next, let’s load our inputs and targets into two NumPy arrays. Since the dataset is very small, we can load everything into memory:
[p337_e05] (code)
```
import numpy as np
import random
img_size = (200, 200)
num_imgs = len(input_img_paths)
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_paths)
def path_to_input_image(path):
    return img_to_array(load_img(path, target_size=img_size))
def path_to_target(path):
    img = img_to_array(
        load_img(path, target_size=img_size, color_mode="grayscale")
    )
    img = img.astype("uint8") - 1
    return img
```

## Code annotations printed next to listings (callouts, in order)
- Shuffles the file paths (they were originally
- sorted by breed). We use the same seed (1337)
- in both statements to ensure that the input
- paths and target paths stay in the same order.
- We resize everything to
- 200 x 200 for this example.
- Total number of
- samples in the data
- Subtracts 1 so that our
- labels become 0, 1, and 2

## Figures
- Figure 11.4: The corresponding target mask  | labels inside the figure: (none - image)

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

## Previous page (PDF 336) text, for continuity
Now, what does one of these inputs and its mask look like? Let’s take a quick look (see figure 11.3).

Figure 11.3 An example image

Let’s look at its target mask as well (see figure 11.4):

## Next page (PDF 338) text, for continuity
As always, let’s split the arrays into a training and a validation set:

11.2.2 Building and training the segmentation model

Now, it’s time to define our model:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.345 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.335 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.339 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.545 — 17 Image generation › 17.2 Diffusion models
- PDF p.366 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.557 — 17 Image generation › 17.3 Text-to-image models
- PDF p.347 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.310 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.335 (printed 311) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 
To get more familiar with segmentation, let’s get started with training a small segmentation model from scratch on your own data.

### PDF p.335 (printed 311) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 11.2.1 Downloading a segmentation dataset
We’ll work with the Oxford-IIIT Pets dataset (https://www.robots.ox.ac.uk/~vgg/data/ pets/), which contains 7,390 pictures of various breeds of cats and dogs, together with foreground-background segmentation masks for each picture. A segmentation mask is the image segmentation equivalent of a label: it’s an image the same size as the input image, with a single color channel where each integer value corresponds to the class of the corresponding pixel in the input image. In our case, the pixels of our segmentation masks can take one of three integer values:

1 (foreground)

2 (background)

3 (contour)

Let’s start by downloading and uncompressing our dataset, using the `wget` and `tar` shell utilities:

### PDF p.352 (printed 328) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › Summary
Image segmentation is one of the main categories of computer vision tasks. It consists of computing segmentation masks that describe the contents of an image at the pixel level.

To build your own segmentation model, use a stack of strided `Conv2D` layers to “compress” the input image into a smaller feature map, followed by a stack of corresponding `Conv2DTranspose` layers to “expand” the feature map into a segmentation mask the same size as the input image.

You can also use a pretrained segmentation model. Segment Anything, included in KerasHub, is a powerful model that supports image prompting, text prompting, point prompting, and box prompting.

### PDF p.343 (printed 319) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3.2 How Segment Anything works
Before we try running some segmentation with the model, let’s talk a little more about how SAM works. Much of the capability of the model comes from the scale of the pretraining dataset. Meta developed the SA-1B dataset along with the model, where the partially trained model was used to assist with the data labeling process. That is, the dataset and model were developed together in a feedback loop of sorts.

The goal with the SA-1B dataset is to create fully segmented images, where every object in an image is given a unique segmentation mask. See figure 11.7 as an example. Each image in the dataset has ~100 masks on average, and some images have over 500 individually masked objects. This was done through a pipeline of increasingly automated data collection. At first, human experts manually segmented a small example

### PDF p.366 (printed 342) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
With that, our data is ready for training.

NOTE This training example shows clearly why a streaming library like `tf.data` is helpful. Loading all the images in this large dataset in one go would overwhelm our system memory (remember an image tensor is much larger than a compressed JPEG file). With `tf.data`, we can load our image data in batch by batch and release the memory when we are done, only mapping in the particular parts of the dataset we need at a given moment. The `prefetch(2)` call will cause `tf.data` to keep two batches buffered and ready before they are used so we don’t interrupt training each batch to load and resize more images.

### PDF p.343 (printed 319) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3.1 Downloading the Segment Anything Model
First, let’s instantiate SAM and download its weights. Once again, we can use the KerasHub package to use this pretrained model without needing to implement it ourselves from scratch.

Remember the `ImageClassifier` task we used in the previous chapter? We can use another KerasHub task `ImageSegmenter` for wrapping pretrained image segmentation models into a high-level model with standard inputs and outputs. Here, we’ll use the `sam_huge_sa1b` pretrained model, where `sam` stands for the model, `huge` refers to the number of parameters in the model, and `sa1b` stands for the SA-1B dataset released along with the model, with 1 billion annotated masks. Let’s download it now:

### PDF p.366 (printed 342) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Lastly, let’s use `tf.data` to load our image data. We will load our images from disk, apply our preprocessing, and batch them. We should also split a validation set to monitor training.

Listing 12.9 Creating a dataset to train on

### PDF p.344 (printed 320) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
dataset of images, which was used to train an initial model. This model was used to help drive a semiautomated stage of data collection, where images were first segmented by SAM and improved by human correction and further annotation.

Figure 11.7 An example image from the SA-1B dataset

The model is trained on `(image, prompt, mask)` triples. `image` and `prompt` are the model inputs. The image can be any input image, and the prompt can take a couple of forms:

A point inside the object to mask

A box around the object to mask

Given the `image` and `prompt` input, the model is expected to produce an accurate predicted mask for the object indicated by the prompt, which is compared with a ground truth `mask` label.

The model consists of a few separate components. An image encoder, similar to the Xception model we used in previous chapters, will take an input image and output a much smaller image embedding. This is something we already know how to build. Next, we add a prompt encoder, which is responsible for mapping prompts in any of the previously mentioned forms to an embedded vector, and a mask decoder, which takes in both the image embedding and prompt embedding and outputs a 
