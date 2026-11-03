# Generation bundle for page_346  (PDF page 346, printed page 322)

Page kind: content
Chapter: 11 Image segmentation
Section: 11.3 Using a pretrained segmentation model
Subsection: 11.3.3 Preparing a test image
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_346.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_346.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p346_e03] (figure_caption) Figure 11.9 Our test image
[p346_e04] (body) SAM expects inputs that are 1024 × 1024. However, forcibly resizing arbitrary images to 1024 × 1024 would distort their aspect ratio—for instance, our image isn’t square. It’s better to first resize the image so that its longest side becomes 1,024 pixels long and then pad the remaining pixels with a filler value, such as 0. We can achieve this with the `pad_to_aspect_ratio` argument in the `keras.ops.image.resize()` operation, like this:
[p346_e05] (code)
```
from keras import ops
image_size = (1024, 1024)
def resize_and_pad(x):
    return ops.image.resize(x, image_size, pad_to_aspect_ratio=True)
image = resize_and_pad(image_array)
```
[p346_e06] (body) Next, let’s define a few utilities that will come in handy when using the model. We’re going to need to
[p346_e07] (bullet) Display images.
[p346_e08] (bullet) Display segmentation masks overlaid on an image.
[p346_e09] (bullet) Highlight specific points on an image.
[p346_e10] (bullet) Display boxes overlaid on an image.
[p346_e11] (body) All our utilities take a Matplotlib `axis` object (noted `ax`) so that they can all write to the same figure:

## Figures
- Figure 11.9: Our test image  | labels inside the figure: (none - image)

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

## Previous page (PDF 345) text, for continuity
Figure 11.8 The Segment Anything high-level architecture overview

All of these subcomponents are trained simultaneously by forming batches of new `(image,` `prompt,` `mask)` triples to train on from the SA-1B image and mask data. The process here is actually quite simple. For a given input image, choose a random mask in the input. Next, randomly choose whether to create a box prompt or a point prompt. To create a point prompt, choose a random pixel inside the mask label. To create a box prompt, draw a box around all points inside the mask label. We can repeat this process indefinitely, sampling a number of `(image,` `prompt,` `mask)` triples from each image input.

11.3.3 Preparing a test image

Let’s make this a little more concrete by trying the model out. We can start by loading a test image for our segmentation work. We’ll use a picture of a bowl of fruits (see figure 11.9):

## Next page (PDF 347) text, for continuity
11.3.4 Prompting the model with a target point

To use SAM, you need to prompt it. This means we need one of the following:

Point prompts—Select a point in an image and let the model segment the object that the point belongs to.

Box prompts—Draw an approximate box around an object (it does not need to be particularly precise) and let the model segment the object in the box.

Let’s start with a point prompt. Points are labeled, with 1 indicating the foreground (the object you want to segment) and 0 indicating the background (everything around the object). In ambiguous cases, to improve your results, you could pass multiple labeled points, instead of a single point, to refine your definition of what should be included (points labeled 1) and what should be excluded (points labeled 0).

We try a single foreground point (see figure 11.10). Here’s a test point:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.335 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.352 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.348 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.271 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.343 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.342 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.349 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.557 — 17 Image generation › 17.3 Text-to-image models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.352 (printed 328) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
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

### PDF p.271 (printed 247) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › Data preprocessing
As you know by now, data should be formatted into appropriately preprocessed floating-point tensors before being fed into the model. Currently, the data sits on a drive as JPEG files, so the steps for getting it into the model are roughly as follows:

1 Read the picture files. 2 Decode the JPEG content to RGB grids of pixels. 3 Convert these into floating-point tensors. 4 Resize them to a shared size (we’ll use 180 x 180). 5 Pack them into batches (we’ll use batches of 32 images). This may seem a bit daunting, but fortunately Keras has utilities to take care of these steps automatically. In particular, Keras features the utility function `image_dataset_` `from_directory`, which lets you quickly set up a data pipeline that can automatically turn image files on disk into batches of preprocessed tensors. This is what you’ll use here. Calling `image_dataset_from_directory(directory)` will first list the subdirectories of `directory` and assume each one contains images from one of your classes. It will then index the image files in each subdirectory. Finally, it will create and return a `tf.data
` `.Dataset` object configured to read these files, shuffle them, decode them to tensors, re

### PDF p.349 (printed 325) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
input_point = np.array([[300, 550]])
input_label = np.array([1])
outputs = model.predict(
    {
        "images": ops.expand_dims(image, axis=0),
        "points": ops.expand_dims(input_point, axis=0),
        "labels": ops.expand_dims(input_label, axis=0),
    }
)
mask = get_mask(outputs, index=0)
plt.figure(figsize=(10, 10))
show_image(image, plt.gca())
show_mask(mask, plt.gca())
show_points(input_point, plt.gca())
plt.show()

### PDF p.352 (printed 328) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › Summary
Image segmentation is one of the main categories of computer vision tasks. It consists of computing segmentation masks that describe the contents of an image at the pixel level.

To build your own segmentation model, use a stack of strided `Conv2D` layers to “compress” the input image into a smaller feature map, followed by a stack of corresponding `Conv2DTranspose` layers to “expand” the feature map into a segmentation mask the same size as the input image.

You can also use a pretrained segmentation model. Segment Anything, included in KerasHub, is a powerful model that supports image prompting, text prompting, point prompting, and box prompting.

### PDF p.348 (printed 324) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
outputs = model.predict(
    {
        "images": ops.expand_dims(image, axis=0),
        "points": ops.expand_dims(input_point, axis=0),
        "labels": ops.expand_dims(input_label, axis=0),
    }
)

### PDF p.335 (printed 311) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 11.2.1 Downloading a segmentation dataset
We’ll work with the Oxford-IIIT Pets dataset (https://www.robots.ox.ac.uk/~vgg/data/ pets/), which contains 7,390 pictures of various breeds of cats and dogs, together with foreground-background segmentation masks for each picture. A segmentation mask is the image segmentation equivalent of a label: it’s an image the same size as the input image, with a single color channel where each integer value corresponds to the class of the corresponding pixel in the input image. In our case, the pixels of our segmentation masks can take one of three integer values:

1 (foreground)

2 (background)

3 (contour)

Let’s start by downloading and uncompressing our dataset, using the `wget` and `tar` shell utilities:

### PDF p.344 (printed 320) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
dataset of images, which was used to train an initial model. This model was used to help drive a semiautomated stage of data collection, where images were first segmented by SAM and improved by human correction and further annotation.

Figure 11.7 An example image from the SA-1B dataset

The model is trained on `(image, prompt, mask)` triples. `image` and `prompt` are the model inputs. The image can be any input image, and the prompt can take a couple of forms:

A point inside the object to mask

A box around the object to mask

Given the `image` and `prompt` input, the model is expected to produce an accurate predicted mask for the object indicated by the prompt, which is compared with a ground truth `mask` label.

The model consists of a few separate components. An image encoder, similar to the Xception model we used in previous chapters, will take an input image and output a much smaller image embedding. This is something we already know how to build. Next, we add a prompt encoder, which is responsible for mapping prompts in any of the previously mentioned forms to an embedded vector, and a mask decoder, which takes in both the image embedding and prompt embedding and outputs a 

### PDF p.257 (printed 233) — 8 Image classification › 8.1 Introduction to convnets › Image data formats in deep learning frameworks
Some deep learning libraries flip the location of channels in image tensors to the first rank (notably much of the PyTorch ecosystem). Rather than passing images with

shape `(height, width, channels)` you would pass `(channels, height, width)`.

This is purely a matter of convention, and in Keras is configurable. You can call `keras
`

`.config.set_image_data_format("channels_first")` to change Keras’ default, or

pass a `data_format` argument to any conv or pooling layer. In general, you can leave the default as is unless you have a specific need for `"channels_first"`.
