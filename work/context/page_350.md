# Generation bundle for page_350  (PDF page 350, printed page 326)

Page kind: content
Chapter: 11 Image segmentation
Section: 11.3 Using a pretrained segmentation model
Subsection: 11.3.4 Prompting the model with a target point
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_350.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_350.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p350_e03] (figure_caption) Figure 11.12 Segmented banana
[p350_e04] (body) Now, what about the other mask candidates? Those can come in handy for ambiguous prompts. Let’s try to plot the other three masks (see figure 11.13):
[p350_e05] (code)
```
fig, axes = plt.subplots(1, 3, figsize=(20, 60))
masks = outputs["masks"][0][1:]
for i, mask in enumerate(masks):
    show_image(image, axes[i])
    show_points(input_point, axes[i])
    mask = get_mask(outputs, index=i + 1)
    show_mask(mask, axes[i])
    axes[i].set_title(f"Mask {i + 1}", fontsize=16)
    axes[i].axis("off")
plt.show()
```
[p350_e06] (figure_caption) Figure 11.13 Alternative segmentation masks for the banana prompt

## Figures
- Figure 11.12: Segmented banana  | labels inside the figure: 0, 200, 400, 600, 800, 1000, 0, 200, 400, 600, 800, 1000
- Figure 11.13: Alternative segmentation masks for the banana prompt  | labels inside the figure: Mask 1, Mask 2, Mask 3

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

## Previous page (PDF 349) text, for continuity
Figure 11.11 Segmented peach

Pretty good!

Next, let’s try a banana. We’ll prompt the model with coordinates `(300, 550)`, which land on the second banana from the left (see figure 11.12):

## Next page (PDF 351) text, for continuity
As you can see here, an alternative segmentation found by the model includes both bananas.

11.3.5 Prompting the model with a target box

Besides providing one or more target points, you can also provide boxes approximating the location of the object to segment. These boxes should be passed via the coordinates of their top-left and bottom-right corners. Here’s a box around the mango (see figure 11.14):

Figure 11.14 Box prompt around the mango

Let’s prompt SAM with it (see figure 11.15):

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.348 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.352 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.335 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.336 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.345 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.342 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.344 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.346 — 11 Image segmentation › 11.3 Using a pretrained segmentation model

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.344 (printed 320) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
dataset of images, which was used to train an initial model. This model was used to help drive a semiautomated stage of data collection, where images were first segmented by SAM and improved by human correction and further annotation.

Figure 11.7 An example image from the SA-1B dataset

The model is trained on `(image, prompt, mask)` triples. `image` and `prompt` are the model inputs. The image can be any input image, and the prompt can take a couple of forms:

A point inside the object to mask

A box around the object to mask

Given the `image` and `prompt` input, the model is expected to produce an accurate predicted mask for the object indicated by the prompt, which is compared with a ground truth `mask` label.

The model consists of a few separate components. An image encoder, similar to the Xception model we used in previous chapters, will take an input image and output a much smaller image embedding. This is something we already know how to build. Next, we add a prompt encoder, which is responsible for mapping prompts in any of the previously mentioned forms to an embedded vector, and a mask decoder, which takes in both the image embedding and prompt embedding and outputs a 

### PDF p.347 (printed 323) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3.4 Prompting the model with a target point
To use SAM, you need to prompt it. This means we need one of the following:

Point prompts—Select a point in an image and let the model segment the object that the point belongs to.

Box prompts—Draw an approximate box around an object (it does not need to be particularly precise) and let the model segment the object in the box.

Let’s start with a point prompt. Points are labeled, with 1 indicating the foreground (the object you want to segment) and 0 indicating the background (everything around the object). In ambiguous cases, to improve your results, you could pass multiple labeled points, instead of a single point, to refine your definition of what should be included (points labeled 1) and what should be excluded (points labeled 0).

We try a single foreground point (see figure 11.10). Here’s a test point:

### PDF p.336 (printed 312) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 
Figure 11.3 An example image

Let’s look at its target mask as well (see figure 11.4):

### PDF p.352 (printed 328) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › Summary
Image segmentation is one of the main categories of computer vision tasks. It consists of computing segmentation masks that describe the contents of an image at the pixel level.

To build your own segmentation model, use a stack of strided `Conv2D` layers to “compress” the input image into a smaller feature map, followed by a stack of corresponding `Conv2DTranspose` layers to “expand” the feature map into a segmentation mask the same size as the input image.

You can also use a pretrained segmentation model. Segment Anything, included in KerasHub, is a powerful model that supports image prompting, text prompting, point prompting, and box prompting.

### PDF p.346 (printed 322) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
Next, let’s define a few utilities that will come in handy when using the model. We’re going to need to

Display images.

Display segmentation masks overlaid on an image.

Highlight specific points on an image.

Display boxes overlaid on an image.

All our utilities take a Matplotlib `axis` object (noted `ax`) so that they can all write to the same figure:

### PDF p.343 (printed 319) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3.2 How Segment Anything works
Before we try running some segmentation with the model, let’s talk a little more about how SAM works. Much of the capability of the model comes from the scale of the pretraining dataset. Meta developed the SA-1B dataset along with the model, where the partially trained model was used to assist with the data labeling process. That is, the dataset and model were developed together in a feedback loop of sorts.

The goal with the SA-1B dataset is to create fully segmented images, where every object in an image is given a unique segmentation mask. See figure 11.7 as an example. Each image in the dataset has ~100 masks on average, and some images have over 500 individually masked objects. This was done through a pipeline of increasingly automated data collection. At first, human experts manually segmented a small example

### PDF p.336 (printed 312) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 
Now, what does one of these inputs and its mask look like? Let’s take a quick look (see figure 11.3).

### PDF p.342 (printed 318) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3 Using a pretrained segmentation model
In the image classification example from chapter 8, you saw how using a pretrained model could significantly boost your accuracy—especially when you only have a few samples to train on. Image segmentation is no different.

The Segment Anything Model ,1 or SAM for short, is a powerful pretrained segmentation model you can use for, well, almost anything. It was developed by Meta AI and released in April 2023. It was trained on 11 million images and their segmentation masks, covering over 1 billion object instances. This massive amount of training data

1 Kirillov et al., “Segment Anything,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, arXiv

(2023), https://arxiv.org/abs/2304.02643.
