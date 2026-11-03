# Generation bundle for page_348  (PDF page 348, printed page 324)

Page kind: content
Chapter: 11 Image segmentation
Section: 11.3 Using a pretrained segmentation model
Subsection: 11.3.4 Prompting the model with a target point
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_348.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_348.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p348_e03] (figure_caption) Figure 11.10 A prompt point, landing on a peach
[p348_e04] (body) Let’s prompt SAM with it:
[p348_e05] (code)
```
outputs = model.predict(
    {
        "images": ops.expand_dims(image, axis=0),
        "points": ops.expand_dims(input_point, axis=0),
        "labels": ops.expand_dims(input_label, axis=0),
    }
)
```
[p348_e06] (body) The return value `outputs` has a `"masks"` field which contains four 256 × 256 candidate masks for the target object, ranked by decreasing match quality. The quality scores of the masks are available under the `"iou_pred"` field as part of the model’s output:
[p348_e07] (code)
```
>>> outputs["masks"].shape
(1, 4, 256, 256)
```
[p348_e08] (body) Let’s overlay the first mask on the image (see figure 11.11):
[p348_e09] (code)
```
def get_mask(sam_outputs, index=0):
    mask = sam_outputs["masks"][0][index]
    mask = np.expand_dims(mask, axis=-1)
    mask = resize_and_pad(mask)
    return ops.convert_to_numpy(mask) > 0.0
mask = get_mask(outputs, index=0)
```

## Figures
- Figure 11.10: A prompt point, landing on a peach  | labels inside the figure: 0, 200, 400, 600, 800, 1000, 0, 200, 400, 600, 800, 1000

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

## Previous page (PDF 347) text, for continuity
11.3.4 Prompting the model with a target point

To use SAM, you need to prompt it. This means we need one of the following:

Point prompts—Select a point in an image and let the model segment the object that the point belongs to.

Box prompts—Draw an approximate box around an object (it does not need to be particularly precise) and let the model segment the object in the box.

Let’s start with a point prompt. Points are labeled, with 1 indicating the foreground (the object you want to segment) and 0 indicating the background (everything around the object). In ambiguous cases, to improve your results, you could pass multiple labeled points, instead of a single point, to refine your definition of what should be included (points labeled 1) and what should be excluded (points labeled 0).

We try a single foreground point (see figure 11.10). Here’s a test point:

## Next page (PDF 349) text, for continuity
Figure 11.11 Segmented peach

Pretty good!

Next, let’s try a banana. We’ll prompt the model with coordinates `(300, 550)`, which land on the second banana from the left (see figure 11.12):

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.352 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.350 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.342 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.336 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.345 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.335 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.340 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
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

### PDF p.350 (printed 326) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
Figure 11.13 Alternative segmentation masks for the banana prompt

### PDF p.350 (printed 326) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
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

### PDF p.345 (printed 321) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
Figure 11.8 The Segment Anything high-level architecture overview

All of these subcomponents are trained simultaneously by forming batches of new `(image,` `prompt,` `mask)` triples to train on from the SA-1B image and mask data. The process here is actually quite simple. For a given input image, choose a random mask in the input. Next, randomly choose whether to create a box prompt or a point prompt. To create a point prompt, choose a random pixel inside the mask label. To create a box prompt, draw a box around all points inside the mask label. We can repeat this process indefinitely, sampling a number of `(image,` `prompt,` `mask)` triples from each image input.

### PDF p.336 (printed 312) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 
Figure 11.3 An example image

Let’s look at its target mask as well (see figure 11.4):

### PDF p.340 (printed 316) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 
simply apply the corresponding sequence of `Conv2DTranspose` layers followed by a final `Conv2D` layer to produce outputs of shape `(200, 200, 3)`.

To evaluate the model, we’ll use a metric named Intersection over Union (IoU). It’s a measure of the match between the ground truth segmentation masks and the predicted masks. It can be computed separately for each class or averaged over multiple classes. Here’s how it works: 1 Compute the intersection between the masks, the area where the prediction and

ground truth overlap. 2 Compute the union of the masks, the total area covered by both masks combined. This is the whole space we’re interested in—the target object and any extra bits your model might have included by mistake. 3 Divide the intersection area by the union area to get the IoU. It’s a number between 0 and 1, where 1 denotes a perfect match, and 0 denotes a complete miss. We can simply use a built-in Keras metric rather than building this ourselves:

### PDF p.351 (printed 327) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3.5 Prompting the model with a target box
Figure 11.14 Box prompt around the mango

Let’s prompt SAM with it (see figure 11.15):

### PDF p.343 (printed 319) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3.2 How Segment Anything works
Before we try running some segmentation with the model, let’s talk a little more about how SAM works. Much of the capability of the model comes from the scale of the pretraining dataset. Meta developed the SA-1B dataset along with the model, where the partially trained model was used to assist with the data labeling process. That is, the dataset and model were developed together in a feedback loop of sorts.

The goal with the SA-1B dataset is to create fully segmented images, where every object in an image is given a unique segmentation mask. See figure 11.7 as an example. Each image in the dataset has ~100 masks on average, and some images have over 500 individually masked objects. This was done through a pipeline of increasingly automated data collection. At first, human experts manually segmented a small example
