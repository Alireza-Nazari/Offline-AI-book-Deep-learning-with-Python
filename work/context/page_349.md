# Generation bundle for page_349  (PDF page 349, printed page 325)

Page kind: content
Chapter: 11 Image segmentation
Section: 11.3 Using a pretrained segmentation model
Subsection: 11.3.4 Prompting the model with a target point
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_349.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_349.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p349_e03] (code)
```
plt.figure(figsize=(10, 10))
show_image(image, plt.gca())
show_mask(mask, plt.gca())
show_points(input_point, plt.gca())
plt.show()
```
[p349_e04] (figure_caption) Figure 11.11 Segmented peach
[p349_e05] (body) Pretty good!
[p349_e06] (body) Next, let’s try a banana. We’ll prompt the model with coordinates `(300, 550)`, which land on the second banana from the left (see figure 11.12):
[p349_e07] (code)
```
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
```

## Figures
- Figure 11.11: Segmented peach  | labels inside the figure: 0, 200, 400, 600, 800, 1000, 0, 200, 400, 600, 800, 1000

## Equation candidates (formulas written inline in the text)
- [p349_e03] plt.figure(figsize=(10, 10))
show_image(image, plt.gca())
show_mask(mask, plt.gca())
show_points(input_point, plt.gca())
plt.show()

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

## Previous page (PDF 348) text, for continuity
Figure 11.10 A prompt point, landing on a peach

Let’s prompt SAM with it:

The return value `outputs` has a `"masks"` field which contains four 256 × 256 candidate masks for the target object, ranked by decreasing match quality. The quality scores of the masks are available under the `"iou_pred"` field as part of the model’s output:

Let’s overlay the first mask on the image (see figure 11.11):

## Next page (PDF 350) text, for continuity
Figure 11.12 Segmented banana

Now, what about the other mask candidates? Those can come in handy for ambiguous prompts. Let’s try to plot the other three masks (see figure 11.13):

Figure 11.13 Alternative segmentation masks for the banana prompt

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.347 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.352 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.351 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.336 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.345 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.342 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.346 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.343 — 11 Image segmentation › 11.3 Using a pretrained segmentation model

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.351 (printed 327) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3.5 Prompting the model with a target box
Figure 11.14 Box prompt around the mango

Let’s prompt SAM with it (see figure 11.15):

### PDF p.351 (printed 327) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3.5 Prompting the model with a target box
Besides providing one or more target points, you can also provide boxes approximating the location of the object to segment. These boxes should be passed via the coordinates of their top-left and bottom-right corners. Here’s a box around the mango (see figure 11.14):

### PDF p.347 (printed 323) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3.4 Prompting the model with a target point
To use SAM, you need to prompt it. This means we need one of the following:

Point prompts—Select a point in an image and let the model segment the object that the point belongs to.

Box prompts—Draw an approximate box around an object (it does not need to be particularly precise) and let the model segment the object in the box.

Let’s start with a point prompt. Points are labeled, with 1 indicating the foreground (the object you want to segment) and 0 indicating the background (everything around the object). In ambiguous cases, to improve your results, you could pass multiple labeled points, instead of a single point, to refine your definition of what should be included (points labeled 1) and what should be excluded (points labeled 0).

We try a single foreground point (see figure 11.10). Here’s a test point:

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

### PDF p.347 (printed 323) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3.4 Prompting the model with a target point
import numpy as np
input_point = np.array([[580, 450]])
input_label = np.array([1])
plt.figure(figsize=(10, 10))
show_image(image, plt.gca())
show_points(input_point, plt.gca())
plt.show()

### PDF p.343 (printed 319) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 11.3.2 How Segment Anything works
Before we try running some segmentation with the model, let’s talk a little more about how SAM works. Much of the capability of the model comes from the scale of the pretraining dataset. Meta developed the SA-1B dataset along with the model, where the partially trained model was used to assist with the data labeling process. That is, the dataset and model were developed together in a feedback loop of sorts.

The goal with the SA-1B dataset is to create fully segmented images, where every object in an image is given a unique segmentation mask. See figure 11.7 as an example. Each image in the dataset has ~100 masks on average, and some images have over 500 individually masked objects. This was done through a pipeline of increasingly automated data collection. At first, human experts manually segmented a small example

### PDF p.336 (printed 312) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 
Figure 11.3 An example image

Let’s look at its target mask as well (see figure 11.4):

### PDF p.352 (printed 328) — 11 Image segmentation › 11.3 Using a pretrained segmentation model › 
Figure 11.15 Segmented mango

SAM can be a powerful tool to quickly create large datasets of images annotated with segmentation masks.
