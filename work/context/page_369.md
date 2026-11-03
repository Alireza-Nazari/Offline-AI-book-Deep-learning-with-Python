# Generation bundle for page_369  (PDF page 369, printed page 345)

Page kind: content
Chapter: 12 Object detection
Section: 12.2 Training a YOLO model from scratch
Subsection: 12.2.4 Training the YOLO model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_369.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_369.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p369_e03] (listing_caption) Listing 12.13 Training the YOLO model
[p369_e04] (code)
```
x, y = next(iter(val_dataset.rebatch(1)))
preds = model.predict(x)
boxes = preds["box"][0]
classes = np.argmax(preds["class"][0], axis=-1)
path = metadata[0]["path"]
draw_prediction(path, boxes, classes, cutoff=0.1)
```
[p369_e05] (figure_caption) Figure 12.6 Predictions for our sample image
[p369_e06] (body) We can see our model is starting to understand box locations and class labels, though it is still not very accurate. Let’s visualize every box predicted by the model (figure 12.7), even those with zero confidence:
[p369_e07] (code)
```
draw_prediction(path, boxes, classes, cutoff=None)
```

## Code annotations printed next to listings (callouts, in order)
- Rebatches our dataset to get
- a single sample instead of 16
- Uses argmax to
- find the most
- likely label at
- each grid location
- Loads the image from
- disk to view it a full size

## Listings
- Listing 12.13: Training the YOLO model (code elements p369_e04)

## Figures
- Figure 12.6: Predictions for our sample image  | labels inside the figure: (none - image)

## Equation candidates (formulas written inline in the text)
- [p369_e07] draw_prediction(path, boxes, classes, cutoff=None)

## Chapter outline
Chapter: 12 Object detection (PDF pages 353-374)
  - 12.1 Single-stage vs. two-stage object detectors (p.354)
      - 12.1.1 Two-stage R-CNN detectors (p.354)
      - 12.1.2 Single-stage detectors (p.356)
  - 12.2 Training a YOLO model from scratch (p.356)
      - 12.2.1 Downloading the COCO dataset (p.356)
      - 12.2.2 Creating a YOLO model (p.360)
      - 12.2.3 Readying the COCO data for the YOLO model (p.363)
      - 12.2.4 Training the YOLO model (p.366)
  - 12.3 Using a pretrained RetinaNet detector (p.370)

## Previous page (PDF 368) text, for continuity
Listing 12.11 Defining the YOLO bounding box loss

We are finally ready to start training our YOLO model. Purely to keep this example short, we will skip over metrics. In a real-world setting, you’d want quite a few metrics here—such as the accuracy of the model at different confidence cutoff levels.

Listing 12.12 Training the YOLO model

Training takes over an hour on the Colab free GPU runtime, and our model is still undertrained (validation loss is still falling!). Let’s try visualizing an output from our model (figure 12.6). We will use a low-confidence cutoff, as our model is not a very good object detector quite yet.

## Next page (PDF 370) text, for continuity
Figure 12.7 Every bounding box predicted by the YOLO model

Our model learns very low-confidence values because it has not yet learned to consistently locate objects in a scene. To further improve the model, we should try a number of things:

Train for more epochs

Use the whole COCO dataset

Data augmentation (e.g., translating and rotating input images and boxes)

Improve our class probability map for overlapping boxes

Predict multiple boxes per grid location using a bigger output grid

All of these would positively affect the model performance and get us closer to the original YOLO training recipe. However, this example is really just to get a feel for object detection training—training an accurate COCO detection model from scratch would take a large amount of compute and time. Instead, to get a sense of a better-performing detection model, let’s try using a pretrained object detection model called RetinaNet.

12.3 Using a pretrained RetinaNet detector

RetinaNet is also a single-stage object detector and operates on the same basic principles as the YOLO model. The biggest conceptual difference between our model and

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.365 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.359 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.364 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.372 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.360 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.357 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.358 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.361 — 12 Object detection › 12.2 Training a YOLO model from scratch

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.364 (printed 340) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Let’s visualize our YOLO training data with our box drawing helpers (figure 12.5). We will draw the entire class activation map over our first input image4 and add the confidence score of a box along with its label.

4 Image from the COCO 2017 dataset, https://cocodataset.org/. Image from Flickr, http://farm9.staticflickr.com /8081/8387882360_5b97a233c4_z.jpg, CC BY 2.0 https://creativecommons.org/licenses/by/2.0/.

### PDF p.360 (printed 336) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2.2 Creating a YOLO model
As mentioned previously, the YOLO model is a single stage detector. Rather than first attempting to identify all candidate objects in a scene and then classifying the object regions, YOLO will propose bounding boxes and object labels in one go.

Our model will divide an image up into a grid and predict two separate outputs at each grid location—a bounding box and a class label. In the original paper by Redmon et al.,3 the model actually predicted multiple boxes per grid location, but we keep things simple and just predict one box in each grid square. Most images will not have objects evenly distributed across a grid, and to account for this, the model will output a confidence score along with each box, as shown in figure 12.4. We’d like this confidence to be high when an object is detected at a location, and zero when there’s no object. Most grid locations will have no object and should report a near-zero confidence. Like many models in computer vision, the YOLO model uses a ConvNet backbone to obtain interesting high-level features for an input image, a concept we first explored in chapter 8. In their paper, the authors created their own backbone model and pretrained it with Image

### PDF p.366 (printed 342) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2.4 Training the YOLO model
We have our model and our training data ready, but there’s one last element we need before we can actually run `fit()`: the loss function. Our model outputs predicted boxes and predicted grid labels. We saw in chapter 7 how we can define multiple losses for each output—Keras will simply sum the losses together during training. We can handle the classification loss with `sparse_categorical_crossentropy` as usual.

The box loss, however, needs some special consideration. The basic loss proposed by the YOLO authors is fairly simple. They use the sum-squared error of the difference between the target box parameters and the predicted ones. We will only compute this error for grid cells with actual boxes in the labeled data. The tricky part of the loss is the box confidence output. The authors wanted the confidence output to reflect not just the presence of an object, but also how good the predicted

### PDF p.365 (printed 341) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Figure 12.5 YOLO outputs a bounding box prediction and class label for each image region.

### PDF p.359 (printed 335) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Figure 12.3 YOLO outputs a bounding box prediction and class label for each image region.

2 Image from the COCO 2017 dataset, https://cocodataset.org/. Image from Flickr, http://farm8.staticflickr.com /7250/7520201840_3e01349e3f_z.jpg, CC BY 2.0 https://creativecommons.org/licenses/by/2.0/.

### PDF p.356 (printed 332) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2 Training a YOLO model from scratch
Overall, building an object detector can be a bit of an undertaking—not that there’s anything theoretically complex about it. There’s just a lot of code needed to handle manipulating bounding boxes and predicted output. To keep things simple, we will recreate the very first YOLO model from 2015. There are 12 YOLO versions as of this writing, but the original is a bit simpler to work with.

### PDF p.364 (printed 340) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Let’s rework our training data so it conforms to this new grid structure. We can create two arrays as long as our dataset with our grid:

The first will contain our class probability map. We will mark all grid cells that intersect with a bounding box with the correct label. To keep our code simple, we won’t worry about overlapping boxes.

The second will contain the boxes themselves. We will translate all boxes to the grid and label the correct grid cell with the coordinates for the box. The confidence for an actual box in our label data will always be one, and the confidence for all other locations will be zero.

Listing 12.7 Creating the YOLO targets

### PDF p.361 (printed 337) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Next, we can turn our backbone into a detection model by adding new layers for outputting box and class predictions. The setup proposed in the YOLO paper is quite simple. Take the output of a ConvNet backbone and feed it through two densely connected layers with an activation in the middle. Then, split the output. The first five numbers will be used for bounding box prediction (four for the box and one for the box confidence). The rest will be used for the class probability map shown in figure 12.4—a classification prediction for each grid location over all possible 91 labels.

Let’s write this out.
