# Generation bundle for page_364  (PDF page 364, printed page 340)

Page kind: content
Chapter: 12 Object detection
Section: 12.2 Training a YOLO model from scratch
Subsection: 12.2.3 Readying the COCO data for the YOLO model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_364.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_364.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p364_e03] (code)
```
    x = (xi + x) / grid_size - w / 2
    y = (yi + y) / grid_size - h / 2
    return (x, y, w, h)
```
[p364_e04] (body) Let’s rework our training data so it conforms to this new grid structure. We can create two arrays as long as our dataset with our grid:
[p364_e05] (bullet) The first will contain our class probability map. We will mark all grid cells that intersect with a bounding box with the correct label. To keep our code simple, we won’t worry about overlapping boxes.
[p364_e06] (bullet) The second will contain the boxes themselves. We will translate all boxes to the grid and label the correct grid cell with the coordinates for the box. The confidence for an actual box in our label data will always be one, and the confidence for all other locations will be zero.
[p364_e07] (listing_caption) Listing 12.7 Creating the YOLO targets
[p364_e08] (code)
```
import numpy as np
import math
class_array = np.zeros((len(metadata), grid_size, grid_size))
box_array = np.zeros((len(metadata), grid_size, grid_size, 5))
for index, sample in enumerate(metadata):
    boxes, labels = sample["boxes"], sample["labels"]
    for box, label in zip(boxes, labels):
        (x, y, w, h) = box
        left, right = math.floor(x * grid_size), math.ceil((x + w) * grid_size)
        bottom, top = math.floor(y * grid_size), math.ceil((y + h) * grid_size)
        class_array[index, bottom:top, left:right] = label
for index, sample in enumerate(metadata):
    boxes, labels = sample["boxes"], sample["labels"]
    for box, label in zip(boxes, labels):
        (xi, yi), (grid_box) = to_grid(box)
        box_array[index, yi, xi] = [*grid_box, 1.0]
        class_array[index, yi, xi] = label
```
[p364_e09] (body) Let’s visualize our YOLO training data with our box drawing helpers (figure 12.5). We will draw the entire class activation map over our first input image4 and add the confidence score of a box along with its label.
[p364_e10] (footnote) 4 Image from the COCO 2017 dataset, https://cocodataset.org/. Image from Flickr, http://farm9.staticflickr.com /8081/8387882360_5b97a233c4_z.jpg, CC BY 2.0 https://creativecommons.org/licenses/by/2.0/.

## Code annotations printed next to listings (callouts, in order)
- Finds all grid cells
- whose center falls
- inside the box
- Transforms the
- box to the grid
- coordinate system
- Makes sure the class label
- for the box’s center location
- matches the box

## Listings
- Listing 12.7: Creating the YOLO targets (code elements p364_e08)

## Equation candidates (formulas written inline in the text)
- [p364_e03]     x = (xi + x) / grid_size - w / 2
    y = (yi + y) / grid_size - h / 2
    return (x, y, w, h)

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

## Previous page (PDF 363) text, for continuity
Our backbone outputs have shape `(batch_size, 14, 14, 2048)`. That is 401,408 output floats per image, a bit too many to feed into our dense layers. We downscale the feature maps with a strided conv layer to `(batch_size, 6, 6, 512)` with a more manageable 18,432 floats per image.

Next, we can add our two densely connected layers. We flatten the entire feature map, pass it through a `Dense` with a `relu` activation and then pass it through a final `Dense` with our exact number of output predictions—5 for the bounding box and confidence and 91 for each object class at each grid location. Finally, we reshape the outputs back to a 6 × 6 grid and split our box and class predictions. As usual for our classification outputs, we apply a softmax. The box outputs will need more special consideration; we will cover this later. Looking good! Note that because we flatten the entire feature map through the classification layer, every grid detector can use the entire image’s features; there’s no locality constraint. This is by design—large objects will not stay contained to a single grid cell.

12.2.3 Readying the COCO data for the YOLO model

Our model is relatively simple, but we still need to preprocess our inputs to align them with the prediction grid. Each grid detector will be responsible for detecting any boxes whose center falls inside the grid box. Our model will output five floats for the box `(x, y, w, h, confidence)`. The `x` and `y` will represent the object’s center relative to the bounds of the grid cell (from 0 to 1). The `w` and `h` will represent the object’s size relative to the image size.

We already have the right `w` and `h` values in our training data. However, we need to translate our `x` and `y` values to and from the grid. Let’s define two utilities:

## Next page (PDF 365) text, for continuity
Listing 12.8 Visualizing a YOLO target

Figure 12.5 YOLO outputs a bounding box prediction and class label for each image region.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.359 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.373 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.360 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.362 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.369 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.358 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.367 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.357 — 12 Object detection › 12.2 Training a YOLO model from scratch

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.359 (printed 335) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Figure 12.3 YOLO outputs a bounding box prediction and class label for each image region.

2 Image from the COCO 2017 dataset, https://cocodataset.org/. Image from Flickr, http://farm8.staticflickr.com /7250/7520201840_3e01349e3f_z.jpg, CC BY 2.0 https://creativecommons.org/licenses/by/2.0/.

### PDF p.370 (printed 346) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 
Figure 12.7 Every bounding box predicted by the YOLO model

Our model learns very low-confidence values because it has not yet learned to consistently locate objects in a scene. To further improve the model, we should try a number of things:

Train for more epochs

Use the whole COCO dataset

Data augmentation (e.g., translating and rotating input images and boxes)

Improve our class probability map for overlapping boxes

Predict multiple boxes per grid location using a bigger output grid

All of these would positively affect the model performance and get us closer to the original YOLO training recipe. However, this example is really just to get a feel for object detection training—training an accurate COCO detection model from scratch would take a large amount of compute and time. Instead, to get a sense of a better-performing detection model, let’s try using a pretrained object detection model called RetinaNet.

### PDF p.360 (printed 336) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2.2 Creating a YOLO model
As mentioned previously, the YOLO model is a single stage detector. Rather than first attempting to identify all candidate objects in a scene and then classifying the object regions, YOLO will propose bounding boxes and object labels in one go.

Our model will divide an image up into a grid and predict two separate outputs at each grid location—a bounding box and a class label. In the original paper by Redmon et al.,3 the model actually predicted multiple boxes per grid location, but we keep things simple and just predict one box in each grid square. Most images will not have objects evenly distributed across a grid, and to account for this, the model will output a confidence score along with each box, as shown in figure 12.4. We’d like this confidence to be high when an object is detected at a location, and zero when there’s no object. Most grid locations will have no object and should report a near-zero confidence. Like many models in computer vision, the YOLO model uses a ConvNet backbone to obtain interesting high-level features for an input image, a concept we first explored in chapter 8. In their paper, the authors created their own backbone model and pretrained it with Image

### PDF p.366 (printed 342) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2.4 Training the YOLO model
We have our model and our training data ready, but there’s one last element we need before we can actually run `fit()`: the loss function. Our model outputs predicted boxes and predicted grid labels. We saw in chapter 7 how we can define multiple losses for each output—Keras will simply sum the losses together during training. We can handle the classification loss with `sparse_categorical_crossentropy` as usual.

The box loss, however, needs some special consideration. The basic loss proposed by the YOLO authors is fairly simple. They use the sum-squared error of the difference between the target box parameters and the predicted ones. We will only compute this error for grid cells with actual boxes in the labeled data. The tricky part of the loss is the box confidence output. The authors wanted the confidence output to reflect not just the presence of an object, but also how good the predicted

### PDF p.361 (printed 337) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Next, we can turn our backbone into a detection model by adding new layers for outputting box and class predictions. The setup proposed in the YOLO paper is quite simple. Take the output of a ConvNet backbone and feed it through two densely connected layers with an activation in the middle. Then, split the output. The first five numbers will be used for bounding box prediction (four for the box and one for the box confidence). The rest will be used for the class probability map shown in figure 12.4—a classification prediction for each grid location over all possible 91 labels.

Let’s write this out.

### PDF p.357 (printed 333) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
We need to do some input massaging before we are ready to use this data. The first download gives us an unlabeled directory of all the COCO images. The second download includes all image metadata via a JSON file. COCO associates each image file with an ID, and each bounding box is paired with one of these IDs. We need to collate all box and image data together.

Each bounding box comes with `x, y, width, height` pixel coordinates starting at the top left corner of the image. As we load our data, we can rescale all bounding box coordinates so they are points in a `[0, 1]` unit square. This will make it easier to manipulate these boxes without needing to check the size of each input image.

Listing 12.2 Parsing the COCO data

### PDF p.358 (printed 334) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
We have 117,266 images. Each image can have anywhere from 1 to 63 objects with an associated bounding box. There are only 91 possible labels for objects, chosen by the COCO dataset creators.

We can use a KerasHub utility `keras_hub.utils.coco_id_to_name(id)` to map these integer labels to human-readable names, similar to the utility we used to decode ImageNet predictions to text labels back in chapter 8. Let’s visualize an example image to make this a little more concrete. We can define a function to draw an image with Matplotlib and another function to draw a labeled bounding box on this image. We will need both of these throughout the chapter. We can use the HSV colorspace as a simple trick to generate new colors for each new label we see. By fixing the saturation and brightness of the color and only updating its hue, we can generate bright new colors that stand out clearly from our image.

Listing 12.4 Visualizing a COCO image with box annotations

### PDF p.369 (printed 345) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Listing 12.13 Training the YOLO model
x, y = next(iter(val_dataset.rebatch(1)))
preds = model.predict(x)
boxes = preds["box"][0]
classes = np.argmax(preds["class"][0], axis=-1)
path = metadata[0]["path"]
draw_prediction(path, boxes, classes, cutoff=0.1)
