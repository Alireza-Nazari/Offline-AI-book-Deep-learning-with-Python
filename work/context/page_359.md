# Generation bundle for page_359  (PDF page 359, printed page 335)

Page kind: content
Chapter: 12 Object detection
Section: 12.2 Training a YOLO model from scratch
Subsection: 12.2.1 Downloading the COCO dataset
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_359.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_359.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p359_e03] (code)
```
    textbox = dict(fc=color, pad=1, ec="none")
    ax.text(x, y, text, c="white", size=10, va="bottom", bbox=textbox)
def draw_image(ax, image):
    ax.set(xlim=(0, 1), ylim=(1, 0), xticks=[], yticks=[], aspect="equal")
    image = plt.imread(image)
    height, width = image.shape[:2]
    hpad = (1 - height / width) / 2 if width > height else 0
    wpad = (1 - width / height) / 2 if height > width else 0
    extent = [wpad, 1 - wpad, 1 - hpad, hpad]
    ax.imshow(image, extent=extent)
```
[p359_e04] (body) Let’s use our new visualization to look at the sample image2 we were inspecting earlier (see figure 12.3):
[p359_e05] (code)
```
sample = metadata[435]
ig, ax = plt.subplots(dpi=300)
draw_image(ax, sample["path"])
for box, label in zip(sample["boxes"], sample["labels"]):
    label_name = keras_hub.utils.coco_id_to_name(label)
    draw_box(ax, box, label_name, label_to_color(label))
plt.show()
```
[p359_e06] (figure_caption) Figure 12.3 YOLO outputs a bounding box prediction and class label for each image region.
[p359_e07] (footnote) 2 Image from the COCO 2017 dataset, https://cocodataset.org/. Image from Flickr, http://farm8.staticflickr.com /7250/7520201840_3e01349e3f_z.jpg, CC BY 2.0 https://creativecommons.org/licenses/by/2.0/.

## Code annotations printed next to listings (callouts, in order)
- Pads the image
- so it fits inside
- the unit cube
- Draws the image on a unit cube
- with (0, 0) at the top left

## Figures
- Figure 12.3: YOLO outputs a bounding box prediction and class label for each image region.  | labels inside the figure: (none - image)

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

## Previous page (PDF 358) text, for continuity
Listing 12.3 Inspecting the COCO data

We have 117,266 images. Each image can have anywhere from 1 to 63 objects with an associated bounding box. There are only 91 possible labels for objects, chosen by the COCO dataset creators.

We can use a KerasHub utility `keras_hub.utils.coco_id_to_name(id)` to map these integer labels to human-readable names, similar to the utility we used to decode ImageNet predictions to text labels back in chapter 8. Let’s visualize an example image to make this a little more concrete. We can define a function to draw an image with Matplotlib and another function to draw a labeled bounding box on this image. We will need both of these throughout the chapter. We can use the HSV colorspace as a simple trick to generate new colors for each new label we see. By fixing the saturation and brightness of the color and only updating its hue, we can generate bright new colors that stand out clearly from our image.

Listing 12.4 Visualizing a COCO image with box annotations

## Next page (PDF 360) text, for continuity
While it would be fun to train on all 18 gigabytes of our input data, we want to keep the examples in this book easily runnable on modest hardware. If we limit ourselves to only images with four or fewer boxes, we will make our training problem easier and a halve the data size. Let’s do this and shuffle our data—the images are grouped by object type, which would be terrible for training:

That’s it for data loading! Let’s start creating our YOLO model.

12.2.2 Creating a YOLO model

As mentioned previously, the YOLO model is a single stage detector. Rather than first attempting to identify all candidate objects in a scene and then classifying the object regions, YOLO will propose bounding boxes and object labels in one go.

Our model will divide an image up into a grid and predict two separate outputs at each grid location—a bounding box and a class label. In the original paper by Redmon et al.,3 the model actually predicted multiple boxes per grid location, but we keep things simple and just predict one box in each grid square. Most images will not have objects evenly distributed across a grid, and to account for this, the model will output a confidence score along with each box, as shown in figure 12.4. We’d like this confidence to be high when an object is detected at a location, and zero when there’s no object. Most grid locations will have no object and should report a near-zero confidence. Like many models in computer vision, the YOLO model uses a ConvNet backbone to obtain interesting high-level features for an input image, a concept we first explored in chapter 8. In their paper, the authors created their own backbone model and pretrained it with ImageNet for classification. Rather than do this ourselves, we can instead use KerasHub to load a pretrained backbone …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.373 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.365 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.357 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.364 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.372 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.369 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.347 — 11 Image segmentation › 11.3 Using a pretrained segmentation model
- PDF p.363 — 12 Object detection › 12.2 Training a YOLO model from scratch

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.364 (printed 340) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Let’s visualize our YOLO training data with our box drawing helpers (figure 12.5). We will draw the entire class activation map over our first input image4 and add the confidence score of a box along with its label.

4 Image from the COCO 2017 dataset, https://cocodataset.org/. Image from Flickr, http://farm9.staticflickr.com /8081/8387882360_5b97a233c4_z.jpg, CC BY 2.0 https://creativecommons.org/licenses/by/2.0/.

### PDF p.365 (printed 341) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Figure 12.5 YOLO outputs a bounding box prediction and class label for each image region.

### PDF p.370 (printed 346) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 
Figure 12.7 Every bounding box predicted by the YOLO model

Our model learns very low-confidence values because it has not yet learned to consistently locate objects in a scene. To further improve the model, we should try a number of things:

Train for more epochs

Use the whole COCO dataset

Data augmentation (e.g., translating and rotating input images and boxes)

Improve our class probability map for overlapping boxes

Predict multiple boxes per grid location using a bigger output grid

All of these would positively affect the model performance and get us closer to the original YOLO training recipe. However, this example is really just to get a feel for object detection training—training an accurate COCO detection model from scratch would take a large amount of compute and time. Instead, to get a sense of a better-performing detection model, let’s try using a pretrained object detection model called RetinaNet.

### PDF p.356 (printed 332) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2.1 Downloading the COCO dataset
Before we start creating our model, we need data to train with. The COCO dataset,1 short for Common Objects in Context, is one of the best-known and most commonly used object detection datasets. It consists of real-world photos from a number of different sources plus human-created annotations. This includes object labels, bounding box annotations, and full segmentation masks. We will disregard the segmentation masks and just use bounding boxes.

Let’s download the 2017 version of the COCO dataset. While not a large dataset by today’s standards, this 18 GB dataset will be the largest dataset we use in the book. If you are running the code as you read, this is a good chance to take a breather.

Listing 12.1 Downloading the 2017 COCO dataset

### PDF p.356 (printed 332) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2.1 Downloading the COCO dataset
1 The COCO 2017 detection dataset can be explored at https://cocodataset.org/. Most images in this chapter are from the dataset.

### PDF p.361 (printed 337) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Next, we can turn our backbone into a detection model by adding new layers for outputting box and class predictions. The setup proposed in the YOLO paper is quite simple. Take the output of a ConvNet backbone and feed it through two densely connected layers with an activation in the middle. Then, split the output. The first five numbers will be used for bounding box prediction (four for the box and one for the box confidence). The rest will be used for the class probability map shown in figure 12.4—a classification prediction for each grid location over all possible 91 labels.

Let’s write this out.

### PDF p.363 (printed 339) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2.3 Readying the COCO data for the YOLO model
Our model is relatively simple, but we still need to preprocess our inputs to align them with the prediction grid. Each grid detector will be responsible for detecting any boxes whose center falls inside the grid box. Our model will output five floats for the box `(x, y, w, h, confidence)`. The `x` and `y` will represent the object’s center relative to the bounds of the grid cell (from 0 to 1). The `w` and `h` will represent the object’s size relative to the image size.

We already have the right `w` and `h` values in our training data. However, we need to translate our `x` and `y` values to and from the grid. Let’s define two utilities:

### PDF p.357 (printed 333) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
We need to do some input massaging before we are ready to use this data. The first download gives us an unlabeled directory of all the COCO images. The second download includes all image metadata via a JSON file. COCO associates each image file with an ID, and each bounding box is paired with one of these IDs. We need to collate all box and image data together.

Each bounding box comes with `x, y, width, height` pixel coordinates starting at the top left corner of the image. As we load our data, we can rescale all bounding box coordinates so they are points in a `[0, 1]` unit square. This will make it easier to manipulate these boxes without needing to check the size of each input image.

Listing 12.2 Parsing the COCO data
