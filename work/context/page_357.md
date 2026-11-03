# Generation bundle for page_357  (PDF page 357, printed page 333)

Page kind: content
Chapter: 12 Object detection
Section: 12.2 Training a YOLO model from scratch
Subsection: 12.2.1 Downloading the COCO dataset
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_357.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_357.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p357_e03] (code)
```
)
annotations_path = keras.utils.get_file(
    "annotations",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    extract=True,
)
```
[p357_e04] (body) We need to do some input massaging before we are ready to use this data. The first download gives us an unlabeled directory of all the COCO images. The second download includes all image metadata via a JSON file. COCO associates each image file with an ID, and each bounding box is paired with one of these IDs. We need to collate all box and image data together.
[p357_e05] (body) Each bounding box comes with `x, y, width, height` pixel coordinates starting at the top left corner of the image. As we load our data, we can rescale all bounding box coordinates so they are points in a `[0, 1]` unit square. This will make it easier to manipulate these boxes without needing to check the size of each input image.
[p357_e06] (listing_caption) Listing 12.2 Parsing the COCO data
[p357_e07] (code)
```
import json
with open(f"{annotations_path}/annotations/instances_train2017.json", "r") as f:
    annotations = json.load(f)
images = {image["id"]: image for image in annotations["images"]}
def scale_box(box, width, height):
    scale = 1.0 / max(width, height)
    x, y, w, h = [v * scale for v in box]
    x += (height - width) * scale / 2 if height > width else 0
    y += (width - height) * scale / 2 if width > height else 0
    return [x, y, w, h]
metadata = {}
for annotation in annotations["annotations"]:
    id = annotation["image_id"]
    if id not in metadata:
        metadata[id] = {"boxes": [], "labels": []}
    image = images[id]
    box = scale_box(annotation["bbox"], image["width"], image["height"])
    metadata[id]["boxes"].append(box)
    metadata[id]["labels"].append(annotation["category_id"])
    metadata[id]["path"] = images_path + "/train2017/" + image["file_name"]
metadata = list(metadata.values())
```
[p357_e08] (body) Let’s take a look at the data we just loaded.

## Code annotations printed next to listings (callouts, in order)
- Sorts image metadata by ID
- Converts bounding box to
- coordinates on a unit square
- Aggregates all bounding box
- annotations by image ID

## Listings
- Listing 12.2: Parsing the COCO data (code elements p357_e07)

## Equation candidates (formulas written inline in the text)
- [p357_e03] )
annotations_path = keras.utils.get_file(
    "annotations",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    extract=True,
)

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

## Previous page (PDF 356) text, for continuity
12.1.2 Single-stage detectors

Around 2015, researchers and practitioners began experimenting with using a single deep learning model to jointly predict bounding box coordinates together with their labels, an architecture called a single-stage detector. The main families of single-stage detectors are RetinaNet, Single Shot MultiBox Detectors (SSD), and the You Only Look Once family, abbreviated as YOLO. Yes, like the meme. That’s on purpose.

Single-stage detectors, especially recent YOLO iterations, boast significantly faster speeds and greater efficiency than their two-stage counterparts, albeit with a minor potential tradeoff in accuracy. Nowadays, YOLO is arguably the most popular object detection model out there, especially when it comes to real-time applications. There’s usually a new version of it that comes out every year—interestingly, each new version tends to be developed by a separate organization. In the next section, we will build a simplified YOLO model from scratch.

12.2 Training a YOLO model from scratch

Overall, building an object detector can be a bit of an undertaking—not that there’s anything theoretically complex about it. There’s just a lot of code needed to handle manipulating bounding boxes and predicted output. To keep things simple, we will recreate the very first YOLO model from 2015. There are 12 YOLO versions as of this writing, but the original is a bit simpler to work with.

12.2.1 Downloading the COCO dataset

Before we start creating our model, we need data to train with. The COCO dataset,1 short for Common Objects in Context, is one of the best-known and most commonly used object detection datasets. It consists of real-world photos from a number of different sources plus human-created annotations. This includes object labels, boundin …

## Next page (PDF 358) text, for continuity
Listing 12.3 Inspecting the COCO data

We have 117,266 images. Each image can have anywhere from 1 to 63 objects with an associated bounding box. There are only 91 possible labels for objects, chosen by the COCO dataset creators.

We can use a KerasHub utility `keras_hub.utils.coco_id_to_name(id)` to map these integer labels to human-readable names, similar to the utility we used to decode ImageNet predictions to text labels back in chapter 8. Let’s visualize an example image to make this a little more concrete. We can define a function to draw an image with Matplotlib and another function to draw a labeled bounding box on this image. We will need both of these throughout the chapter. We can use the HSV colorspace as a simple trick to generate new colors for each new label we see. By fixing the saturation and brightness of the color and only updating its hue, we can generate bright new colors that stand out clearly from our image.

Listing 12.4 Visualizing a COCO image with box annotations

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.359 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.364 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.369 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.372 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.360 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.363 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.365 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.366 — 12 Object detection › 12.2 Training a YOLO model from scratch

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.359 (printed 335) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Figure 12.3 YOLO outputs a bounding box prediction and class label for each image region.

2 Image from the COCO 2017 dataset, https://cocodataset.org/. Image from Flickr, http://farm8.staticflickr.com /7250/7520201840_3e01349e3f_z.jpg, CC BY 2.0 https://creativecommons.org/licenses/by/2.0/.

### PDF p.363 (printed 339) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2.3 Readying the COCO data for the YOLO model
Our model is relatively simple, but we still need to preprocess our inputs to align them with the prediction grid. Each grid detector will be responsible for detecting any boxes whose center falls inside the grid box. Our model will output five floats for the box `(x, y, w, h, confidence)`. The `x` and `y` will represent the object’s center relative to the bounds of the grid cell (from 0 to 1). The `w` and `h` will represent the object’s size relative to the image size.

We already have the right `w` and `h` values in our training data. However, we need to translate our `x` and `y` values to and from the grid. Let’s define two utilities:

### PDF p.364 (printed 340) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Let’s visualize our YOLO training data with our box drawing helpers (figure 12.5). We will draw the entire class activation map over our first input image4 and add the confidence score of a box along with its label.

4 Image from the COCO 2017 dataset, https://cocodataset.org/. Image from Flickr, http://farm9.staticflickr.com /8081/8387882360_5b97a233c4_z.jpg, CC BY 2.0 https://creativecommons.org/licenses/by/2.0/.

### PDF p.370 (printed 346) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 
Figure 12.7 Every bounding box predicted by the YOLO model

Our model learns very low-confidence values because it has not yet learned to consistently locate objects in a scene. To further improve the model, we should try a number of things:

Train for more epochs

Use the whole COCO dataset

Data augmentation (e.g., translating and rotating input images and boxes)

Improve our class probability map for overlapping boxes

Predict multiple boxes per grid location using a bigger output grid

All of these would positively affect the model performance and get us closer to the original YOLO training recipe. However, this example is really just to get a feel for object detection training—training an accurate COCO detection model from scratch would take a large amount of compute and time. Instead, to get a sense of a better-performing detection model, let’s try using a pretrained object detection model called RetinaNet.

### PDF p.363 (printed 339) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2.3 Readying the COCO data for the YOLO model
def to_grid(box):
    x, y, w, h = box
    cx, cy = (x + w / 2) * grid_size, (y + h / 2) * grid_size
    ix, iy = int(cx), int(cy)
    return (ix, iy), (cx - ix, cy - iy, w, h)
def from_grid(loc, box):
    (xi, yi), (x, y, w, h) = loc, box

### PDF p.360 (printed 336) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2.2 Creating a YOLO model
As mentioned previously, the YOLO model is a single stage detector. Rather than first attempting to identify all candidate objects in a scene and then classifying the object regions, YOLO will propose bounding boxes and object labels in one go.

Our model will divide an image up into a grid and predict two separate outputs at each grid location—a bounding box and a class label. In the original paper by Redmon et al.,3 the model actually predicted multiple boxes per grid location, but we keep things simple and just predict one box in each grid square. Most images will not have objects evenly distributed across a grid, and to account for this, the model will output a confidence score along with each box, as shown in figure 12.4. We’d like this confidence to be high when an object is detected at a location, and zero when there’s no object. Most grid locations will have no object and should report a near-zero confidence. Like many models in computer vision, the YOLO model uses a ConvNet backbone to obtain interesting high-level features for an input image, a concept we first explored in chapter 8. In their paper, the authors created their own backbone model and pretrained it with Image

### PDF p.364 (printed 340) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Let’s rework our training data so it conforms to this new grid structure. We can create two arrays as long as our dataset with our grid:

The first will contain our class probability map. We will mark all grid cells that intersect with a bounding box with the correct label. To keep our code simple, we won’t worry about overlapping boxes.

The second will contain the boxes themselves. We will translate all boxes to the grid and label the correct grid cell with the coordinates for the box. The confidence for an actual box in our label data will always be one, and the confidence for all other locations will be zero.

Listing 12.7 Creating the YOLO targets

### PDF p.372 (printed 348) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 
You’ll note we pass an extra argument to specify the bounding box format. We can do this for most Keras models and layers that support bounding boxes. We pass `"rel_` `xywh"` to use the same format as we did for the YOLO model, so we can use the same box drawing utilities. Here, `rel` stands for relative to the image size (e.g., from [0, 1]). Let’s inspect the prediction we just made:
