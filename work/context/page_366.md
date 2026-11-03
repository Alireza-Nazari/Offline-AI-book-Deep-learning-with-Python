# Generation bundle for page_366  (PDF page 366, printed page 342)

Page kind: content
Chapter: 12 Object detection
Section: 12.2 Training a YOLO model from scratch
Subsection: 12.2.4 Training the YOLO model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_366.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_366.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p366_e03] (body) Lastly, let’s use `tf.data` to load our image data. We will load our images from disk, apply our preprocessing, and batch them. We should also split a validation set to monitor training.
[p366_e04] (listing_caption) Listing 12.9 Creating a dataset to train on
[p366_e05] (code)
```
import tensorflow as tf
def load_image(path):
    x = tf.io.read_file(path)
    x = tf.image.decode_jpeg(x, channels=3)
    return preprocessor(x)
images = tf.data.Dataset.from_tensor_slices([x["path"] for x in metadata])
images = images.map(load_image, num_parallel_calls=8)
labels = {"box": box_array, "class": class_array}
labels = tf.data.Dataset.from_tensor_slices(labels)
dataset = tf.data.Dataset.zip(images, labels).batch(16).prefetch(2)
val_dataset, train_dataset = dataset.take(500), dataset.skip(500)
```
[p366_e06] (body) With that, our data is ready for training.
[p366_e07] (body) NOTE This training example shows clearly why a streaming library like `tf.data` is helpful. Loading all the images in this large dataset in one go would overwhelm our system memory (remember an image tensor is much larger than a compressed JPEG file). With `tf.data`, we can load our image data in batch by batch and release the memory when we are done, only mapping in the particular parts of the dataset we need at a given moment. The `prefetch(2)` call will cause `tf.data` to keep two batches buffered and ready before they are used so we don’t interrupt training each batch to load and resize more images.
[p366_e08] (heading_subsection) 12.2.4 Training the YOLO model
[p366_e09] (body) We have our model and our training data ready, but there’s one last element we need before we can actually run `fit()`: the loss function. Our model outputs predicted boxes and predicted grid labels. We saw in chapter 7 how we can define multiple losses for each output—Keras will simply sum the losses together during training. We can handle the classification loss with `sparse_categorical_crossentropy` as usual.
[p366_e10] (body) The box loss, however, needs some special consideration. The basic loss proposed by the YOLO authors is fairly simple. They use the sum-squared error of the difference between the target box parameters and the predicted ones. We will only compute this error for grid cells with actual boxes in the labeled data. The tricky part of the loss is the box confidence output. The authors wanted the confidence output to reflect not just the presence of an object, but also how good the predicted

## Code annotations printed next to listings (callouts, in order)
- Loads and resizes the
- model with tf.data
- Creates a merged
- dataset and batches it
- Splits off some validation data

## Listings
- Listing 12.9: Creating a dataset to train on (code elements p366_e05)

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

## Previous page (PDF 365) text, for continuity
Listing 12.8 Visualizing a YOLO target

Figure 12.5 YOLO outputs a bounding box prediction and class label for each image region.

## Next page (PDF 367) text, for continuity
box is. To create a smooth estimate of how good a box prediction is, the authors propose using the Intersection over Union (IoU) metric we saw last chapter. If a grid cell is empty, the predicted confidence at the location should be zero. However, if a grid cell contains an object, we can use the IoU score between the current box prediction and the actual box as the target confidence value. This way, as the model becomes better at predicting box locations, the IoU score and the learned confidence values will go up.

This calls for a custom loss function. We can start be defining a utility to compute IoU scores for target and predicted boxes.

Listing 12.10 Computing IoU for two boxes

Let’s use this utility to define our custom loss. Redmon et al. propose a couple of loss scaling tricks to improve the quality of training:

They scale up the box placement loss by a factor of five, so it becomes a more important part of overall training.

Since most grid cells are empty, they also scale down the confidence loss in empty locations by a factor of two. This keeps these zero confidence predictions from overwhelming the loss.

They take the square root of the width and height before computing the loss. This is to stop large boxes from mattering disproportionately more than small boxes. We will use a `sqrt` function that preserves the sign of the input, since our model might predict negative widths and heights at the start of training.

Let’s write this out.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.360 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.272 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.357 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.274 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.363 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.364 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.271 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.283 — 8 Image classification › 8.3 Using a pretrained model

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.360 (printed 336) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2.2 Creating a YOLO model
As mentioned previously, the YOLO model is a single stage detector. Rather than first attempting to identify all candidate objects in a scene and then classifying the object regions, YOLO will propose bounding boxes and object labels in one go.

Our model will divide an image up into a grid and predict two separate outputs at each grid location—a bounding box and a class label. In the original paper by Redmon et al.,3 the model actually predicted multiple boxes per grid location, but we keep things simple and just predict one box in each grid square. Most images will not have objects evenly distributed across a grid, and to account for this, the model will output a confidence score along with each box, as shown in figure 12.4. We’d like this confidence to be high when an object is detected at a location, and zero when there’s no object. Most grid locations will have no object and should report a near-zero confidence. Like many models in computer vision, the YOLO model uses a ConvNet backbone to obtain interesting high-level features for an input image, a concept we first explored in chapter 8. In their paper, the authors created their own backbone model and pretrained it with Image

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
Our model is relatively simple, but we still need to preprocess our inputs to align them with the prediction grid. Each grid detector will be responsible for detecting any boxes whose center falls inside the grid box. Our model will output five floats for the box `(x, y, w, h, confidence)`. The `x` and `y` will represent the object’s center relative to the bounds of the grid cell (from 0 to 1). The `w` and `h` will represent the object’s size relative to the image size.

We already have the right `w` and `h` values in our training data. However, we need to translate our `x` and `y` values to and from the grid. Let’s define two utilities:

### PDF p.368 (printed 344) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Listing 12.12 Training the YOLO model
model.compile(
    optimizer=keras.optimizers.Adam(2e-4),
    loss={"box": box_loss, "class": "sparse_categorical_crossentropy"},
)
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=4,
)

### PDF p.426 (printed 402) — 14 Text classification › 14.4 Set models › Preprocessing text efficiently
use it throughout this chapter—`keras.utils.text_dataset_from_directory()` will

load a `tf.data.Dataset`, and `map()` will transform our input data, for example, by applying a `TextVectorization` layer. `tf.data` works by running text preprocessing in parallel multiple CPU cores, which is generally sufficient to avoid bottlenecking accelerators during a training run. It is important to note that the code in this chapter is still multi-backend (in fact, we generated the outputs for this chapter using Jax). You can use `tf.data` with PyTorch, JAX, or TensorFlow itself—Keras will automatically convert input Tensors to the correct format for a given backend.

### PDF p.364 (printed 340) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Let’s rework our training data so it conforms to this new grid structure. We can create two arrays as long as our dataset with our grid:

The first will contain our class probability map. We will mark all grid cells that intersect with a bounding box with the correct label. To keep our code simple, we won’t worry about overlapping boxes.

The second will contain the boxes themselves. We will translate all boxes to the grid and label the correct grid cell with the coordinates for the box. The confidence for an actual box in our label data will always be one, and the confidence for all other locations will be zero.

Listing 12.7 Creating the YOLO targets

### PDF p.272 (printed 248) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › Understanding TensorFlow Dataset objects
TensorFlow makes available the `tf.data` API to create efficient input pipelines for machine learning models. Its core class is `tf.data.Dataset`.

The `Dataset` class can be used for data loading and preprocessing in any framework—not just TensorFlow. You can use it together with JAX or PyTorch. When you use it with a Keras model, it works the same, independently of the backend you’re currently using. A `Dataset` object is an iterator: you can use it in a `for` loop. It will typically return batches of input data and labels. You can pass a `Dataset` object directly to the `fit()` method of a Keras model. The `Dataset` class handles many key features that would otherwise be cumbersome to implement yourself, in particular parallelization of the preprocessing logic across multiple CPU cores, as well as asynchronous data prefetching (preprocessing the next batch of data while the previous one is being handled by the model, which keeps execution flowing without interruptions). The `Dataset` class also exposes a functional-style API for modifying datasets. Here’s a quick example: let’s create a `Dataset` instance from a NumPy array of random numbers. We’ll consider 1,000 samples, where 

### PDF p.368 (printed 344) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
We are finally ready to start training our YOLO model. Purely to keep this example short, we will skip over metrics. In a real-world setting, you’d want quite a few metrics here—such as the accuracy of the model at different confidence cutoff levels.

Listing 12.12 Training the YOLO model
