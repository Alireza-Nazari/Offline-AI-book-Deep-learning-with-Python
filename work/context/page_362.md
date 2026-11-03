# Generation bundle for page_362  (PDF page 362, printed page 338)

Page kind: content
Chapter: 12 Object detection
Section: 12.2 Training a YOLO model from scratch
Subsection: 12.2.2 Creating a YOLO model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_362.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_362.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p362_e03] (listing_caption) Listing 12.6 Attaching a YOLO prediction head
[p362_e04] (code)
```
from keras import layers
grid_size = 6
num_labels = 91
inputs = keras.Input(shape=(image_size, image_size, 3))
x = backbone(inputs)
x = layers.Conv2D(512, (3, 3), strides=(2, 2))(x)
x = keras.layers.Flatten()(x)
x = layers.Dense(2048, activation="relu", kernel_initializer="glorot_normal")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(grid_size * grid_size * (num_labels + 5))(x)
x = layers.Reshape((grid_size, grid_size, num_labels + 5))(x)
box_predictions = x[..., :5]
class_predictions = layers.Activation("softmax")(x[..., 5:])
outputs = {"box": box_predictions, "class": class_predictions}
model = keras.Model(inputs, outputs)
```
[p362_e05] (body) We can get a better sense of the model by looking at the model summary:
[p362_e06] (code)
```
>>> model.summary()
Model: "functional_3"
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)          ┃ Output Shape      ┃     Param # ┃ Connected to       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ input_layer_7         │ (None, 448, 448,  │           0 │ -                  │
│ (InputLayer)          │ 3)                │             │                    │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ res_net_backbone_12   │ (None, 14, 14,    │  23,580,512 │ input_layer_7[0][… │
│ (ResNetBackbone)      │ 2048)             │             │                    │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ conv2d_3 (Conv2D)     │ (None, 6, 6, 512) │   9,437,696 │ res_net_backbone_… │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ flatten_3 (Flatten)   │ (None, 18432)     │           0 │ conv2d_3[0][0]     │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ dense_6 (Dense)       │ (None, 2048)      │  37,750,784 │ flatten_3[0][0]    │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ dropout_3 (Dropout)   │ (None, 2048)      │           0 │ dense_6[0][0]      │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ dense_7 (Dense)       │ (None, 3456)      │   7,081,344 │ dropout_3[0][0]    │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ reshape_3 (Reshape)   │ (None, 6, 6, 96)  │           0 │ dense_7[0][0]      │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ get_item_7 (GetItem)  │ (None, 6, 6, 91)  │           0 │ reshape_3[0][0]    │
```

## Code annotations printed next to listings (callouts, in order)
- Passes our flattened feature maps
- through two densely connected layers
- Makes our backbone
- outputs smaller and then
- flattens the output features
- Split box and class predictions
- Reshapes outputs to a 6 × 6 grid

## Listings
- Listing 12.6: Attaching a YOLO prediction head (code elements p362_e04)

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

## Previous page (PDF 361) text, for continuity
Figure 12.4 YOLO outputs as visualized in the first YOLO paper

Listing 12.5 Loading the ResNet model

Next, we can turn our backbone into a detection model by adding new layers for outputting box and class predictions. The setup proposed in the YOLO paper is quite simple. Take the output of a ConvNet backbone and feed it through two densely connected layers with an activation in the middle. Then, split the output. The first five numbers will be used for bounding box prediction (four for the box and one for the box confidence). The rest will be used for the class probability map shown in figure 12.4—a classification prediction for each grid location over all possible 91 labels.

Let’s write this out.

## Next page (PDF 363) text, for continuity
Our backbone outputs have shape `(batch_size, 14, 14, 2048)`. That is 401,408 output floats per image, a bit too many to feed into our dense layers. We downscale the feature maps with a strided conv layer to `(batch_size, 6, 6, 512)` with a more manageable 18,432 floats per image.

Next, we can add our two densely connected layers. We flatten the entire feature map, pass it through a `Dense` with a `relu` activation and then pass it through a final `Dense` with our exact number of output predictions—5 for the bounding box and confidence and 91 for each object class at each grid location. Finally, we reshape the outputs back to a 6 × 6 grid and split our box and class predictions. As usual for our classification outputs, we apply a softmax. The box outputs will need more special consideration; we will cover this later. Looking good! Note that because we flatten the entire feature map through the classification layer, every grid detector can use the entire image’s features; there’s no locality constraint. This is by design—large objects will not stay contained to a single grid cell.

12.2.3 Readying the COCO data for the YOLO model

Our model is relatively simple, but we still need to preprocess our inputs to align them with the prediction grid. Each grid detector will be responsible for detecting any boxes whose center falls inside the grid box. Our model will output five floats for the box `(x, y, w, h, confidence)`. The `x` and `y` will represent the object’s center relative to the bounds of the grid cell (from 0 to 1). The `w` and `h` will represent the object’s size relative to the image size.

We already have the right `w` and `h` values in our training data. However, we need to translate our `x` and `y` values to and from the grid. Let’s define two utilities:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.264 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.220 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.270 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.365 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.364 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.257 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.539 — 17 Image generation › 17.1 Deep learning for image generation
- PDF p.369 — 12 Object detection › 12.2 Training a YOLO model from scratch

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.356 (printed 332) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2 Training a YOLO model from scratch
Overall, building an object detector can be a bit of an undertaking—not that there’s anything theoretically complex about it. There’s just a lot of code needed to handle manipulating bounding boxes and predicted output. To keep things simple, we will recreate the very first YOLO model from 2015. There are 12 YOLO versions as of this writing, but the original is a bit simpler to work with.

### PDF p.368 (printed 344) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
We are finally ready to start training our YOLO model. Purely to keep this example short, we will skip over metrics. In a real-world setting, you’d want quite a few metrics here—such as the accuracy of the model at different confidence cutoff levels.

Listing 12.12 Training the YOLO model

### PDF p.370 (printed 346) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 
Figure 12.7 Every bounding box predicted by the YOLO model

Our model learns very low-confidence values because it has not yet learned to consistently locate objects in a scene. To further improve the model, we should try a number of things:

Train for more epochs

Use the whole COCO dataset

Data augmentation (e.g., translating and rotating input images and boxes)

Improve our class probability map for overlapping boxes

Predict multiple boxes per grid location using a bigger output grid

All of these would positively affect the model performance and get us closer to the original YOLO training recipe. However, this example is really just to get a feel for object detection training—training an accurate COCO detection model from scratch would take a large amount of compute and time. Instead, to get a sense of a better-performing detection model, let’s try using a pretrained object detection model called RetinaNet.

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

### PDF p.369 (printed 345) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Listing 12.13 Training the YOLO model
x, y = next(iter(val_dataset.rebatch(1)))
preds = model.predict(x)
boxes = preds["box"][0]
classes = np.argmax(preds["class"][0], axis=-1)
path = metadata[0]["path"]
draw_prediction(path, boxes, classes, cutoff=0.1)

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

### PDF p.369 (printed 345) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Listing 12.13 Training the YOLO model

### PDF p.360 (printed 336) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
That’s it for data loading! Let’s start creating our YOLO model.
