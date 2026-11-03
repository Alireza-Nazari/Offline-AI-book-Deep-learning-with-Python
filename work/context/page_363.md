# Generation bundle for page_363  (PDF page 363, printed page 339)

Page kind: content
Chapter: 12 Object detection
Section: 12.2 Training a YOLO model from scratch
Subsection: 12.2.3 Readying the COCO data for the YOLO model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_363.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_363.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p363_e03] (code)
```
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ get_item_6 (GetItem)  │ (None, 6, 6, 5)   │           0 │ reshape_3[0][0]    │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ activation_33         │ (None, 6, 6, 91)  │           0 │ get_item_7[0][0]   │
│ (Activation)          │                   │             │                    │
└───────────────────────┴───────────────────┴─────────────┴────────────────────┘
 Total params: 77,850,336 (296.98 MB)
 Trainable params: 77,797,088 (296.77 MB)
 Non-trainable params: 53,248 (208.00 KB)
```
[p363_e04] (body) Our backbone outputs have shape `(batch_size, 14, 14, 2048)`. That is 401,408 output floats per image, a bit too many to feed into our dense layers. We downscale the feature maps with a strided conv layer to `(batch_size, 6, 6, 512)` with a more manageable 18,432 floats per image.
[p363_e05] (body) Next, we can add our two densely connected layers. We flatten the entire feature map, pass it through a `Dense` with a `relu` activation and then pass it through a final `Dense` with our exact number of output predictions—5 for the bounding box and confidence and 91 for each object class at each grid location. Finally, we reshape the outputs back to a 6 × 6 grid and split our box and class predictions. As usual for our classification outputs, we apply a softmax. The box outputs will need more special consideration; we will cover this later. Looking good! Note that because we flatten the entire feature map through the classification layer, every grid detector can use the entire image’s features; there’s no locality constraint. This is by design—large objects will not stay contained to a single grid cell.
[p363_e06] (heading_subsection) 12.2.3 Readying the COCO data for the YOLO model
[p363_e07] (body) Our model is relatively simple, but we still need to preprocess our inputs to align them with the prediction grid. Each grid detector will be responsible for detecting any boxes whose center falls inside the grid box. Our model will output five floats for the box `(x, y, w, h, confidence)`. The `x` and `y` will represent the object’s center relative to the bounds of the grid cell (from 0 to 1). The `w` and `h` will represent the object’s size relative to the image size.
[p363_e08] (body) We already have the right `w` and `h` values in our training data. However, we need to translate our `x` and `y` values to and from the grid. Let’s define two utilities:
[p363_e09] (code)
```
def to_grid(box):
    x, y, w, h = box
    cx, cy = (x + w / 2) * grid_size, (y + h / 2) * grid_size
    ix, iy = int(cx), int(cy)
    return (ix, iy), (cx - ix, cy - iy, w, h)
def from_grid(loc, box):
    (xi, yi), (x, y, w, h) = loc, box
```

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

## Previous page (PDF 362) text, for continuity
Listing 12.6 Attaching a YOLO prediction head

We can get a better sense of the model by looking at the model summary:

## Next page (PDF 364) text, for continuity
Let’s rework our training data so it conforms to this new grid structure. We can create two arrays as long as our dataset with our grid:

The first will contain our class probability map. We will mark all grid cells that intersect with a bounding box with the correct label. To keep our code simple, we won’t worry about overlapping boxes.

The second will contain the boxes themselves. We will translate all boxes to the grid and label the correct grid cell with the coordinates for the box. The confidence for an actual box in our label data will always be one, and the confidence for all other locations will be zero.

Listing 12.7 Creating the YOLO targets

Let’s visualize our YOLO training data with our box drawing helpers (figure 12.5). We will draw the entire class activation map over our first input image4 and add the confidence score of a box along with its label.

4 Image from the COCO 2017 dataset, https://cocodataset.org/. Image from Flickr, http://farm9.staticflickr.com /8081/8387882360_5b97a233c4_z.jpg, CC BY 2.0 https://creativecommons.org/licenses/by/2.0/.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.361 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.360 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.372 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.365 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.367 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.366 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.370 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.369 — 12 Object detection › 12.2 Training a YOLO model from scratch

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.360 (printed 336) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2.2 Creating a YOLO model
As mentioned previously, the YOLO model is a single stage detector. Rather than first attempting to identify all candidate objects in a scene and then classifying the object regions, YOLO will propose bounding boxes and object labels in one go.

Our model will divide an image up into a grid and predict two separate outputs at each grid location—a bounding box and a class label. In the original paper by Redmon et al.,3 the model actually predicted multiple boxes per grid location, but we keep things simple and just predict one box in each grid square. Most images will not have objects evenly distributed across a grid, and to account for this, the model will output a confidence score along with each box, as shown in figure 12.4. We’d like this confidence to be high when an object is detected at a location, and zero when there’s no object. Most grid locations will have no object and should report a near-zero confidence. Like many models in computer vision, the YOLO model uses a ConvNet backbone to obtain interesting high-level features for an input image, a concept we first explored in chapter 8. In their paper, the authors created their own backbone model and pretrained it with Image

### PDF p.361 (printed 337) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Next, we can turn our backbone into a detection model by adding new layers for outputting box and class predictions. The setup proposed in the YOLO paper is quite simple. Take the output of a ConvNet backbone and feed it through two densely connected layers with an activation in the middle. Then, split the output. The first five numbers will be used for bounding box prediction (four for the box and one for the box confidence). The rest will be used for the class probability map shown in figure 12.4—a classification prediction for each grid location over all possible 91 labels.

Let’s write this out.

### PDF p.370 (printed 346) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 
Figure 12.7 Every bounding box predicted by the YOLO model

Our model learns very low-confidence values because it has not yet learned to consistently locate objects in a scene. To further improve the model, we should try a number of things:

Train for more epochs

Use the whole COCO dataset

Data augmentation (e.g., translating and rotating input images and boxes)

Improve our class probability map for overlapping boxes

Predict multiple boxes per grid location using a bigger output grid

All of these would positively affect the model performance and get us closer to the original YOLO training recipe. However, this example is really just to get a feel for object detection training—training an accurate COCO detection model from scratch would take a large amount of compute and time. Instead, to get a sense of a better-performing detection model, let’s try using a pretrained object detection model called RetinaNet.

### PDF p.366 (printed 342) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2.4 Training the YOLO model
We have our model and our training data ready, but there’s one last element we need before we can actually run `fit()`: the loss function. Our model outputs predicted boxes and predicted grid labels. We saw in chapter 7 how we can define multiple losses for each output—Keras will simply sum the losses together during training. We can handle the classification loss with `sparse_categorical_crossentropy` as usual.

The box loss, however, needs some special consideration. The basic loss proposed by the YOLO authors is fairly simple. They use the sum-squared error of the difference between the target box parameters and the predicted ones. We will only compute this error for grid cells with actual boxes in the labeled data. The tricky part of the loss is the box confidence output. The authors wanted the confidence output to reflect not just the presence of an object, but also how good the predicted

### PDF p.359 (printed 335) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Figure 12.3 YOLO outputs a bounding box prediction and class label for each image region.

2 Image from the COCO 2017 dataset, https://cocodataset.org/. Image from Flickr, http://farm8.staticflickr.com /7250/7520201840_3e01349e3f_z.jpg, CC BY 2.0 https://creativecommons.org/licenses/by/2.0/.

### PDF p.258 (printed 234) — 8 Image classification › 8.1 Introduction to convnets › 
After the last `Conv2D` layer, we end up with an output of shape `(3, 3, 256)`—a 3 × 3 feature map of 256 channels. The next step is to feed this output into a densely connected classifier like those you’re already familiar with: a stack of `Dense` layers. These classifiers process vectors, which are 1D, whereas the current output is a rank-3 tensor. To bridge the gap, we flatten the 3D outputs to 1D with a `GlobalAveragePooling2D` layer before adding the `Dense` layers. This layer will take the average of each 3 × 3 feature map in the tensor of shape `(3,` `3,` `256)`, resulting in an output vector of shape `(256,)`. Finally, we’ll do 10-way classification, so our last layer has 10 outputs and a softmax activation.

Now, let’s train the ConvNet on the MNIST digits. We’ll reuse a lot of the code from the MNIST example in chapter 2. Because we’re doing 10-way classification with a softmax output, we’ll use the categorical crossentropy loss, and because our labels are integers, we’ll use the sparse version, `sparse_categorical_crossentropy`.

Listing 8.3 Training the ConvNet on MNIST images

### PDF p.372 (printed 348) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 
You’ll note we pass an extra argument to specify the bounding box format. We can do this for most Keras models and layers that support bounding boxes. We pass `"rel_` `xywh"` to use the same format as we did for the YOLO model, so we can use the same box drawing utilities. Here, `rel` stands for relative to the image size (e.g., from [0, 1]). Let’s inspect the prediction we just made:

### PDF p.640 (printed 616) — index ›  › O
building model 108–110

object detection 309, 329 IMDb dataset 106

creating YOLO models 336 preparing data 107 downloading COCO dataset 332 using trained model to generate predictions on new data 115 RetinaNet detector 347–349 validating approach 111–115 single-stage vs. two-stage detectors 330–332 MSE (mean squared error) 116, 134, 360, 602 training YOLO model 342 multiclass classification 105, 116–126 training YOLO models from scratch 332 building model 118 YOLO (You Only Look Once) model, preparing further experiments 126

COCO data for 339 generating predictions on new data 124 objective function 9 handling labels and loss 124 objective maximization/minimization 542 importance of having sufficiently large Occam’s razor 165 intermediate layers 125 OCR (Optical Character Recognition) 329 preparing data 118 on_batch_* method 211 Reuters dataset 116 one-hot encoding 118, 406 validating approach 120–123 on_epoch_* method 211 multi-GPU distributed training 548 ONNX (Open Neural Network Exchange) 185 MultiHeadAttention layer 443–447 on-the-fly adaptation 580 multi-hot encoding 406 on-the-fly recombination 578 multi-input, multi-output models 197 ops.argmax() function 302 training 198
