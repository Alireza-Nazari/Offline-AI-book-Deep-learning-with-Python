# Generation bundle for page_356  (PDF page 356, printed page 332)

Page kind: content
Chapter: 12 Object detection
Section: 12.2 Training a YOLO model from scratch
Subsection: 12.2.1 Downloading the COCO dataset
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_356.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_356.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p356_e03] (heading_subsection) 12.1.2 Single-stage detectors
[p356_e04] (body) Around 2015, researchers and practitioners began experimenting with using a single deep learning model to jointly predict bounding box coordinates together with their labels, an architecture called a single-stage detector. The main families of single-stage detectors are RetinaNet, Single Shot MultiBox Detectors (SSD), and the You Only Look Once family, abbreviated as YOLO. Yes, like the meme. That’s on purpose.
[p356_e05] (body) Single-stage detectors, especially recent YOLO iterations, boast significantly faster speeds and greater efficiency than their two-stage counterparts, albeit with a minor potential tradeoff in accuracy. Nowadays, YOLO is arguably the most popular object detection model out there, especially when it comes to real-time applications. There’s usually a new version of it that comes out every year—interestingly, each new version tends to be developed by a separate organization. In the next section, we will build a simplified YOLO model from scratch.
[p356_e06] (heading_section) 12.2 Training a YOLO model from scratch
[p356_e07] (body) Overall, building an object detector can be a bit of an undertaking—not that there’s anything theoretically complex about it. There’s just a lot of code needed to handle manipulating bounding boxes and predicted output. To keep things simple, we will recreate the very first YOLO model from 2015. There are 12 YOLO versions as of this writing, but the original is a bit simpler to work with.
[p356_e08] (heading_subsection) 12.2.1 Downloading the COCO dataset
[p356_e09] (body) Before we start creating our model, we need data to train with. The COCO dataset,1 short for Common Objects in Context, is one of the best-known and most commonly used object detection datasets. It consists of real-world photos from a number of different sources plus human-created annotations. This includes object labels, bounding box annotations, and full segmentation masks. We will disregard the segmentation masks and just use bounding boxes.
[p356_e10] (body) Let’s download the 2017 version of the COCO dataset. While not a large dataset by today’s standards, this 18 GB dataset will be the largest dataset we use in the book. If you are running the code as you read, this is a good chance to take a breather.
[p356_e11] (listing_caption) Listing 12.1 Downloading the 2017 COCO dataset
[p356_e12] (code)
```
import keras
import keras_hub
images_path = keras.utils.get_file(
    "coco",
    "http://images.cocodataset.org/zips/train2017.zip",
    extract=True,
```
[p356_e13] (footnote) 1 The COCO 2017 detection dataset can be explored at https://cocodataset.org/. Most images in this chapter are from the dataset.

## Listings
- Listing 12.1: Downloading the 2017 COCO dataset (code elements p356_e12)

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

## Previous page (PDF 355) text, for continuity
areas that look object-like. These boxes are called region proposals. This stage isn’t very smart, so at that point we aren’t quite sure whether the proposed regions do contain objects and, if so, what objects they contain.

That’s the job of the second stage—a ConvNet that looks at each region proposal and classifies it into a number of predetermined classes, just like the models you’ve seen in chapter 9 (see figure 12.2). Region proposals that have a low score across all classes considered are discarded. We are then left with a much smaller set of boxes, each with a high class-presence score for one particular class. Finally, bounding boxes around each object are further refined to eliminate duplicates and make each bounding box as precise as possible.

Figure 12.2 An R-CNN first extracts region proposals and then classifies the proposals with a ConvNet (a CNN).

In early R-CNN versions, the first stage was a heuristic model called Selective Search that used some definition of spatial consistency to identify object-like areas. Heuristic is a term you’ll hear quite a lot in machine learning—it simply means “a bundle of hard-coded rules someone made up.” It’s usually used in opposition to learned models (where the rules are automatically derived) or theory-derived models. In later versions of R-CNN, such as Faster-R-CNN, the box generation stage became a deep learning model, called a Region Proposal Network.

The two-stage approach of R-CNN works very well in practice, but it’s quite computationally expensive, most notably because it requires you to classify thousands of patches—for every single image you process. That makes it unsuitable for most real-time applications and for embedded systems. My take is that, in practical applications, you generally don’t ever need a …

## Next page (PDF 357) text, for continuity
We need to do some input massaging before we are ready to use this data. The first download gives us an unlabeled directory of all the COCO images. The second download includes all image metadata via a JSON file. COCO associates each image file with an ID, and each bounding box is paired with one of these IDs. We need to collate all box and image data together.

Each bounding box comes with `x, y, width, height` pixel coordinates starting at the top left corner of the image. As we load our data, we can rescale all bounding box coordinates so they are points in a `[0, 1]` unit square. This will make it easier to manipulate these boxes without needing to check the size of each input image.

Listing 12.2 Parsing the COCO data

Let’s take a look at the data we just loaded.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.370 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.360 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.359 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.374 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector
- PDF p.358 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.354 — 12 Object detection › 12.1 Single-stage vs. two-stage object detectors
- PDF p.13 — contents › 
- PDF p.372 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.370 (printed 346) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 
Figure 12.7 Every bounding box predicted by the YOLO model

Our model learns very low-confidence values because it has not yet learned to consistently locate objects in a scene. To further improve the model, we should try a number of things:

Train for more epochs

Use the whole COCO dataset

Data augmentation (e.g., translating and rotating input images and boxes)

Improve our class probability map for overlapping boxes

Predict multiple boxes per grid location using a bigger output grid

All of these would positively affect the model performance and get us closer to the original YOLO training recipe. However, this example is really just to get a feel for object detection training—training an accurate COCO detection model from scratch would take a large amount of compute and time. Instead, to get a sense of a better-performing detection model, let’s try using a pretrained object detection model called RetinaNet.

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

### PDF p.374 (printed 350) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › Summary
Object detection identifies and locates objects within an image using bounding boxes. It’s basically a weaker version of image segmentation, but one that can be run much more efficiently.

There are two primary approaches to object detection: – Region-based Convolutional Neural Networks (R-CNNs), which are two-stage models that first propose regions of interest and then classify them with a ConvNet. – Single-stage detectors (like RetinaNet and YOLO), which perform both tasks in a single step. Single-stage detectors are generally faster and more efficient, making them suitable for real-time applications (e.g., self-driving cars).

YOLO computes two separate outputs simultaneously during training—possible bounding boxes and a class probability map: – Each candidate bounding box is paired with a confidence score, which is trained to target the Intersection over Union of the predicted box and the ground truth box. – The class probability map classifies different regions of an image as belonging to different objects.

RetinaNet builds on this idea by using a feature pyramid network (FPN), which combines features from multiple ConvNet layers to create feature maps at different scales, al

### PDF p.640 (printed 616) — index ›  › O
building model 108–110

object detection 309, 329 IMDb dataset 106

creating YOLO models 336 preparing data 107 downloading COCO dataset 332 using trained model to generate predictions on new data 115 RetinaNet detector 347–349 validating approach 111–115 single-stage vs. two-stage detectors 330–332 MSE (mean squared error) 116, 134, 360, 602 training YOLO model 342 multiclass classification 105, 116–126 training YOLO models from scratch 332 building model 118 YOLO (You Only Look Once) model, preparing further experiments 126

COCO data for 339 generating predictions on new data 124 objective function 9 handling labels and loss 124 objective maximization/minimization 542 importance of having sufficiently large Occam’s razor 165 intermediate layers 125 OCR (Optical Character Recognition) 329 preparing data 118 on_batch_* method 211 Reuters dataset 116 one-hot encoding 118, 406 validating approach 120–123 on_epoch_* method 211 multi-GPU distributed training 548 ONNX (Open Neural Network Exchange) 185 MultiHeadAttention layer 443–447 on-the-fly adaptation 580 multi-hot encoding 406 on-the-fly recombination 578 multi-input, multi-output models 197 ops.argmax() function 302 training 198

### PDF p.360 (printed 336) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.2.2 Creating a YOLO model
As mentioned previously, the YOLO model is a single stage detector. Rather than first attempting to identify all candidate objects in a scene and then classifying the object regions, YOLO will propose bounding boxes and object labels in one go.

Our model will divide an image up into a grid and predict two separate outputs at each grid location—a bounding box and a class label. In the original paper by Redmon et al.,3 the model actually predicted multiple boxes per grid location, but we keep things simple and just predict one box in each grid square. Most images will not have objects evenly distributed across a grid, and to account for this, the model will output a confidence score along with each box, as shown in figure 12.4. We’d like this confidence to be high when an object is detected at a location, and zero when there’s no object. Most grid locations will have no object and should report a near-zero confidence. Like many models in computer vision, the YOLO model uses a ConvNet backbone to obtain interesting high-level features for an input image, a concept we first explored in chapter 8. In their paper, the authors created their own backbone model and pretrained it with Image

### PDF p.359 (printed 335) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Figure 12.3 YOLO outputs a bounding box prediction and class label for each image region.

2 Image from the COCO 2017 dataset, https://cocodataset.org/. Image from Flickr, http://farm8.staticflickr.com /7250/7520201840_3e01349e3f_z.jpg, CC BY 2.0 https://creativecommons.org/licenses/by/2.0/.

### PDF p.370 (printed 346) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 12.3 Using a pretrained RetinaNet detector
RetinaNet is also a single-stage object detector and operates on the same basic principles as the YOLO model. The biggest conceptual difference between our model and

### PDF p.371 (printed 347) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 
Feature pyramid networks can substantially boost performance by building effective features for both small and large objects in terms of pixel footprint. Recent versions of YOLO also use the same setup.

Let’s go ahead and try out the RetinaNet model, which was also trained on the COCO dataset. To make this a little more interesting, let’s try an image that is out-of-distribution for the model, the Pointillist painting A Sunday Afternoon on the Island of La Grande Jatte. We can start by downloading the image and converting it to a NumPy array:
