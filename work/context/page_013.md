# Generation bundle for page_013  (PDF page 13, printed page xiii)

Page kind: front_matter
Chapter: contents
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_013.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_013.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p013_e04] (body) 11.3
[p013_e05] (body) Using a pretrained segmentation model 318
[p013_e06] (quote) Downloading the Segment Anything Model 319 ■ How Segment Anything works 319 ■ Preparing a test image 321 ■ Prompting the model with a target point 323 ■ Prompting the model with a target box 327
[p013_e07] (body) 12
[p013_e08] (body) Object detection 329 12.1
[p013_e09] (body) Single-stage vs. two-stage object detectors 330
[p013_e10] (quote) Two-stage R-CNN detectors 330 ■ Single-stage detectors 332
[p013_e11] (body) 12.2
[p013_e12] (body) Training a YOLO model from scratch 332
[p013_e13] (quote) Downloading the COCO dataset 332 ■ Creating a YOLO model 336 ■ Readying the COCO data for the YOLO model 339 Training the YOLO model 342
[p013_e14] (body) 12.3
[p013_e15] (body) Using a pretrained RetinaNet detector 346 13 Timeseries forecasting 351 13.1 Different kinds of timeseries tasks 351 13.2 A temperature forecasting example 352
[p013_e16] (quote) Preparing the data 356 ■ A commonsense, non-machine-learning baseline 359 ■ Let’s try a basic machine learning model 360 Let’s try a 1D convolutional model 362
[p013_e17] (body) 13.3
[p013_e18] (body) Recurrent neural networks 364
[p013_e19] (quote) Understanding recurrent neural networks 365 ■ A recurrent layer in Keras 368 ■ Getting the most out of recurrent neural networks 372 ■ Using recurrent dropout to fight overfitting 372 Stacking recurrent layers 375 ■ Using bidirectional RNNs 377
[p013_e20] (body) 13.4
[p013_e21] (body) Going even further 379 14 Text classification 381 14.1 A brief history of natural language processing 381 14.2 Preparing text data 384
[p013_e22] (quote) Character and word tokenization 387 ■ Subword tokenization 390
[p013_e23] (body) 14.3
[p013_e24] (body) Sets vs. sequences 395
[p013_e25] (quote) Loading the IMDb classification dataset 396
[p013_e26] (body) 14.4
[p013_e27] (body) Set models 398
[p013_e28] (quote) Training a bag-of-words model 399 ■ Training a bigram model 403

## Chapter outline
Chapter: contents (PDF pages 8-16)

## Previous page (PDF 12) text, for continuity
8.2

Training a ConvNet from scratch on a small dataset 241

The relevance of deep learning for small-data problems 242 Downloading the data 242 ■ Building your model 245 Data preprocessing 247 ■ Using data augmentation 252

8.3

Using a pretrained model 256

Feature extraction with a pretrained model 256 ■ Fine-tuning a pretrained model 264

9

ConvNet architecture patterns 268 9.1

Modularity, hierarchy, and reuse 269 9.2 Residual connections 272 9.3 Batch normalization 276 9.4 Depthwise separable convolutions 278 9.5 Putting it together: A mini Xception-like model 280 9.6 Beyond convolution: Vision Transformers 282 10 Interpreting what ConvNets learn 284 10.1 Visualizing intermediate activations 285 10.2 Visualizing ConvNet filters 291

Gradient ascent in TensorFlow 294 ■ Gradient ascent in PyTorch 295 ■ Gradient ascent in JAX 295 ■ The filter visualization loop 296

10.3

Visualizing heatmaps of class activation 299

Getting the gradient of the top class: TensorFlow version 302 Getting the gradient of the top class: PyTorch version 302 Getting the gradient of the top class: JAX version 303 Displaying the class activation heatmap 304

10.4 Visualizing the latent space of a ConvNet 306 11 Image segmentation 308 11.1

Computer vision tasks 308

Types of image segmentation 310

11.2

Training a segmentation model from scratch 311

Downloading a segmentation dataset 311 ■ Building and training the segmentation model 314

## Next page (PDF 14) text, for continuity
14.5

Sequence models 405

Training a recurrent model 406 ■ Understanding word embeddings 409 ■ Using a word embedding 410 ■ Pretraining a word embedding 414 ■ Using the pretrained embedding for classification 418

15

Language models and the Transformer 421 15.1

The language model 421

Training a Shakespeare language model 422 ■ Generating Shakespeare 426

15.2

Sequence-to-sequence learning 428

English-to-Spanish translation 430 ■ Sequence-to-sequence learning with RNNs 432

15.3

The Transformer architecture 437

Dot-product attention 439 ■ Transformer encoder block 444 Transformer decoder block 446 ■ Sequence-to-sequence learning with a Transformer 448 ■ Embedding positional information 451

15.4

Classification with a pretrained Transformer 454

Pretraining a Transformer encoder 454 ■ Loading a pretrained Transformer 455 ■ Preprocessing IMDb movie reviews 458 Fine-tuning a pretrained Transformer 460

15.5

What makes the Transformer effective? 461 16 Text generation 466 16.1 A brief history of sequence generation 468 16.2 Training a mini-GPT 470

Building the model 473 ■ Pretraining the model 476 Generative decoding 478 ■ Sampling strategies 480

16.3

Using a pretrained LLM 484

Text generation with the Gemma model 485 ■ Instruction fine-tuning 488 ■ Low-Rank Adaptation (LoRA) 490

16.4

Going further with LLMs 495

Reinforcement Learning with Human Feedback (RLHF) 495 Multimodal LLMs 498 ■ Retrieval Augmented Generation (RAG) 501 ■ “Reasoning” models 502

16.5

Where are LLMs heading next? 504

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.640 — index › 
- PDF p.642 — index › 
- PDF p.637 — index › 
- PDF p.356 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.643 — index › 
- PDF p.645 — index › 
- PDF p.634 — index › 
- PDF p.370 — 12 Object detection › 12.3 Using a pretrained RetinaNet detector

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.640 (printed 616) — index ›  › O
building model 108–110

object detection 309, 329 IMDb dataset 106

creating YOLO models 336 preparing data 107 downloading COCO dataset 332 using trained model to generate predictions on new data 115 RetinaNet detector 347–349 validating approach 111–115 single-stage vs. two-stage detectors 330–332 MSE (mean squared error) 116, 134, 360, 602 training YOLO model 342 multiclass classification 105, 116–126 training YOLO models from scratch 332 building model 118 YOLO (You Only Look Once) model, preparing further experiments 126

COCO data for 339 generating predictions on new data 124 objective function 9 handling labels and loss 124 objective maximization/minimization 542 importance of having sufficiently large Occam’s razor 165 intermediate layers 125 OCR (Optical Character Recognition) 329 preparing data 118 on_batch_* method 211 Reuters dataset 116 one-hot encoding 118, 406 validating approach 120–123 on_epoch_* method 211 multi-GPU distributed training 548 ONNX (Open Neural Network Exchange) 185 MultiHeadAttention layer 443–447 on-the-fly adaptation 580 multi-hot encoding 406 on-the-fly recombination 578 multi-input, multi-output models 197 ops.argmax() function 302 training 198

### PDF p.645 (printed 621) — index ›  › Y
Wigner, Eugene 175

YOLO (You Only Look Once) models Word2Vec 462–464

creating 336 word embeddings 409, 410 preparing COCO data for object detection 339 pretraining 414 training 342 word tokenization 387–390 training from scratch 332 workflow, machine learning choosing measure of success 178

### PDF p.642 (printed 618) — index ›  › 
RetinaNet detector 347–349

self-supervised learning 11, 214, 516 return_sequences argument 368 semantic segmentation 310 reusability 11 SeparableConv1D layer 362 RLHF (Reinforcement Learning with Human SeparableConv2D layer 292, 362, 602 Feedback) 495–498 separable convolution 269 using chatbots trained with 496–498 seq2seq (sequence-to-sequence) learning 428–437 RMSE (root mean squared error) 207

English-to-Spanish translation 430–432 RNNs (recurrent neural networks) 352, 364–378, with RNNs 433–437 591, 604 sequence generation, history of 468–470 advanced features of 372 sequence models 395, 405 bidirectional RNNs 377–378 sequences, sets vs. 395–398 overview of 365–367 sequence-to-sequence language model 464 recurrent dropout 373–375 sequence-to-sequence learning 603 recurrent layers in Keras 368–372 with Transformer 449 seq2seq learning with 433–437 sequential stacking recurrent layers 375 architecture 191 RoBERTa 464 class 230 ROC AUC (receiver operating characteristic model 191, 192 curve) 178 Shakespeare language model 422–428

### PDF p.641 (printed 617) — index ›  › P
randomized A/B testing 188 padding 238, 239, 322 RandomSearch tuner 542 panoptic segmentation 310 R-CNN (region-based convolutional neural parameters, defined 539

networks), two-stage detectors 331 pattern recognition 586 real-world best practices, lower-precision computation 556–562 perception, defined 586 faster inference with quantization 561 pip install tensorboard command 213 float8 training 560 plot_model() utility 199 float16 inference 558 PositionalEmbedding layer 452, 474 positional encoding 603 floating-point precision 557 position embedding 451 loss scaling with mixed precision 560 positive prompt 532 mixed-precision training 559 predict() function 191, 301, 479, 489 recombination 578 prediction error or loss value 105 reconstruction loss 512 prediction or output 105 recurrent dropout 372–375 predict() method 115, 124, 203, 259, 260 recurrent layers in Keras 368–372 premade search spaces 546 recurrent models, training 406 preparing data 179–180 region proposals 331 preprocessor layer 489 regression 104, 126–134, 602 pretrained embedding, using for classification 418 building model 128 pretrained models 256–266 California Housing Price dataset 126 feature extraction with

### PDF p.374 (printed 350) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › Summary
Object detection identifies and locates objects within an image using bounding boxes. It’s basically a weaker version of image segmentation, but one that can be run much more efficiently.

There are two primary approaches to object detection: – Region-based Convolutional Neural Networks (R-CNNs), which are two-stage models that first propose regions of interest and then classify them with a ConvNet. – Single-stage detectors (like RetinaNet and YOLO), which perform both tasks in a single step. Single-stage detectors are generally faster and more efficient, making them suitable for real-time applications (e.g., self-driving cars).

YOLO computes two separate outputs simultaneously during training—possible bounding boxes and a class probability map: – Each candidate bounding box is paired with a confidence score, which is trained to target the Intersection over Union of the predicted box and the ground truth box. – The class probability map classifies different regions of an image as belonging to different objects.

RetinaNet builds on this idea by using a feature pyramid network (FPN), which combines features from multiple ConvNet layers to create feature maps at different scales, al

### PDF p.370 (printed 346) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 
Figure 12.7 Every bounding box predicted by the YOLO model

Our model learns very low-confidence values because it has not yet learned to consistently locate objects in a scene. To further improve the model, we should try a number of things:

Train for more epochs

Use the whole COCO dataset

Data augmentation (e.g., translating and rotating input images and boxes)

Improve our class probability map for overlapping boxes

Predict multiple boxes per grid location using a bigger output grid

All of these would positively affect the model performance and get us closer to the original YOLO training recipe. However, this example is really just to get a feel for object detection training—training an accurate COCO detection model from scratch would take a large amount of compute and time. Instead, to get a sense of a better-performing detection model, let’s try using a pretrained object detection model called RetinaNet.

### PDF p.356 (printed 332) — 12 Object detection › 12.2 Training a YOLO model from scratch › 12.1.2 Single-stage detectors
Around 2015, researchers and practitioners began experimenting with using a single deep learning model to jointly predict bounding box coordinates together with their labels, an architecture called a single-stage detector. The main families of single-stage detectors are RetinaNet, Single Shot MultiBox Detectors (SSD), and the You Only Look Once family, abbreviated as YOLO. Yes, like the meme. That’s on purpose.

Single-stage detectors, especially recent YOLO iterations, boast significantly faster speeds and greater efficiency than their two-stage counterparts, albeit with a minor potential tradeoff in accuracy. Nowadays, YOLO is arguably the most popular object detection model out there, especially when it comes to real-time applications. There’s usually a new version of it that comes out every year—interestingly, each new version tends to be developed by a separate organization. In the next section, we will build a simplified YOLO model from scratch.

### PDF p.370 (printed 346) — 12 Object detection › 12.3 Using a pretrained RetinaNet detector › 12.3 Using a pretrained RetinaNet detector
RetinaNet is also a single-stage object detector and operates on the same basic principles as the YOLO model. The biggest conceptual difference between our model and
