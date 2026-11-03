# Generation bundle for page_012  (PDF page 12, printed page xii)

Page kind: front_matter
Chapter: contents
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_012.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_012.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p012_e04] (body) 8.2
[p012_e05] (body) Training a ConvNet from scratch on a small dataset 241
[p012_e06] (quote) The relevance of deep learning for small-data problems 242 Downloading the data 242 ■ Building your model 245 Data preprocessing 247 ■ Using data augmentation 252
[p012_e07] (body) 8.3
[p012_e08] (body) Using a pretrained model 256
[p012_e09] (quote) Feature extraction with a pretrained model 256 ■ Fine-tuning a pretrained model 264
[p012_e10] (body) 9
[p012_e11] (body) ConvNet architecture patterns 268 9.1
[p012_e12] (body) Modularity, hierarchy, and reuse 269 9.2 Residual connections 272 9.3 Batch normalization 276 9.4 Depthwise separable convolutions 278 9.5 Putting it together: A mini Xception-like model 280 9.6 Beyond convolution: Vision Transformers 282 10 Interpreting what ConvNets learn 284 10.1 Visualizing intermediate activations 285 10.2 Visualizing ConvNet filters 291
[p012_e13] (quote) Gradient ascent in TensorFlow 294 ■ Gradient ascent in PyTorch 295 ■ Gradient ascent in JAX 295 ■ The filter visualization loop 296
[p012_e14] (body) 10.3
[p012_e15] (body) Visualizing heatmaps of class activation 299
[p012_e16] (quote) Getting the gradient of the top class: TensorFlow version 302 Getting the gradient of the top class: PyTorch version 302 Getting the gradient of the top class: JAX version 303 Displaying the class activation heatmap 304
[p012_e17] (body) 10.4 Visualizing the latent space of a ConvNet 306 11 Image segmentation 308 11.1
[p012_e18] (body) Computer vision tasks 308
[p012_e19] (quote) Types of image segmentation 310
[p012_e20] (body) 11.2
[p012_e21] (body) Training a segmentation model from scratch 311
[p012_e22] (quote) Downloading a segmentation dataset 311 ■ Building and training the segmentation model 314

## Chapter outline
Chapter: contents (PDF pages 8-16)

## Previous page (PDF 11) text, for continuity
5.4

Improving generalization 158

Dataset curation 159 ■ Feature engineering 159 ■ Using early stopping 161 ■ Regularizing your model 161

6

The universal workflow of machine learning 171 6.1

Defining the task 172

Framing the problem 172 ■ Collecting a dataset 174 Understanding your data 178 ■ Choosing a measure of success 178

6.2

Developing a model 179

Preparing the data 179 ■ Choosing an evaluation protocol 180 Beating a baseline 181 ■ Scaling up: Developing a model that overfits 182 ■ Regularizing and tuning your model 183

6.3

Deploying your model 183

Explaining your work to stakeholders and setting expectations 184 Shipping an inference model 184 ■ Monitoring your model in the wild 188 ■ Maintaining your model 188

7

A deep dive on Keras 190 7.1

A spectrum of workflows 191 7.2 Different ways to build Keras models 192

The Sequential model 192 ■ The Functional API 195 Subclassing the Model class 202 ■ Mixing and matching different components 204 ■ Remember: Use the right tool for the job 205

7.3

Using built-in training and evaluation loops 206

Writing your own metrics 207 ■ Using callbacks 208 Writing your own callbacks 210 ■ Monitoring and visualization with TensorBoard 212

7.4

Writing your own training and evaluation loops 214

Training vs. inference 215 ■ Writing custom training step functions 216 ■ Low-level usage of metrics 221 ■ Using fit() with a custom training loop 222 ■ Handling metrics in a custom train_step() 226

8

Image classification 231 8.1

Introduction to ConvNets 232

The convolution operation 234 ■ The max-pooling operation 239

## Next page (PDF 13) text, for continuity
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

Going even further 379 14 Text classification 381 14.1 A brief history of natural language processing 381 14.2 Preparing text data 384

Character and word tokenization 387 ■ Subword tokenization 390

14.3

Sets vs. sequences 395

Loading the IMDb classification dataset 396

14.4

Set models 398

Training a bag-of-words model 399 ■ Training a bigram model 403

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.635 — index › 
- PDF p.309 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.323 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation
- PDF p.637 — index › 
- PDF p.315 — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters
- PDF p.644 — index › 
- PDF p.7 — brief contents › 
- PDF p.326 — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.637 (printed 613) — index ›  › 
distributed training, accessing multiple

building model 245 GPUs 551 ConvNets 232–241 multi-GPU training 548–550 data augmentation 252 grad attribute 217 data preprocessing 247, 248, 250 .grad() function 302 deep learning for small-data problems 242 gradient ascent downloading data 242 in input space 291 pretrained models 256–266 in JAX 295 training ConvNet from scratch on small in PyTorch 295

dataset 241 in TensorFlow 294 ImageClassifier 301 gradient-based optimization 39–50 ImageConverter layer 258 backpropagation algorithm 46–50 image generation 508 derivatives 41–43 deep learning for 509–518 stochastic gradient descent 43–46 diffusion models 518–530 gradient descent 40, 291, 571, 596 text-to-image models 531–537 tuning key parameters 154 image normalization layer 525 GradientTape 294, 302 image segmentation 308, 309 scope 216 computer vision tasks 308–311 ground-truth or annotations 105 preparing test images 321 GRU (Gated Recurrent Unit) 375, 424, 426, 428, prompting model with target points 323–327 439 training segmentation model from scratch layers 434, 444, 449, 604 311–318 types of 310

### PDF p.635 (printed 611) — index ›  › 
ConvNet filters 285

achievements of 12 architecture patterns 268, 272–276, 280–282 advantages of 597 convolution operation 235–239 AI winters 14 depthwise separable convolutions 278–280 artificial intelligence, machine learning,

and 2–3, 599 interpreting what they learn 284 depth of model 7 max-pooling operation 239–241 enabling technologies 598 modularity, hierarchy, and reuse 269 for image generation 509–518 training from scratch on small dataset 241 frameworks 61–62 visualizing filters 291–299 generative AI 11 visualizing heatmaps of class activation 299–306 guided program search 606 visualizing intermediate activations 285–291 image classification for small-data problems 242 visualizing latent space of 306 integrating modules into hybrid systems 590 ViTs (Vision Transformers) 282 Keras 206–214 convolutional base 257 limitations of 564–570, 605 convolution kernel 237 network architectures 600–605 convolution operation 235–239, 550 object detection, training YOLO model 342 border effects and padding 238 overview of 8, 597 strides 239 properties of 10 Core Knowledge priors 581 short-term hype 13–14 cosine distance 462 staying up to date 607–608 cost function 9 text classificatio

### PDF p.309 (printed 285) — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations › 
wide array of techniques has been developed for visualizing and interpreting these representations. We won’t survey all of them, but we’ll cover three of the most accessible and useful ones:

Visualizing intermediate ConvNet outputs (intermediate activations)—Useful for understanding how successive ConvNet layers transform their input, and for getting a first idea of the meaning of individual ConvNet filters

Visualizing ConvNets filters—Useful for understanding precisely what visual pattern or concept each filter in a ConvNet is receptive to

Visualizing heatmaps of class activation in an image—Useful for understanding which parts of an image were identified as belonging to a given class, thus allowing you to localize objects in images

Visualizing the way ConvNets learn semantically organized manifolds of images—Useful for detecting outliers and getting a sense for what images are seen by your model as semantically ambiguous, thus allowing you to clean and improve your training dataset as well as refine your evaluation dataset

For the first method—activation visualization—you’ll use the small ConvNet that you trained from scratch on the dogs-versus-cats classification problem in

### PDF p.315 (printed 291) — 10 Interpreting what ConvNets learn › 10.2 Visualizing ConvNet filters › 10.2 Visualizing ConvNet filters
Another easy way to inspect the filters learned by ConvNets is to display the visual pattern that each filter is meant to respond to. This can be done with gradient ascent in input space, applying gradient descent to the value of the input image of a ConvNet so as to maximize the response of a specific filter, starting from a blank input image. The resulting input image will be one that the chosen filter is maximally responsive to.

Let’s try this with the filters of the Xception model. The process is simple: we’ll build a loss function that maximizes the value of a given filter in a given convolution layer, and then we’ll use stochastic gradient descent to adjust the values of the input image so as to maximize this activation value. This will be your second example of a low-level

### PDF p.7 (printed vii) — brief contents ›  › brief contents
1

■ What is deep learning? 1

2

■ The mathematical building blocks of neural networks 16

3

■ Introduction to TensorFlow, PyTorch, JAX, and Keras 60

4

■ Classification and regression 104

5

■ Fundamentals of machine learning 136

6

■ The universal workflow of machine learning 171

7

■ A deep dive on Keras 190

8

■ Image classification 231

9

■ ConvNet architecture patterns 268

10

■ Interpreting what ConvNets learn 284

11

■ Image segmentation 308

12

■ Object detection 329

13

■ Timeseries forecasting 351

14

■ Text classification 381

15

■ Language models and the Transformer 421

16

■ Text generation 466

17

■ Image generation 508

18

■ Best practices for the real world 538

19

■ The future of AI 564

20

■ Conclusions 595

v

### PDF p.266 (printed 242) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › The relevance of deep learning for small-data problems
What qualifies as “enough samples” to train a model is relative—relative to the size and depth of the model you’re trying to train, for starters. It isn’t possible to train a ConvNet to solve a complex problem with just a few tens of samples, but a few hundred can potentially suffice if the model is small and well regularized and the task is simple. Because ConvNets learn local, translation-invariant features, they’re highly data efficient on perceptual problems. Training a ConvNet from scratch on a very small image dataset will still yield reasonable results despite a relative lack of data, without the need for any custom feature engineering. You’ll see this in action in this section.

What’s more, deep learning models are by nature highly repurposable: you can take, say, an image-classification or speech-to-text model trained on a large-scale dataset and reuse it on a significantly different problem with only minor changes. Specifically, in the case of computer vision, many pretrained classification models are publicly available for download and can be used to bootstrap powerful vision models out of very little data. This is one of the greatest strengths of deep learning: feature

### PDF p.323 (printed 299) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 10.3 Visualizing heatmaps of class activation
1 Ramprasaath R. Selvaraju, et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,” arxiv (2019), https://arxiv.org/abs/1610.02391.

### PDF p.644 (printed 620) — index ›  › V
training step functions 216

VAEs (variational autoencoders) 509–512 JAX 218

implementing with Keras 513–518 PyTorch 217 validation, K-fold 129–134 TensorFlow 216 validation_data argument 111, 250 training, validation, and test sets 149–152 validation set overfitting 545 iterated K-fold validation with shuffling 152 validation_split argument 112 K-fold validation 151 val_loss metric 250 simple hold-out validation 150 value-centric analogies 585 train_step() function 216, 222, 223, 230, 516 value normalization 179 handling metrics in custom train_step() value vectors 480 226–228 vanishing gradient problem 271–272, 369 Transformer 421, 465, 603 variable_dtype attribute 559 classification with pretrained 454–461 variable path 553, 554 decoder block 447 vector database 502 effectiveness of 462–464 vectorization 179, 598 embedding positional information 451 vector regression 105 encoder block 444 versatility, defined 11 fine-tuning pretrained 460 visual concepts, representation of 284–285 language models 421–428 VisualizationCallback callback 529 loading pretrained 455–457 visualizing ConvNet filters 291–299 preprocessing IMDb movie reviews 458–460 filter visualization loop 296–299 pre
