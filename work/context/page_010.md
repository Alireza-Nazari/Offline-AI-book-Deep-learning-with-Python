# Generation bundle for page_010  (PDF page 10, printed page x)

Page kind: front_matter
Chapter: contents
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_010.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_010.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p010_e04] (body) 3.6
[p010_e05] (body) Introduction to Keras 90
[p010_e06] (quote) First steps with Keras 91 ■ Layers: The building blocks of deep learning 92 ■ From layers to models 96 ■ The “compile” step: Configuring the learning process 97 ■ Picking a loss function 99 Understanding the fit method 100 ■ Monitoring loss and metrics on validation data 101 ■ Inference: Using a model after training 102
[p010_e07] (body) 4
[p010_e08] (body) Classification and regression 104 4.1
[p010_e09] (body) Classifying movie reviews: A binary classification example 106
[p010_e10] (quote) The IMDb dataset 106 ■ Preparing the data 107 ■ Building your model 108 ■ Validating your approach 111 ■ Using a trained model to generate predictions on new data 115 Further experiments 115 ■ Wrapping up 116
[p010_e11] (body) 4.2
[p010_e12] (body) Classifying newswires: A multiclass classification example 116
[p010_e13] (quote) The Reuters dataset 116 ■ Preparing the data 118 ■ Building your model 118 ■ Validating your approach 120 ■ Generating predictions on new data 124 ■ A different way to handle the labels and the loss 124 ■ The importance of having sufficiently large intermediate layers 125 ■ Further experiments 125 Wrapping up 126
[p010_e14] (body) 4.3
[p010_e15] (body) Predicting house prices: A regression example 126
[p010_e16] (quote) The California Housing Price dataset 126 ■ Preparing the data 128 ■ Building your model 128 ■ Validating your approach using K-fold validation 129 ■ Generating predictions on new data 134 ■ Wrapping up 134
[p010_e17] (body) 5
[p010_e18] (body) Fundamentals of machine learning 136 5.1
[p010_e19] (body) Generalization: The goal of machine learning 136
[p010_e20] (quote) Underfitting and overfitting 137 ■ The nature of generalization in deep learning 143
[p010_e21] (body) 5.2
[p010_e22] (body) Evaluating machine-learning models 149
[p010_e23] (quote) Training, validation, and test sets 149 ■ Beating a common-sense baseline 152 ■ Things to keep in mind about model evaluation 152
[p010_e24] (body) 5.3
[p010_e25] (body) Improving model fit 153
[p010_e26] (quote) Tuning key gradient descent parameters 153 ■ Using better architecture priors 155 ■ Increasing model capacity 155

## Chapter outline
Chapter: contents (PDF pages 8-16)

## Previous page (PDF 9) text, for continuity
2

The mathematical building blocks of neural networks 16 2.1

A first look at a neural network 17 2.2 Data representations for neural networks 21

Scalars (rank-0 tensors) 22 ■ Vectors (rank-1 tensors) 22 Matrices (rank-2 tensors) 22 ■ Rank-3 tensors and higher-rank tensors 23 ■ Key attributes 23 ■ Manipulating tensors in NumPy 25 ■ The notion of data batches 25 Real-world examples of data tensors 26

2.3

The gears of neural networks: Tensor operations 28

Element-wise operations 29 ■ Broadcasting 30 Tensor product 32 ■ Tensor reshaping 34 ■ Geometric interpretation of tensor operations 35 ■ A geometric interpretation of deep learning 38

2.4

The engine of neural networks: Gradient-based optimization 39

What’s a derivative? 41 ■ Derivative of a tensor operation: The gradient 42 ■ Stochastic gradient descent 43 ■ Chaining derivatives: The Backpropagation algorithm 46

2.5

Looking back at our first example 51

Reimplementing our first example from scratch 53 Running one training step 55 ■ The full training loop 57 Evaluating the model 58

3

Introduction to TensorFlow, PyTorch, JAX, and Keras 60 3.1

A brief history of deep learning frameworks 61 3.2 How these frameworks relate to each other 63 3.3 Introduction to TensorFlow 63

First steps with TensorFlow 64 ■ An end-to-end example: A linear classifier in pure TensorFlow 69 ■ What makes the TensorFlow approach unique 74

3.4

Introduction to PyTorch 74

First steps with PyTorch 75 ■ An end-to-end example: A linear classifier in pure PyTorch 78 ■ What makes the PyTorch approach unique 81

3.5

Introduction to JAX 82

First steps with JAX 82 ■ Tensors in JAX 83 ■ Random number generation in JAX 83 ■ An end-to-end example: A linear classifier in pure JAX 88 ■ What makes the JAX approach unique 90

## Next page (PDF 11) text, for continuity
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

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.160 — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning
- PDF p.640 — index › 
- PDF p.7 — brief contents › 
- PDF p.193 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.179 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.639 — index › 
- PDF p.21 — about this book › How this book is organized: A road map
- PDF p.623 — 20 Conclusions › 20.1 Key concepts in review

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.641 (printed 617) — index ›  › P
randomized A/B testing 188 padding 238, 239, 322 RandomSearch tuner 542 panoptic segmentation 310 R-CNN (region-based convolutional neural parameters, defined 539

networks), two-stage detectors 331 pattern recognition 586 real-world best practices, lower-precision computation 556–562 perception, defined 586 faster inference with quantization 561 pip install tensorboard command 213 float8 training 560 plot_model() utility 199 float16 inference 558 PositionalEmbedding layer 452, 474 positional encoding 603 floating-point precision 557 position embedding 451 loss scaling with mixed precision 560 positive prompt 532 mixed-precision training 559 predict() function 191, 301, 479, 489 recombination 578 prediction error or loss value 105 reconstruction loss 512 prediction or output 105 recurrent dropout 372–375 predict() method 115, 124, 203, 259, 260 recurrent layers in Keras 368–372 premade search spaces 546 recurrent models, training 406 preparing data 179–180 region proposals 331 preprocessor layer 489 regression 104, 126–134, 602 pretrained embedding, using for classification 418 building model 128 pretrained models 256–266 California Housing Price dataset 126 feature extraction with

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

### PDF p.640 (printed 616) — index ›  › O
building model 108–110

object detection 309, 329 IMDb dataset 106

creating YOLO models 336 preparing data 107 downloading COCO dataset 332 using trained model to generate predictions on new data 115 RetinaNet detector 347–349 validating approach 111–115 single-stage vs. two-stage detectors 330–332 MSE (mean squared error) 116, 134, 360, 602 training YOLO model 342 multiclass classification 105, 116–126 training YOLO models from scratch 332 building model 118 YOLO (You Only Look Once) model, preparing further experiments 126

COCO data for 339 generating predictions on new data 124 objective function 9 handling labels and loss 124 objective maximization/minimization 542 importance of having sufficiently large Occam’s razor 165 intermediate layers 125 OCR (Optical Character Recognition) 329 preparing data 118 on_batch_* method 211 Reuters dataset 116 one-hot encoding 118, 406 validating approach 120–123 on_epoch_* method 211 multi-GPU distributed training 548 ONNX (Open Neural Network Exchange) 185 MultiHeadAttention layer 443–447 on-the-fly adaptation 580 multi-hot encoding 406 on-the-fly recombination 578 multi-input, multi-output models 197 ops.argmax() function 302 training 198

### PDF p.639 (printed 615) — index ›  › M
model interpretability 299 machine learning (ML) 2, 3, 136, 171 model.layers property 200 evaluating models 149–153 ModelParallel distribution instance 556 generalization 137–148 model parallelism 548, 549 improving generalization 158–168 models improving model fit 153–158

best practices for 539–548 markets and 379 building in Keras 192 universal workflow of 179–189 dropout 167–168 workflow 599 reducing size of 161–164 workflow of, choosing measure of success 178 training recurrent 406 MAE (mean absolute error) 129, 131, 359 training recurrent model 406

### PDF p.160 (printed 136) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › Generalization: The goal of machine learning
In the three examples presented in chapter 4—predicting movie reviews, topic classification, and house-price regression—we split the data into a training set, a

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

### PDF p.638 (printed 614) — index ›  › K
classification with pretrained Transformer

454–461 Kaggle 607 dot-product attention 439 kagglehub package 242, 485 embedding positional information 451 kaleidoscope hypothesis 577 generating Shakespeare 426–428 Keras 60, 190 loss function 462 building models in 192 seq2seq learning 428–437 ecosystem 608 sequence-to-sequence learning with evaluation loops 206–214 Transformer 449 fit() function 222– 225 training Shakespeare language model 422–426 Functional API 195, 197–200 Transformer 444, 447, 462–464 implementing VAEs with 513–518 Transformer architecture 437 low-level usage of metrics 221 last_conv_layer_output 302 mixing and matching different components 204 latent space 509, 568 relationship between frameworks 63 of images 509 Sequential model 192 subclassing Model class 202 of text-to-image models 533–537 summary 230 visualizing 306 training and evaluation loops 214 lateral connections 347 training loops 206–214 Layer API 191 training step functions 216–218 layered representations learning 7 using right tool for job 205 LayerNormalization layer 445 workflows 191 layer_output layer 167 Keras API, recurrent layers in 368–372 layers 596 keras.callbacks.Callback class 210 LayoutM

### PDF p.21 (printed xxi) — about this book › How this book is organized: A road map › How this book is organized: A road map
This book contains 20 chapters, organized as follows:

Chapters 1–7: Foundations of deep learning—These chapters give you a solid grounding in deep learning. We discuss the history of the field, provide an introduction to coding with the major deep learning frameworks currently available (JAX, PyTorch, and TensorFlow), and outline a practical guide to applying machine learning to real-world problems. We look closely at the Keras library for training machine learning models.

Chapters 8–12: Computer vision—These chapters cover machine learning on image data. We train models to classify entire images and models to recognize and separate out individual objects in images. We look closely at ConvNets, a model architecture well suited to images and the first big success of the deep learning era.

Chapter 13: Timeseries forecasting—This chapter covers working with timeseries data. We take data that is measured over time, like the weather or sales data, and use past patterns to predict future trends.

Chapters 14–15: Natural language processing—These chapters cover applying deep learning to text. We build sentiment classifiers to predict whether text is positive or negative. We build an En
