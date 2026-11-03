# Generation bundle for page_645  (PDF page 645, printed page 621)

Page kind: index
Chapter: index
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_645.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_645.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p645_e03] (heading_subsubsection) W
[p645_e04] (body) investing in data-annotation infrastructure 175 non-representative data 176 weights 9, 39, 596 task definition 172–178 pruning 188 quantization 188 understanding data 178 regularization 165–166
[p645_e05] (heading_subsubsection) X
[p645_e06] (body) regularizer instances 165 weight.value.grad method 217
[p645_e07] (body) Xception model 258, 299 wget shell utility 311 while statements 591
[p645_e08] (heading_subsubsection) Y
[p645_e09] (body) Wigner, Eugene 175
[p645_e10] (body) YOLO (You Only Look Once) models Word2Vec 462–464
[p645_e11] (body) creating 336 word embeddings 409, 410 preparing COCO data for object detection 339 pretraining 414 training 342 word tokenization 387–390 training from scratch 332 workflow, machine learning choosing measure of success 178
[p645_e12] (heading_subsubsection) Z
[p645_e13] (body) collecting dataset 174–177 framing problem 172–174
[p645_e14] (body) zero_grad() method 217

## Chapter outline
Chapter: index (PDF pages 633-650)

## Previous page (PDF 644) text, for continuity
top_pred_index 302

tuning model 183 torch.nn.Module class 217 Turing test, defined 3 TPU (Tensor Processing Unit) training 555 two-stage R-CNN detectors 331 step fusing 556 trainable

U

attribute 262

underfitting 137–142 parameters 39

ambiguous features 139 weights 215 noisy training data 138 training rare features and spurious correlations 140–142 diffusion models 525 U-Net denoising autoencoder 522 inference vs. 215 unigrams 403 lower-precision computation 556–562 universal workflow of machine learning training arguments 215, 218 deploying model 183–189 training data 147 developing model 179–183 training loops 10, 39, 206–214 update_state() method 207, 222 callbacks 208, 210 upsampling stage 522 monitoring and visualization with use_casual_mask 448 TensorBoard 212 using fit() with custom training loops 222–225

V

training step functions 216

VAEs (variational autoencoders) 509–512 JAX 218

implementing with Keras 513–518 PyTorch 217 validation, K-fold 129–134 TensorFlow 216 validation_data argument 111, 250 training, validation, and test sets 149–152 validation set overfitting 545 iterated K-fold validation with shuffling 152 validation_split argument 112 K-fold validation 151 val_loss metric 250 simple hold-out validation 150 value-centric analogies 585 train_step() function 216, 222, 223, 230, 516 value normalization 179 handling metrics in custom train_step() value vectors 480 226–228 vanishing gradient problem 271–272, 369 Transformer 421, 465, 603 variable_dtype attribute 559 classification with pretrained 454–461 variable path 553, 554 decoder block 447 vector database 502 effectiveness of 462–464 vectorization 179, 598 embedding positional information 451 vector regression 105 encoder block 444 versatility, defined 11 fine-tuning pretrained 460 visual conc …

## Next page (PDF 646) text, for continuity
RELATED MANNING TITLES

Grokking Artificial Intelligence Algorithms by Rishal Hurbans

ISBN 9781617296185 392 pages, $59.99 July 2020

Deep Learning and the Game of Go by Max Pumperla and Kevin Ferguson Foreword by Thore Graepel

ISBN: 9781617295324 384 pages, $54.99 January 2019

Transfer Learning for Natural Language Processing by Paul Azunre

ISBN: 9781617297267 272 pages, $49.99 July 2021

Deep Learning with JavaScript by Shanqing Cai, Stanley Bileschi, Eric D. Nielsen, and François Chollet Foreword by Nikhil Thorat and Daniel Smilkov

ISBN 9781617296178 560 pages, $49.99 January 2020

For ordering information, go to www.manning.com

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.640 — index › 
- PDF p.641 — index › 
- PDF p.643 — index › 
- PDF p.13 — contents › 
- PDF p.11 — contents › 
- PDF p.639 — index › 
- PDF p.637 — index › 
- PDF p.636 — index › 

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

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

### PDF p.11 (printed xi) — contents ›  › 
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

Writing your own metrics 207 ■ Using callbacks 208 Writing your own callbacks 210 ■ Monitoring and visualization with Tens

### PDF p.640 (printed 616) — index ›  › O
building model 108–110

object detection 309, 329 IMDb dataset 106

creating YOLO models 336 preparing data 107 downloading COCO dataset 332 using trained model to generate predictions on new data 115 RetinaNet detector 347–349 validating approach 111–115 single-stage vs. two-stage detectors 330–332 MSE (mean squared error) 116, 134, 360, 602 training YOLO model 342 multiclass classification 105, 116–126 training YOLO models from scratch 332 building model 118 YOLO (You Only Look Once) model, preparing further experiments 126

COCO data for 339 generating predictions on new data 124 objective function 9 handling labels and loss 124 objective maximization/minimization 542 importance of having sufficiently large Occam’s razor 165 intermediate layers 125 OCR (Optical Character Recognition) 329 preparing data 118 on_batch_* method 211 Reuters dataset 116 one-hot encoding 118, 406 validating approach 120–123 on_epoch_* method 211 multi-GPU distributed training 548 ONNX (Open Neural Network Exchange) 185 MultiHeadAttention layer 443–447 on-the-fly adaptation 580 multi-hot encoding 406 on-the-fly recombination 578 multi-input, multi-output models 197 ops.argmax() function 302 training 198

### PDF p.438 (printed 414) — 14 Text classification › 14.5 Sequence models › 14.5.4 Pretraining a word embedding
The last decade of rapid advancement in NLP has coincided with the rise of pretraining as the dominant approach for text modeling problems. Once we move past simple set-based regression models to sequence models with millions or even billions of parameters, text models become incredibly data-hungry. We are usually limited by our ability to find labeled examples for a particular problem in the text domain.

The idea is to devise an unsupervised task to train model parameters that do not need labeled data. Pretraining data can be text in a similar domain to our final task, or even just arbitrary text in the languages we are interested in working with. Pretraining allows us to learn general patterns in language, effectively priming our model before we specialize it to the final task we are interested in. Word embeddings were one of the first big successes with text pretraining, and we will show how to pretrain a word embedding in this section. Remember the `unsup/` directory we ignored in our IMDb dataset preparation? It contains another 25,000 reviews— the same size as our training data. We will combine all our training data together and show how to pretrain the parameters of an `Emb

### PDF p.643 (printed 619) — index ›  › 
stemming, defined 386–387

TensorFlow Serving 185 step fusing 556 tensor operations, derivatives of 42–43 stochastic gradient descent 43–46 tensors, defined 63 strided convolutions 239 test images, preparing 321 subclassing Model class 202 test-time training 582 rewriting previous example as subclassed text classification 381 model 202

augmenting text data 413 what subclassed models don't support 204 NLP (natural language processing) 382–384 subgraph isomorphism 586 preparing text data 384–394 subword tokenization 390–394 pretraining word embeddings 414 summary() method 193 sequence models 405 supervised sets vs. sequences 395–398 fine-tuning 495 setting models 398–404 learning 214 training recurrent model 406 symbolic AI 3 using pretrained embedding for symbolic tensor 196

classification 418 symbols 584–594 word embeddings 409, 410 blending deep learning and program text-conditioning 509 synthesis 590–592 text_dataset_from_directory utility 397 cognition as combination of both kinds of abstraction 587 text_embeddings input 531 deep learning and 588 text generation 466 long-term vision 594 history of sequence generation 468–470 modular component recombination and lifelong LLMs (l

### PDF p.640 (printed 616) — index ›  › 
weight regularization 165–166

convolutional networks 602 model.stateless_call() method 222 densely connected networks 601 model subclassing 192 RNNs 604 model training, scaling up with multiple Transformers 603 devices 548–556 networks, reducing size of 161–164 distributed training 551–555 neural attention 603 multi-GPU training 548–550 neural networks 7 TPU training 555 gradient-based optimization 39–50 model.zero_grad() method 217 modular component recombination and lifelong mathematical building blocks of 16 learning 592 n-gram 403 modularity, hierarchy, and reuse 269 NLP (natural language processing) 381–384, 421 modular recombination 607 noise_rates input 522 Module API 217 noisy_images input 531 modules 269 noisy training data 138 momentum, defined 45 non-trainable weights 216 monitoring, with TensorBoard 212 movie reviews, classifying 106–116

### PDF p.10 (printed x) — contents ›  › 
3.6

Introduction to Keras 90

First steps with Keras 91 ■ Layers: The building blocks of deep learning 92 ■ From layers to models 96 ■ The “compile” step: Configuring the learning process 97 ■ Picking a loss function 99 Understanding the fit method 100 ■ Monitoring loss and metrics on validation data 101 ■ Inference: Using a model after training 102

4

Classification and regression 104 4.1

Classifying movie reviews: A binary classification example 106

The IMDb dataset 106 ■ Preparing the data 107 ■ Building your model 108 ■ Validating your approach 111 ■ Using a trained model to generate predictions on new data 115 Further experiments 115 ■ Wrapping up 116

4.2

Classifying newswires: A multiclass classification example 116

The Reuters dataset 116 ■ Preparing the data 118 ■ Building your model 118 ■ Validating your approach 120 ■ Generating predictions on new data 124 ■ A different way to handle the labels and the loss 124 ■ The importance of having sufficiently large intermediate layers 125 ■ Further experiments 125 Wrapping up 126

4.3

Predicting house prices: A regression example 126

The California Housing Price dataset 126 ■ Preparing the data 128 ■ Building your model 1

### PDF p.639 (printed 615) — index ›  › M
model interpretability 299 machine learning (ML) 2, 3, 136, 171 model.layers property 200 evaluating models 149–153 ModelParallel distribution instance 556 generalization 137–148 model parallelism 548, 549 improving generalization 158–168 models improving model fit 153–158

best practices for 539–548 markets and 379 building in Keras 192 universal workflow of 179–189 dropout 167–168 workflow 599 reducing size of 161–164 workflow of, choosing measure of success 178 training recurrent 406 MAE (mean absolute error) 129, 131, 359 training recurrent model 406
