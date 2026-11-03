# Generation bundle for page_011  (PDF page 11, printed page xi)

Page kind: front_matter
Chapter: contents
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_011.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_011.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p011_e04] (body) 5.4
[p011_e05] (body) Improving generalization 158
[p011_e06] (quote) Dataset curation 159 ■ Feature engineering 159 ■ Using early stopping 161 ■ Regularizing your model 161
[p011_e07] (body) 6
[p011_e08] (body) The universal workflow of machine learning 171 6.1
[p011_e09] (body) Defining the task 172
[p011_e10] (quote) Framing the problem 172 ■ Collecting a dataset 174 Understanding your data 178 ■ Choosing a measure of success 178
[p011_e11] (body) 6.2
[p011_e12] (body) Developing a model 179
[p011_e13] (quote) Preparing the data 179 ■ Choosing an evaluation protocol 180 Beating a baseline 181 ■ Scaling up: Developing a model that overfits 182 ■ Regularizing and tuning your model 183
[p011_e14] (body) 6.3
[p011_e15] (body) Deploying your model 183
[p011_e16] (quote) Explaining your work to stakeholders and setting expectations 184 Shipping an inference model 184 ■ Monitoring your model in the wild 188 ■ Maintaining your model 188
[p011_e17] (body) 7
[p011_e18] (body) A deep dive on Keras 190 7.1
[p011_e19] (body) A spectrum of workflows 191 7.2 Different ways to build Keras models 192
[p011_e20] (quote) The Sequential model 192 ■ The Functional API 195 Subclassing the Model class 202 ■ Mixing and matching different components 204 ■ Remember: Use the right tool for the job 205
[p011_e21] (body) 7.3
[p011_e22] (body) Using built-in training and evaluation loops 206
[p011_e23] (quote) Writing your own metrics 207 ■ Using callbacks 208 Writing your own callbacks 210 ■ Monitoring and visualization with TensorBoard 212
[p011_e24] (body) 7.4
[p011_e25] (body) Writing your own training and evaluation loops 214
[p011_e26] (quote) Training vs. inference 215 ■ Writing custom training step functions 216 ■ Low-level usage of metrics 221 ■ Using fit() with a custom training loop 222 ■ Handling metrics in a custom train_step() 226
[p011_e27] (body) 8
[p011_e28] (body) Image classification 231 8.1
[p011_e29] (body) Introduction to ConvNets 232
[p011_e30] (quote) The convolution operation 234 ■ The max-pooling operation 239

## Chapter outline
Chapter: contents (PDF pages 8-16)

## Previous page (PDF 10) text, for continuity
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

The California Housing Price dataset 126 ■ Preparing the data 128 ■ Building your model 128 ■ Validating your approach using K-fold validation 129 ■ Generating predictions on new data 134 ■ Wrapping up 134

5

Fundamentals of machine learning 136 5.1

Generalization: The goal of machine learning 136

Underfitting and overfitting 137 ■ The nature of generalization in deep learning 143

5.2

Evaluating machine-learning models 149

Training, validation, and test sets 149 ■ Beating a common-sense baseline 152 ■ Things to keep in mind about model evaluation 152

5.3

Improving model fit 153

Tuning key gradient descent parameters 153 ■ Using better architecture priors 155 ■ Increasing  …

## Next page (PDF 12) text, for continuity
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

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.638 — index › 
- PDF p.644 — index › 
- PDF p.635 — index › 
- PDF p.639 — index › 
- PDF p.636 — index › 
- PDF p.15 — contents › 
- PDF p.7 — brief contents › 
- PDF p.640 — index › 

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.639 (printed 615) — index ›  › M
model interpretability 299 machine learning (ML) 2, 3, 136, 171 model.layers property 200 evaluating models 149–153 ModelParallel distribution instance 556 generalization 137–148 model parallelism 548, 549 improving generalization 158–168 models improving model fit 153–158

best practices for 539–548 markets and 379 building in Keras 192 universal workflow of 179–189 dropout 167–168 workflow 599 reducing size of 161–164 workflow of, choosing measure of success 178 training recurrent 406 MAE (mean absolute error) 129, 131, 359 training recurrent model 406

### PDF p.644 (printed 620) — index ›  › U
attribute 262

underfitting 137–142 parameters 39

ambiguous features 139 weights 215 noisy training data 138 training rare features and spurious correlations 140–142 diffusion models 525 U-Net denoising autoencoder 522 inference vs. 215 unigrams 403 lower-precision computation 556–562 universal workflow of machine learning training arguments 215, 218 deploying model 183–189 training data 147 developing model 179–183 training loops 10, 39, 206–214 update_state() method 207, 222 callbacks 208, 210 upsampling stage 522 monitoring and visualization with use_casual_mask 448 TensorBoard 212 using fit() with custom training loops 222–225

### PDF p.196 (printed 172) — 6 The universal workflow of machine learning › 6.1 Defining the task › 
Building a music recommendation system for users of an online radio.

Detecting credit card fraud for an e-commerce website.

Predicting display ad click-through rate to decide which ad to serve to a given user at a given time.

Flagging anomalous cookies on the conveyor belt of a cookie-manufacturing line.

Using satellite images to predict the location of as-yet unknown archaeological sites.

It would be very convenient if you could import the correct dataset from `keras
` `.datasets` and start fitting some deep learning models. Unfortunately, in the real world, you’ll have to start from scratch.

In this chapter, you’ll learn about the universal step-by-step blueprint that you can use to approach and solve any machine learning problem, like those previously listed. This template will bring together and consolidate everything you’ve learned in chapters 4 and 5 and give you the wider context that should anchor what you will learn in the next chapters. The universal workflow of machine learning is broadly structured in three parts:

Define the task—Understand the problem domain and the business logic underlying what the customer asked. Collect a dataset, understand what the data re

### PDF p.213 (printed 189) — 6 The universal workflow of machine learning › 6.3 Deploying your model › Summary
When you take on a new machine learning project, first, define the problem at hand: – Understand the broader context of what you’re setting out to do—what’s the end goal and what are the constraints? – Collect and annotate a dataset; make sure you understand your data in depth. – Choose how you’ll measure success on your problem. What metrics will you monitor on your validation data?

Once you understand the problem and you have an appropriate dataset, develop a model: – Prepare your data. – Pick your evaluation protocol. Hold-out validation? K-fold validation? Which portion of the data should you use for validation? – Achieve statistical power: beat a simple baseline. – Scale up: develop a model that can overfit. – Regularize your model and tune its hyperparameters, based on performance on the validation data. A lot of machine learning research tends to focus only on this step—but keep the big picture in mind.

When your model is ready and yields good performance on the test data, it’s time for deployment: – First, make sure to set appropriate expectations with stakeholders. – Optimize a final model for inference, and ship the model to the deployment environment of choice—web serv

### PDF p.214 (printed 190) — 7 A deep dive on Keras ›  › This chapter covers
The different ways to create Keras models: the

`Sequential` class, the Functional API, and model

subclassing

How to use the built-in Keras training and

evaluation loops, including how to use custom

metrics and custom losses

Using Keras callbacks to further customize how

training proceeds

Using TensorBoard for monitoring your training

and evaluation metrics over time

How to write your own training and evaluation

loops from scratch

You’re starting to have some amount of experience with Keras. You’re familiar with the Sequential model, `Dense` layers, and built-in APIs for training, evaluation, and inference—`compile()`, `fit()`, `evaluate()`, and `predict()`. You’ve even learned in chapter 3 how to inherit from the `Layer` class to create custom layers, and how to

### PDF p.641 (printed 617) — index ›  › P
randomized A/B testing 188 padding 238, 239, 322 RandomSearch tuner 542 panoptic segmentation 310 R-CNN (region-based convolutional neural parameters, defined 539

networks), two-stage detectors 331 pattern recognition 586 real-world best practices, lower-precision computation 556–562 perception, defined 586 faster inference with quantization 561 pip install tensorboard command 213 float8 training 560 plot_model() utility 199 float16 inference 558 PositionalEmbedding layer 452, 474 positional encoding 603 floating-point precision 557 position embedding 451 loss scaling with mixed precision 560 positive prompt 532 mixed-precision training 559 predict() function 191, 301, 479, 489 recombination 578 prediction error or loss value 105 reconstruction loss 512 prediction or output 105 recurrent dropout 372–375 predict() method 115, 124, 203, 259, 260 recurrent layers in Keras 368–372 premade search spaces 546 recurrent models, training 406 preparing data 179–180 region proposals 331 preprocessor layer 489 regression 104, 126–134, 602 pretrained embedding, using for classification 418 building model 128 pretrained models 256–266 California Housing Price dataset 126 feature extraction with

### PDF p.638 (printed 614) — index ›  › K
classification with pretrained Transformer

454–461 Kaggle 607 dot-product attention 439 kagglehub package 242, 485 embedding positional information 451 kaleidoscope hypothesis 577 generating Shakespeare 426–428 Keras 60, 190 loss function 462 building models in 192 seq2seq learning 428–437 ecosystem 608 sequence-to-sequence learning with evaluation loops 206–214 Transformer 449 fit() function 222– 225 training Shakespeare language model 422–426 Functional API 195, 197–200 Transformer 444, 447, 462–464 implementing VAEs with 513–518 Transformer architecture 437 low-level usage of metrics 221 last_conv_layer_output 302 mixing and matching different components 204 latent space 509, 568 relationship between frameworks 63 of images 509 Sequential model 192 subclassing Model class 202 of text-to-image models 533–537 summary 230 visualizing 306 training and evaluation loops 214 lateral connections 347 training loops 206–214 Layer API 191 training step functions 216–218 layered representations learning 7 using right tool for job 205 LayerNormalization layer 445 workflows 191 layer_output layer 167 Keras API, recurrent layers in 368–372 layers 596 keras.callbacks.Callback class 210 LayoutM

### PDF p.254 (printed 230) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Summary
Keras offers a spectrum of different workflows, based on the principle of progressive disclosure of complexity. They all smoothly interoperate.

You can build models via the `Sequential` class, via the Functional API, or by subclassing the `Model` class. Most of the time, you’ll be using the Functional API.

The simplest way to train and evaluate a model is via the default `fit()` and `evaluate()` methods.

Keras callbacks provide a simple way to monitor models during your call to `fit()` and automatically take action based on the state of the model.

You can also fully control what `fit()` does by overriding the `train_step()` method, using APIs from your backend of choice—JAX, TensorFlow, or PyTorch.

Beyond `fit()`, you can also write your own training loops entirely from scratch, in a backend-native way. This is useful for researchers implementing brand-new training algorithms.
