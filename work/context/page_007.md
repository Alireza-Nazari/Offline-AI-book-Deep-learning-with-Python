# Generation bundle for page_007  (PDF page 7, printed page vii)

Page kind: front_matter
Chapter: brief contents
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_007.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_007.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p007_e01] (chapter_title) brief contents
[p007_e02] (body) 1
[p007_e03] (body) ■ What is deep learning? 1
[p007_e04] (body) 2
[p007_e05] (body) ■ The mathematical building blocks of neural networks 16
[p007_e06] (body) 3
[p007_e07] (body) ■ Introduction to TensorFlow, PyTorch, JAX, and Keras 60
[p007_e08] (body) 4
[p007_e09] (body) ■ Classification and regression 104
[p007_e10] (body) 5
[p007_e11] (body) ■ Fundamentals of machine learning 136
[p007_e12] (body) 6
[p007_e13] (body) ■ The universal workflow of machine learning 171
[p007_e14] (body) 7
[p007_e15] (body) ■ A deep dive on Keras 190
[p007_e16] (body) 8
[p007_e17] (body) ■ Image classification 231
[p007_e18] (body) 9
[p007_e19] (body) ■ ConvNet architecture patterns 268
[p007_e20] (body) 10
[p007_e21] (body) ■ Interpreting what ConvNets learn 284
[p007_e22] (body) 11
[p007_e23] (body) ■ Image segmentation 308
[p007_e24] (body) 12
[p007_e25] (body) ■ Object detection 329
[p007_e26] (body) 13
[p007_e27] (body) ■ Timeseries forecasting 351
[p007_e28] (body) 14
[p007_e29] (body) ■ Text classification 381
[p007_e30] (body) 15
[p007_e31] (body) ■ Language models and the Transformer 421
[p007_e32] (body) 16
[p007_e33] (body) ■ Text generation 466
[p007_e34] (body) 17
[p007_e35] (body) ■ Image generation 508
[p007_e36] (body) 18
[p007_e37] (body) ■ Best practices for the real world 538
[p007_e38] (body) 19
[p007_e39] (body) ■ The future of AI 564
[p007_e40] (body) 20
[p007_e41] (body) ■ Conclusions 595
[p007_e42] (body) v

## Chapter outline
Chapter: brief contents (PDF pages 7-7)

## Previous page (PDF 6) text, for continuity
For online information and ordering of this and other Manning books, please visit www.manning.com. The publisher offers discounts on this book when ordered in quantity. For more information, please contact

Special Sales Department Manning Publications Co. 20 Baldwin Road PO Box 761 Shelter Island, NY 11964 Email: orders@manning.com

© 2026 Manning Publications Co. All rights reserved.

No part of this publication may be reproduced, stored in a retrieval system, or transmitted, in any form or by means electronic, mechanical, photocopying, or otherwise, without prior written permission of the publisher.

Many of the designations used by manufacturers and sellers to distinguish their products are claimed as trademarks. Where those designations appear in the book, and Manning Publications was aware of a trademark claim, the designations have been printed in initial caps or all caps.

∞

Recognizing the importance of preserving what has been written, it is Manning’s policy to have the books we publish printed on acid-free paper, and we exert our best efforts to that end. Recognizing also our responsibility to conserve the resources of our planet, Manning books are printed on paper that is at least 15 percent recycled and processed without the use of elemental chlorine.

The author and publisher have made every effort to ensure that the information in this book was correct at press time. The author and publisher do not assume and hereby disclaim any liability to any party for any loss, damage, or disruption caused by errors or omissions, whether such errors or omissions result from negligence, accident, or any other cause, or from any usage of the information herein.

Manning Publications Co.

Development editor: Ian Hough 20 Baldwin Road

Review editor: Dunja NikitoviÊ Prod …

## Next page (PDF 8) text, for continuity
contents

preface xv acknowledgments xvii about this book xviii about the authors xxi about the cover illustration xxii 1 What is deep learning? 1 1.1 Artificial intelligence, machine learning, and deep learning 2 1.2 Artificial intelligence 2 1.3 Machine learning 3 1.4 Learning rules and representations from data 4 1.5 The “deep” in “deep learning” 7 1.6 Understanding how deep learning works, in three figures 8 1.7 What makes deep learning different 10 1.8 The age of generative AI 11 1.9 What deep learning has achieved so far 11 1.10 Beware of the short-term hype 12 1.11 Summer can turn to winter 14 1.12 The promise of AI 14

vi

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.21 — about this book › How this book is organized: A road map
- PDF p.15 — contents › 
- PDF p.635 — index › 
- PDF p.12 — contents › 
- PDF p.11 — contents › 
- PDF p.16 — contents › 
- PDF p.9 — contents › 
- PDF p.10 — contents › 

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.21 (printed xxi) — about this book › How this book is organized: A road map › How this book is organized: A road map
This book contains 20 chapters, organized as follows:

Chapters 1–7: Foundations of deep learning—These chapters give you a solid grounding in deep learning. We discuss the history of the field, provide an introduction to coding with the major deep learning frameworks currently available (JAX, PyTorch, and TensorFlow), and outline a practical guide to applying machine learning to real-world problems. We look closely at the Keras library for training machine learning models.

Chapters 8–12: Computer vision—These chapters cover machine learning on image data. We train models to classify entire images and models to recognize and separate out individual objects in images. We look closely at ConvNets, a model architecture well suited to images and the first big success of the deep learning era.

Chapter 13: Timeseries forecasting—This chapter covers working with timeseries data. We take data that is measured over time, like the weather or sales data, and use past patterns to predict future trends.

Chapters 14–15: Natural language processing—These chapters cover applying deep learning to text. We build sentiment classifiers to predict whether text is positive or negative. We build an En

### PDF p.15 (printed xv) — contents ›  › 
17

Image generation 508 17.1

Deep learning for image generation 508

Sampling from latent spaces of images 509 ■ Variational autoencoders 510 ■ Implementing a VAE with Keras 513

17.2

Diffusion models 518

The Oxford Flowers dataset 520 ■ A U-Net denoising autoencoder 521 ■ The concepts of diffusion time and diffusion schedule 523 ■ The training process 525 ■ The generation process 527 ■ Visualizing results with a custom callback 528 It’s go time! 529

17.3

Text-to-image models 531

Exploring the latent space of a text-to-image model 533

18

Best practices for the real world 538 18.1

Getting the most out of your models 539

Hyperparameter optimization 539 ■ Model ensembling 546

18.2

Scaling up model training with multiple devices 548

Multi-GPU training 548 ■ Distributed training in practice 550 TPU training 555

18.3

Speeding up training and inference with lower-precision computation 556

Understanding floating-point precision 556 ■ Float16 inference 558 ■ Mixed-precision training 559 ■ Using loss scaling with mixed precision 559 ■ Beyond mixed precision: float8 training 560 ■ Faster inference with quantization 561

19

The future of AI 564 19.1

The limitations of deep l

### PDF p.9 (printed ix) — contents ›  › 
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

A brief history of deep learning frameworks 61 3.2 How these frameworks relate 

### PDF p.635 (printed 611) — index ›  › 
ConvNet filters 285

achievements of 12 architecture patterns 268, 272–276, 280–282 advantages of 597 convolution operation 235–239 AI winters 14 depthwise separable convolutions 278–280 artificial intelligence, machine learning,

and 2–3, 599 interpreting what they learn 284 depth of model 7 max-pooling operation 239–241 enabling technologies 598 modularity, hierarchy, and reuse 269 for image generation 509–518 training from scratch on small dataset 241 frameworks 61–62 visualizing filters 291–299 generative AI 11 visualizing heatmaps of class activation 299–306 guided program search 606 visualizing intermediate activations 285–291 image classification for small-data problems 242 visualizing latent space of 306 integrating modules into hybrid systems 590 ViTs (Vision Transformers) 282 Keras 206–214 convolutional base 257 limitations of 564–570, 605 convolution kernel 237 network architectures 600–605 convolution operation 235–239, 550 object detection, training YOLO model 342 border effects and padding 238 overview of 8, 597 strides 239 properties of 10 Core Knowledge priors 581 short-term hype 13–14 cosine distance 462 staying up to date 607–608 cost function 9 text classificatio

### PDF p.12 (printed xii) — contents ›  › 
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

10.4 Visualizing the latent space of a ConvN

### PDF p.16 (printed xvi) — contents ›  › 
19.3

How to build intelligence 576

The kaleidoscope hypothesis 577 ■ The essence of intelligence: Abstraction acquisition and recombination 578 The importance of setting the right target 578 ■ A new target: On-the-fly adaptation 580 ■ ARC Prize 581 ■ The test-time adaptation era 582 ■ ARC-AGI 2 583

19.4

The missing ingredients: Search and symbols 584

The two poles of abstraction 585 ■ Cognition as a combination of both kinds of abstraction 587 ■ Why deep learning isn’t a complete answer to abstraction generation 588 ■ An alternative approach to AI: Program synthesis 589 ■ Blending deep learning and program synthesis 590 ■ Modular component recombination and lifelong learning 592 ■ The long-term vision 593

20

Conclusions 595 20.1

Key concepts in review 595

Various approaches to artificial intelligence 596 ■ What makes deep learning special within the field of machine learning 596 How to think about deep learning 597 ■ Key enabling technologies 598 ■ The universal machine learning workflow 599 ■ Key network architectures 600

20.2

Limitations of deep learning 605 20.3 What might lie ahead 606 20.4 Staying up to date in a fast-moving field 607

Practice on real-world problem

### PDF p.11 (printed xi) — contents ›  › 
8

Image classification 231 8.1

Introduction to ConvNets 232

The convolution operation 234 ■ The max-pooling operation 239

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
