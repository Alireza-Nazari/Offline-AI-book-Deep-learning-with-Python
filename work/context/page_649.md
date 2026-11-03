# Generation bundle for page_649  (PDF page 649, printed page 625)

Page kind: index
Chapter: index
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_649.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_649.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p649_e01] (figure_caption) Task Last-layer activation Loss function Metrics
[p649_e02] (table_text) Binary classification Sigmoid Binary crossentropy Binary accuracy, ROC AUC Multiclass, single-label Softmax Categorical crossentropy Categorical accuracy, classification Top-k categorical accuracy, ROC AUC Multiclass, multi-label Sigmoid Binary crossentropy Binary accuracy, ROC classification AUC Regression None Mean squared error Mean absolute error
[p649_e03] (figure_caption) Input Output Model
[p649_e04] (table_text) Vector data Class probability, regression value Densely connected network Timeseries Class probability, regression value RNN, Transformer Images Class probability, regression value ConvNet Text Class probability, regression value Transformer Text, images Text Transformer Text, images Images VAE, diffusion model

## Figures
- Figure None: Task Last-layer activation Loss function Metrics  | labels inside the figure: (none - image)
- Figure None: Input Output Model  | labels inside the figure: (none - image)

## Chapter outline
Chapter: index (PDF pages 633-650)

## Previous page (PDF 648) text, for continuity
The Manning Early Access Program

Don’t wait to start learning! In MEAP, the Manning Early Access Program, you can read

books as they’re being created and long before they’re available in stores.

Here’s how MEAP works.

• Start now. Buy a MEAP and you’ll get all available chapters in PDF, ePub, Kindle,

and liveBook formats.

• Regular updates. New chapters are released as soon as they’re written. We’ll

let you know when fresh content is available.

• Finish faster. MEAP customers are the first to get final versions of all books!

Pre-order the print book, and it’ll ship as soon as it’s off the press.

• Contribute to the process. The feedback you share with authors makes the end

product better.

• No risk. You get a full refund or exchange if we ever have to cancel a MEAP.

Explore dozens of titles in MEAP at www.manning.com.

## Next page (PDF 650) text, for continuity
“

Perfect for anyone interested

DEEP LEARNING with Python Third Edition

in learning by doing from one

of the industry greats.”

Chollet ● Watson

—Anthony Goldbloom

I

n less than a decade, deep learning has changed the world

Founder of Kaggle

—twice. First, Python-based libraries like Keras, Tensor-

Flow, and PyTorch elevated neural networks from lab

“

A sharp, deeply practical

experiments to high-performance production systems deployed

guide that teaches you how

at scale. And now, through Large Language Models and other

to think from fi rst

generative AI tools, deep learning is again transforming busi-

principles to build models

ness and society. In this new edition, Keras creator François

that actually work.”

Chollet invites you into this amazing subject in the fl uid,

mentoring style of a true insider.

—Santiago Valdarrama

Founder of ml.school

Deep Learning with Python, Third Edition makes the concepts

behind deep learning and generative AI understandable and

“

Th e most up-to-date and

approachable. Th is complete rewrite of the bestselling original

includes fresh chapters on transformers, building your own

complete guide to deep

learning you’ll fi nd today!”

GPT-like LLM, and generating images with diff usion models.

Each chapter introduces practical projects and code examples

—Aran Komatsuzaki

that build your understanding of deep learning, layer by layer.

EleutherAI

What’s Inside

“

Masterfully conveys the

● Hands-on, code-fi rst learning

true essence of neural

● Comprehensive, from basics to generative AI

networks. A rare case in

recent years of outstanding

● Intuitive and easy math explanations

technical writing.”

● Examples in Keras, PyTorch, JAX, and TensorFlow

—Salvatore Sanfi lippo

For readers with intermediate …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.206 — 6 The universal workflow of machine learning › 6.2 Developing a model
- PDF p.625 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.153 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.634 — index › 
- PDF p.143 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.134 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.258 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.205 — 6 The universal workflow of machine learning › 6.2 Developing a model

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.206 (printed 182) — 6 The universal workflow of machine learning › 6.2 Developing a model › 
Table 6.1 Which loss, last-layer activation, and metrics to use for different tasks

Task Last-layer activation Loss function Metrics

Binary classification Sigmoid Binary crossentropy Binary accuracy, ROC AUC Multiclass, single-label Softmax Categorical Categorical accuracy, classification crossentropy top-k categorical accuracy, ROC AUC Multiclass, multi-label Sigmoid Binary crossentropy Binary accuracy, ROC classification AUC Regression None Mean squared error Mean absolute error

For most problems, there are existing templates you can start from. You’re not the first person to try to build a spam detector, a music recommendation engine, or an image classifier. Make sure to research prior art to identify the feature engineering techniques and model architectures that are most likely to perform well on your task.

Note that it’s not always possible to achieve statistical power. If you can’t beat a simple baseline after trying multiple reasonable architectures, it may be that the answer to the question you’re asking isn’t present in the input data. Remember that you’re making two hypotheses:

You hypothesize that your outputs can be predicted given your inputs.

You hypothesize th

### PDF p.150 (printed 126) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Wrapping up
Here’s what you should take away from this example:

If you’re trying to classify data points among N classes, your model should end with a `Dense` layer of size N.

In a single-label, multiclass classification problem, your model should end with a `softmax` activation so that it will output a probability distribution over the N output classes.

Categorical crossentropy is almost always the loss function you should use for such problems. It minimizes the distance between the probability distributions output by the model and the true distribution of the targets.

There are two ways to handle labels in multiclass classification: – Encoding the labels via categorical encoding (also known as one-hot encoding) and using `categorical_crossentropy` as a loss function – Encoding the labels as integers and using the `sparse_categorical_` `crossentropy` loss function

If you need to classify data into a large number of categories, you should avoid creating information bottlenecks in your model due to intermediate layers that are too small.

### PDF p.143 (printed 119) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › 
There are two other things you should note about this architecture:

You end the model with a `Dense` layer of size 46. This means for each input sample, the network will output a 46-dimensional vector. Each entry in this vector (each dimension) will encode a different output class.

The last layer uses a `softmax` activation. You saw this pattern in the MNIST example. It means the model will output a probability distribution over the 46 different output classes—for every input sample, the model will produce a 46-dimensional output vector, where `output[i]` is the probability that the sample belongs to class `i`. The 46 scores will sum to 1.

The best loss function to use in this case is `categorical_crossentropy`. It measures the distance between two probability distributions—here, between the probability distribution outputted by the model and the true distribution of the labels. By minimizing the distance between these two distributions, you train the model to output something as close as possible to the true labels.

Like last time, we’ll also monitor accuracy. However, accuracy is a bit of a crude metric in this case: if the model has the correct class as its second choice for

### PDF p.205 (printed 181) — 6 The universal workflow of machine learning › 6.2 Developing a model › Picking the right loss function
It’s often not possible to directly optimize for the metric that measures success on a problem. Sometimes there is no easy way to turn a metric into a loss function; loss functions, after all, need to be computable given only a mini-batch of data (ideally, a loss function should be computable for as little as a single data point) and must be differentiable (otherwise, you can’t use backpropagation to train your network). For instance, the widely used classification metric ROC AUC can’t be directly optimized. Hence, in classification tasks, it’s common to optimize for a proxy metric of ROC AUC, such as crossentropy. In general, you can hope that the lower the crossentropy gets, the higher the ROC AUC will be. Table 6.1 can help you choose a last-layer activation, a loss function, and metrics for a few common problem types.

### PDF p.634 (printed 610) — index ›  › B
categorical classification 105 categorical_crossentropy function 119, 126, 601 backbone 257, 258, 336 categorical encoding 118 model 489 causal language model 455 backend engine 63 CausalLM task 485, 532 backpropagation 63 CBOW (Continuous Bag of Words) 462 algorithm 9, 46–50

model 415 .backward() function 302 chain rule 46 bag-of-words model, training 399–402 character tokenization 387–390 baseline, beating 181 CharTokenizer 394 .batch() method 249 class activation, visualizing heatmaps of 299–306 batch normalization 276–278 displaying class activation heatmap 304 BatchNormalization layer 216, 265, 445, 603 JAX version 303 BayesianOptimization tuner 542 behavioral programs 572 TensorFlow version 302 BERT (Bidirectional Encoder Representations from classes 105 Transformers) 455 classification 104, 352 best practices 538 movie reviews 106–116 scaling up model training with multiple multiclass classification 116–126 devices 548–556 of images, building model 245 best practices for models regression 126–134 hyperparameter optimization 539–546 classification and regression, glossary 105 model ensembling 546–548 classifier_model 302 bidirectional class probability map 337 layer 378, 434

### PDF p.159 (printed 135) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Summary
The three most common kinds of machine learning tasks on vector data are binary classification, multiclass classification, and scalar regression. Each task uses different loss functions: – `binary_crossentropy` for binary classification – `categorical_crossentropy` for multiclass classification – `mean_squared_error` for scalar regression

You’ll usually need to preprocess raw data before feeding it into a neural network.

When your data has features with different ranges, scale each feature independently as part of preprocessing.

As training progresses, neural networks eventually begin to overfit and obtain worse results on never-before-seen data.

If you don’t have much training data, use a small model with only one or two intermediate layers, to avoid severe overfitting.

If your data is divided into many categories, you may cause information bottlenecks if you make the intermediate layers too small.

When you’re working with little data, K-fold validation can help reliably evaluate your model.

### PDF p.134 (printed 110) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › 
So the layer could only learn linear transformations (affine transformations) of the input data: the hypothesis space of the layer would be the set of all possible linear transformations of the input data into a 16-dimensional space. Such a hypothesis space is too restricted and wouldn’t benefit from multiple layers of representations because a deep stack of linear layers would still implement a linear operation: adding more layers wouldn’t extend the hypothesis space (as you saw in chapter 2). To get access to a much richer hypothesis space that would benefit from deep representations, you need a non-linearity or activation function. `relu` is the most popular activation function in deep learning, but there are many other candidates, which all come with similarly strange names: `prelu`, `elu`, and so on.

Figure 4.2 The rectified linear unit function Figure 4.3 The sigmoid function

Finally, you need to choose a loss function and an optimizer. Because you’re facing a binary classification problem and the output of your model is a probability (you end your model with a single-unit layer with a sigmoid activation), it’s best to use the `binary_crossentropy` loss. It isn’t the only v

### PDF p.258 (printed 234) — 8 Image classification › 8.1 Introduction to convnets › 
After the last `Conv2D` layer, we end up with an output of shape `(3, 3, 256)`—a 3 × 3 feature map of 256 channels. The next step is to feed this output into a densely connected classifier like those you’re already familiar with: a stack of `Dense` layers. These classifiers process vectors, which are 1D, whereas the current output is a rank-3 tensor. To bridge the gap, we flatten the 3D outputs to 1D with a `GlobalAveragePooling2D` layer before adding the `Dense` layers. This layer will take the average of each 3 × 3 feature map in the tensor of shape `(3,` `3,` `256)`, resulting in an output vector of shape `(256,)`. Finally, we’ll do 10-way classification, so our last layer has 10 outputs and a softmax activation.

Now, let’s train the ConvNet on the MNIST digits. We’ll reuse a lot of the code from the MNIST example in chapter 2. Because we’re doing 10-way classification with a softmax output, we’ll use the categorical crossentropy loss, and because our labels are integers, we’ll use the sparse version, `sparse_categorical_crossentropy`.

Listing 8.3 Training the ConvNet on MNIST images
