# Generation bundle for page_206  (PDF page 206, printed page 182)

Page kind: content
Chapter: 6 The universal workflow of machine learning
Section: 6.2 Developing a model
Subsection: 6.2.4 Scaling up: Developing a model that overfits
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_206.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_206.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p206_e03] (table_caption) Table 6.1 Which loss, last-layer activation, and metrics to use for different tasks
[p206_e04] (figure_caption) Task Last-layer activation Loss function Metrics
[p206_e05] (table_text) Binary classification Sigmoid Binary crossentropy Binary accuracy, ROC AUC Multiclass, single-label Softmax Categorical Categorical accuracy, classification crossentropy top-k categorical accuracy, ROC AUC Multiclass, multi-label Sigmoid Binary crossentropy Binary accuracy, ROC classification AUC Regression None Mean squared error Mean absolute error
[p206_e06] (body) For most problems, there are existing templates you can start from. You’re not the first person to try to build a spam detector, a music recommendation engine, or an image classifier. Make sure to research prior art to identify the feature engineering techniques and model architectures that are most likely to perform well on your task.
[p206_e07] (body) Note that it’s not always possible to achieve statistical power. If you can’t beat a simple baseline after trying multiple reasonable architectures, it may be that the answer to the question you’re asking isn’t present in the input data. Remember that you’re making two hypotheses:
[p206_e08] (bullet) You hypothesize that your outputs can be predicted given your inputs.
[p206_e09] (bullet) You hypothesize that the available data is sufficiently informative to learn the relationship between inputs and outputs.
[p206_e10] (body) It may well be that these hypotheses are false, in which case you must go back to the drawing board.
[p206_e11] (heading_subsection) 6.2.4
[p206_e12] (heading_subsection) Scaling up: Developing a model that overfits
[p206_e13] (body) Once you’ve obtained a model that has statistical power, the question becomes, is your model sufficiently powerful? Does it have enough layers and parameters to properly model the problem at hand? For instance, a logistic regression model has statistical power on MNIST but wouldn’t be sufficient to solve the problem well. Remember that the universal tension in machine learning is between optimization and generalization; the ideal model is one that stands right at the border between underfitting and overfitting, between undercapacity and overcapacity. To figure out where this border lies, first you must cross it.
[p206_e14] (body) To figure out how big a model you’ll need, you must develop a model that overfits. This is fairly easy, as you learned in chapter 5:
[p206_e15] (bullet) Add layers.
[p206_e16] (bullet) Make the layers bigger.
[p206_e17] (bullet) Train for more epochs.

## Figures
- Figure None: Task Last-layer activation Loss function Metrics  | labels inside the figure: (none - image)

## Table extraction (raw, may be noisy - verify against the page image)
[{"bbox": [93.0, 75.6, 464.8, 193.5], "rows": [["Task", "Last-layer activation", "Lossfunction", "Metrics"], ["Binaryclassifi cation\nM clau slt sic ifla cass s ningle-label\nti,\ni\no\nM clau slt sic ifla cass m nulti -label\nti,\ni\no\nRegression", "Sigmoid\nSoftmax\nSigmoid\nNone", "Binarycrossentropy\nCategorical\ncrossentropy\nBinarycrossentropy\nMeansquarederror", "Binaryaccuracy ROC\n,\nAUC\nCategoricalaccuracy\n,\ntop-k categorical accu\nracy ROCAUC\n,\nBinaryaccuracy ROC\n,\nAUC\nMeanabsoluteerror"]]}]

## Chapter outline
Chapter: 6 The universal workflow of machine learning (PDF pages 195-213)
  - 6.1 Defining the task (p.196)
      - 6.1.1 Framing the problem (p.196)
      - 6.1.2 Collecting a dataset (p.198)
      - 6.1.3 Understanding your data (p.202)
      - 6.1.4 Choosing a measure of success (p.202)
  - 6.2 Developing a model (p.203)
      - 6.2.1 Preparing the data (p.203)
      - 6.2.2 Choosing an evaluation protocol (p.204)
      - 6.2.3 Beating a baseline (p.205)
      - 6.2.4 Scaling up: Developing a model that overfits (p.206)
      - 6.2.5 Regularizing and tuning your model (p.207)
  - 6.3 Deploying your model (p.207)
      - 6.3.1 Explaining your work to stakeholders and setting expectations (p.208)
      - 6.3.2 Shipping an inference model (p.208)
      - 6.3.3 Monitoring your model in the wild (p.212)
      - 6.3.4 Maintaining your model (p.212)

## Previous page (PDF 205) text, for continuity
performance. The goal of your validation protocol is to accurately estimate what your success metric of choice (such as accuracy) will be on actual production data. The reliability of that process is critical to building a useful model.

In chapter 5, we reviewed three common evaluation protocols:

Maintaining a hold-out validation set—The way to go when you have plenty of data

Doing K-fold cross-validation—The right choice when you have too few samples for hold-out validation to be reliable

Doing iterated K-fold validation—For performing highly accurate model evaluation when little data is available

Just pick one of these. In most cases, the first will work well enough. As you’ve learned before, always be mindful of the representativity of your validation set(s) and be careful not to have redundant samples between your training set and your validation set(s).

6.2.3

Beating a baseline

As you start working on the model itself, your initial goal is to achieve statistical power, as you saw in chapter 5—that is, to develop a small model that is capable of beating a simple baseline.

At this stage, these are the three most important things you should focus on:

Feature engineering—Filter out uninformative features (feature selection) and use your knowledge of the problem to develop new features that are likely to be useful.

Selecting the correct architecture priors—What type of model architecture will you use? A densely connected network, a ConvNet, a recurrent neural network, a Transformer? Is deep learning even a good approach for the task, or should you use something else?

Selecting a good enough training configuration—What loss function should you use? What batch size and learning rate?

Picking the right loss function

It’s often not possible to directly optimiz …

## Next page (PDF 207) text, for continuity
Always monitor the training loss and validation loss, as well as the training and validation values for any metrics you care about. When you see that the model’s performance on the validation data begins to degrade, you’ve achieved overfitting.

6.2.5

Regularizing and tuning your model

Once you’ve achieved statistical power and you’re able to overfit, you know you’re on the right path. At this point, your goal becomes to maximize generalization performance.

This phase will take the most time: you’ll repeatedly modify your model, train it, evaluate on your validation data (not the test data, at this point), modify it again, and repeat, until the model is as good as it can get. Here are some things you should try:

Try different architectures; add or remove layers.

Add dropout.

If your model is small, add L1 or L2 regularization.

Try different hyperparameters (such as the number of units per layer or the learning rate of the optimizer) to find the optimal configuration.

Optionally, iterate on data curation or feature engineering: collect and annotate more data, develop better features, or remove features that don’t seem to be informative.

It’s possible to automate a large chunk of this work by using automated hyperparameter-tuning software, such as KerasTuner. We’ll cover this in chapter 18.

Be mindful of the following: every time you use feedback from your validation process to tune your model, you leak information about the validation process into the model. Repeated just a few times, this is innocuous; however, done systematically over many iterations, it will eventually cause your model to overfit to the validation process (even though no model is directly trained on any of the validation data). This makes the evaluation process less reliable. Once you’ve dev …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.649 — index › 
- PDF p.153 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.134 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.198 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.181 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.202 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.197 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.143 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.649 (printed 625) — index ›  › 
Task Last-layer activation Loss function Metrics

Binary classification Sigmoid Binary crossentropy Binary accuracy, ROC AUC Multiclass, single-label Softmax Categorical crossentropy Categorical accuracy, classification Top-k categorical accuracy, ROC AUC Multiclass, multi-label Sigmoid Binary crossentropy Binary accuracy, ROC classification AUC Regression None Mean squared error Mean absolute error

Input Output Model

Vector data Class probability, regression value Densely connected network Timeseries Class probability, regression value RNN, Transformer Images Class probability, regression value ConvNet Text Class probability, regression value Transformer Text, images Text Transformer Text, images Images VAE, diffusion model

### PDF p.150 (printed 126) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Wrapping up
Here’s what you should take away from this example:

If you’re trying to classify data points among N classes, your model should end with a `Dense` layer of size N.

In a single-label, multiclass classification problem, your model should end with a `softmax` activation so that it will output a probability distribution over the N output classes.

Categorical crossentropy is almost always the loss function you should use for such problems. It minimizes the distance between the probability distributions output by the model and the true distribution of the targets.

There are two ways to handle labels in multiclass classification: – Encoding the labels via categorical encoding (also known as one-hot encoding) and using `categorical_crossentropy` as a loss function – Encoding the labels as integers and using the `sparse_categorical_` `crossentropy` loss function

If you need to classify data into a large number of categories, you should avoid creating information bottlenecks in your model due to intermediate layers that are too small.

### PDF p.153 (printed 129) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › 
The model ends with a single unit and no activation: it will be a linear layer. This is a typical setup for scalar regression—a regression where you’re trying to predict a single continuous value. Applying an activation function would constrain the range the output can take; for instance, if you applied a `sigmoid` activation function to the last layer, the model could only learn to predict values between 0 and 1. Here, because the last layer is purely linear, the model is free to learn to predict values in any range.

Note that you compile the model with the `mean_squared_error` loss function—mean squared error, the square of the difference between the predictions and the targets. This is a widely used loss function for regression problems. You’re also monitoring a new metric during training: mean absolute error (MAE). It’s the absolute value of the difference between the predictions and the targets. For instance, an MAE of 0.5 on this problem would mean your predictions are off by $50,000 on average (remember the target scaling of factor 100,000).

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

### PDF p.271 (printed 247) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
For the compilation step, you’ll go with the `adam` optimizer, as usual. Because you ended the model with a single sigmoid unit, you’ll use binary crossentropy as the loss (as a reminder, check out table 6.1 in chapter 6 for a cheat sheet on what loss function to use in various situations).

Listing 8.8 Configuring the model for training

### PDF p.140 (printed 116) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › Wrapping up
Here’s what you should take away from this example:

You usually need to do quite a bit of preprocessing on your raw data to be able to feed it—as tensors—into a neural network. Sequences of words can be encoded as binary vectors, but there are other encoding options, too.

Stacks of `Dense` layers with `relu` activations can solve a wide range of problems (including sentiment classification), and you’ll use them frequently.

In a binary classification problem (two output classes), your model should end with a `Dense` layer with one unit and a `sigmoid` activation: the output of your model should be a scalar between 0 and 1, encoding a probability.

With such a scalar sigmoid output on a binary classification problem, the loss function you should use is `binary_crossentropy`.

The `adam` optimizer is generally a good enough choice, whatever your problem. That’s one less thing for you to worry about.

As they get better on their training data, neural networks eventually start overfitting and end up obtaining increasingly worse results on data they’ve never seen before. Be sure to always monitor performance on data that is outside of the training set!

### PDF p.143 (printed 119) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › 
There are two other things you should note about this architecture:

You end the model with a `Dense` layer of size 46. This means for each input sample, the network will output a 46-dimensional vector. Each entry in this vector (each dimension) will encode a different output class.

The last layer uses a `softmax` activation. You saw this pattern in the MNIST example. It means the model will output a probability distribution over the 46 different output classes—for every input sample, the model will produce a 46-dimensional output vector, where `output[i]` is the probability that the sample belongs to class `i`. The 46 scores will sum to 1.

The best loss function to use in this case is `categorical_crossentropy`. It measures the distance between two probability distributions—here, between the probability distribution outputted by the model and the true distribution of the labels. By minimizing the distance between these two distributions, you train the model to output something as close as possible to the true labels.

Like last time, we’ll also monitor accuracy. However, accuracy is a bit of a crude metric in this case: if the model has the correct class as its second choice for
