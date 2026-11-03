# Generation bundle for page_275  (PDF page 275, printed page 251)

Page kind: content
Chapter: 8 Image classification
Section: 8.2 Training a convnet from scratch on a small dataset
Subsection: 8.2.4 Data preprocessing
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_275.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_275.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p275_e03] (listing_caption) Listing 8.16 Displaying curves of loss and accuracy during training
[p275_e04] (code)
```
import matplotlib.pyplot as plt
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "r--", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "r--", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()
```
[p275_e05] (figure_caption) Figure 8.9 Training and validation metrics for a simple ConvNet
[p275_e06] (body) These plots are characteristic of overfitting. The training accuracy increases linearly over time, until it reaches nearly 100%, whereas the validation accuracy peaks around 80%. The validation loss reaches its minimum after only 10 epochs and then stalls, whereas the training loss keeps decreasing linearly as training proceeds.
[p275_e07] (body) Let’s check the test accuracy. We’ll reload the model from its saved file to evaluate it as it was before it started overfitting.

## Listings
- Listing 8.16: Displaying curves of loss and accuracy during training (code elements p275_e04)

## Figures
- Figure 8.9: Training and validation metrics for a simple ConvNet  | labels inside the figure: (none - image)

## Chapter outline
Chapter: 8 Image classification (PDF pages 255-291)
  - 8.1 Introduction to convnets (p.256)
      - 8.1.1 The convolution operation (p.258)
      - 8.1.2 The max-pooling operation (p.263)
  - 8.2 Training a convnet from scratch on a small dataset (p.265)
      - 8.2.1 The relevance of deep learning for small-data problems (p.266)
      - 8.2.2 Downloading the data (p.266)
      - 8.2.3 Building your model (p.269)
      - 8.2.4 Data preprocessing (p.271)
      - 8.2.5 Using data augmentation (p.276)
  - 8.3 Using a pretrained model (p.280)
      - 8.3.1 Feature extraction with a pretrained model (p.280)
      - 8.3.2 Fine-tuning a pretrained model (p.288)

## Previous page (PDF 274) text, for continuity
You’re about to see more `map()` action over the next chapters.

Fitting the model

Let’s look at the output of one of these `Dataset` objects: it yields batches of 180 × 180 RGB images (shape `(32, 180, 180, 3)`) and integer labels (shape `(32,)`). There are 32 samples in each batch (the batch size).

Listing 8.14 Displaying the shapes yielded by the `Dataset`

Let’s fit the model on our dataset. We use the `validation_data` argument in `fit()` to monitor validation metrics on a separate `Dataset` object.

Note that we also use a `ModelCheckpoint` callback to save the model after each epoch. We configure it with the path where to save the file, as well as the arguments `save_best_` `only=True` and `monitor="val_loss"`: they tell the callback to only save a new file (overwriting any previous one) when the current value of the `val_loss` metric is lower than at any previous time during training. This guarantees that your saved file will always contain the state of the model corresponding to its best-performing training epoch, in terms of its performance on the validation data. As a result, we won’t have to retrain a new model for a lower number of epochs if we start overfitting: we can just reload our saved file.

Listing 8.15 Fitting the model using a `Dataset`

Let’s plot the loss and accuracy of the model over the training and validation data during training (see figure 8.9).

## Next page (PDF 276) text, for continuity
Listing 8.17 Evaluating the model on the test set

We get a test accuracy of 78.6% (due to the randomness of neural network initializations, you may get numbers within 1 percentage point of that).

Because you have relatively few training samples (2,000), overfitting will be your number-one concern. You already know about a number of techniques that can help mitigate overfitting, such as dropout and weight decay (L2 regularization). We’re now going to work with a new one, specific to computer vision and used almost universally when processing images with deep learning models: data augmentation.

8.2.5

Using data augmentation

Overfitting is caused by having too few samples to learn from, rendering you unable to train a model that can generalize to new data. Given infinite data, your model would be exposed to every possible aspect of the data distribution at hand: you would never overfit. Data augmentation takes the approach of generating more training data from existing training samples, by augmenting the samples via a number of random transformations that yield believable-looking images. The goal is that at training time, your model will never see the exact same picture twice. This helps expose the model to more aspects of the data and generalize better.

In Keras, this can be done via data augmentation layers. Such layers could be added in one of two ways:

At the start of the model—Inside the model. In our case, the layers would come right before the `Rescaling` layer.

Inside the data pipeline—Outside the model. In our case, we’d apply them to our `Dataset` via a `map()` call.

The main difference between these two options is that data augmentation done inside the model would be running on the GPU, just like the rest of the model. Meanwhile, data augmentation done  …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.285 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.137 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.146 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.341 — 11 Image segmentation › 11.2 Training a segmentation model from scratch
- PDF p.144 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.425 — 14 Text classification › 14.4 Set models
- PDF p.284 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.279 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.138 (printed 114) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › 
Figure 4.5 Training and validation accuracy

As you can see, the training loss decreases with every epoch, and the training accuracy increases with every epoch. That’s what you would expect when running gradient-descent optimization—the quantity you’re trying to minimize should be less with every iteration. But that isn’t the case for the validation loss and accuracy: they seem to peak at the fourth epoch. This is an example of what we warned against earlier: a model that performs better on the training data isn’t necessarily a model that will do better on data it has never seen before. In precise terms, what you’re seeing is overfitting: after the fourth epoch, you’re overoptimizing on the training data, and you end up learning representations that are specific to the training data and don’t generalize to data outside of the training set.

In this case, to prevent overfitting, you could stop training after four epochs. In general, you can use a range of techniques to mitigate overfitting, which we’ll cover in chapter 5. Let’s train a new model from scratch for four epochs and then evaluate it on the test data.

Listing 4.10 Training the model for four epochs

### PDF p.279 (printed 255) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
Let’s plot the results again; see figure 8.11. Thanks to data augmentation and dropout, we start overfitting much later, around epochs 60–70 (compared to epoch 10 for the original model). The validation accuracy ends up peaking above 85%—a big improvement over our first try.

Figure 8.11 Training and validation metrics with data augmentation

Let’s check the test accuracy.

Listing 8.22 Evaluating the model on the test set

### PDF p.285 (printed 261) — 8 Image classification › 8.3 Using a pretrained model › 
Figure 8.13 Training and validation metrics for plain feature extraction

You reach a validation accuracy of slightly over 98%—much better than you achieved in the previous section with the small model trained from scratch. This is a bit of an unfair comparison, however, because ImageNet contains many dog and cat instances, which means that our pretrained model already has the exact knowledge required for the task at hand. This won’t always be the case when you use pretrained features.

However, the plots also indicate that you’re overfitting almost from the start—despite using dropout with a fairly large rate. That’s because this technique doesn’t use data augmentation, which is essential for preventing overfitting with small image datasets. Let’s check the test accuracy:

### PDF p.341 (printed 317) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 
Figure 11.5 Displaying training and validation loss curves

You can see that we start overfitting midway, around epoch 25. Let’s reload our best-performing model according to validation loss and demonstrate how to use it to predict a segmentation mask (see figure 11.6):

### PDF p.181 (printed 157) — 5 Fundamentals of machine learning › 5.3 Improving model fit › 
The training curves now look exactly like they should: the model fits fast and starts overfitting after eight epochs (see figure 5.15).

Figure 5.15 Validation loss for a model with appropriate capacity

### PDF p.285 (printed 261) — 8 Image classification › 8.3 Using a pretrained model › 
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "r--", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "r--", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

### PDF p.161 (printed 137) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › Underfitting and overfitting
For all models you’ve seen in the previous chapter, performance on the held-out validation data initially improved as training went on and then inevitably peaked after a while. This pattern (illustrated in figure 5.1) is universal. You’ll see it with any model type and any dataset.

Figure 5.1 Canonical overfitting behavior

At the beginning of training, optimization and generalization are correlated: the lower the loss on training data, the lower the loss on test data. While this is happening, your model is said to be underfit: there is still progress to be made; the network hasn’t yet modeled all relevant patterns in the training data. But after a certain number of iterations on the training data, generalization stops improving, and validation metrics stall and then begin to degrade: the model is starting to overfit. That is, it’s beginning to

### PDF p.188 (printed 164) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
Figure 5.19 shows how the bigger model fares compared to the reference model. The bigger model starts overfitting almost immediately, after just one epoch, and it overfits much more severely. Its validation loss is also noisier. It gets training loss near zero very quickly. The more capacity the model has, the more quickly it can model the training data (resulting in a low training loss), but the more susceptible it is to overfitting (resulting in a large difference between the training and validation loss).

Figure 5.19 Original model vs. much larger model on IMDB review classification
