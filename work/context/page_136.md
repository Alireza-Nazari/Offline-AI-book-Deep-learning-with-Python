# Generation bundle for page_136  (PDF page 136, printed page 112)

Page kind: content
Chapter: 4 Classification and regression
Section: 4.1 Classifying movie reviews: A binary classification example
Subsection: 4.1.4 Validating your approach
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_136.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_136.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p136_e03] (heading_subsection) The validation_split argument
[p136_e04] (sidebar) Instead of manually splitting out validation data from your training data and passing it as the `validation_data` argument, you can also use the `validation_split` argument in `fit()`. It specifies a fraction of the training data to use as validation data, like this:
[p136_e05] (code)
```
history = model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=512,
    validation_split=0.2,
)
```
[p136_e06] (sidebar) In this example, 20% of the samples in the `x_train` and `y_train` arrays are being held out from training and used as validation data.
[p136_e07] (body) On CPU, this will take less than 2 seconds per epoch—training is over in 20 seconds. At the end of every epoch, there is a slight pause as the model computes its loss and accuracy on the 10,000 samples of the validation data.
[p136_e08] (body) Note that the call to `model.fit()` returns a `History` object, as you’ve seen in chapter 3. This object has a member `history`, which is a dictionary containing data about everything that happened during training. Let’s look at it:
[p136_e09] (code)
```
>>> history_dict = history.history
>>> history_dict.keys()
dict_keys(["accuracy", "loss", "val_accuracy", "val_loss"])
```
[p136_e10] (body) The dictionary contains four entries: one per metric that was being monitored during training and during validation. In the following two listings, let’s use Matplotlib to plot the training and validation loss side by side (see figure 4.4), as well as the training and validation accuracy (see figure 4.5). Note that your own results may vary slightly due to a different random initialization of your model.
[p136_e11] (listing_caption) Listing 4.8 Plotting the training and validation loss
[p136_e12] (code)
```
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "r--", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("[IMDB] Training and validation loss")
plt.xlabel("Epochs")
```

## Code annotations printed next to listings (callouts, in order)
- “b” is for “solid blue line.”
- “r--” is for “dashed red line.”

## Listings
- Listing 4.8: Plotting the training and validation loss (code elements p136_e12)

## Equation candidates (formulas written inline in the text)
- [p136_e05] history = model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=512,
    validation_split=0.2,
)

## Chapter outline
Chapter: 4 Classification and regression (PDF pages 128-159)
  - 4.1 Classifying movie reviews: A binary classification example (p.130)
      - 4.1.1 The IMDb dataset (p.130)
      - 4.1.2 Preparing the data (p.131)
      - 4.1.3 Building your model (p.132)
      - 4.1.4 Validating your approach (p.135)
      - 4.1.5 Using a trained model to generate predictions on new data (p.139)
      - 4.1.6 Further experiments (p.139)
      - 4.1.7 Wrapping up (p.140)
  - 4.2 Classifying newswires: A multiclass classification example (p.140)
      - 4.2.1 The Reuters dataset (p.140)
      - 4.2.2 Preparing the data (p.142)
      - 4.2.3 Building your model (p.142)
      - 4.2.4 Validating your approach (p.144)
      - 4.2.5 Generating predictions on new data (p.148)
      - 4.2.6 A different way to handle the labels and the loss (p.148)
      - 4.2.7 The importance of having sufficiently large intermediate layers (p.149)
      - 4.2.8 Further experiments (p.149)
      - 4.2.9 Wrapping up (p.150)
  - 4.3 Predicting house prices: A regression example (p.150)
      - 4.3.1 The California Housing Price dataset (p.150)
      - 4.3.2 Preparing the data (p.152)
      - 4.3.3 Building your model (p.152)
      - 4.3.4 Validating your approach using K-fold validation (p.153)
      - 4.3.5 Generating predictions on new data (p.158)
      - 4.3.6 Wrapping up (p.158)

## Previous page (PDF 135) text, for continuity
Listing 4.5 Compiling the model

4.1.4

Validating your approach

As you learned in chapter 3, a deep learning model should never be evaluated on its training data—it’s standard practice to use a “validation set” to monitor the accuracy of the model during training. Here, you’ll create a validation set by setting apart 10,000 samples from the original training data.

You might ask, why not simply use the test data to evaluate the model? That seems like it would be easier. The reason is that you’re going to want to use the results you get on the validation set to inform your next choices to improve training—for instance, your choice of what model size to use or how many epochs to train for. When you start doing this, your validation scores stop being an accurate reflection of the performance of the model on brand-new data, since the model has been deliberately modified to perform better on the validation data. It’s good to keep around a set of never-before-seen samples that you can use to perform the final evaluation round in a completely unbiased way, and that’s exactly what the test set is. We’ll talk more about this in the next chapter.

Listing 4.6 Setting aside a validation set

You’ll now train the model for 20 epochs (20 iterations over all samples in the training data), in mini-batches of 512 samples. At the same time, you’ll monitor loss and accuracy on the 10,000 samples that you set apart. You do so by passing the validation data as the `validation_data` argument to `model.fit()`.

Listing 4.7 Training your model

## Next page (PDF 137) text, for continuity
Figure 4.4 Training and validation loss

Listing 4.9 Plotting the training and validation accuracy

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.144 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.285 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.275 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.146 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.157 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.425 — 14 Text classification › 14.4 Set models
- PDF p.156 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.153 — 4 Classification and regression › 4.3 Predicting house prices: A regression example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.153 (printed 129) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Validating your approach using K-fold validation
To evaluate your model while you keep adjusting its parameters (such as the number of epochs used for training), you could split the data into a training set and a validation set, as you did in the previous examples. But because you have so few data points, the validation set would end up being very small (for instance, about 100 examples). As a consequence, the validation scores might change a lot depending on which data points you chose to use for validation and which you chose for training: the validation scores might have a high variance with regard to the validation split. This would prevent you from reliably evaluating your model.

The best practice in such situations is to use K-fold cross-validation (see figure 4.9). It consists of splitting the available data into K partitions (typically K = 4 or 5), instantiating K identical models, and training each one on K – 1 partitions while evaluating on the remaining partition. The validation score for the model used is then the average of the K validation scores obtained. In terms of code, this is straightforward.

### PDF p.125 (printed 101) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Monitoring loss and metrics on validation data
The goal of machine learning is not to obtain models that perform well on the training data, which is easy—all you have to do is follow the gradient. The goal is to obtain models that perform well in general, particularly on data points that the model has never encountered before. Just because a model performs well on its training data doesn’t mean it will perform well on data it has never seen! For instance, it’s possible that your model could end up merely memorizing a mapping between your training samples and their targets, which would be useless for the task of predicting targets for data the model has never seen before. We’ll go over this point in much more detail in the chapter 5.

To keep an eye on how the model does on new data, it’s standard practice to reserve a subset of the training data as “validation data”: you won’t be training the model on this data, but you will use it to compute a loss value and metrics value. You do this by using the `validation_data` argument in `fit()`. Like the training data, the validation data could be passed as NumPy arrays or as a TensorFlow `Dataset` object.

Listing 3.35 Using the validation data argument

### PDF p.274 (printed 250) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › Fitting the model
Let’s fit the model on our dataset. We use the `validation_data` argument in `fit()` to monitor validation metrics on a separate `Dataset` object.

Note that we also use a `ModelCheckpoint` callback to save the model after each epoch. We configure it with the path where to save the file, as well as the arguments `save_best_` `only=True` and `monitor="val_loss"`: they tell the callback to only save a new file (overwriting any previous one) when the current value of the `val_loss` metric is lower than at any previous time during training. This guarantees that your saved file will always contain the state of the model corresponding to its best-performing training epoch, in terms of its performance on the validation data. As a result, we won’t have to retrain a new model for a lower number of epochs if we start overfitting: we can just reload our saved file.

Listing 8.15 Fitting the model using a `Dataset`

### PDF p.138 (printed 114) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › 
Figure 4.5 Training and validation accuracy

As you can see, the training loss decreases with every epoch, and the training accuracy increases with every epoch. That’s what you would expect when running gradient-descent optimization—the quantity you’re trying to minimize should be less with every iteration. But that isn’t the case for the validation loss and accuracy: they seem to peak at the fourth epoch. This is an example of what we warned against earlier: a model that performs better on the training data isn’t necessarily a model that will do better on data it has never seen before. In precise terms, what you’re seeing is overfitting: after the fourth epoch, you’re overoptimizing on the training data, and you end up learning representations that are specific to the training data and don’t generalize to data outside of the training set.

In this case, to prevent overfitting, you could stop training after four epochs. In general, you can use a range of techniques to mitigate overfitting, which we’ll cover in chapter 5. Let’s train a new model from scratch for four epochs and then evaluate it on the test data.

Listing 4.10 Training the model for four epochs

### PDF p.421 (printed 397) — 14 Text classification › 14.3 Sets vs. sequences › 
Before we begin tokenizing our input text, we will make a copy of our training data with a few important modifications. We can ignore the unsupervised reviews for now and create a separate validation set to monitor our accuracy while training. We do this by splitting 20% of the training text files into a new directory.

Listing 14.10 Splitting validation from the IMDb dataset

### PDF p.144 (printed 120) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › Validating your approach
Let’s set apart 1,000 samples in the training data to use as a validation set.

Listing 4.17 Setting aside a validation set

### PDF p.144 (printed 120) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › Validating your approach
Listing 4.19 Plotting the training and validation loss
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "r--", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.xticks(epochs)
plt.ylabel("Loss")
plt.legend()
plt.show()

### PDF p.126 (printed 102) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
data strictly separate: the purpose of validation is to monitor whether what the model is learning is actually useful on new data. If any of the validation data has been seen by the model during training, your validation loss and metrics will be flawed.

If you want to compute the validation loss and metrics after training is complete, you can call the `evaluate` method:
