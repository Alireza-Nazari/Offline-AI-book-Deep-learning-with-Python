# Generation bundle for page_137  (PDF page 137, printed page 113)

Page kind: content
Chapter: 4 Classification and regression
Section: 4.1 Classifying movie reviews: A binary classification example
Subsection: 4.1.4 Validating your approach
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_137.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_137.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p137_e03] (code)
```
plt.xticks(epochs)
plt.ylabel("Loss")
plt.legend()
plt.show()
```
[p137_e04] (figure_caption) Figure 4.4 Training and validation loss
[p137_e05] (listing_caption) Listing 4.9 Plotting the training and validation accuracy
[p137_e06] (code)
```
plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "r--", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("[IMDB] Training and validation accuracy")
plt.xlabel("Epochs")
plt.xticks(epochs)
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

## Code annotations printed next to listings (callouts, in order)
- Clears the figure

## Listings
- Listing 4.9: Plotting the training and validation accuracy (code elements p137_e06)

## Figures
- Figure 4.4: Training and validation loss  | labels inside the figure: (none - image)

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

## Previous page (PDF 136) text, for continuity
The validation_split argument

Instead of manually splitting out validation data from your training data and passing it as the `validation_data` argument, you can also use the `validation_split` argument in `fit()`. It specifies a fraction of the training data to use as validation data, like this:

In this example, 20% of the samples in the `x_train` and `y_train` arrays are being held out from training and used as validation data.

On CPU, this will take less than 2 seconds per epoch—training is over in 20 seconds. At the end of every epoch, there is a slight pause as the model computes its loss and accuracy on the 10,000 samples of the validation data.

Note that the call to `model.fit()` returns a `History` object, as you’ve seen in chapter 3. This object has a member `history`, which is a dictionary containing data about everything that happened during training. Let’s look at it:

The dictionary contains four entries: one per metric that was being monitored during training and during validation. In the following two listings, let’s use Matplotlib to plot the training and validation loss side by side (see figure 4.4), as well as the training and validation accuracy (see figure 4.5). Note that your own results may vary slightly due to a different random initialization of your model.

Listing 4.8 Plotting the training and validation loss

## Next page (PDF 138) text, for continuity
Figure 4.5 Training and validation accuracy

As you can see, the training loss decreases with every epoch, and the training accuracy increases with every epoch. That’s what you would expect when running gradient-descent optimization—the quantity you’re trying to minimize should be less with every iteration. But that isn’t the case for the validation loss and accuracy: they seem to peak at the fourth epoch. This is an example of what we warned against earlier: a model that performs better on the training data isn’t necessarily a model that will do better on data it has never seen before. In precise terms, what you’re seeing is overfitting: after the fourth epoch, you’re overoptimizing on the training data, and you end up learning representations that are specific to the training data and don’t generalize to data outside of the training set.

In this case, to prevent overfitting, you could stop training after four epochs. In general, you can use a range of techniques to mitigate overfitting, which we’ll cover in chapter 5. Let’s train a new model from scratch for four epochs and then evaluate it on the test data.

Listing 4.10 Training the model for four epochs

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.146 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.144 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.275 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.156 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.285 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.145 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.157 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.425 — 14 Text classification › 14.4 Set models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.130 (printed 106) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › Classifying movie reviews: A binary classification example
Two-class classification, or binary classification, is one of the most common kinds of machine learning problem. In this example, you’ll learn to classify movie reviews as positive or negative, based on the text content of the reviews.

### PDF p.144 (printed 120) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › Validating your approach
And finally, let’s display its loss and accuracy curves (see figures 4.6 and 4.7).

Listing 4.19 Plotting the training and validation loss

### PDF p.146 (printed 122) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › 
Figure 4.8 Training and validation top-3 accuracy

Listing 4.21 Plotting the training and validation top-3 accuracy

### PDF p.146 (printed 122) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › 
Listing 4.20 Plotting the training and validation accuracy

### PDF p.145 (printed 121) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › 
Figure 4.6 Training and validation loss Figure 4.7 Training and validation accuracy

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

### PDF p.156 (printed 132) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › 
Let’s plot this; see figure 4.10.

Listing 4.31 Plotting validation scores

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
