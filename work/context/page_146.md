# Generation bundle for page_146  (PDF page 146, printed page 122)

Page kind: content
Chapter: 4 Classification and regression
Section: 4.2 Classifying newswires: A multiclass classification example
Subsection: 4.2.4 Validating your approach
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_146.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_146.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p146_e03] (listing_caption) Listing 4.20 Plotting the training and validation accuracy
[p146_e04] (code)
```
plt.clf()
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "r--", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.xticks(epochs)
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```
[p146_e05] (figure_caption) Figure 4.8 Training and validation top-3 accuracy
[p146_e06] (listing_caption) Listing 4.21 Plotting the training and validation top-3 accuracy
[p146_e07] (code)
```
plt.clf()
acc = history.history["top_3_accuracy"]
val_acc = history.history["val_top_3_accuracy"]
plt.plot(epochs, acc, "r--", label="Training top-3 accuracy")
plt.plot(epochs, val_acc, "b", label="Validation top-3 accuracy")
plt.title("Training and validation top-3 accuracy")
plt.xlabel("Epochs")
plt.xticks(epochs)
plt.ylabel("Top-3 accuracy")
plt.legend()
plt.show()
```

## Listings
- Listing 4.20: Plotting the training and validation accuracy (code elements p146_e04)
- Listing 4.21: Plotting the training and validation top-3 accuracy (code elements p146_e07)

## Figures
- Figure 4.8: Training and validation top-3 accuracy  | labels inside the figure: (none - image)

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

## Previous page (PDF 145) text, for continuity
Figure 4.6 Training and validation loss Figure 4.7 Training and validation accuracy

## Next page (PDF 147) text, for continuity
The model begins to overfit after nine epochs. Let’s train a new model from scratch for nine epochs and then evaluate it on the test set.

Listing 4.22 Retraining a model from scratch

Here are the final results:

This approach reaches an accuracy of approximately 80%. With a balanced binary classification problem, the accuracy reached by a purely random classifier would be 50%. But in this case, we have 46 classes, and they may not be equally represented. What would be the accuracy of a random baseline? We could try quickly implementing one to check this empirically:

As you can see, a random classifier would score around 19% classification accuracy, so the results of our model seem pretty good in that light.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.137 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.144 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.275 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.285 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.156 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.425 — 14 Text classification › 14.4 Set models
- PDF p.157 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.136 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.137 (printed 113) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › 
Figure 4.4 Training and validation loss

Listing 4.9 Plotting the training and validation accuracy

### PDF p.144 (printed 120) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › Validating your approach
And finally, let’s display its loss and accuracy curves (see figures 4.6 and 4.7).

Listing 4.19 Plotting the training and validation loss

### PDF p.137 (printed 113) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › 
Listing 4.9 Plotting the training and validation accuracy
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

### PDF p.136 (printed 112) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › The validation_split argument
The dictionary contains four entries: one per metric that was being monitored during training and during validation. In the following two listings, let’s use Matplotlib to plot the training and validation loss side by side (see figure 4.4), as well as the training and validation accuracy (see figure 4.5). Note that your own results may vary slightly due to a different random initialization of your model.

Listing 4.8 Plotting the training and validation loss

### PDF p.156 (printed 132) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › 
Let’s plot this; see figure 4.10.

Listing 4.31 Plotting validation scores

### PDF p.140 (printed 116) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › Classifying newswires: A multiclass classification example
In the previous section, you saw how to classify vector inputs into two mutually exclusive classes using a densely connected neural network. But what happens when you have more than two classes?

In this section, you’ll build a model to classify Reuters newswires into 46 mutually exclusive topics. Because you have many classes, this problem is an instance of multiclass classification, and because each data point should be classified into only one category, the problem is more specifically an instance of single-label, multiclass classification. If each data point could belong to multiple categories (in this case, topics), you’d be facing a multilabel, multiclass classification problem.

### PDF p.165 (printed 141) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › 
Listing 5.3 Plotting a validation accuracy comparison

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
