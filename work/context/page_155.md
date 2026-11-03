# Generation bundle for page_155  (PDF page 155, printed page 131)

Page kind: content
Chapter: 4 Classification and regression
Section: 4.3 Predicting house prices: A regression example
Subsection: 4.3.4 Validating your approach using K-fold validation
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_155.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_155.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p155_e03] (code)
```
>>> [round(value, 3) for value in all_scores]
[0.298, 0.349, 0.232, 0.305]
>>> round(np.mean(all_scores), 3)
0.296
```
[p155_e04] (body) The different runs do indeed show meaningfully different validation scores, from 0.232 to 0.349. The average (0.296) is a much more reliable metric than any single score—that’s the entire point of K-fold cross-validation. In this case, you’re off by $29,600 on average, which is significant considering that the prices range from $60,000 to $500,000.
[p155_e05] (body) Let’s try training the model a bit longer: 200 epochs. To keep a record of how well the model does at each epoch, you’ll modify the training loop to save the per-epoch validation score log.
[p155_e06] (listing_caption) Listing 4.29 Saving the validation logs at each fold
[p155_e07] (code)
```
k = 4
num_val_samples = len(x_train) // k
num_epochs = 200
all_mae_histories = []
for i in range(k):
    print(f"Processing fold #{i + 1}")
    fold_x_val = x_train[i * num_val_samples : (i + 1) * num_val_samples]
    fold_y_val = y_train[i * num_val_samples : (i + 1) * num_val_samples]
    fold_x_train = np.concatenate(
        [x_train[: i * num_val_samples], x_train[(i + 1) * num_val_samples :]],
        axis=0,
    )
    fold_y_train = np.concatenate(
        [y_train[: i * num_val_samples], y_train[(i + 1) * num_val_samples :]],
        axis=0,
    )
    model = get_model()
    history = model.fit(
        fold_x_train,
        fold_y_train,
        validation_data=(fold_x_val, fold_y_val),
        epochs=num_epochs,
        batch_size=16,
        verbose=0,
    )
    mae_history = history.history["val_mean_absolute_error"]
    all_mae_histories.append(mae_history)
```
[p155_e08] (body) You can then compute the average of the per-epoch mean absolute error (MAE) scores for all folds.

## Code annotations printed next to listings (callouts, in order)
- Prepares the training data: data
- from all other partitions
- Prepares the validation
- data: data from partition #k
- Builds the Keras model
- (already compiled)
- Trains the model

## Listings
- Listing 4.29: Saving the validation logs at each fold (code elements p155_e07)

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

## Previous page (PDF 154) text, for continuity
Figure 4.9 Three-fold cross-validation

Listing 4.28 K-fold validation

Running this with `num_epochs = 50` yields the following results:

## Next page (PDF 156) text, for continuity
Listing 4.30 Building the history of successive mean K-fold validation scores

Let’s plot this; see figure 4.10.

Listing 4.31 Plotting validation scores

Figure 4.10 Validation MAE by epoch

It may be a little difficult to read the plot due to a scaling issue: the validation MAE for the first few epochs is dramatically higher than the values that follow. Let’s omit the first 10 data points, which are on a different scale than the rest of the curve.

Listing 4.32 Plotting validation scores, excluding the first 10 data points

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.135 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.144 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.153 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.175 — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models
- PDF p.157 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.147 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.139 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.567 — 18 Best practices for the real world › 18.1 Getting the most out of your models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.153 (printed 129) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Validating your approach using K-fold validation
To evaluate your model while you keep adjusting its parameters (such as the number of epochs used for training), you could split the data into a training set and a validation set, as you did in the previous examples. But because you have so few data points, the validation set would end up being very small (for instance, about 100 examples). As a consequence, the validation scores might change a lot depending on which data points you chose to use for validation and which you chose for training: the validation scores might have a high variance with regard to the validation split. This would prevent you from reliably evaluating your model.

The best practice in such situations is to use K-fold cross-validation (see figure 4.9). It consists of splitting the available data into K partitions (typically K = 4 or 5), instantiating K identical models, and training each one on K – 1 partitions while evaluating on the remaining partition. The validation score for the model used is then the average of the K validation scores obtained. In terms of code, this is straightforward.

### PDF p.175 (printed 151) — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models › 
Your final score is then the averages of the K scores obtained. This method is helpful when the performance of your model shows significant variance based on your train/ test split. Like hold-out validation, this method doesn’t exempt you from using a distinct validation set for model calibration.

Schematically, K-fold cross-validation looks like figure 5.13. Listing 5.6 shows a simple implementation.

Figure 5.13 Three-fold validation

Listing 5.6 K-fold cross-validation (note that labels are omitted for simplicity)

### PDF p.175 (printed 151) — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models › 
Listing 5.6 K-fold cross-validation (note that labels are omitted for simplicity)
k = 3
num_validation_samples = len(data) // k
np.random.shuffle(data)
validation_scores = []
for fold in range(k):
    validation_data = data[
        num_validation_samples * fold : num_validation_samples * (fold + 1)
    ]
    training_data = np.concatenate(
        data[: num_validation_samples * fold],
        data[num_validation_samples * (fold + 1) :],
    )
    model = get_model()
    model.fit(training_data, ...)
    validation_score = model.evaluate(validation_data, ...)
    validation_scores.append(validation_score)
validation_score = np.average(validation_scores)
model = get_model()
model.fit(data, ...)
test_score = model.evaluate(test_data, ...)

### PDF p.176 (printed 152) — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models › Iterated K-fold validation with shuffling
This one is for situations in which you have relatively little data available and you need to evaluate your model as precisely as possible. I’ve found it to be extremely helpful in Kaggle competitions. It consists of applying K-fold validation multiple times, shuffling the data every time before splitting it `K` ways. The final score is the average of the scores obtained at each run of K-fold validation. Note that you end up training and evaluating `P * K` models (where `P` is the number of iterations you use), which can be very expensive.

### PDF p.205 (printed 181) — 6 The universal workflow of machine learning › 6.2 Developing a model › 
performance. The goal of your validation protocol is to accurately estimate what your success metric of choice (such as accuracy) will be on actual production data. The reliability of that process is critical to building a useful model.

In chapter 5, we reviewed three common evaluation protocols:

Maintaining a hold-out validation set—The way to go when you have plenty of data

Doing K-fold cross-validation—The right choice when you have too few samples for hold-out validation to be reliable

Doing iterated K-fold validation—For performing highly accurate model evaluation when little data is available

Just pick one of these. In most cases, the first will work well enough. As you’ve learned before, always be mindful of the representativity of your validation set(s) and be careful not to have redundant samples between your training set and your validation set(s).

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

### PDF p.157 (printed 133) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › 
According to this plot (see figure 4.11), validation MAE stops improving significantly after 120–140 epochs (this number includes the 10 epochs we omitted). Past that point, you start overfitting.

Once you’re finished tuning other parameters of the model (in addition to the number of epochs, you could also adjust the size of the intermediate layers), you can train a final production model on all of the training data, with the best parameters, and then look at its performance on the test data.

Figure 4.11 Validation MAE by epoch, excluding the first 10 data points

Listing 4.33 Training the final model

### PDF p.174 (printed 150) — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models › K-fold validation
With this approach, you split your data into `K` partitions of equal size. For each partition `i`, train a model on the remaining `K - 1` partitions and evaluate it on partition `i`.
