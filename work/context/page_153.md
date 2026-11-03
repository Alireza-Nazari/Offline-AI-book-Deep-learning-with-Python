# Generation bundle for page_153  (PDF page 153, printed page 129)

Page kind: content
Chapter: 4 Classification and regression
Section: 4.3 Predicting house prices: A regression example
Subsection: 4.3.4 Validating your approach using K-fold validation
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_153.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_153.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p153_e03] (code)
```
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model
```
[p153_e04] (body) The model ends with a single unit and no activation: it will be a linear layer. This is a typical setup for scalar regression—a regression where you’re trying to predict a single continuous value. Applying an activation function would constrain the range the output can take; for instance, if you applied a `sigmoid` activation function to the last layer, the model could only learn to predict values between 0 and 1. Here, because the last layer is purely linear, the model is free to learn to predict values in any range.
[p153_e05] (body) Note that you compile the model with the `mean_squared_error` loss function—mean squared error, the square of the difference between the predictions and the targets. This is a widely used loss function for regression problems. You’re also monitoring a new metric during training: mean absolute error (MAE). It’s the absolute value of the difference between the predictions and the targets. For instance, an MAE of 0.5 on this problem would mean your predictions are off by $50,000 on average (remember the target scaling of factor 100,000).
[p153_e06] (heading_subsection) 4.3.4
[p153_e07] (heading_subsection) Validating your approach using K-fold validation
[p153_e08] (body) To evaluate your model while you keep adjusting its parameters (such as the number of epochs used for training), you could split the data into a training set and a validation set, as you did in the previous examples. But because you have so few data points, the validation set would end up being very small (for instance, about 100 examples). As a consequence, the validation scores might change a lot depending on which data points you chose to use for validation and which you chose for training: the validation scores might have a high variance with regard to the validation split. This would prevent you from reliably evaluating your model.
[p153_e09] (body) The best practice in such situations is to use K-fold cross-validation (see figure 4.9). It consists of splitting the available data into K partitions (typically K = 4 or 5), instantiating K identical models, and training each one on K – 1 partitions while evaluating on the remaining partition. The validation score for the model used is then the average of the K validation scores obtained. In terms of code, this is straightforward.

## Equation candidates (formulas written inline in the text)
- [p153_e09] It consists of splitting the available data into K partitions (typically K = 4 or 5), instantiating K identical models, and training each one on K – 1 partitions while evaluating on the remaining partition.

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

## Previous page (PDF 152) text, for continuity
4.3.2

Preparing the data

It would be problematic to feed into a neural network values that all take wildly different ranges. The model might be able to automatically adapt to such heterogeneous data, but it would definitely make learning more difficult. A widespread best practice to deal with such data is to do feature-wise normalization: for each feature in the input data (a column in the input data matrix), you subtract the mean of the feature and divide by the standard deviation, so that the feature is centered around 0 and has a unit standard deviation. This is easily done in NumPy.

Listing 4.25 Normalizing the data

Note that the quantities used for normalizing the test data are computed using the training data. You should never use in your workflow any quantity computed on the test data, even for something as simple as data normalization.

In addition, we should also scale the targets. Our normalized inputs have their value in a small range close to 0, and our model’s weights are initialized with small random values. This means that our model’s prediction will also be small values when we start training. If the targets are in the range 60,000–500,000, the model is going to need very large weight values to output those. With a small learning rate, it would take a very long time to get there. The simplest fix is to divide all target values by 100,000, so that the smallest target becomes 0.6, and the largest becomes 5. We can then convert the model’s predictions back to dollar values by multiplying them by 100,000 accordingly.

Listing 4.26 Scaling the targets

4.3.3

Building your model

Because so few samples are available, you’ll use a very small model with two intermediate layers, each with 64 units. In general, the less training data you have, the worse overf …

## Next page (PDF 154) text, for continuity
Figure 4.9 Three-fold cross-validation

Listing 4.28 K-fold validation

Running this with `num_epochs = 50` yields the following results:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.155 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.158 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.134 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.135 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.626 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.157 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.156 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.144 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.158 (printed 134) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Wrapping up
Here’s what you should take away from this scalar regression example:

Regression is done using a different loss function than what we used for classification. Mean squared error (MSE) is a loss function commonly used for regression.

Similarly, evaluation metrics to be used for regression differ from those used for classification; naturally, the concept of accuracy doesn’t apply for regression. A common regression metric is MAE.

When features in the input data have values in different ranges, each feature should be scaled independently as a preprocessing step.

When there is little data available, using K-fold validation is a great way to reliably evaluate a model.

When little training data is available, it’s preferable to use a small model with few intermediate layers (typically only one or two), in order to avoid severe overfitting.

### PDF p.155 (printed 131) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › 
The different runs do indeed show meaningfully different validation scores, from 0.232 to 0.349. The average (0.296) is a much more reliable metric than any single score—that’s the entire point of K-fold cross-validation. In this case, you’re off by $29,600 on average, which is significant considering that the prices range from $60,000 to $500,000.

Let’s try training the model a bit longer: 200 epochs. To keep a record of how well the model does at each epoch, you’ll modify the training loop to save the per-epoch validation score log.

Listing 4.29 Saving the validation logs at each fold

### PDF p.175 (printed 151) — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models › 
Your final score is then the averages of the K scores obtained. This method is helpful when the performance of your model shows significant variance based on your train/ test split. Like hold-out validation, this method doesn’t exempt you from using a distinct validation set for model calibration.

Schematically, K-fold cross-validation looks like figure 5.13. Listing 5.6 shows a simple implementation.

Figure 5.13 Three-fold validation

Listing 5.6 K-fold cross-validation (note that labels are omitted for simplicity)

### PDF p.626 (printed 602) — 20 Conclusions › 20.1 Key concepts in review › 
To perform regression toward a vector of continuous values, end your stack of layers with a `Dense` layer with a number of units equal to the number of values you’re trying to predict (often a single one, such as the price of a house) and no activation. Various losses can be used for regression—most commonly `mean_squared_error` (MSE):

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

### PDF p.384 (printed 360) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 13.2.3 Let’s try a basic machine learning model
In the same way that it’s useful to establish a commonsense baseline before trying machine learning approaches, it’s useful to try simple, cheap, machine learning models (such as small, densely connected networks) before looking into complicated and computationally expensive models such as RNNs. This is the best way to make sure any further complexity you throw at the problem is legitimate and delivers real benefits.

Listing 13.10 shows a fully connected model that starts by flattening the data and then runs it through two `Dense` layers. Note the lack of activation function on the last `Dense` layer, which is typical for a regression problem. We use mean squared error (MSE) as the loss, rather than MAE, because unlike MAE, it’s smooth around zero, a useful property for gradient descent. We will monitor MAE by adding it as a metric in `compile()`.

Listing 13.10 Training and evaluating a densely connected model

### PDF p.157 (printed 133) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › 
According to this plot (see figure 4.11), validation MAE stops improving significantly after 120–140 epochs (this number includes the 10 epochs we omitted). Past that point, you start overfitting.

Once you’re finished tuning other parameters of the model (in addition to the number of epochs, you could also adjust the size of the intermediate layers), you can train a final production model on all of the training data, with the best parameters, and then look at its performance on the test data.

Figure 4.11 Validation MAE by epoch, excluding the first 10 data points

Listing 4.33 Training the final model

### PDF p.174 (printed 150) — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models › K-fold validation
With this approach, you split your data into `K` partitions of equal size. For each partition `i`, train a model on the remaining `K - 1` partitions and evaluate it on partition `i`.
