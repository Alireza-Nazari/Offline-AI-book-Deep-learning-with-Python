# Generation bundle for page_151  (PDF page 151, printed page 127)

Page kind: content
Chapter: 4 Classification and regression
Section: 4.3 Predicting house prices: A regression example
Subsection: 4.3.1 The California Housing Price dataset
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_151.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_151.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p151_e03] (bullet) The longitude and latitude of the approximate geographic center of the area.
[p151_e04] (bullet) The median age of houses in the district.
[p151_e05] (bullet) The population of the district. The districts are pretty small: the average population is 1,425.5.
[p151_e06] (bullet) The total number of households.
[p151_e07] (bullet) The median income of those households.
[p151_e08] (bullet) The total number of rooms in the district, across all homes located there. This is typically in the low thousands.
[p151_e09] (bullet) The total number of bedrooms in the district.
[p151_e10] (body) That’s eight variables in total (longitude and latitude count as two variables). The goal is to use these variables to predict the median value of the houses in the district. Let’s get started by loading the data.
[p151_e11] (listing_caption) Listing 4.24 Loading the California Housing Prices dataset
[p151_e12] (code)
```
from keras.datasets import california_housing
```
[p151_e13] (code)
```
(train_data, train_targets), (test_data, test_targets) = (
    california_housing.load_data(version="small")
)
```
[p151_e14] (body) Let’s look at the data:
[p151_e15] (code)
```
>>> train_data.shape
(480, 8)
>>> test_data.shape
(120, 8)
```
[p151_e16] (body) As you can see, we have 480 training samples and 120 test samples, each with 8 numerical features. The targets are the median values of homes in the district considered, in dollars:
[p151_e17] (code)
```
>>> train_targets
array([252300., 146900., 290900., ..., 140500., 217100.],
      dtype=float32)
```
[p151_e18] (body) The prices are between $60,000 and $500,000. If that sounds cheap, remember that this was in 1990, and these prices aren’t adjusted for inflation.

## Code annotations printed next to listings (callouts, in order)
- Make sure to pass version="small"
- to get the right dataset.

## Listings
- Listing 4.24: Loading the California Housing Prices dataset (code elements p151_e12, p151_e13)

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

## Previous page (PDF 150) text, for continuity
4.2.9

Wrapping up

Here’s what you should take away from this example:

If you’re trying to classify data points among N classes, your model should end with a `Dense` layer of size N.

In a single-label, multiclass classification problem, your model should end with a `softmax` activation so that it will output a probability distribution over the N output classes.

Categorical crossentropy is almost always the loss function you should use for such problems. It minimizes the distance between the probability distributions output by the model and the true distribution of the targets.

There are two ways to handle labels in multiclass classification: – Encoding the labels via categorical encoding (also known as one-hot encoding) and using `categorical_crossentropy` as a loss function – Encoding the labels as integers and using the `sparse_categorical_` `crossentropy` loss function

If you need to classify data into a large number of categories, you should avoid creating information bottlenecks in your model due to intermediate layers that are too small.

4.3

Predicting house prices: A regression example

The two previous examples were considered classification problems, where the goal was to predict a single discrete label of an input data point. Another common type of machine learning problem is regression, which consists of predicting a continuous value instead of a discrete label: for instance, predicting the temperature tomorrow given meteorological data, or predicting the time that a software project will take to complete given its specifications.

NOTE Confusingly, logistic regression isn’t a regression algorithm—it’s a classification algorithm.

4.3.1

The California Housing Price dataset

You’ll attempt to predict the median price of homes in different areas of Cal …

## Next page (PDF 152) text, for continuity
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

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.158 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.141 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.129 — 4 Classification and regression › 
- PDF p.130 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.204 — 6 The universal workflow of machine learning › 6.2 Developing a model
- PDF p.634 — index › 
- PDF p.155 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.42 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.158 (printed 134) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Generating predictions on new data
The first district in the test set is predicted to have an average home price of about $283,000.

### PDF p.203 (printed 179) — 6 The universal workflow of machine learning › 6.2 Developing a model › Value normalization
In the MNIST digit-classification example from chapter 2, you started from image data encoded as integers in the 0–255 range, encoding grayscale values. Before you fed this data into your network, you had to cast it to `float32` and divide by 255 so you’d end up with floating-point values in the 0–1 range. Similarly, when predicting house prices, you started from features that took a variety of ranges—some features had small floating-point values, others had fairly large integer values. Before you fed this data into your network, you had to normalize each feature independently so that it had a standard deviation of 1 and a mean of 0.

In general, it isn’t safe to feed into a neural network data that takes relatively large values (for example, multi-digit integers, which are much larger than the initial values taken by the weights of a network) or data that is heterogeneous (for example, data where one feature is in the range 0–1, and another is in the range 100–200). Doing so can trigger large gradient updates that will prevent the network from

### PDF p.634 (printed 610) — index ›  › C
confidence score 336 C4 (Colossal Clean Crawled Corpus) 470 constraints 372 California Housing Price dataset 126 continuous function 41 callbacks 208 Conv1D layer 362, 602 writing 210 Conv2D layer 232, 233, 234, 239, 245, 292, 362, 602 call() method 215, 475 Conv2DTranspose layers 316, 347 CAM (class activation map) 299 Conv3D layer 362, 602

### PDF p.129 (printed 105) — 4 Classification and regression ›  › Classification and regression glossary
dicting house prices is a good example: the different target prices form a continuous space.

Vector regression—A task where the target is a set of continuous values: for

example, a continuous vector. If you’re doing regression against multiple values (such as the coordinates of a bounding box in an image), then you’re doing vector regression.

Mini-batch or just batch—A small set of samples (typically between 8 and 128)

that are processed simultaneously by the model. The number of samples is often a power of 2, to facilitate memory allocation on GPU. When training, a mini-batch is used to compute a single gradient-descent update applied to the weights of the model.

### PDF p.626 (printed 602) — 20 Conclusions › 20.1 Key concepts in review › 
To perform regression toward a vector of continuous values, end your stack of layers with a `Dense` layer with a number of units equal to the number of values you’re trying to predict (often a single one, such as the price of a house) and no activation. Various losses can be used for regression—most commonly `mean_squared_error` (MSE):

### PDF p.204 (printed 180) — 6 The universal workflow of machine learning › 6.2 Developing a model › Handling missing values
You may sometimes have missing values in your data. For instance, in the house price example, the second feature was the median age of houses in the district. What if this feature wasn’t available for all samples? You’d then have missing values in the training or test data.

You could just discard the feature entirely, but you don’t necessarily have to:

If the feature is categorical, it’s safe to create a new category that means “the value is missing.” The model will automatically learn what this implies with respect to the targets.

If the feature is numerical, avoid inputting an arbitrary value like 0 because it may create a discontinuity in the latent space formed by your features, making it harder for a model trained on it to generalize. Instead, consider replacing the missing value with the average or median value for the feature in the dataset. You could also train a model to predict the feature value given the values of other features.

Note that if you’re expecting missing categorical features in the test data, but the network was trained on data without any missing values, the network won’t have learned to ignore missing values! In this situation, you should artificially 

### PDF p.158 (printed 134) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Generating predictions on new data
When calling `predict()` on our binary classification model, we retrieved a scalar score between 0 and 1 for each input sample. With our multiclass classification model, we retrieved a probability distribution over all classes for each sample. Now, with this scalar regression model, `predict()` returns the model’s guess for the sample’s price in hundreds of thousands of dollars:

### PDF p.160 (printed 136) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › Generalization: The goal of machine learning
In the three examples presented in chapter 4—predicting movie reviews, topic classification, and house-price regression—we split the data into a training set, a
