# Generation bundle for page_139  (PDF page 139, printed page 115)

Page kind: content
Chapter: 4 Classification and regression
Section: 4.1 Classifying movie reviews: A binary classification example
Subsection: 4.1.6 Further experiments
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_139.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_139.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p139_e03] (code)
```
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
```
[p139_e04] (body) The final results are as follows:
[p139_e05] (code)
```
>>> results
[0.2929924130630493, 0.88327999999999995]
```
[p139_e06] (body) This fairly naive approach achieves an accuracy of 88%. With state-of-the-art approaches, you should be able to get close to 95%.
[p139_e07] (heading_subsection) 4.1.5
[p139_e08] (heading_subsection) Using a trained model to generate predictions on new data
[p139_e09] (body) After having trained a model, you’ll want to use it in a practical setting. You can generate the likelihood of reviews being positive by using the `predict` method, as you’ve learned in chapter 3:
[p139_e10] (code)
```
>>> model.predict(x_test)
array([[ 0.98006207]
       [ 0.99758697]
       [ 0.99975556]
       ...,
       [ 0.82167041]
       [ 0.02885115]
       [ 0.65371346]], dtype=float32)
```
[p139_e11] (body) As you can see, the model is confident for some samples (0.99 or more, or 0.01 or less) but less confident for others (0.6, 0.4).
[p139_e12] (heading_subsection) 4.1.6
[p139_e13] (heading_subsection) Further experiments
[p139_e14] (body) The following experiments will help convince you that the architecture choices you’ve made are all fairly reasonable, although there’s still room for improvement:
[p139_e15] (bullet) You used two representation layers before the final classification layer. Try using one or three representation layers and see how doing so affects validation and test accuracy.

## Code annotations printed next to listings (callouts, in order)
- The first number, 0.29, is the test
- loss, and the second number,
- 0.88, is the test accuracy.

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

## Previous page (PDF 138) text, for continuity
Figure 4.5 Training and validation accuracy

As you can see, the training loss decreases with every epoch, and the training accuracy increases with every epoch. That’s what you would expect when running gradient-descent optimization—the quantity you’re trying to minimize should be less with every iteration. But that isn’t the case for the validation loss and accuracy: they seem to peak at the fourth epoch. This is an example of what we warned against earlier: a model that performs better on the training data isn’t necessarily a model that will do better on data it has never seen before. In precise terms, what you’re seeing is overfitting: after the fourth epoch, you’re overoptimizing on the training data, and you end up learning representations that are specific to the training data and don’t generalize to data outside of the training set.

In this case, to prevent overfitting, you could stop training after four epochs. In general, you can use a range of techniques to mitigate overfitting, which we’ll cover in chapter 5. Let’s train a new model from scratch for four epochs and then evaluate it on the test data.

Listing 4.10 Training the model for four epochs

## Next page (PDF 140) text, for continuity
Try using layers with more units or fewer units: 32 units, 64 units, and so on.

Try using the `mean_squared_error` loss function instead of `binary_crossentropy`.

Try using the `tanh` activation (an activation that was popular in the early days of neural networks) instead of `relu`.

4.1.7

Wrapping up

Here’s what you should take away from this example:

You usually need to do quite a bit of preprocessing on your raw data to be able to feed it—as tensors—into a neural network. Sequences of words can be encoded as binary vectors, but there are other encoding options, too.

Stacks of `Dense` layers with `relu` activations can solve a wide range of problems (including sentiment classification), and you’ll use them frequently.

In a binary classification problem (two output classes), your model should end with a `Dense` layer with one unit and a `sigmoid` activation: the output of your model should be a scalar between 0 and 1, encoding a probability.

With such a scalar sigmoid output on a binary classification problem, the loss function you should use is `binary_crossentropy`.

The `adam` optimizer is generally a good enough choice, whatever your problem. That’s one less thing for you to worry about.

As they get better on their training data, neural networks eventually start overfitting and end up obtaining increasingly worse results on data they’ve never seen before. Be sure to always monitor performance on data that is outside of the training set!

4.2

Classifying newswires: A multiclass classification example

In the previous section, you saw how to classify vector inputs into two mutually exclusive classes using a densely connected neural network. But what happens when you have more than two classes?

In this section, you’ll build a model to classify Reuters newsw …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.147 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.149 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.157 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.135 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.133 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.153 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.187 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.158 — 4 Classification and regression › 4.3 Predicting house prices: A regression example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

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

### PDF p.130 (printed 106) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › Classifying movie reviews: A binary classification example
Two-class classification, or binary classification, is one of the most common kinds of machine learning problem. In this example, you’ll learn to classify movie reviews as positive or negative, based on the text content of the reviews.

### PDF p.149 (printed 125) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › Further experiments
Like in the previous example, we encourage you to try out the following experiments to train your intuition about the kind of configuration decisions you have to make with such models:

Try using larger or smaller layers: 32 units, 128 units, and so on.

You used two intermediate layers before the final softmax classification layer. Now try using a single intermediate layer, or three intermediate layers.

### PDF p.44 (printed 20) — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network › 
Two quantities are displayed during training: the loss of the model over the training data and the accuracy of the model over the training data. We quickly reach an accuracy of 0.989 (98.9%) on the training data.

Now that we have a trained model, we can use it to predict class probabilities for new digits—images that weren’t part of the training data, like those from the test set.

Listing 2.6 Using the model to make predictions

### PDF p.280 (printed 256) — 8 Image classification › 8.3 Using a pretrained model › 
We get a test accuracy of 83.9%. It’s starting to look good! If you’re using Colab, make sure to download the saved file (`convnet_from_scratch_with_augmentation.keras`), as we will use it for some experiments in the next chapter.

By further tuning the model’s configuration (such as the number of filters per convolution layer or the number of layers in the model), you may be able to get an even better accuracy, likely up to 90%. But it would prove difficult to go any higher just by training your own ConvNet from scratch because you have so little data to work with. As a next step to improve your accuracy on this problem, you’ll have to use a pretrained model, which is the focus of the next two sections.

### PDF p.290 (printed 266) — 8 Image classification › 8.3 Using a pretrained model › 
Figure 8.15 Training and validation metrics for fine-tuning

Here, you get a test accuracy of 98.6% (again, your own results may be within half a percentage point). In the original Kaggle competition around this dataset, this would have been one of the top results. It’s not quite a fair comparison, however, since you used pretrained features that already contained prior knowledge about cats and dogs, which competitors couldn’t use at the time.

On the positive side, by using modern deep learning techniques, you managed to reach this result using only a small fraction of the training data that was available for the competition (about 10%). There is a huge difference between being able to train on 20,000 samples compared to 2,000 samples! Now you have a solid set of tools for dealing with image-classification problems—in particular, with small datasets.

### PDF p.135 (printed 111) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › Validating your approach
As you learned in chapter 3, a deep learning model should never be evaluated on its training data—it’s standard practice to use a “validation set” to monitor the accuracy of the model during training. Here, you’ll create a validation set by setting apart 10,000 samples from the original training data.

You might ask, why not simply use the test data to evaluate the model? That seems like it would be easier. The reason is that you’re going to want to use the results you get on the validation set to inform your next choices to improve training—for instance, your choice of what model size to use or how many epochs to train for. When you start doing this, your validation scores stop being an accurate reflection of the performance of the model on brand-new data, since the model has been deliberately modified to perform better on the validation data. It’s good to keep around a set of never-before-seen samples that you can use to perform the final evaluation round in a completely unbiased way, and that’s exactly what the test set is. We’ll talk more about this in the next chapter.

Listing 4.6 Setting aside a validation set

### PDF p.445 (printed 421) — 15 Language models and the Transformer › 15.1 The language model › 15.1 The language model
In the previous chapter, we learned how to convert text data to numeric inputs, and we used this numeric representation to classify movie reviews. However, text classification is, in many ways, a uniquely simple problem. We only need to output a single floating-point number for binary classification and, at worst, N numbers for N-way classification.
