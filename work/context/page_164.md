# Generation bundle for page_164  (PDF page 164, printed page 140)

Page kind: content
Chapter: 5 Fundamentals of machine learning
Section: 5.1 Generalization: The goal of machine learning
Subsection: 5.1.1 Underfitting and overfitting
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_164.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_164.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p164_e03] (heading_subsubsection) Rare features and spurious correlations
[p164_e04] (body) If you’ve only ever seen two orange tabby cats in your life, and they both happened to be terribly antisocial, you might infer that orange tabby cats are generally likely to be antisocial. That’s overfitting: if you had been exposed to a wider variety of cats, including more orange ones, you’d have learned that cat color is not well correlated with character.
[p164_e05] (body) Likewise, machine learning models trained on datasets that include rare feature values are highly susceptible to overfitting. In a sentiment classification task, if the word “cherimoya” (a fruit native to the Andes) only appears in one text in the training data, and this text happens to be negative in sentiment, a poorly regularized model might put a very high weight on this word and always classify new texts that mention cherimoyas as negative, whereas, objectively, there’s nothing negative about the cherimoya.1 Importantly, a feature value doesn’t need to occur only a couple of times to lead to spurious correlations. Consider a word that occurs in 100 samples in your training data, and that’s associated with a positive sentiment 54% of the time and with a negative sentiment 46% of the time. That difference may well be a complete statistical fluke, yet your model is likely to learn to use that feature for its classification task. This is one of the most common sources of overfitting. Here’s a striking example. Take MNIST. Create a new training set by concatenating 784 white noise dimensions to the existing 784 dimensions of the data—so half of the data is now noise. For comparison, also create an equivalent dataset by concatenating 784 all-zeros dimensions. Our concatenation of meaningless features does not at all affect the information content of the data: we’re only adding irrelevant data points. Human classification accuracy wouldn’t be affected by these transformations at all.
[p164_e06] (listing_caption) Listing 5.1 Adding white noise channels or all-zeros channels to MNIST
[p164_e07] (code)
```
from keras.datasets import mnist
import numpy as np
(train_images, train_labels), _ = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
train_images_with_noise_channels = np.concatenate(
    [train_images, np.random.random((len(train_images), 784))], axis=1
)
train_images_with_zeros_channels = np.concatenate(
    [train_images, np.zeros((len(train_images), 784))], axis=1
)
```
[p164_e08] (body) Now, let’s train the model from chapter 2 on both of these training sets.
[p164_e09] (footnote) 1 Mark Twain even called it “the most delicious fruit known to men.”

## Listings
- Listing 5.1: Adding white noise channels or all-zeros channels to MNIST (code elements p164_e07)

## Chapter outline
Chapter: 5 Fundamentals of machine learning (PDF pages 160-194)
  - 5.1 Generalization: The goal of machine learning (p.160)
      - 5.1.1 Underfitting and overfitting (p.161)
      - 5.1.2 The nature of generalization in deep learning (p.167)
  - 5.2 Evaluating machine-learning models (p.173)
      - 5.2.1 Training, validation, and test sets (p.173)
      - 5.2.2 Beating a common-sense baseline (p.176)
      - 5.2.3 Things to keep in mind about model evaluation (p.176)
  - 5.3 Improving model fit (p.177)
      - 5.3.1 Tuning key gradient descent parameters (p.177)
      - 5.3.2 Using better architecture priors (p.179)
      - 5.3.3 Increasing model capacity (p.179)
  - 5.4 Improving generalization (p.182)
      - 5.4.1 Dataset curation (p.183)
      - 5.4.2 Feature engineering (p.183)
      - 5.4.3 Using early stopping (p.185)
      - 5.4.4 Regularizing your model (p.185)

## Previous page (PDF 163) text, for continuity
Figure 5.4 Dealing with outliers: robust fit vs. overfitting

Ambiguous features

Not all data noise comes from inaccuracies—even perfectly clean and neatly labeled data can be noisy when the problem involves uncertainty and ambiguity (see figure 5.5). In classification tasks, it is often the case that some regions of the input feature space are associated with multiple classes at the same time. Let’s say you’re developing a model that takes an image of a banana and predicts whether the banana is unripened, ripe, or rotten. These categories have no objective boundaries, so the same picture might be classified as either unripened or ripe by different human labelers. Similarly, many problems involve randomness. You could use atmospheric pressure data to predict whether it will rain tomorrow, but the exact same measurements may be followed sometimes by rain, sometimes by a clear sky—with some probability.

Figure 5.5 Robust fit vs. overfitting giving an ambiguous area of the feature space

A model could overfit to such probabilistic data by being too confident about ambiguous regions of the feature space, like in figure 5.6. A more robust fit would ignore individual data points and look at the bigger picture.

## Next page (PDF 165) text, for continuity
Listing 5.2 Training the same model on MNIST data with noise channels or all-zero channels

Listing 5.3 Plotting a validation accuracy comparison

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.166 — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning
- PDF p.178 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.162 — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning
- PDF p.640 — index › 
- PDF p.204 — 6 The universal workflow of machine learning › 6.2 Developing a model
- PDF p.180 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.167 — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning
- PDF p.44 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.162 (printed 138) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › 
learn patterns that are specific to the training data but that are misleading or irrelevant when it comes to new data.

Overfitting is particularly likely to occur when your data is noisy, if it involves uncertainty, or if it includes rare features. Let’s look at concrete examples.

### PDF p.161 (printed 137) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › 
validation set, and a test set. The reason not to evaluate the models on the same data they were trained on quickly became evident: after just a few epochs, performance on never-before-seen data started diverging from performance on the training data, which always improves as training progresses. The models started to overfit. Overfitting happens in every machine-learning problem.

The fundamental issue in machine learning is the tension between optimization and generalization. Optimization refers to the process of adjusting a model to get the best performance possible on the training data (the learning in machine learning), whereas generalization refers to how well the trained model performs on data it has never seen before. The goal of the game is to get good generalization, of course, but you don’t control generalization; you can only fit the model to its training data. If you do that too well, overfitting kicks in and generalization suffers. But what causes overfitting? How can we achieve good generalization?

### PDF p.166 (printed 142) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › 
Despite the data holding the same information in both cases, the validation accuracy of the model trained with noise channels ends up about one percentage point lower— purely through the influence of spurious correlations (figure 5.6). The more noise channels you might add, the further accuracy would degrade.

Figure 5.6 Effect of noise channels on validation accuracy

Noisy features inevitably lead to overfitting. As such, in cases where you aren’t sure whether the features you have are informative or distracting, it’s common to do feature selection before training. Restricting the IMDB data to the top 10,000 most common words was a crude form of feature selection, for instance. The typical way to do feature selection is to compute some usefulness score for each feature available—a measure of how informative the feature is with respect to the task, such as the mutual information between the feature and the labels—and only keep features that are above some threshold. Doing this would filter out the white noise channels in the preceding example.

### PDF p.640 (printed 616) — index ›  › N
overfitting 137–142, 182 negative prompt 532

ambiguous features 139 Nelder-Mead algorithm 547 noisy training data 138 network architectures 600–605 rare features and spurious correlations 140–142

### PDF p.161 (printed 137) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › Underfitting and overfitting
For all models you’ve seen in the previous chapter, performance on the held-out validation data initially improved as training went on and then inevitably peaked after a while. This pattern (illustrated in figure 5.1) is universal. You’ll see it with any model type and any dataset.

Figure 5.1 Canonical overfitting behavior

At the beginning of training, optimization and generalization are correlated: the lower the loss on training data, the lower the loss on test data. While this is happening, your model is said to be underfit: there is still progress to be made; the network hasn’t yet modeled all relevant patterns in the training data. But after a certain number of iterations on the training data, generalization stops improving, and validation metrics stall and then begin to degrade: the model is starting to overfit. That is, it’s beginning to

### PDF p.159 (printed 135) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Summary
The three most common kinds of machine learning tasks on vector data are binary classification, multiclass classification, and scalar regression. Each task uses different loss functions: – `binary_crossentropy` for binary classification – `categorical_crossentropy` for multiclass classification – `mean_squared_error` for scalar regression

You’ll usually need to preprocess raw data before feeding it into a neural network.

When your data has features with different ranges, scale each feature independently as part of preprocessing.

As training progresses, neural networks eventually begin to overfit and obtain worse results on never-before-seen data.

If you don’t have much training data, use a small model with only one or two intermediate layers, to avoid severe overfitting.

If your data is divided into many categories, you may cause information bottlenecks if you make the intermediate layers too small.

When you’re working with little data, K-fold validation can help reliably evaluate your model.

### PDF p.285 (printed 261) — 8 Image classification › 8.3 Using a pretrained model › 
Figure 8.13 Training and validation metrics for plain feature extraction

You reach a validation accuracy of slightly over 98%—much better than you achieved in the previous section with the small model trained from scratch. This is a bit of an unfair comparison, however, because ImageNet contains many dog and cat instances, which means that our pretrained model already has the exact knowledge required for the task at hand. This won’t always be the case when you use pretrained features.

However, the plots also indicate that you’re overfitting almost from the start—despite using dropout with a fairly large rate. That’s because this technique doesn’t use data augmentation, which is essential for preventing overfitting with small image datasets. Let’s check the test accuracy:

### PDF p.10 (printed x) — contents ›  › 
5

Fundamentals of machine learning 136 5.1

Generalization: The goal of machine learning 136

Underfitting and overfitting 137 ■ The nature of generalization in deep learning 143

5.2

Evaluating machine-learning models 149

Training, validation, and test sets 149 ■ Beating a common-sense baseline 152 ■ Things to keep in mind about model evaluation 152

5.3

Improving model fit 153

Tuning key gradient descent parameters 153 ■ Using better architecture priors 155 ■ Increasing model capacity 155
