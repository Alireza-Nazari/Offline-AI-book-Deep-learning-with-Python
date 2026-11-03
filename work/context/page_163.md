# Generation bundle for page_163  (PDF page 163, printed page 139)

Page kind: content
Chapter: 5 Fundamentals of machine learning
Section: 5.1 Generalization: The goal of machine learning
Subsection: 5.1.1 Underfitting and overfitting
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_163.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_163.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p163_e03] (figure_caption) Figure 5.4 Dealing with outliers: robust fit vs. overfitting
[p163_e04] (heading_subsubsection) Ambiguous features
[p163_e05] (body) Not all data noise comes from inaccuracies—even perfectly clean and neatly labeled data can be noisy when the problem involves uncertainty and ambiguity (see figure 5.5). In classification tasks, it is often the case that some regions of the input feature space are associated with multiple classes at the same time. Let’s say you’re developing a model that takes an image of a banana and predicts whether the banana is unripened, ripe, or rotten. These categories have no objective boundaries, so the same picture might be classified as either unripened or ripe by different human labelers. Similarly, many problems involve randomness. You could use atmospheric pressure data to predict whether it will rain tomorrow, but the exact same measurements may be followed sometimes by rain, sometimes by a clear sky—with some probability.
[p163_e06] (figure_caption) Figure 5.5 Robust fit vs. overfitting giving an ambiguous area of the feature space
[p163_e07] (body) A model could overfit to such probabilistic data by being too confident about ambiguous regions of the feature space, like in figure 5.6. A more robust fit would ignore individual data points and look at the bigger picture.

## Code annotations printed next to listings (callouts, in order)
- Area of uncertainty

## Figures
- Figure 5.4: Dealing with outliers: robust fit vs. overfitting  | labels inside the figure: (none - image)
- Figure 5.5: Robust fit vs. overfitting giving an ambiguous area of the feature space  | labels inside the figure: (none - image)

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

## Previous page (PDF 162) text, for continuity
learn patterns that are specific to the training data but that are misleading or irrelevant when it comes to new data.

Overfitting is particularly likely to occur when your data is noisy, if it involves uncertainty, or if it includes rare features. Let’s look at concrete examples.

Noisy training data

In real-world datasets, it’s fairly common for some inputs to be invalid. Perhaps a MNIST digit could be an all-black image, for instance—or something like figure 5.2.

Figure 5.2 Some pretty weird MNIST training samples

What are these? We don’t know either. But they’re all part of the MNIST training set. What’s even worse, however, is having perfectly valid inputs that end up mislabeled, like those shown in figure 5.3.

Figure 5.3 Mislabeled MNIST training samples

If a model goes out of its way to incorporate such outliers, its generalization performance will degrade, as shown in figure 5.4. For instance, a 4 that looks very close to the mislabeled 4 in figure 5.3 may end up getting classified as a 9.

## Next page (PDF 164) text, for continuity
Rare features and spurious correlations

If you’ve only ever seen two orange tabby cats in your life, and they both happened to be terribly antisocial, you might infer that orange tabby cats are generally likely to be antisocial. That’s overfitting: if you had been exposed to a wider variety of cats, including more orange ones, you’d have learned that cat color is not well correlated with character.

Likewise, machine learning models trained on datasets that include rare feature values are highly susceptible to overfitting. In a sentiment classification task, if the word “cherimoya” (a fruit native to the Andes) only appears in one text in the training data, and this text happens to be negative in sentiment, a poorly regularized model might put a very high weight on this word and always classify new texts that mention cherimoyas as negative, whereas, objectively, there’s nothing negative about the cherimoya.1 Importantly, a feature value doesn’t need to occur only a couple of times to lead to spurious correlations. Consider a word that occurs in 100 samples in your training data, and that’s associated with a positive sentiment 54% of the time and with a negative sentiment 46% of the time. That difference may well be a complete statistical fluke, yet your model is likely to learn to use that feature for its classification task. This is one of the most common sources of overfitting. Here’s a striking example. Take MNIST. Create a new training set by concatenating 784 white noise dimensions to the existing 784 dimensions of the data—so half of the data is now noise. For comparison, also create an equivalent dataset by concatenating 784 all-zeros dimensions. Our concatenation of meaningless features does not at all affect the information content of the data: we’re only addi …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.166 — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning
- PDF p.188 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.134 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.276 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.204 — 6 The universal workflow of machine learning › 6.2 Developing a model
- PDF p.171 — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning
- PDF p.183 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.640 — index › 

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.166 (printed 142) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › 
Despite the data holding the same information in both cases, the validation accuracy of the model trained with noise channels ends up about one percentage point lower— purely through the influence of spurious correlations (figure 5.6). The more noise channels you might add, the further accuracy would degrade.

Figure 5.6 Effect of noise channels on validation accuracy

Noisy features inevitably lead to overfitting. As such, in cases where you aren’t sure whether the features you have are informative or distracting, it’s common to do feature selection before training. Restricting the IMDB data to the top 10,000 most common words was a crude form of feature selection, for instance. The typical way to do feature selection is to compute some usefulness score for each feature available—a measure of how informative the feature is with respect to the task, such as the mutual information between the feature and the labels—and only keep features that are above some threshold. Doing this would filter out the white noise channels in the preceding example.

### PDF p.161 (printed 137) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › 
validation set, and a test set. The reason not to evaluate the models on the same data they were trained on quickly became evident: after just a few epochs, performance on never-before-seen data started diverging from performance on the training data, which always improves as training progresses. The models started to overfit. Overfitting happens in every machine-learning problem.

The fundamental issue in machine learning is the tension between optimization and generalization. Optimization refers to the process of adjusting a model to get the best performance possible on the training data (the learning in machine learning), whereas generalization refers to how well the trained model performs on data it has never seen before. The goal of the game is to get good generalization, of course, but you don’t control generalization; you can only fit the model to its training data. If you do that too well, overfitting kicks in and generalization suffers. But what causes overfitting? How can we achieve good generalization?

### PDF p.644 (printed 620) — index ›  › U
attribute 262

underfitting 137–142 parameters 39

ambiguous features 139 weights 215 noisy training data 138 training rare features and spurious correlations 140–142 diffusion models 525 U-Net denoising autoencoder 522 inference vs. 215 unigrams 403 lower-precision computation 556–562 universal workflow of machine learning training arguments 215, 218 deploying model 183–189 training data 147 developing model 179–183 training loops 10, 39, 206–214 update_state() method 207, 222 callbacks 208, 210 upsampling stage 522 monitoring and visualization with use_casual_mask 448 TensorBoard 212 using fit() with custom training loops 222–225

### PDF p.177 (printed 153) — 5 Fundamentals of machine learning › 5.3 Improving model fit › Improving model fit
To achieve the perfect fit, you must first overfit. Since you don’t know in advance where the boundary lies, you must cross it to find it. Thus, your initial goal as you start working on a problem is to achieve a model that shows some generalization power, and that is able to overfit. Once you have such a model, you’ll focus on refining generalization by fighting overfitting.

There are three common problems you’ll encounter at this stage:

Training doesn’t get started: your training loss doesn’t go down over time.

Training gets started just fine, but your model doesn’t meaningfully generalize: you can’t beat the common-sense baseline you set.

Training and validation loss both go down over time, and you can beat your baseline, but you don’t seem to be able to overfit, which indicates you’re still underfitting.

Let’s see how you can address these issues to achieve the first big milestone of a machine learning project: getting a model that has some generalization power (it can beat a trivial baseline) and is able to overfit.

### PDF p.161 (printed 137) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › Underfitting and overfitting
For all models you’ve seen in the previous chapter, performance on the held-out validation data initially improved as training went on and then inevitably peaked after a while. This pattern (illustrated in figure 5.1) is universal. You’ll see it with any model type and any dataset.

Figure 5.1 Canonical overfitting behavior

At the beginning of training, optimization and generalization are correlated: the lower the loss on training data, the lower the loss on test data. While this is happening, your model is said to be underfit: there is still progress to be made; the network hasn’t yet modeled all relevant patterns in the training data. But after a certain number of iterations on the training data, generalization stops improving, and validation metrics stall and then begin to degrade: the model is starting to overfit. That is, it’s beginning to

### PDF p.183 (printed 159) — 5 Fundamentals of machine learning › 5.4 Improving generalization › Dataset curation
You’ve already learned that generalization in deep learning originates from the latent structure of your data. If your data makes it possible to smoothly interpolate between samples, then you will be able to train a deep learning model that generalizes. If your problem is overly noisy or fundamentally discrete, like, say, list sorting, deep learning will not help you. Deep learning is curve fitting, not magic.

As such, it is essential that you make sure that you’re working with an appropriate dataset. Spending more effort and money on data collection almost always yields a much greater return on investment than spending the same on developing a better model:

Make sure you have enough data. Remember that you need a dense sampling of the input-cross-output space. More data will yield a better model. Sometimes, problems that seem impossible at first become solvable with a larger dataset.

Minimize labeling errors—visualize your inputs to check for anomalies, and proofread your labels.

Clean your data and deal with missing values (we cover this in the next chapter).

If you have many features and you aren’t sure which ones are actually useful, do feature selection.

A particularly i

### PDF p.285 (printed 261) — 8 Image classification › 8.3 Using a pretrained model › 
Figure 8.13 Training and validation metrics for plain feature extraction

You reach a validation accuracy of slightly over 98%—much better than you achieved in the previous section with the small model trained from scratch. This is a bit of an unfair comparison, however, because ImageNet contains many dog and cat instances, which means that our pretrained model already has the exact knowledge required for the task at hand. This won’t always be the case when you use pretrained features.

However, the plots also indicate that you’re overfitting almost from the start—despite using dropout with a fairly large rate. That’s because this technique doesn’t use data augmentation, which is essential for preventing overfitting with small image datasets. Let’s check the test accuracy:

### PDF p.192 (printed 168) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
Figure 5.22 shows a plot of the results. This is a clear improvement over the reference model. It also seems to be working much better than L2 regularization since the lowest validation loss reached has improved.

To recap, these are the most common ways to maximize generalization and prevent overfitting in neural networks:

Getting more training data, or better training data

Developing better features
