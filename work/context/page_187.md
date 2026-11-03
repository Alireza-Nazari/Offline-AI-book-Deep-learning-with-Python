# Generation bundle for page_187  (PDF page 187, printed page 163)

Page kind: content
Chapter: 5 Fundamentals of machine learning
Section: 5.4 Improving generalization
Subsection: 5.4.4 Regularizing your model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_187.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_187.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p187_e03] (code)
```
        layers.Dense(4, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history_smaller_model = model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_split=0.4,
)
```
[p187_e04] (body) Figure 5.18 shows a comparison of the validation losses of the original model and the smaller model.
[p187_e05] (figure_caption) Figure 5.18 Original model vs. smaller model on IMDb review classification
[p187_e06] (body) As you can see, the smaller model starts overfitting later than the reference model (after six epochs rather than four), and its performance degrades more slowly once it starts overfitting.
[p187_e07] (body) Now, let’s add to our benchmark a model that has much more capacity—far more than the problem warrants.
[p187_e08] (listing_caption) Listing 5.12 Version of the model with higher capacity
[p187_e09] (code)
```
model = keras.Sequential(
    [
        layers.Dense(512, activation="relu"),
```

## Listings
- Listing 5.12: Version of the model with higher capacity (code elements p187_e09)

## Figures
- Figure 5.18: Original model vs. smaller model on IMDb review classification  | labels inside the figure: (none - image)

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

## Previous page (PDF 186) text, for continuity
Unfortunately, there is no magical formula to determine the right number of layers or the right size for each layer. You must evaluate an array of different architectures (on your validation set, not on your test set, of course) to find the correct model size for your data. The general workflow to find an appropriate model size is to start with relatively few layers and parameters and increase the size of the layers or add new layers until you see diminishing returns with regard to validation loss. Let’s try this on the movie-review classification model. Here’s a condensed version of the model from chapter 4.

Listing 5.10 Original model

Now let’s try to replace it with this smaller model.

Listing 5.11 Version of the model with lower capacity

## Next page (PDF 188) text, for continuity
Figure 5.19 shows how the bigger model fares compared to the reference model. The bigger model starts overfitting almost immediately, after just one epoch, and it overfits much more severely. Its validation loss is also noisier. It gets training loss near zero very quickly. The more capacity the model has, the more quickly it can model the training data (resulting in a low training loss), but the more susceptible it is to overfitting (resulting in a large difference between the training and validation loss).

Figure 5.19 Original model vs. much larger model on IMDB review classification

Adding weight regularization

You may be familiar with the principle of Occam’s razor: given two explanations for something, the explanation most likely to be correct is the simplest one—the one that makes fewer assumptions. This idea also applies to the models learned by neural

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.181 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.182 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.192 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.180 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.178 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.149 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.626 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.143 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.193 (printed 169) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
Figure 5.22 Effect of dropout on validation loss

Reducing the capacity of the model

Adding weight regularization (for smaller models)

Adding dropout

### PDF p.194 (printed 170) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
separate test set for final model evaluation, since information leaks from your validation data to your model may have occurred.

When you start working on a model, your goal is first to achieve a model that has some generalization power and that can overfit. Best practices to do this include tuning your learning rate and batch size, using better architecture priors, increasing model capacity, or simply training longer.

As your model starts overfitting, your goal switches to improving generalization through model regularization. You can reduce your model’s capacity, add dropout or weight regularization, and use early stopping. And naturally, a larger or better dataset is always the number one way to help a model generalize.

### PDF p.192 (printed 168) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
Figure 5.22 shows a plot of the results. This is a clear improvement over the reference model. It also seems to be working much better than L2 regularization since the lowest validation loss reached has improved.

To recap, these are the most common ways to maximize generalization and prevent overfitting in neural networks:

Getting more training data, or better training data

Developing better features

### PDF p.181 (printed 157) — 5 Fundamentals of machine learning › 5.3 Improving model fit › 
The training curves now look exactly like they should: the model fits fast and starts overfitting after eight epochs (see figure 5.15).

Figure 5.15 Validation loss for a model with appropriate capacity

### PDF p.190 (printed 166) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
Note that weight regularization is more typically used for smaller deep learning models. Large deep learning models tend to be so overparameterized that imposing constraints on weight values does not have much effect on model capacity and generalization. In these cases, a different regularization technique is preferred: dropout.

### PDF p.161 (printed 137) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › Underfitting and overfitting
For all models you’ve seen in the previous chapter, performance on the held-out validation data initially improved as training went on and then inevitably peaked after a while. This pattern (illustrated in figure 5.1) is universal. You’ll see it with any model type and any dataset.

Figure 5.1 Canonical overfitting behavior

At the beginning of training, optimization and generalization are correlated: the lower the loss on training data, the lower the loss on test data. While this is happening, your model is said to be underfit: there is still progress to be made; the network hasn’t yet modeled all relevant patterns in the training data. But after a certain number of iterations on the training data, generalization stops improving, and validation metrics stall and then begin to degrade: the model is starting to overfit. That is, it’s beginning to

### PDF p.279 (printed 255) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
Let’s plot the results again; see figure 8.11. Thanks to data augmentation and dropout, we start overfitting much later, around epochs 60–70 (compared to epoch 10 for the original model). The validation accuracy ends up peaking above 85%—a big improvement over our first try.

Figure 8.11 Training and validation metrics with data augmentation

Let’s check the test accuracy.

Listing 8.22 Evaluating the model on the test set

### PDF p.161 (printed 137) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › 
validation set, and a test set. The reason not to evaluate the models on the same data they were trained on quickly became evident: after just a few epochs, performance on never-before-seen data started diverging from performance on the training data, which always improves as training progresses. The models started to overfit. Overfitting happens in every machine-learning problem.

The fundamental issue in machine learning is the tension between optimization and generalization. Optimization refers to the process of adjusting a model to get the best performance possible on the training data (the learning in machine learning), whereas generalization refers to how well the trained model performs on data it has never seen before. The goal of the game is to get good generalization, of course, but you don’t control generalization; you can only fit the model to its training data. If you do that too well, overfitting kicks in and generalization suffers. But what causes overfitting? How can we achieve good generalization?
