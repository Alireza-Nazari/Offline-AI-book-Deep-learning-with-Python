# Generation bundle for page_180  (PDF page 180, printed page 156)

Page kind: content
Chapter: 5 Fundamentals of machine learning
Section: 5.3 Improving model fit
Subsection: 5.3.3 Increasing model capacity
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_180.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_180.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p180_e03] (body) Consider the following small model—a simple logistic regression—trained on MNIST pixels.
[p180_e04] (listing_caption) Listing 5.9 A simple logistic regression on MNIST
[p180_e05] (code)
```
model = keras.Sequential([layers.Dense(10, activation="softmax")])
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history_small_model = model.fit(
    train_images, train_labels, epochs=20, batch_size=128, validation_split=0.2
)
```
[p180_e06] (body) You get loss curves that look like this (see figure 5.14):
[p180_e07] (code)
```
import matplotlib.pyplot as plt
val_loss = history_small_model.history["val_loss"]
epochs = range(1, 21)
plt.plot(epochs, val_loss, "b-", label="Validation loss")
plt.title("Validation loss for a model with insufficient capacity")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
```
[p180_e08] (figure_caption) Figure 5.14 Effect of insufficient model capacity on loss curves
[p180_e09] (body) Validation metrics seem to stall or to improve very slowly, instead of peaking and reversing course. The validation loss goes to 0.26 and just stays there. You can fit, but

## Listings
- Listing 5.9: A simple logistic regression on MNIST (code elements p180_e05)

## Figures
- Figure 5.14: Effect of insufficient model capacity on loss curves  | labels inside the figure: (none - image)

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

## Previous page (PDF 179) text, for continuity
The model is now able to train.

If you find yourself in a similar situation, try

Lowering or increasing the learning rate. A learning rate that is too high may lead to updates that vastly overshoot a proper fit, like in the previous example, and a learning rate that is too low may make training so slow that it appears to stall.

Increasing the batch size. A batch with more samples will lead to gradients that are more informative and less noisy (lower variance).

You will, eventually, find a configuration that gets training started.

5.3.2

Using better architecture priors

You have a model that fits, but for some reason your validation metrics aren’t improving at all. They remain no better than what a random classifier would achieve: your model trains, but doesn’t generalize. What’s going on?

This is perhaps the worst machine learning situation you can find yourself in. It indicates that something is fundamentally wrong with your approach, and it may not be easy to tell what. Here are some tips. First, it may be that the input data you’re using simply doesn’t contain sufficient information to predict your targets: the problem as formulated is not solvable. This is what happened earlier when we tried to fit an MNIST model where the labels were shuffled: the model would train just fine, but validation accuracy would stay stuck at 10%, because it was plainly impossible to generalize with such a dataset. It may also be that the kind of model you’re using is not suited for the problem at hand. For instance, in chapter 13, you’ll see an example of a timeseries prediction problem where a densely connected architecture isn’t able to beat a trivial baseline, whereas a more appropriate recurrent architecture does manage to generalize well. Using a model that makes the right as …

## Next page (PDF 181) text, for continuity
you can’t clearly overfit, even after many iterations over the training data. You’re likely to encounter similar curves often in your career.

Remember that it should always be possible to overfit. Much like the problem “the training loss doesn’t go down,” this is an issue that can always be solved. If you can’t seem to be able to overfit, it’s likely a problem with the representational power of your model: you’re going to need a bigger model, one with more capacity—that is, able to store more information. You can increase representational power by adding more layers, using bigger layers (layers with more parameters), or using kinds of layers that are more appropriate for the problem at hand (better architecture priors). Let’s try training a bigger model, one with two intermediate layers with 128 units each:

The training curves now look exactly like they should: the model fits fast and starts overfitting after eight epochs (see figure 5.15).

Figure 5.15 Validation loss for a model with appropriate capacity

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.178 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.165 — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning
- PDF p.187 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.182 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.275 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.285 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.188 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.186 — 5 Fundamentals of machine learning › 5.4 Improving generalization

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.182 (printed 158) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
Figure 5.16 Effect of excessive model capacity on validation loss

### PDF p.193 (printed 169) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
Figure 5.22 Effect of dropout on validation loss

Reducing the capacity of the model

Adding weight regularization (for smaller models)

Adding dropout

### PDF p.187 (printed 163) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
Figure 5.18 shows a comparison of the validation losses of the original model and the smaller model.

Figure 5.18 Original model vs. smaller model on IMDb review classification

As you can see, the smaller model starts overfitting later than the reference model (after six epochs rather than four), and its performance degrades more slowly once it starts overfitting.

Now, let’s add to our benchmark a model that has much more capacity—far more than the problem warrants.

Listing 5.12 Version of the model with higher capacity

### PDF p.167 (printed 143) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › The nature of generalization in deep learning
A remarkable fact about deep learning models is that they can be trained to fit anything, as long as they have enough representational power.

Don’t believe me? Try shuffling the order of the MNIST labels and train a model on that. Even though there is no relationship whatsoever between the inputs and the shuffled labels, the training loss goes down just fine, even with a relatively small model. Naturally, the validation loss does not improve at all over time, since there is no possibility of generalization in this setting.

Listing 5.4 Fitting an MNIST model with randomly shuffled labels

### PDF p.188 (printed 164) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
Figure 5.19 shows how the bigger model fares compared to the reference model. The bigger model starts overfitting almost immediately, after just one epoch, and it overfits much more severely. Its validation loss is also noisier. It gets training loss near zero very quickly. The more capacity the model has, the more quickly it can model the training data (resulting in a low training loss), but the more susceptible it is to overfitting (resulting in a large difference between the training and validation loss).

Figure 5.19 Original model vs. much larger model on IMDB review classification

### PDF p.182 (printed 158) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
Note that while it is standard to work with models that are way overparameterized for the problem at hand, there can definitely be such a thing as too much memorization capacity. You’ll know your model is too large if it starts overfitting right away. Here’s what happens for an MNIST model with three intermediate layers with 2,048 units each (see figure 5.16):

### PDF p.541 (printed 517) — 17 Image generation › 17.1 Deep learning for image generation › 
Finally, you’re ready to instantiate and train the model on MNIST digits. Because `compute_loss()` already takes care of the loss, you don’t specify an external loss at compile time (`loss=None`), which, in turn, means you won’t pass target data during training (as you can see, you only pass `x_train` to the model in `fit`).

Listing 17.5 Training the VAE

### PDF p.425 (printed 401) — 14 Text classification › 14.4 Set models › 
Figure 14 2 Training and validation metrics for our bag of words model

We can see that validation performance levels off rather than significantly declining; our model is so simple it cannot really overfit. Let’s try evaluating it on our test set.
