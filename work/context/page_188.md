# Generation bundle for page_188  (PDF page 188, printed page 164)

Page kind: content
Chapter: 5 Fundamentals of machine learning
Section: 5.4 Improving generalization
Subsection: 5.4.4 Regularizing your model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_188.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_188.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p188_e03] (code)
```
        layers.Dense(512, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history_larger_model = model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_split=0.4,
)
```
[p188_e04] (body) Figure 5.19 shows how the bigger model fares compared to the reference model. The bigger model starts overfitting almost immediately, after just one epoch, and it overfits much more severely. Its validation loss is also noisier. It gets training loss near zero very quickly. The more capacity the model has, the more quickly it can model the training data (resulting in a low training loss), but the more susceptible it is to overfitting (resulting in a large difference between the training and validation loss).
[p188_e05] (figure_caption) Figure 5.19 Original model vs. much larger model on IMDB review classification
[p188_e06] (heading_subsubsection) Adding weight regularization
[p188_e07] (body) You may be familiar with the principle of Occam’s razor: given two explanations for something, the explanation most likely to be correct is the simplest one—the one that makes fewer assumptions. This idea also applies to the models learned by neural

## Figures
- Figure 5.19: Original model vs. much larger model on IMDB review classification  | labels inside the figure: (none - image)

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

## Previous page (PDF 187) text, for continuity
Figure 5.18 shows a comparison of the validation losses of the original model and the smaller model.

Figure 5.18 Original model vs. smaller model on IMDb review classification

As you can see, the smaller model starts overfitting later than the reference model (after six epochs rather than four), and its performance degrades more slowly once it starts overfitting.

Now, let’s add to our benchmark a model that has much more capacity—far more than the problem warrants.

Listing 5.12 Version of the model with higher capacity

## Next page (PDF 189) text, for continuity
networks: given some training data and a network architecture, multiple sets of weight values (multiple models) could explain the data. Simpler models are less likely to overfit than complex ones.

A simple model in this context is a model where the distribution of parameter values has less entropy (or a model with fewer parameters, as you saw in the previous section). Thus a common way to mitigate overfitting is to put constraints on the complexity of a model by forcing its weights to take only small values, which makes the distribution of weight values more regular. This is called weight regularization, and it’s done by adding to the loss function of the model a cost associated with having large weights. This cost comes in two flavors:

L1 regularization—The cost added is proportional to the absolute value of the weight coefficients (the L1 norm of the weights).

L2 regularization—The cost added is proportional to the square of the value of the weight coefficients (the L2 norm of the weights). L2 regularization is also called weight decay in the context of neural networks. Don’t let the different name confuse you: weight decay is mathematically the same as L2 regularization.

In Keras, weight regularization is added by passing weight regularizer instances to layers as keyword arguments. Let’s add L2 weight regularization to the movie review classification model.

Listing 5.13 Adding L2 weight regularization to the model

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.181 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.186 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.182 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.192 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.180 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.178 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.149 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.153 — 4 Classification and regression › 4.3 Predicting house prices: A regression example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.192 (printed 168) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
Figure 5.22 shows a plot of the results. This is a clear improvement over the reference model. It also seems to be working much better than L2 regularization since the lowest validation loss reached has improved.

To recap, these are the most common ways to maximize generalization and prevent overfitting in neural networks:

Getting more training data, or better training data

Developing better features

### PDF p.194 (printed 170) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
separate test set for final model evaluation, since information leaks from your validation data to your model may have occurred.

When you start working on a model, your goal is first to achieve a model that has some generalization power and that can overfit. Best practices to do this include tuning your learning rate and batch size, using better architecture priors, increasing model capacity, or simply training longer.

As your model starts overfitting, your goal switches to improving generalization through model regularization. You can reduce your model’s capacity, add dropout or weight regularization, and use early stopping. And naturally, a larger or better dataset is always the number one way to help a model generalize.

### PDF p.161 (printed 137) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › Underfitting and overfitting
For all models you’ve seen in the previous chapter, performance on the held-out validation data initially improved as training went on and then inevitably peaked after a while. This pattern (illustrated in figure 5.1) is universal. You’ll see it with any model type and any dataset.

Figure 5.1 Canonical overfitting behavior

At the beginning of training, optimization and generalization are correlated: the lower the loss on training data, the lower the loss on test data. While this is happening, your model is said to be underfit: there is still progress to be made; the network hasn’t yet modeled all relevant patterns in the training data. But after a certain number of iterations on the training data, generalization stops improving, and validation metrics stall and then begin to degrade: the model is starting to overfit. That is, it’s beginning to

### PDF p.190 (printed 166) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
`l2(0.002)` means every coefficient in the weight matrix of the layer will add `0.002 * ` `weight_coefficient_value ** 2` to the total loss of the model. Note that because this penalty is only added at training time, the loss for this model will be much higher at training than at test time.

Figure 5.20 shows the effect of the L2 regularization penalty. As you can see, the model with L2 regularization has become much more resistant to overfitting than the reference model, even though both models have the same number of parameters.

Figure 5.20 Effect of L2 weight regularization on validation loss

As an alternative to L2 regularization, you can use one of the following Keras weight regularizers.

Listing 5.14 Different weight regularizers available in Keras

### PDF p.138 (printed 114) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › 
Figure 4.5 Training and validation accuracy

As you can see, the training loss decreases with every epoch, and the training accuracy increases with every epoch. That’s what you would expect when running gradient-descent optimization—the quantity you’re trying to minimize should be less with every iteration. But that isn’t the case for the validation loss and accuracy: they seem to peak at the fourth epoch. This is an example of what we warned against earlier: a model that performs better on the training data isn’t necessarily a model that will do better on data it has never seen before. In precise terms, what you’re seeing is overfitting: after the fourth epoch, you’re overoptimizing on the training data, and you end up learning representations that are specific to the training data and don’t generalize to data outside of the training set.

In this case, to prevent overfitting, you could stop training after four epochs. In general, you can use a range of techniques to mitigate overfitting, which we’ll cover in chapter 5. Let’s train a new model from scratch for four epochs and then evaluate it on the test data.

Listing 4.10 Training the model for four epochs

### PDF p.193 (printed 169) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
Figure 5.22 Effect of dropout on validation loss

Reducing the capacity of the model

Adding weight regularization (for smaller models)

Adding dropout

### PDF p.181 (printed 157) — 5 Fundamentals of machine learning › 5.3 Improving model fit › 
The training curves now look exactly like they should: the model fits fast and starts overfitting after eight epochs (see figure 5.15).

Figure 5.15 Validation loss for a model with appropriate capacity

### PDF p.181 (printed 157) — 5 Fundamentals of machine learning › 5.3 Improving model fit › 
you can’t clearly overfit, even after many iterations over the training data. You’re likely to encounter similar curves often in your career.

Remember that it should always be possible to overfit. Much like the problem “the training loss doesn’t go down,” this is an issue that can always be solved. If you can’t seem to be able to overfit, it’s likely a problem with the representational power of your model: you’re going to need a bigger model, one with more capacity—that is, able to store more information. You can increase representational power by adding more layers, using bigger layers (layers with more parameters), or using kinds of layers that are more appropriate for the problem at hand (better architecture priors). Let’s try training a bigger model, one with two intermediate layers with 128 units each:
