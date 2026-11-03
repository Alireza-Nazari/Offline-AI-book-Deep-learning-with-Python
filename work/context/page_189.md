# Generation bundle for page_189  (PDF page 189, printed page 165)

Page kind: content
Chapter: 5 Fundamentals of machine learning
Section: 5.4 Improving generalization
Subsection: 5.4.4 Regularizing your model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_189.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_189.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p189_e03] (body) networks: given some training data and a network architecture, multiple sets of weight values (multiple models) could explain the data. Simpler models are less likely to overfit than complex ones.
[p189_e04] (body) A simple model in this context is a model where the distribution of parameter values has less entropy (or a model with fewer parameters, as you saw in the previous section). Thus a common way to mitigate overfitting is to put constraints on the complexity of a model by forcing its weights to take only small values, which makes the distribution of weight values more regular. This is called weight regularization, and it’s done by adding to the loss function of the model a cost associated with having large weights. This cost comes in two flavors:
[p189_e05] (bullet) L1 regularization—The cost added is proportional to the absolute value of the weight coefficients (the L1 norm of the weights).
[p189_e06] (bullet) L2 regularization—The cost added is proportional to the square of the value of the weight coefficients (the L2 norm of the weights). L2 regularization is also called weight decay in the context of neural networks. Don’t let the different name confuse you: weight decay is mathematically the same as L2 regularization.
[p189_e07] (body) In Keras, weight regularization is added by passing weight regularizer instances to layers as keyword arguments. Let’s add L2 weight regularization to the movie review classification model.
[p189_e08] (listing_caption) Listing 5.13 Adding L2 weight regularization to the model
[p189_e09] (code)
```
from keras.regularizers import l2
model = keras.Sequential(
    [
        layers.Dense(16, kernel_regularizer=l2(0.002),
activation="relu"),
        layers.Dense(16, kernel_regularizer=l2(0.002),
activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history_l2_reg = model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_split=0.4,
)
```

## Listings
- Listing 5.13: Adding L2 weight regularization to the model (code elements p189_e09)

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

## Previous page (PDF 188) text, for continuity
Figure 5.19 shows how the bigger model fares compared to the reference model. The bigger model starts overfitting almost immediately, after just one epoch, and it overfits much more severely. Its validation loss is also noisier. It gets training loss near zero very quickly. The more capacity the model has, the more quickly it can model the training data (resulting in a low training loss), but the more susceptible it is to overfitting (resulting in a large difference between the training and validation loss).

Figure 5.19 Original model vs. much larger model on IMDB review classification

Adding weight regularization

You may be familiar with the principle of Occam’s razor: given two explanations for something, the explanation most likely to be correct is the simplest one—the one that makes fewer assumptions. This idea also applies to the models learned by neural

## Next page (PDF 190) text, for continuity
`l2(0.002)` means every coefficient in the weight matrix of the layer will add `0.002 * ` `weight_coefficient_value ** 2` to the total loss of the model. Note that because this penalty is only added at training time, the loss for this model will be much higher at training than at test time.

Figure 5.20 shows the effect of the L2 regularization penalty. As you can see, the model with L2 regularization has become much more resistant to overfitting than the reference model, even though both models have the same number of parameters.

Figure 5.20 Effect of L2 weight regularization on validation loss

As an alternative to L2 regularization, you can use one of the following Keras weight regularizers.

Listing 5.14 Different weight regularizers available in Keras

Note that weight regularization is more typically used for smaller deep learning models. Large deep learning models tend to be so overparameterized that imposing constraints on weight values does not have much effect on model capacity and generalization. In these cases, a different regularization technique is preferred: dropout.

Adding dropout

Dropout, developed by Geoff Hinton and his students at the University of Toronto, is one of the most effective and most commonly used regularization techniques for

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.192 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.193 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.187 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.181 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.182 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.276 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.186 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.185 — 5 Fundamentals of machine learning › 5.4 Improving generalization

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.194 (printed 170) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
separate test set for final model evaluation, since information leaks from your validation data to your model may have occurred.

When you start working on a model, your goal is first to achieve a model that has some generalization power and that can overfit. Best practices to do this include tuning your learning rate and batch size, using better architecture priors, increasing model capacity, or simply training longer.

As your model starts overfitting, your goal switches to improving generalization through model regularization. You can reduce your model’s capacity, add dropout or weight regularization, and use early stopping. And naturally, a larger or better dataset is always the number one way to help a model generalize.

### PDF p.192 (printed 168) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
Figure 5.22 shows a plot of the results. This is a clear improvement over the reference model. It also seems to be working much better than L2 regularization since the lowest validation loss reached has improved.

To recap, these are the most common ways to maximize generalization and prevent overfitting in neural networks:

Getting more training data, or better training data

Developing better features

### PDF p.193 (printed 169) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
Figure 5.22 Effect of dropout on validation loss

Reducing the capacity of the model

Adding weight regularization (for smaller models)

Adding dropout

### PDF p.185 (printed 161) — 5 Fundamentals of machine learning › 5.4 Improving generalization › Regularizing your model
Regularization techniques are a set of best practices that actively impede the model’s ability to fit perfectly to the training data, with the goal of making the model perform better during validation. This is called “regularizing” the model because it tends to make the model simpler, more “regular,” its curve smoother, and more “generic”—thus less specific to the training set and better able to generalize by more closely approximating the latent manifold of the data. Keep in mind that “regularizing” a model is a process that should always be guided by an accurate evaluation procedure. You will only achieve generalization if you can measure it.

Let’s review some of the most common regularization techniques and apply them in practice to improve the movie classification model from chapter 4.

### PDF p.276 (printed 252) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
We get a test accuracy of 78.6% (due to the randomness of neural network initializations, you may get numbers within 1 percentage point of that).

Because you have relatively few training samples (2,000), overfitting will be your number-one concern. You already know about a number of techniques that can help mitigate overfitting, such as dropout and weight decay (L2 regularization). We’re now going to work with a new one, specific to computer vision and used almost universally when processing images with deep learning models: data augmentation.

### PDF p.172 (printed 148) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › 
Figure 5.11 A dense sampling of the input space is necessary to learn a model capable of accurate generalization.

past training inputs, without having to use common-sense, abstract reasoning, or external knowledge about the world—all things that machine learning models have no access to. As such, you should always keep in mind that the best way to improve a deep learning model is to train it on more data or better data (of course, adding overly noisy or inaccurate data will harm generalization). A denser coverage of the input data manifold will yield a model that generalizes better. You should never expect a deep learning model to perform anything more than crude interpolation between its training samples, and thus, you should do everything you can to make interpolation as easy as possible. The only thing you will find in a deep learning model is what you put into it: the priors encoded in its architecture and the data it was trained on.

When getting more data isn’t possible, the next-best solution is to modulate the quantity of information that your model is allowed to store, or to add constraints on the smoothness of the model curve. If a network can only afford to memorize a s

### PDF p.287 (printed 263) — 8 Image classification › 8.3 Using a pretrained model › 
With this setup, only the weights from the two `Dense` layers that you added will be trained. That’s a total of four weight tensors: two per layer (the main weight matrix and the bias vector). Note that for these changes to take effect, you must first compile the model. If you ever modify weight trainability after compilation, you should then recompile the model, or these changes will be ignored.

Let’s train our model. We’ll reuse our augmented dataset `augmented_train_dataset`. Thanks to data augmentation, it will take much longer for the model to start overfitting, so we can train for more epochs—let’s do 30:

### PDF p.212 (printed 188) — 6 The universal workflow of machine learning › 6.3 Deploying your model › 
the memory and compute footprint of your model at a small cost in performance metrics. By tuning how much pruning you want to apply, you are in control of the tradeoff between size and accuracy.

Weight quantization—Deep learning models are trained with single-precision floating-point (`float32`) weights. However, it’s possible to quantize weights to 8-bit signed integers (`int8`) to get an inference-only model that’s four times smaller but remains near the accuracy of the original model. Keras models come with a built-in `quantize()` API that can help with this. Simply call `model
` `.quantize("int8")` to compress each weight in your model to a single byte.
