# Generation bundle for page_191  (PDF page 191, printed page 167)

Page kind: content
Chapter: 5 Fundamentals of machine learning
Section: 5.4 Improving generalization
Subsection: 5.4.4 Regularizing your model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_191.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_191.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p191_e03] (body) neural networks. Dropout, applied to a layer, consists of randomly dropping out (setting to zero) a number of output features of the layer during training. Let’s say a given layer would normally return a vector `[0.2, 0.5, 1.3, 0.8, 1.1]` for a given input sample during training. After applying dropout, this vector will have a few zero entries distributed at random: for example, `[0, 0.5, 1.3, 0, 1.1]`. The dropout rate is the fraction of the features that are zeroed out; it’s usually set between 0.2 and 0.5. At test time, no units are dropped out; instead, the layer’s output values are scaled down by a factor equal to the dropout rate, to balance for the fact that more units are active than at training time.
[p191_e04] (body) Consider a NumPy matrix containing the output of a layer, `layer_output`, of shape `(batch_size, features)`. At training time, we zero-out at random a fraction of the values in the matrix:
[p191_e05] (code)
```
layer_output *= np.random.randint(low=0, high=2, size=layer_output.shape)
```
[p191_e06] (body) At test time, we scale down the output by the dropout rate. Here, we scale by 0.5 (because we previously dropped half the units):
[p191_e07] (code)
```
layer_output *= 0.5
```
[p191_e08] (body) Note that this process can be implemented by doing both operations at training time and leaving the output unchanged at test time, which is often the way it’s implemented in practice (see figure 5.21):
[p191_e09] (code)
```
layer_output *= np.random.randint(low=0, high=2, size=layer_output.shape)
layer_output /= 0.5
```
[p191_e10] (figure_caption) Figure 5.21 Dropout applied to an activation matrix at training time, with rescaling happening during training. At test time, the activation matrix is unchanged.

## Code annotations printed next to listings (callouts, in order)
- At training time, drops out 50%
- of the units in the output
- At test time
- At training time
- Note that we’re scaling up
- rather scaling down in this case.

## Figures
- Figure 5.21: Dropout applied to an activation matrix at training time, with rescaling happening during training. At test time, the activation matrix is unchanged.  | labels inside the figure: 0.3, * 2, 0.6, 0.2, 0.7, 0.2, 0.1, 1.9, 0.5, 1.5, 0.0, 0.3, 1.0, 0.0, 0.3, 1.2, 0.0, 0.0, 50%, dropout, 0.6, 0.0, 0.7, 0.2, 0.1, 1.9, 0.0, 1.5, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0

## Equation candidates (formulas written inline in the text)
- [p191_e05] layer_output *= np.random.randint(low=0, high=2, size=layer_output.shape)
- [p191_e07] layer_output *= 0.5
- [p191_e09] layer_output *= np.random.randint(low=0, high=2, size=layer_output.shape)
layer_output /= 0.5

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

## Previous page (PDF 190) text, for continuity
`l2(0.002)` means every coefficient in the weight matrix of the layer will add `0.002 * ` `weight_coefficient_value ** 2` to the total loss of the model. Note that because this penalty is only added at training time, the loss for this model will be much higher at training than at test time.

Figure 5.20 shows the effect of the L2 regularization penalty. As you can see, the model with L2 regularization has become much more resistant to overfitting than the reference model, even though both models have the same number of parameters.

Figure 5.20 Effect of L2 weight regularization on validation loss

As an alternative to L2 regularization, you can use one of the following Keras weight regularizers.

Listing 5.14 Different weight regularizers available in Keras

Note that weight regularization is more typically used for smaller deep learning models. Large deep learning models tend to be so overparameterized that imposing constraints on weight values does not have much effect on model capacity and generalization. In these cases, a different regularization technique is preferred: dropout.

Adding dropout

Dropout, developed by Geoff Hinton and his students at the University of Toronto, is one of the most effective and most commonly used regularization techniques for

## Next page (PDF 192) text, for continuity
This technique may seem strange and arbitrary. Why would this help reduce overfitting? Hinton says he was inspired by, among other things, a fraud-prevention mechanism used by banks:

I went to my bank. The tellers kept changing and I asked one of them why. He said he didn’t know but they got moved around a lot. I figured it must be because it would require cooperation between employees to successfully defraud the bank. This made me realize that randomly removing a different subset of neurons on each example would prevent conspiracies and thus reduce overfitting.

The core idea is that introducing noise in the output values of a layer can break up happenstance patterns that aren’t significant (what Hinton refers to as conspiracies), which the model will start memorizing if no noise is present.

In Keras, you can introduce dropout in a model via the `Dropout` layer, which is applied to the output of the layer right before it. Let’s add two `Dropout` layers in the IMDB model to see how well they do at reducing overfitting.

Listing 5.15 Adding dropout to the IMDB model

Figure 5.22 shows a plot of the results. This is a clear improvement over the reference model. It also seems to be working much better than L2 regularization since the lowest validation loss reached has improved.

To recap, these are the most common ways to maximize generalization and prevent overfitting in neural networks:

Getting more training data, or better training data

Developing better features

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.133 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.397 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.143 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.626 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.204 — 6 The universal workflow of machine learning › 6.2 Developing a model
- PDF p.152 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.178 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.177 — 5 Fundamentals of machine learning › 5.3 Improving model fit

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.397 (printed 373) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
curves, it’s evident that the model is quickly overfitting, despite only having very few units: the training and validation losses start to diverge considerably after a few epochs. You’re already familiar with a classic technique for fighting this phenomenon: dropout, which randomly zeros out input units of a layer to break happenstance correlations in the training data that the layer is exposed to. But how to correctly apply dropout in recurrent networks isn’t a trivial question.

It has long been known that applying dropout before a recurrent layer hinders learning rather than helping with regularization. In 2015, Yarin Gal, as part of his PhD thesis on Bayesian deep learning,5 determined the proper way to use dropout with a recurrent network: the same dropout mask (the same pattern of dropped units) should be applied at every timestep, instead of a dropout mask that varies randomly from timestep to timestep. What’s more, to regularize the representations formed by the recurrent gates of layers such as `GRU` and `LSTM`, a temporally constant dropout mask should be applied to the inner recurrent activations of the layer (a recurrent dropout mask). Using the same dropout mask at ev

### PDF p.278 (printed 254) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
Let’s train the model using data augmentation and dropout. Because we expect overfitting to occur much later during training, we will train for twice as many epochs— 100. Note that we evaluate on images that aren’t augmented—data augmentation is usually only performed at training time, as it is a regularization technique.

### PDF p.133 (printed 109) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › 
Having 16 units means the weight matrix `W` will have shape `(input_dimension,` `16)`: the dot product with `W` will project the input data onto a 16-dimensional representation space (and then you’ll add the bias vector `b` and apply the `relu` operation). You can intuitively understand the dimensionality of your representation space as “how much freedom you’re allowing the model to have when learning internal representations.” Having more units (a higher-dimensional representation space) allows your model to learn more complex representations, but it makes the model more computationally expensive and may lead to learning unwanted patterns (patterns that will improve performance on the training data but not on the test data).

The intermediate layers use `relu` as their activation function, and the final layer uses a sigmoid activation to output a probability (a score between 0 and 1, indicating how likely the review is to be positive). A `relu` (rectified linear unit) is a function meant to zero-out negative values (see figure 4.2), whereas a sigmoid “squashes” arbitrary values into the `[0, 1]` interval (see figure 4.3), outputting something that can be interpreted as a probabili

### PDF p.239 (printed 215) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Training vs. inference
In the low-level training loop examples you’ve seen so far, step 1 (the forward pass) was done via `predictions = model(inputs)`, and step 2 (retrieving the gradients computed by the gradient tape) was done via a backend-specific API, such as

`gradients = tape.gradient(loss, model.weights)` in TensorFlow

`loss.backward()` in PyTorch

`jax.value_and_grad()` in JAX

In the general case, there are actually two subtleties you need to take into account.

Some Keras layers, such as the `Dropout` layer, have different behaviors during training and during inference (when you use them to generate predictions). Such layers expose a `training` Boolean argument in their `call()` method. Calling `dropout(inputs, ` `training=True)` will drop some activation entries, while calling `dropout(inputs, ` `training=False)` does nothing. By extension, Functional models and Sequential models also expose this `training` argument in their `call()` methods. Remember to pass `training=True` when you call a Keras model during the forward pass! Our forward pass thus becomes `predictions = model(inputs, training=True)`. In addition, note that when you retrieve the gradients of the weights of your model, you s

### PDF p.193 (printed 169) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
Figure 5.22 Effect of dropout on validation loss

Reducing the capacity of the model

Adding weight regularization (for smaller models)

Adding dropout

### PDF p.152 (printed 128) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Preparing the data
Note that the quantities used for normalizing the test data are computed using the training data. You should never use in your workflow any quantity computed on the test data, even for something as simple as data normalization.

In addition, we should also scale the targets. Our normalized inputs have their value in a small range close to 0, and our model’s weights are initialized with small random values. This means that our model’s prediction will also be small values when we start training. If the targets are in the range 60,000–500,000, the model is going to need very large weight values to output those. With a small learning rate, it would take a very long time to get there. The simplest fix is to divide all target values by 100,000, so that the smallest target becomes 0.6, and the largest becomes 5. We can then convert the model’s predictions back to dollar values by multiplying them by 100,000 accordingly.

Listing 4.26 Scaling the targets

### PDF p.404 (printed 380) — 13 Timeseries forecasting › 13.4 Going even further › Summary
As you first learned in chapter 6, when approaching a new problem, it’s good to first establish commonsense baselines for your metric of choice. If you don’t have a baseline to beat, you can’t tell whether you’re making real progress.

Try simple models before expensive ones, to justify the additional expense. Sometimes a simple model will turn out to be your best option.

When you have data where ordering matters—in particular, for timeseries data— recurrent networks are a great fit and easily outperform models that first flatten the temporal data. The two essential RNN layers available in Keras are the `LSTM` layer and the `GRU` layer.

To use dropout with recurrent networks, you should use a time-constant dropout mask and recurrent dropout mask. These are built into Keras recurrent layers, so all you have to do is use the `recurrent_dropout` arguments of recurrent layers.

Stacked RNNs provide more representational power than a single RNN layer. They’re also much more expensive and thus not always worth it. Although they offer clear gains on complex problems (such as machine translation), they may not always be relevant to smaller, simpler problems.

### PDF p.397 (printed 373) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
Listing 13.22 Training and evaluating a dropout-regularized LSTM
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.LSTM(32, recurrent_dropout=0.25)(inputs)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "jena_lstm_dropout.keras", save_best_only=True
    )
]
