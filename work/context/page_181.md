# Generation bundle for page_181  (PDF page 181, printed page 157)

Page kind: content
Chapter: 5 Fundamentals of machine learning
Section: 5.3 Improving model fit
Subsection: 5.3.3 Increasing model capacity
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_181.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_181.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p181_e03] (body) you can’t clearly overfit, even after many iterations over the training data. You’re likely to encounter similar curves often in your career.
[p181_e04] (body) Remember that it should always be possible to overfit. Much like the problem “the training loss doesn’t go down,” this is an issue that can always be solved. If you can’t seem to be able to overfit, it’s likely a problem with the representational power of your model: you’re going to need a bigger model, one with more capacity—that is, able to store more information. You can increase representational power by adding more layers, using bigger layers (layers with more parameters), or using kinds of layers that are more appropriate for the problem at hand (better architecture priors). Let’s try training a bigger model, one with two intermediate layers with 128 units each:
[p181_e05] (code)
```
model = keras.Sequential(
    [
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history_large_model = model.fit(
    train_images,
    train_labels,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
)
```
[p181_e06] (body) The training curves now look exactly like they should: the model fits fast and starts overfitting after eight epochs (see figure 5.15).
[p181_e07] (figure_caption) Figure 5.15 Validation loss for a model with appropriate capacity

## Figures
- Figure 5.15: Validation loss for a model with appropriate capacity  | labels inside the figure: (none - image)

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

## Previous page (PDF 180) text, for continuity
Consider the following small model—a simple logistic regression—trained on MNIST pixels.

Listing 5.9 A simple logistic regression on MNIST

You get loss curves that look like this (see figure 5.14):

Figure 5.14 Effect of insufficient model capacity on loss curves

Validation metrics seem to stall or to improve very slowly, instead of peaking and reversing course. The validation loss goes to 0.26 and just stays there. You can fit, but

## Next page (PDF 182) text, for continuity
Note that while it is standard to work with models that are way overparameterized for the problem at hand, there can definitely be such a thing as too much memorization capacity. You’ll know your model is too large if it starts overfitting right away. Here’s what happens for an MNIST model with three intermediate layers with 2,048 units each (see figure 5.16):

Figure 5.16 Effect of excessive model capacity on validation loss

5.4

Improving generalization

Once your model has shown to have some generalization power and to be able to overfit, it’s time to switch your focus toward maximizing generalization.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.178 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.187 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.188 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.192 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.186 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.149 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.177 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.189 — 5 Fundamentals of machine learning › 5.4 Improving generalization

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.177 (printed 153) — 5 Fundamentals of machine learning › 5.3 Improving model fit › Improving model fit
To achieve the perfect fit, you must first overfit. Since you don’t know in advance where the boundary lies, you must cross it to find it. Thus, your initial goal as you start working on a problem is to achieve a model that shows some generalization power, and that is able to overfit. Once you have such a model, you’ll focus on refining generalization by fighting overfitting.

There are three common problems you’ll encounter at this stage:

Training doesn’t get started: your training loss doesn’t go down over time.

Training gets started just fine, but your model doesn’t meaningfully generalize: you can’t beat the common-sense baseline you set.

Training and validation loss both go down over time, and you can beat your baseline, but you don’t seem to be able to overfit, which indicates you’re still underfitting.

Let’s see how you can address these issues to achieve the first big milestone of a machine learning project: getting a model that has some generalization power (it can beat a trivial baseline) and is able to overfit.

### PDF p.194 (printed 170) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
separate test set for final model evaluation, since information leaks from your validation data to your model may have occurred.

When you start working on a model, your goal is first to achieve a model that has some generalization power and that can overfit. Best practices to do this include tuning your learning rate and batch size, using better architecture priors, increasing model capacity, or simply training longer.

As your model starts overfitting, your goal switches to improving generalization through model regularization. You can reduce your model’s capacity, add dropout or weight regularization, and use early stopping. And naturally, a larger or better dataset is always the number one way to help a model generalize.

### PDF p.185 (printed 161) — 5 Fundamentals of machine learning › 5.4 Improving generalization › Reducing the network’s size
You’ve already learned that a model that is too small will not overfit. The simplest way to mitigate overfitting is to reduce the size of the model (the number of learnable parameters in the model, determined by the number of layers and the number of units per layer). If the model has limited memorization resources, it won’t be able to simply memorize its training data. To minimize its loss, it will have to resort to learning compressed representations that have predictive power regarding the targets—precisely the type of representations we’re interested in. At the same time, keep in mind that you should use models that have enough parameters that they don’t underfit: your model shouldn’t be starved for memorization resources. There is a compromise to be found between too much capacity and not enough capacity.

### PDF p.188 (printed 164) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
Figure 5.19 shows how the bigger model fares compared to the reference model. The bigger model starts overfitting almost immediately, after just one epoch, and it overfits much more severely. Its validation loss is also noisier. It gets training loss near zero very quickly. The more capacity the model has, the more quickly it can model the training data (resulting in a low training loss), but the more susceptible it is to overfitting (resulting in a large difference between the training and validation loss).

Figure 5.19 Original model vs. much larger model on IMDB review classification

### PDF p.179 (printed 155) — 5 Fundamentals of machine learning › 5.3 Improving model fit › Increasing model capacity
If you manage to get to a model that fits, where validation metrics are going down, and that seems to achieve at least some level of generalization power, congratulations: you’re almost there. Next, you need to get your model to start overfitting.

### PDF p.624 (printed 600) — 20 Conclusions › 20.1 Key concepts in review › 
Develop a first model that beats a trivial common-sense baseline, thus demonstrating that

machine learning can work on your problem. This may not always be the case!

Gradually refine your model architecture by tuning hyperparameters and adding regularization. Make changes based on performance on the validation data only, not the test data or the training data. Remember that you should get your model to overfit (thus identifying a model capacity level that’s greater than you need) and only then begin to add regularization or downsize your model. Beware of validation-set overfitting when tuning hyperparameters—the fact that your hyperparameters may end up being overspecialized to the validation set. Avoiding this is the purpose of having a separate test set!

Deploy your final model in production—as a web API, as part of a JavaScript or C++ application, on an embedded device, etc. Keep monitoring its performance on real-world data and use your findings to refine the next iteration of the model!

### PDF p.399 (printed 375) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3.5 Stacking recurrent layers
Because you’re no longer overfitting, but seem to have hit a performance bottleneck, you should consider increasing the capacity and expressive power of the network. Recall the description of the universal machine learning workflow: it’s generally a good idea to increase the capacity of your model until overfitting becomes the primary obstacle (assuming you’re already taking basic steps to mitigate overfitting, such as using dropout). As long as you aren’t overfitting too badly, you’re likely under capacity.

Increasing network capacity is typically done by increasing the number of units in the layers or adding more layers. Recurrent layer stacking is a classic way to build more powerful recurrent networks: for instance, not too long ago the Google Translate algorithm was powered by a stack of seven large `LSTM` layers—that’s huge. To stack recurrent layers on top of each other in Keras, all intermediate layers should return their full sequence of outputs (a rank-3 tensor) rather than their output at the last timestep. As you’ve already learned, this is done by specifying `return_sequences=True`. In the following example, we’ll try a stack of two dropout-regularized recurrent layer

### PDF p.157 (printed 133) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › 
According to this plot (see figure 4.11), validation MAE stops improving significantly after 120–140 epochs (this number includes the 10 epochs we omitted). Past that point, you start overfitting.

Once you’re finished tuning other parameters of the model (in addition to the number of epochs, you could also adjust the size of the intermediate layers), you can train a final production model on all of the training data, with the best parameters, and then look at its performance on the test data.

Figure 4.11 Validation MAE by epoch, excluding the first 10 data points

Listing 4.33 Training the final model
