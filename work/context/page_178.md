# Generation bundle for page_178  (PDF page 178, printed page 154)

Page kind: content
Chapter: 5 Fundamentals of machine learning
Section: 5.3 Improving model fit
Subsection: 5.3.1 Tuning key gradient descent parameters
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_178.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_178.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p178_e03] (body) data. Even if nothing about your problem makes sense, you should still be able to train something—if only by memorizing the training data.
[p178_e04] (body) When this happens, it’s always a problem with the configuration of the gradient descent process: your choice of optimizer, the distribution of initial values in the weights of your model, your learning rate, or your batch size. All these parameters are interdependent, and as such, it is usually sufficient to tune the learning rate and the batch size while maintaining the rest of the parameters constant. Let’s look at a concrete example: let’s train the MNIST model from chapter 2 with an inappropriately large learning rate, of value 1.
[p178_e05] (listing_caption) Listing 5.7 Training an MNIST model with an incorrectly high learning rate
[p178_e06] (code)
```
(train_images, train_labels), _ = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
model = keras.Sequential(
    [
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1.0),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_images, train_labels, epochs=10, batch_size=128, validation_split=0.2
)
```
[p178_e07] (body) The model quickly reaches a training and validation accuracy in the 20% to 40% range, but cannot get past that. Let’s try to lower the learning rate to a more reasonable value of `1e-2`.
[p178_e08] (listing_caption) Listing 5.8 The same model with a more appropriate learning rate
[p178_e09] (code)
```
model = keras.Sequential(
    [
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-2),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

## Listings
- Listing 5.7: Training an MNIST model with an incorrectly high learning rate (code elements p178_e06)
- Listing 5.8: The same model with a more appropriate learning rate (code elements p178_e09)

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

## Previous page (PDF 177) text, for continuity
ordered by their class, taking the first 80% of the array as your training set and the remaining 20% as your test set will result in your training set containing only classes 0–7, whereas your test set contains only classes 8–9. This seems like a ridiculous mistake, but it’s surprisingly common. For this reason, you usually should randomly shuffle your data before splitting it into training and test sets.

The arrow of time—If you’re trying to predict the future given the past (for example, tomorrow’s weather, stock movements, and so on), you should not randomly shuffle your data before splitting it because doing so will create a temporal leak: your model will effectively be trained on data from the future. In such situations, you should always make sure all data in your test set is posterior to the data in the training set.

Redundancy in your data—If some data points in your data appear twice (fairly common with real-world data), then shuffling the data and splitting it into a training set and a validation set will result in redundancy between the training and validation sets. In effect, you’ll be testing on part of your training data, which is the worst thing you can do! Make sure your training set and validation set are disjoint.

Having a reliable way to evaluate the performance of your model is how you’ll be able to monitor the tension at the heart of machine learning—between optimization and generalization, underfitting and overfitting.

5.3

Improving model fit

To achieve the perfect fit, you must first overfit. Since you don’t know in advance where the boundary lies, you must cross it to find it. Thus, your initial goal as you start working on a problem is to achieve a model that shows some generalization power, and that is able to overfit. Once you have such  …

## Next page (PDF 179) text, for continuity
The model is now able to train.

If you find yourself in a similar situation, try

Lowering or increasing the learning rate. A learning rate that is too high may lead to updates that vastly overshoot a proper fit, like in the previous example, and a learning rate that is too low may make training so slow that it appears to stall.

Increasing the batch size. A batch with more samples will lead to gradients that are more informative and less noisy (lower variance).

You will, eventually, find a configuration that gets training started.

5.3.2

Using better architecture priors

You have a model that fits, but for some reason your validation metrics aren’t improving at all. They remain no better than what a random classifier would achieve: your model trains, but doesn’t generalize. What’s going on?

This is perhaps the worst machine learning situation you can find yourself in. It indicates that something is fundamentally wrong with your approach, and it may not be easy to tell what. Here are some tips. First, it may be that the input data you’re using simply doesn’t contain sufficient information to predict your targets: the problem as formulated is not solvable. This is what happened earlier when we tried to fit an MNIST model where the labels were shuffled: the model would train just fine, but validation accuracy would stay stuck at 10%, because it was plainly impossible to generalize with such a dataset. It may also be that the kind of model you’re using is not suited for the problem at hand. For instance, in chapter 13, you’ll see an example of a timeseries prediction problem where a densely connected architecture isn’t able to beat a trivial baseline, whereas a more appropriate recurrent architecture does manage to generalize well. Using a model that makes the right as …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.181 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.182 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.180 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.186 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.165 — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning
- PDF p.187 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.188 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.192 — 5 Fundamentals of machine learning › 5.4 Improving generalization

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.68 (printed 44) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
Instead, you can use the four-step algorithm outlined at the beginning of this section: modify the parameters little by little based on the current loss value on a random batch of data. Because you’re dealing with a differentiable function, you can compute its gradient, which gives you an efficient way to implement step 4. If you update the weights in the opposite direction from the gradient, the loss will be a little less every time: 1 Draw a batch of training samples `x` and corresponding targets `y_true`. 2 Run the model on `x` to obtain predictions `y_pred` (this is called the forward pass). 3 Compute the loss of the model on the batch, a measure of the mismatch between

`y_pred` and `y_true`. 4 Compute the gradient of the loss with regard to the model’s parameters (this is

called the backward pass).

5 Move the parameters a little in the opposite direction from the gradient—for

example, `W -= learning_rate * gradient`—thus reducing the loss on the batch a bit. The learning rate (`learning_rate` here) would be a scalar factor modulating the “speed” of the gradient descent process. Easy enough! What we just described is called mini-batch stochastic gradient descent (mini-batch

### PDF p.83 (printed 59) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › 
move is defined by the learning rate) in the opposite direction from the gradient.

This is called mini-batch gradient descent.

The entire learning process is made possible by the fact that all tensor operations in neural networks are differentiable, and thus it’s possible to apply the chain rule of derivation to find the gradient function mapping the current parameters and current batch of data to a gradient value. This is called backpropagation.

Two key concepts you’ll see frequently in future chapters are loss and optimizers. These are the two things you need to define before you begin feeding data into a model: – The loss is the quantity you’ll attempt to minimize during training, so it should represent a measure of success for the task you’re trying to solve. – The optimizer specifies the exact way in which the gradient of the loss will be used to update parameters: for instance, it could be the RMSProp optimizer, SGD with momentum, and so on.

### PDF p.207 (printed 183) — 6 The universal workflow of machine learning › 6.3 Deploying your model › Regularizing and tuning your model
Once you’ve achieved statistical power and you’re able to overfit, you know you’re on the right path. At this point, your goal becomes to maximize generalization performance.

This phase will take the most time: you’ll repeatedly modify your model, train it, evaluate on your validation data (not the test data, at this point), modify it again, and repeat, until the model is as good as it can get. Here are some things you should try:

Try different architectures; add or remove layers.

Add dropout.

If your model is small, add L1 or L2 regularization.

Try different hyperparameters (such as the number of units per layer or the learning rate of the optimizer) to find the optimal configuration.

Optionally, iterate on data curation or feature engineering: collect and annotate more data, develop better features, or remove features that don’t seem to be informative.

It’s possible to automate a large chunk of this work by using automated hyperparameter-tuning software, such as KerasTuner. We’ll cover this in chapter 18.

Be mindful of the following: every time you use feedback from your validation process to tune your model, you leak information about the validation process into the mode

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

### PDF p.564 (printed 540) — 18 Best practices for the real world › 18.1 Getting the most out of your models › 
The key to this process is the algorithm that analyzes the relationship between validation performance and various hyperparameter values to choose the next set of hyperparameters to evaluate. Many different techniques are possible: Bayesian optimization, genetic algorithms, simple random search, and so on.

Training the weights of a model is relatively easy: you compute a loss function on a mini-batch of data and then use backpropagation to move the weights in the right direction. Updating hyperparameters, on the other hand, presents unique challenges. Consider that

The hyperparameter space is typically made of discrete decisions and thus isn’t continuous or differentiable. Hence, you typically can’t do gradient descent in hyperparameter space. Instead, you must rely on gradient-free optimization techniques, which, naturally, are far less efficient than gradient descent.

Computing the feedback signal of this optimization process (does this set of hyperparameters lead to a high-performing model on this task?) can be extremely expensive: it requires creating and training a new model from scratch on your dataset.

The feedback signal may be noisy: if a training run performs 0.2% bet

### PDF p.96 (printed 72) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › 
For simplicity, we’ll do batch training instead of mini-batch training: we’ll run each training step (gradient computation and weight update) on the entire data, rather than iterate over the data in small batches. On one hand, this means that each training step will take much longer to run, since we compute the forward pass and the gradients for 2,000 samples at once. On the other hand, each gradient update will be much more effective at reducing the loss on the training data, since it will encompass information from all training samples instead of, say, only 128 random samples. As a result, we will need many fewer steps of training, and we should use a larger learning rate than what we would typically use for mini-batch training (we’ll use `learning_rate = 0.1`, as previously defined).

Listing 3.21 The batch training loop

### PDF p.194 (printed 170) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
separate test set for final model evaluation, since information leaks from your validation data to your model may have occurred.

When you start working on a model, your goal is first to achieve a model that has some generalization power and that can overfit. Best practices to do this include tuning your learning rate and batch size, using better architecture priors, increasing model capacity, or simply training longer.

As your model starts overfitting, your goal switches to improving generalization through model regularization. You can reduce your model’s capacity, add dropout or weight regularization, and use early stopping. And naturally, a larger or better dataset is always the number one way to help a model generalize.

### PDF p.500 (printed 476) — 16 Text generation › 16.2 Training a mini-GPT › 16.2.2 Pretraining the model
Training a large Transformer is famously finicky—the model is sensitive to initializations of parameters and choice of optimizer. When many Transformer layers are stacked, it is easy to suffer from exploding gradients, where parameters update too quickly and our loss function does not converge. A trick that works well is to linearly ease into a full learning rate over a number of warmup steps, so our initial updates to our model parameters are small. This is easy to implement in Keras with a `LearningRateSchedule`.

Listing 16.7 Defining a custom learning rate schedule
