# Generation bundle for page_568  (PDF page 568, printed page 544)

Page kind: content
Chapter: 18 Best practices for the real world
Section: 18.1 Getting the most out of your models
Subsection: 18.1.1 Hyperparameter optimization
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_568.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_568.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p568_e03] (listing_caption) Listing 18.3 Querying the best hyperparameter configurations
[p568_e04] (code)
```
top_n = 4
best_hps = tuner.get_best_hyperparameters(top_n)
```
[p568_e05] (body) Usually, when retraining these models, you may want to include the validation data as part of the training data since you won’t be making any further hyperparameter changes, and thus you will no longer be evaluating performance on the validation data. In our example, we’d train these final models on the totality of the original MNIST training data, without reserving a validation set.
[p568_e06] (body) Before we can train on the full training data, though, there’s one last parameter we need to settle: the optimal number of epochs to train for. Typically, you’ll want to train the new models for longer than you did during the search: using an aggressive `patience` value in the `EarlyStopping` callback saves time during the search, but may lead to underfitted models. Just use the validation set to find the best epoch:
[p568_e07] (code)
```
def get_best_epoch(hp):
    model = build_model(hp)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=10
        )
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=128,
        callbacks=callbacks,
    )
    val_loss_per_epoch = history.history["val_loss"]
    best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
    print(f"Best epoch: {best_epoch}")
    return best_epoch
```
[p568_e08] (body) And finally, train on the full dataset for just a bit longer than this epoch count, since you’re training on more data—20% more, in this case:
[p568_e09] (code)
```
def get_best_trained_model(hp):
    best_epoch = get_best_epoch(hp)
    model = build_model(hp)
    model.fit(
```

## Code annotations printed next to listings (callouts, in order)
- Returns a list of HyperParameters
- objects, which you can pass to the
- model-building function
- Note the very high
- patience value.

## Listings
- Listing 18.3: Querying the best hyperparameter configurations (code elements p568_e04)

## Chapter outline
Chapter: 18 Best practices for the real world (PDF pages 562-587)
  - 18.1 Getting the most out of your models (p.563)
      - 18.1.1 Hyperparameter optimization (p.563)
      - 18.1.2 Model ensembling (p.570)
  - 18.2 Scaling up model training with multiple devices (p.572)
      - 18.2.1 Multi-GPU training (p.572)
      - 18.2.2 Distributed training in practice (p.574)
      - 18.2.3 TPU training (p.579)
  - 18.3 Speeding up training and inference with lower-precision computation (p.580)
      - 18.3.1 Understanding floating-point precision (p.580)
      - 18.3.2 Float16 inference (p.582)
      - 18.3.3 Mixed-precision training (p.583)
      - 18.3.4 Using loss scaling with mixed precision (p.583)
      - 18.3.5 Beyond mixed precision: float8 training (p.584)
      - 18.3.6 Faster inference with quantization (p.585)

## Previous page (PDF 567) text, for continuity
Finally, let’s launch the search. Don’t forget to pass validation data and make sure not to use your test set as validation data—otherwise, you’d quickly start overfitting to your test data, and you wouldn’t be able to trust your test metrics anymore:

The previous example will run in just a few minutes since we’re only looking at a few possible choices and we’re training on MNIST. However, with a typical search space and dataset, you’ll often find yourself letting the hyperparameter search run overnight or even over several days. If your search process crashes, you can always restart it—just specify `overwrite=False` in the tuner so that it can resume from the trial logs stored on disk.

Once the search is complete, you can query the best hyperparameter configurations, which you can use to create high-performing models that you can then retrain.

## Next page (PDF 569) text, for continuity
If you’re not worried about slightly underperforming, there’s a shortcut you can take: just use the tuner to reload the top-performing models with the best weights saved during the hyperparameter search, without retraining new models from scratch:

NOTE One important issue to keep in mind when doing automatic hyperparameter optimization at scale is validation-set overfitting. Because you’re updating hyperparameters based on a signal that is computed using your validation data, you’re effectively training them on the validation data, and thus, they will quickly overfit to the validation data. Always keep this in mind.

The art of crafting the right search space

Overall, hyperparameter optimization is a powerful technique that is an absolute requirement to get to state-of-the-art models on any task or to win machine learning competitions. Think about it: once upon a time, people handcrafted the features that went into shallow machine learning models. That was very suboptimal. Now deep learning automates the task of hierarchical feature engineering—features are learned using a feedback signal, not hand-tuned, and that’s the way it should be. In the same way, you shouldn’t handcraft your model architectures; you should optimize them in a principled way.

However, doing hyperparameter tuning is not a replacement for being familiar with model architecture best practices: search spaces grow combinatorially with the number of choices, so it would be far too expensive to turn everything into a hyperparameter and let the tuner sort it out. You need to be smart about designing the right search space. Hyperparameter tuning is automation, not magic: you use it to automate experiments that you would otherwise have run by hand, but you still need to handpick experiment configurations …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.233 — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops
- PDF p.135 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.564 — 18 Best practices for the real world › 18.1 Getting the most out of your models
- PDF p.157 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.563 — 18 Best practices for the real world › 18.1 Getting the most out of your models
- PDF p.138 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.234 — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops
- PDF p.287 — 8 Image classification › 8.3 Using a pretrained model

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.233 (printed 209) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › The EarlyStopping and ModelCheckpoint callbacks
When you’re training a model, there are many things you can’t predict at the start. In particular, you can’t tell how many epochs will be needed to get to an optimal validation loss. Our examples so far have adopted the strategy of training for enough epochs that you begin overfitting, using the first run to figure out the optimal number of epochs, and then finally launching a new training run from scratch using this optimal number. Of course, this approach is wasteful. A much better way to handle this is to stop training when you measure that the validation loss is no longer improving. This can be achieved using the `EarlyStopping` callback.

The `EarlyStopping` callback interrupts training once a target metric being monitored has stopped improving for a fixed number of epochs. For instance, this callback allows you to interrupt training as soon as you start overfitting, thus avoiding having to retrain your model for a smaller number of epochs. This callback is typically used in combination with `ModelCheckpoint`, which lets you continually save the model during training (and, optionally, save only the current best model so far: the version of the model that achieved the best perf

### PDF p.157 (printed 133) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › 
According to this plot (see figure 4.11), validation MAE stops improving significantly after 120–140 epochs (this number includes the 10 epochs we omitted). Past that point, you start overfitting.

Once you’re finished tuning other parameters of the model (in addition to the number of epochs, you could also adjust the size of the intermediate layers), you can train a final production model on all of the training data, with the best parameters, and then look at its performance on the test data.

Figure 4.11 Validation MAE by epoch, excluding the first 10 data points

Listing 4.33 Training the final model

### PDF p.173 (printed 149) — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models › Training, validation, and test sets
You may ask, why not have two sets: a training set and a test set? You’d train on the training data and evaluate on the test data. Much simpler! The reason is that developing a model always involves tuning its configuration: for example, choosing the number of layers or the size of the layers (called the hyperparameters of the model, to distinguish them from the parameters, which are the network’s weights). You do this tuning by using as a feedback signal the performance of the model on the validation data. In essence, this tuning is a form of learning: a search for a good configuration in some parameter space. As a result, tuning the configuration of the model based on its performance on the validation set can quickly result in overfitting to the validation set, even though your model is never directly trained on it. Central to this phenomenon is the notion of information leaks. Every time you tune a hyperparameter of your model based on the model’s performance on the validation set, some information about the validation data leaks into the model. If you do this only once, for one parameter, then very few bits of information will leak, and your validation set will remain reliable 

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

### PDF p.138 (printed 114) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › 
Figure 4.5 Training and validation accuracy

As you can see, the training loss decreases with every epoch, and the training accuracy increases with every epoch. That’s what you would expect when running gradient-descent optimization—the quantity you’re trying to minimize should be less with every iteration. But that isn’t the case for the validation loss and accuracy: they seem to peak at the fourth epoch. This is an example of what we warned against earlier: a model that performs better on the training data isn’t necessarily a model that will do better on data it has never seen before. In precise terms, what you’re seeing is overfitting: after the fourth epoch, you’re overoptimizing on the training data, and you end up learning representations that are specific to the training data and don’t generalize to data outside of the training set.

In this case, to prevent overfitting, you could stop training after four epochs. In general, you can use a range of techniques to mitigate overfitting, which we’ll cover in chapter 5. Let’s train a new model from scratch for four epochs and then evaluate it on the test data.

Listing 4.10 Training the model for four epochs

### PDF p.185 (printed 161) — 5 Fundamentals of machine learning › 5.4 Improving generalization › Using early stopping
In deep learning, we always use models that are vastly overparameterized: they have way more degrees of freedom than the minimum necessary to fit to the latent manifold of the data. This overparameterization is not an issue because you never fully fit a deep learning model. Such a fit wouldn’t generalize at all. You will always interrupt training long before you’ve reached the minimum possible training loss.

Finding the exact point during training where you’ve reached the most generalizable fit—the exact boundary between an underfit curve and an overfit curve—is one of the most effective things you can do to improve generalization. In the examples from the previous chapter, we would start by training our models for longer than needed to figure out the number of epochs that yielded the best validation metrics, then we would retrain a new model for exactly that number of epochs. This is pretty standard. However, it requires you to do redundant work, which can sometimes be expensive. Naturally, you could just save your model at the end of each epoch, then once you’ve found the best epoch, reuse the closest saved model you have. In Keras, it’s typical to do this with an `EarlyStopping

### PDF p.135 (printed 111) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › Validating your approach
As you learned in chapter 3, a deep learning model should never be evaluated on its training data—it’s standard practice to use a “validation set” to monitor the accuracy of the model during training. Here, you’ll create a validation set by setting apart 10,000 samples from the original training data.

You might ask, why not simply use the test data to evaluate the model? That seems like it would be easier. The reason is that you’re going to want to use the results you get on the validation set to inform your next choices to improve training—for instance, your choice of what model size to use or how many epochs to train for. When you start doing this, your validation scores stop being an accurate reflection of the performance of the model on brand-new data, since the model has been deliberately modified to perform better on the validation data. It’s good to keep around a set of never-before-seen samples that you can use to perform the final evaluation round in a completely unbiased way, and that’s exactly what the test set is. We’ll talk more about this in the next chapter.

Listing 4.6 Setting aside a validation set

### PDF p.564 (printed 540) — 18 Best practices for the real world › 18.1 Getting the most out of your models › 
The key to this process is the algorithm that analyzes the relationship between validation performance and various hyperparameter values to choose the next set of hyperparameters to evaluate. Many different techniques are possible: Bayesian optimization, genetic algorithms, simple random search, and so on.

Training the weights of a model is relatively easy: you compute a loss function on a mini-batch of data and then use backpropagation to move the weights in the right direction. Updating hyperparameters, on the other hand, presents unique challenges. Consider that

The hyperparameter space is typically made of discrete decisions and thus isn’t continuous or differentiable. Hence, you typically can’t do gradient descent in hyperparameter space. Instead, you must rely on gradient-free optimization techniques, which, naturally, are far less efficient than gradient descent.

Computing the feedback signal of this optimization process (does this set of hyperparameters lead to a high-performing model on this task?) can be extremely expensive: it requires creating and training a new model from scratch on your dataset.

The feedback signal may be noisy: if a training run performs 0.2% bet
