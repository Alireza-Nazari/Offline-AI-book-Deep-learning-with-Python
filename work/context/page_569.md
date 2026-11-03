# Generation bundle for page_569  (PDF page 569, printed page 545)

Page kind: content
Chapter: 18 Best practices for the real world
Section: 18.1 Getting the most out of your models
Subsection: 18.1.1 Hyperparameter optimization
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_569.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_569.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p569_e03] (code)
```
        x_train_full, y_train_full, batch_size=128, epochs=int(best_epoch * 1.2)
    )
    return model
best_models = []
for hp in best_hps:
    model = get_best_trained_model(hp)
    model.evaluate(x_test, y_test)
    best_models.append(model)
```
[p569_e04] (body) If you’re not worried about slightly underperforming, there’s a shortcut you can take: just use the tuner to reload the top-performing models with the best weights saved during the hyperparameter search, without retraining new models from scratch:
[p569_e05] (code)
```
best_models = tuner.get_best_models(top_n)
```
[p569_e06] (body) NOTE One important issue to keep in mind when doing automatic hyperparameter optimization at scale is validation-set overfitting. Because you’re updating hyperparameters based on a signal that is computed using your validation data, you’re effectively training them on the validation data, and thus, they will quickly overfit to the validation data. Always keep this in mind.
[p569_e07] (heading_subsubsection) The art of crafting the right search space
[p569_e08] (body) Overall, hyperparameter optimization is a powerful technique that is an absolute requirement to get to state-of-the-art models on any task or to win machine learning competitions. Think about it: once upon a time, people handcrafted the features that went into shallow machine learning models. That was very suboptimal. Now deep learning automates the task of hierarchical feature engineering—features are learned using a feedback signal, not hand-tuned, and that’s the way it should be. In the same way, you shouldn’t handcraft your model architectures; you should optimize them in a principled way.
[p569_e09] (body) However, doing hyperparameter tuning is not a replacement for being familiar with model architecture best practices: search spaces grow combinatorially with the number of choices, so it would be far too expensive to turn everything into a hyperparameter and let the tuner sort it out. You need to be smart about designing the right search space. Hyperparameter tuning is automation, not magic: you use it to automate experiments that you would otherwise have run by hand, but you still need to handpick experiment configurations that have the potential to yield good metrics. The good news: by using hyperparameter tuning, the configuration decisions you have to make graduate from micro-decisions (What number of units do I pick for this layer?) to higher-level architecture decisions (Should I use residual connections throughout this model?). And while micro-decisions are specific to a certain model

## Equation candidates (formulas written inline in the text)
- [p569_e05] best_models = tuner.get_best_models(top_n)

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

## Previous page (PDF 568) text, for continuity
Listing 18.3 Querying the best hyperparameter configurations

Usually, when retraining these models, you may want to include the validation data as part of the training data since you won’t be making any further hyperparameter changes, and thus you will no longer be evaluating performance on the validation data. In our example, we’d train these final models on the totality of the original MNIST training data, without reserving a validation set.

Before we can train on the full training data, though, there’s one last parameter we need to settle: the optimal number of epochs to train for. Typically, you’ll want to train the new models for longer than you did during the search: using an aggressive `patience` value in the `EarlyStopping` callback saves time during the search, but may lead to underfitted models. Just use the validation set to find the best epoch:

And finally, train on the full dataset for just a bit longer than this epoch count, since you’re training on more data—20% more, in this case:

## Next page (PDF 570) text, for continuity
and a certain dataset, higher-level decisions generalize better across different tasks and datasets: for instance, pretty much every image classification problem can be solved via the same sort of search space template.

Following this logic, KerasTuner attempts to provide premade search spaces that are relevant to broad categories of problems—such as image classification. Just add data, run the search, and get a pretty good model. You can try the hypermodels `kt.applications
` `.HyperXception` and `kt.applications.HyperResNet`, which are effectively tunable versions of Keras Applications models.

18.1.2 Model ensembling

Another powerful technique for obtaining the best possible results on a task is model ensembling. Ensembling consists of pooling together the predictions of a set of different models to produce better predictions. If you look at machine learning competitions— in particular, on Kaggle—you’ll see that the winners use very large ensembles of models that inevitably beat any single model, no matter how good.

Ensembling relies on the assumption that different well-performing models trained independently are likely to be good for different reasons: each model looks at slightly different aspects of the data to make its predictions, getting part of the “truth” but not all of it. You may be familiar with the ancient parable of the blind men and the elephant: a group of blind men come across an elephant for the first time and try to understand what the elephant is by touching it. Each man touches a different part of the elephant’s body—just one part, such as the trunk or a leg. Then the men describe to each other what an elephant is: “It’s like a snake,” “Like a pillar or a tree,” and so on. The blind men are essentially machine learning models trying to underst …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.563 — 18 Best practices for the real world › 18.1 Getting the most out of your models
- PDF p.567 — 18 Best practices for the real world › 18.1 Getting the most out of your models
- PDF p.564 — 18 Best practices for the real world › 18.1 Getting the most out of your models
- PDF p.565 — 18 Best practices for the real world › 18.1 Getting the most out of your models
- PDF p.566 — 18 Best practices for the real world › 18.1 Getting the most out of your models
- PDF p.587 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation
- PDF p.207 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.173 — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.564 (printed 540) — 18 Best practices for the real world › 18.1 Getting the most out of your models › 
The key to this process is the algorithm that analyzes the relationship between validation performance and various hyperparameter values to choose the next set of hyperparameters to evaluate. Many different techniques are possible: Bayesian optimization, genetic algorithms, simple random search, and so on.

Training the weights of a model is relatively easy: you compute a loss function on a mini-batch of data and then use backpropagation to move the weights in the right direction. Updating hyperparameters, on the other hand, presents unique challenges. Consider that

The hyperparameter space is typically made of discrete decisions and thus isn’t continuous or differentiable. Hence, you typically can’t do gradient descent in hyperparameter space. Instead, you must rely on gradient-free optimization techniques, which, naturally, are far less efficient than gradient descent.

Computing the feedback signal of this optimization process (does this set of hyperparameters lead to a high-performing model on this task?) can be extremely expensive: it requires creating and training a new model from scratch on your dataset.

The feedback signal may be noisy: if a training run performs 0.2% bet

### PDF p.563 (printed 539) — 18 Best practices for the real world › 18.1 Getting the most out of your models › 18.1.1 Hyperparameter optimization
When building a deep learning model, you have to make many seemingly arbitrary decisions: How many layers should you stack? How many units or filters should go in each layer? Should you use `relu` as an activation, or a different function? Should you use `BatchNormalization` after a given layer? How much dropout should you use? And so on. These architecture-level parameters are called hyperparameters to distinguish them from the parameters of a model, which are trained via backpropagation.

In practice, experienced machine learning engineers and researchers build intuition over time as to what works and what doesn’t when it comes to these choices—they develop hyperparameter-tuning skills. But there are no formal rules. If you want to get to the very limit of what can be achieved on a given task, you can’t be content with such arbitrary choices. Your initial decisions are almost always suboptimal, even if you have very good intuition. You can refine your choices by tweaking them by hand and retraining the model repeatedly—that’s what machine learning engineers and researchers spend most of their time doing. But it shouldn’t be your job as a human to fiddle with hyperparameters all d

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

### PDF p.173 (printed 149) — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models › Training, validation, and test sets
You may ask, why not have two sets: a training set and a test set? You’d train on the training data and evaluate on the test data. Much simpler! The reason is that developing a model always involves tuning its configuration: for example, choosing the number of layers or the size of the layers (called the hyperparameters of the model, to distinguish them from the parameters, which are the network’s weights). You do this tuning by using as a feedback signal the performance of the model on the validation data. In essence, this tuning is a form of learning: a search for a good configuration in some parameter space. As a result, tuning the configuration of the model based on its performance on the validation set can quickly result in overfitting to the validation set, even though your model is never directly trained on it. Central to this phenomenon is the notion of information leaks. Every time you tune a hyperparameter of your model based on the model’s performance on the validation set, some information about the validation data leaks into the model. If you do this only once, for one parameter, then very few bits of information will leak, and your validation set will remain reliable 

### PDF p.587 (printed 563) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › Summary
You can use hyperparameter tuning and KerasTuner to automate the tedium out of finding the best model configuration. But be mindful of validation-set overfitting!

An ensemble of diverse models can often significantly improve the quality of your predictions.

To further scale your workflows, you can use data parallelism to train a model on multiple devices, as long as the model is small enough to fit on a single device.

For larger models, you can also use model parallelism to split your model’s variables and computation across several devices.

You can speed up model training on GPUs or TPUs by turning on mixed precision—you’ll generally get a nice speed boost at virtually no cost.

You can also speed up inference by using `float16` precision or even `int8` quantization.

### PDF p.564 (printed 540) — 18 Best practices for the real world › 18.1 Getting the most out of your models › Using KerasTuner
The key idea that KerasTuner is built upon is to let you replace hardcoded hyperparameter values, such as `units=32`, with a range of possible choices, such as `Int(name="units",` `min_value=16,` `max_value=64,` `step=16)`. The set of such choices in a given model is called the search space of the hyperparameter tuning process.

To specify a search space, define a model-building function (see the next listing). It takes an `hp` argument, from which you can sample hyperparameter ranges, and it returns a compiled Keras model.

Listing 18.1 A KerasTuner model-building function

### PDF p.563 (printed 539) — 18 Best practices for the real world › 18.1 Getting the most out of your models › 
This chapter is about bridging that gap and giving you the best practices you’ll need as you go from machine learning student to a fully fledged machine learning engineer. We’ll review essential techniques for systematically improving model performance: hyperparameter tuning and model ensembling. Then we’ll look at how you can speed up and scale up model training, with multi-GPU and TPU training, mixed precision, and quantization.

### PDF p.624 (printed 600) — 20 Conclusions › 20.1 Key concepts in review › 
Develop a first model that beats a trivial common-sense baseline, thus demonstrating that

machine learning can work on your problem. This may not always be the case!

Gradually refine your model architecture by tuning hyperparameters and adding regularization. Make changes based on performance on the validation data only, not the test data or the training data. Remember that you should get your model to overfit (thus identifying a model capacity level that’s greater than you need) and only then begin to add regularization or downsize your model. Beware of validation-set overfitting when tuning hyperparameters—the fact that your hyperparameters may end up being overspecialized to the validation set. Avoiding this is the purpose of having a separate test set!

Deploy your final model in production—as a web API, as part of a JavaScript or C++ application, on an embedded device, etc. Keep monitoring its performance on real-world data and use your findings to refine the next iteration of the model!
