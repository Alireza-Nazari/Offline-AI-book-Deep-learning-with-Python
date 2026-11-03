# Generation bundle for page_566  (PDF page 566, printed page 542)

Page kind: content
Chapter: 18 Best practices for the real world
Section: 18.1 Getting the most out of your models
Subsection: 18.1.1 Hyperparameter optimization
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_566.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_566.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p566_e03] (body) KerasTuner has several built-in tuners available—`RandomSearch`, `BayesianOptimization`, and `Hyperband`. Let’s try `BayesianOptimization`, a tuner that attempts to make smart predictions for which new hyperparameter values are likely to perform best given the outcome of previous choices:
[p566_e04] (code)
```
tuner = kt.BayesianOptimization(
    build_model,
    objective="val_accuracy",
    max_trials=20,
    executions_per_trial=2,
    directory="mnist_kt_test",
    overwrite=True,
)
```
[p566_e05] (body) You can display an overview of the search space via `search_space_summary()`:
[p566_e06] (code)
```
>>> tuner.search_space_summary()
Search space summary
Default search space size: 2
units (Int)
{"default": None,
 "conditions": [],
 "min_value": 128,
 "max_value": 1024,
 "step": 128,
 "sampling": None}
optimizer (Choice)
{"default": "rmsprop",
 "conditions": [],
 "values": ["rmsprop", "adam"],
 "ordered": False}
```
[p566_e07] (heading_subsection) Objective maximization and minimization
[p566_e08] (sidebar) For built-in metrics (like accuracy, in our case), the direction of the metric (accuracy should be maximized, but a loss should be minimized) is inferred by KerasTuner. However, for a custom metric, you should specify it yourself, like this:

## Code annotations printed next to listings (callouts, in order)
- Specifies the metric that the tuner
- will seek to optimize. Always
- specify validation metrics since
- Specifies the model-building
- the goal of the search process is to
- function (or hypermodel instance)
- find models that generalize!
- Maximum number of different
- model configurations (“trials”)
- to try before ending the search
- Where to store search logs
- Whether to overwrite data in the directory
- To reduce metrics variance, you can
- to start a new search. Set this to True if you’ve
- train the same model multiple times
- modified the model-building function or to False
- and average the results. executions_
- to resume a previously started search with the
- per_trial is how many training
- same model-building function.
- rounds (executions) to run for each
- model configuration (trial).

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

## Previous page (PDF 565) text, for continuity
If you want to adopt a more modular and configurable approach to model-building, you can also subclass the `HyperModel` class and define a `build` method.

Listing 18.2 A KerasTuner `HyperModel`

The next step is to define a “tuner.” Schematically, you can think of a tuner as a `for` loop, which will repeatedly

Pick a set of hyperparameter values

Call the model-building function with these values to create a model

Train the model and record its metrics

## Next page (PDF 567) text, for continuity
Finally, let’s launch the search. Don’t forget to pass validation data and make sure not to use your test set as validation data—otherwise, you’d quickly start overfitting to your test data, and you wouldn’t be able to trust your test metrics anymore:

The previous example will run in just a few minutes since we’re only looking at a few possible choices and we’re training on MNIST. However, with a typical search space and dataset, you’ll often find yourself letting the hyperparameter search run overnight or even over several days. If your search process crashes, you can always restart it—just specify `overwrite=False` in the tuner so that it can resume from the trial logs stored on disk.

Once the search is complete, you can query the best hyperparameter configurations, which you can use to create high-performing models that you can then retrain.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.564 — 18 Best practices for the real world › 18.1 Getting the most out of your models
- PDF p.569 — 18 Best practices for the real world › 18.1 Getting the most out of your models
- PDF p.570 — 18 Best practices for the real world › 18.1 Getting the most out of your models
- PDF p.232 — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops
- PDF p.563 — 18 Best practices for the real world › 18.1 Getting the most out of your models
- PDF p.245 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.122 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.641 — index › 

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.564 (printed 540) — 18 Best practices for the real world › 18.1 Getting the most out of your models › Using KerasTuner
The key idea that KerasTuner is built upon is to let you replace hardcoded hyperparameter values, such as `units=32`, with a range of possible choices, such as `Int(name="units",` `min_value=16,` `max_value=64,` `step=16)`. The set of such choices in a given model is called the search space of the hyperparameter tuning process.

To specify a search space, define a model-building function (see the next listing). It takes an `hp` argument, from which you can sample hyperparameter ranges, and it returns a compiled Keras model.

Listing 18.1 A KerasTuner model-building function

### PDF p.569 (printed 545) — 18 Best practices for the real world › 18.1 Getting the most out of your models › The art of crafting the right search space
Overall, hyperparameter optimization is a powerful technique that is an absolute requirement to get to state-of-the-art models on any task or to win machine learning competitions. Think about it: once upon a time, people handcrafted the features that went into shallow machine learning models. That was very suboptimal. Now deep learning automates the task of hierarchical feature engineering—features are learned using a feedback signal, not hand-tuned, and that’s the way it should be. In the same way, you shouldn’t handcraft your model architectures; you should optimize them in a principled way.

However, doing hyperparameter tuning is not a replacement for being familiar with model architecture best practices: search spaces grow combinatorially with the number of choices, so it would be far too expensive to turn everything into a hyperparameter and let the tuner sort it out. You need to be smart about designing the right search space. Hyperparameter tuning is automation, not magic: you use it to automate experiments that you would otherwise have run by hand, but you still need to handpick experiment configurations that have the potential to yield good metrics. The good news: by usin

### PDF p.564 (printed 540) — 18 Best practices for the real world › 18.1 Getting the most out of your models › 
The key to this process is the algorithm that analyzes the relationship between validation performance and various hyperparameter values to choose the next set of hyperparameters to evaluate. Many different techniques are possible: Bayesian optimization, genetic algorithms, simple random search, and so on.

Training the weights of a model is relatively easy: you compute a loss function on a mini-batch of data and then use backpropagation to move the weights in the right direction. Updating hyperparameters, on the other hand, presents unique challenges. Consider that

The hyperparameter space is typically made of discrete decisions and thus isn’t continuous or differentiable. Hence, you typically can’t do gradient descent in hyperparameter space. Instead, you must rely on gradient-free optimization techniques, which, naturally, are far less efficient than gradient descent.

Computing the feedback signal of this optimization process (does this set of hyperparameters lead to a high-performing model on this task?) can be extremely expensive: it requires creating and training a new model from scratch on your dataset.

The feedback signal may be noisy: if a training run performs 0.2% bet

### PDF p.563 (printed 539) — 18 Best practices for the real world › 18.1 Getting the most out of your models › 18.1.1 Hyperparameter optimization
When building a deep learning model, you have to make many seemingly arbitrary decisions: How many layers should you stack? How many units or filters should go in each layer? Should you use `relu` as an activation, or a different function? Should you use `BatchNormalization` after a given layer? How much dropout should you use? And so on. These architecture-level parameters are called hyperparameters to distinguish them from the parameters of a model, which are trained via backpropagation.

In practice, experienced machine learning engineers and researchers build intuition over time as to what works and what doesn’t when it comes to these choices—they develop hyperparameter-tuning skills. But there are no formal rules. If you want to get to the very limit of what can be achieved on a given task, you can’t be content with such arbitrary choices. Your initial decisions are almost always suboptimal, even if you have very good intuition. You can refine your choices by tweaking them by hand and retraining the model repeatedly—that’s what machine learning engineers and researchers spend most of their time doing. But it shouldn’t be your job as a human to fiddle with hyperparameters all d

### PDF p.569 (printed 545) — 18 Best practices for the real world › 18.1 Getting the most out of your models › 
If you’re not worried about slightly underperforming, there’s a shortcut you can take: just use the tuner to reload the top-performing models with the best weights saved during the hyperparameter search, without retraining new models from scratch:

### PDF p.570 (printed 546) — 18 Best practices for the real world › 18.1 Getting the most out of your models › 
and a certain dataset, higher-level decisions generalize better across different tasks and datasets: for instance, pretty much every image classification problem can be solved via the same sort of search space template.

Following this logic, KerasTuner attempts to provide premade search spaces that are relevant to broad categories of problems—such as image classification. Just add data, run the search, and get a pretty good model. You can try the hypermodels `kt.applications
` `.HyperXception` and `kt.applications.HyperResNet`, which are effectively tunable versions of Keras Applications models.

### PDF p.232 (printed 208) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › 
Custom metrics can be used just like built-in ones. Let’s test-drive our own metric:

### PDF p.587 (printed 563) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › Summary
You can use hyperparameter tuning and KerasTuner to automate the tedium out of finding the best model configuration. But be mindful of validation-set overfitting!

An ensemble of diverse models can often significantly improve the quality of your predictions.

To further scale your workflows, you can use data parallelism to train a model on multiple devices, as long as the model is small enough to fit on a single device.

For larger models, you can also use model parallelism to split your model’s variables and computation across several devices.

You can speed up model training on GPUs or TPUs by turning on mixed precision—you’ll generally get a nice speed boost at virtually no cost.

You can also speed up inference by using `float16` precision or even `int8` quantization.
