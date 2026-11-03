# Generation bundle for page_570  (PDF page 570, printed page 546)

Page kind: content
Chapter: 18 Best practices for the real world
Section: 18.1 Getting the most out of your models
Subsection: 18.1.2 Model ensembling
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_570.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_570.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p570_e03] (body) and a certain dataset, higher-level decisions generalize better across different tasks and datasets: for instance, pretty much every image classification problem can be solved via the same sort of search space template.
[p570_e04] (body) Following this logic, KerasTuner attempts to provide premade search spaces that are relevant to broad categories of problems—such as image classification. Just add data, run the search, and get a pretty good model. You can try the hypermodels `kt.applications
` `.HyperXception` and `kt.applications.HyperResNet`, which are effectively tunable versions of Keras Applications models.
[p570_e05] (heading_subsection) 18.1.2 Model ensembling
[p570_e06] (body) Another powerful technique for obtaining the best possible results on a task is model ensembling. Ensembling consists of pooling together the predictions of a set of different models to produce better predictions. If you look at machine learning competitions— in particular, on Kaggle—you’ll see that the winners use very large ensembles of models that inevitably beat any single model, no matter how good.
[p570_e07] (body) Ensembling relies on the assumption that different well-performing models trained independently are likely to be good for different reasons: each model looks at slightly different aspects of the data to make its predictions, getting part of the “truth” but not all of it. You may be familiar with the ancient parable of the blind men and the elephant: a group of blind men come across an elephant for the first time and try to understand what the elephant is by touching it. Each man touches a different part of the elephant’s body—just one part, such as the trunk or a leg. Then the men describe to each other what an elephant is: “It’s like a snake,” “Like a pillar or a tree,” and so on. The blind men are essentially machine learning models trying to understand the manifold of the training data, each from its own perspective, using its own assumptions (provided by the unique architecture of the model and the unique random weight initialization). Each of them gets part of the truth of the data, but not the whole truth. By pooling their perspectives together, you can get a far more accurate description of the data. The elephant is a combination of parts: no single blind man gets it quite right, but interviewed together, they can tell a fairly accurate story. Let’s use classification as an example. The easiest way to pool the predictions of a set of classifiers (to ensemble the classifiers) is to average their predictions at inference time:
[p570_e08] (code)
```
preds_a = model_a.predict(x_val)
preds_b = model_b.predict(x_val)
preds_c = model_c.predict(x_val)
preds_d = model_d.predict(x_val)
final_preds = 0.25 * (preds_a + preds_b + preds_c + preds_d)
```

## Code annotations printed next to listings (callouts, in order)
- Uses four different models to
- compute initial predictions
- This new prediction array
- should be more accurate
- than any of the initial ones.

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

## Previous page (PDF 569) text, for continuity
If you’re not worried about slightly underperforming, there’s a shortcut you can take: just use the tuner to reload the top-performing models with the best weights saved during the hyperparameter search, without retraining new models from scratch:

NOTE One important issue to keep in mind when doing automatic hyperparameter optimization at scale is validation-set overfitting. Because you’re updating hyperparameters based on a signal that is computed using your validation data, you’re effectively training them on the validation data, and thus, they will quickly overfit to the validation data. Always keep this in mind.

The art of crafting the right search space

Overall, hyperparameter optimization is a powerful technique that is an absolute requirement to get to state-of-the-art models on any task or to win machine learning competitions. Think about it: once upon a time, people handcrafted the features that went into shallow machine learning models. That was very suboptimal. Now deep learning automates the task of hierarchical feature engineering—features are learned using a feedback signal, not hand-tuned, and that’s the way it should be. In the same way, you shouldn’t handcraft your model architectures; you should optimize them in a principled way.

However, doing hyperparameter tuning is not a replacement for being familiar with model architecture best practices: search spaces grow combinatorially with the number of choices, so it would be far too expensive to turn everything into a hyperparameter and let the tuner sort it out. You need to be smart about designing the right search space. Hyperparameter tuning is automation, not magic: you use it to automate experiments that you would otherwise have run by hand, but you still need to handpick experiment configurations …

## Next page (PDF 571) text, for continuity
However, this will work only if the classifiers are more or less equally good. If one of them is significantly worse than the others, the final predictions may not be as good as the best classifier of the group.

A smarter way to ensemble classifiers is to do a weighted average, where the weights are learned on the validation data—typically, the better classifiers are given a higher weight, and the worse classifiers are given a lower weight. To search for a good set of ensembling weights, you can use random search or a simple optimization algorithm, such as the Nelder-Mead algorithm:

There are many possible variants: you can do an average of an exponential of the predictions, for instance. In general, a simple weighted average with weights optimized on the validation data provides a very strong baseline.

The key to making ensembling work is the diversity of the set of classifiers. Diversity is strength. If all the blind men only touched the elephant’s trunk, they would agree that elephants are like snakes, and they would forever stay ignorant of the truth of the elephant. Diversity is what makes ensembling work. In machine learning terms, if all of your models are biased in the same way, then your ensemble will retain this same bias. If your models are biased in different ways, the biases will cancel each other out, and the ensemble will be more robust and more accurate. For this reason, you should ensemble models that are as good as possible while being as different as possible. This typically means using very different architectures or even different brands of machine learning approaches. One thing that is largely not worth doing is ensembling the same network trained several times independently, from different random initializations. If the only difference between  …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.572 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.176 — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models
- PDF p.562 — 18 Best practices for the real world › 
- PDF p.563 — 18 Best practices for the real world › 18.1 Getting the most out of your models
- PDF p.228 — 7 A deep dive on Keras › 7.2 Different ways to build Keras models
- PDF p.44 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network
- PDF p.283 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.134 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.572 (printed 548) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › 
was so different from every other model: it provided information that the other models didn’t have access to. That’s precisely the point of ensembling. It’s not so much about how good your best model is; it’s about the diversity of your set of candidate models.

### PDF p.562 (printed 538) — 18 Best practices for the real world ›  › This chapter covers
Hyperparameter tuning

Model ensembling

Training Keras models on multiple GPUs or on TPU

Mixed-precision training

Quantization

You’ve come quite far since the beginning of this book. You can now train image classification models, image segmentation models, models for classification or regression on vector data, timeseries forecasting models, text classification models, sequence-to-sequence models, and even generative models for text and images. You’ve got all the bases covered.

However, your models so far have all been trained at a small scale—on small datasets, with a single GPU—and they generally haven’t reached the best achievable performance on each dataset we’ve looked at. This book is, after all, an introductory book. If you are to go out into the real world and achieve state-of-the-art results on brand new problems, there’s still a bit of a chasm that you’ll need to cross.

### PDF p.10 (printed x) — contents ›  › 
3.6

Introduction to Keras 90

First steps with Keras 91 ■ Layers: The building blocks of deep learning 92 ■ From layers to models 96 ■ The “compile” step: Configuring the learning process 97 ■ Picking a loss function 99 Understanding the fit method 100 ■ Monitoring loss and metrics on validation data 101 ■ Inference: Using a model after training 102

4

Classification and regression 104 4.1

Classifying movie reviews: A binary classification example 106

The IMDb dataset 106 ■ Preparing the data 107 ■ Building your model 108 ■ Validating your approach 111 ■ Using a trained model to generate predictions on new data 115 Further experiments 115 ■ Wrapping up 116

4.2

Classifying newswires: A multiclass classification example 116

The Reuters dataset 116 ■ Preparing the data 118 ■ Building your model 118 ■ Validating your approach 120 ■ Generating predictions on new data 124 ■ A different way to handle the labels and the loss 124 ■ The importance of having sufficiently large intermediate layers 125 ■ Further experiments 125 Wrapping up 126

4.3

Predicting house prices: A regression example 126

The California Housing Price dataset 126 ■ Preparing the data 128 ■ Building your model 1

### PDF p.325 (printed 301) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 
To visualize which parts of the image are the most African elephant–like, let’s set up the Grad-CAM process.

You will note that we didn’t need to preprocess our image before calling the task model. That’s because the KerasHub `ImageClassifier` is preprocessing inputs for us as part of `predict()`. Let’s preprocess the image ourselves so we can use the preprocessed inputs directly:

### PDF p.293 (printed 269) — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse › 
images are translation-invariant. To effectively learn from data, you need to make assumptions about what you’re looking for.

Model architecture is often the difference between success and failure. If you make inappropriate architecture choices, your model may be stuck with suboptimal metrics, and no amount of training data will save it. Inversely, a good model architecture will accelerate learning and will enable your model to make efficient use of the training data available, reducing the need for large datasets. A good model architecture is one that reduces the size of the search space or otherwise makes it easier to converge to a good point of the search space. Just like feature engineering and data curation, model architecture is all about making the problem simpler for gradient descent to solve—and remember that gradient descent is a pretty stupid search process, so it needs all the help it can get. Model architecture is more an art than a science. Experienced machine learning engineers are able to intuitively cobble together high-performing models on their first try, while beginners often struggle to create a model that trains at all. The keyword here is intuitively: no one

### PDF p.115 (printed 91) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
Waymo self-driving cars rely on Keras models for processing sensor data. Keras is also a popular framework on Kaggle, the machine learning competition website.

Because Keras has a diverse user base, it doesn’t force you to follow a single “true” way of building and training models. Rather, it enables a wide range of different workflows, from the very high-level to the very low-level, corresponding to different user profiles. For instance, you have an array of ways to build models and an array of ways to train them, each representing a certain tradeoff between usability and flexibility. In chapter 7, we’ll review in detail a good fraction of this spectrum of workflows.

### PDF p.219 (printed 195) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › The Functional API
The Sequential model is easy to use, but its applicability is extremely limited: it can only express models with a single input and a single output, applying one layer after the other in a sequential fashion. In practice, it’s pretty common to encounter models with multiple inputs (say, an image and its metadata), multiple outputs (different things you want to predict about the data), or a nonlinear topology.

In such cases, you’d build your model using the Functional API. This is what most Keras models you’ll encounter in the wild use. It’s fun and powerful—it feels like playing with LEGO bricks.

### PDF p.324 (printed 300) — 10 Interpreting what ConvNets learn › 10.3 Visualizing heatmaps of class activation › 
Figure 10.7 Test picture of African elephants

So far, we have only used KerasHub to instantiate a pretrained feature extractor network using the backbone class. For Grad-CAM, we need the entire Xception model including the classification head—recall that Xception was trained on the ImageNet dataset with ~1 million labeled images belonging to 1,000 different classes.

KerasHub provides a high-level task API for common end-to-end workflows like image classification, text classification, image generation, and so on. A task wraps preprocessing, a feature extraction network, and a task-specific head into a single class that is easy to use. Let’s try it out:
