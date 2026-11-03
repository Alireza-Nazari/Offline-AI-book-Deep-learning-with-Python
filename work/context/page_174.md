# Generation bundle for page_174  (PDF page 174, printed page 150)

Page kind: content
Chapter: 5 Fundamentals of machine learning
Section: 5.2 Evaluating machine-learning models
Subsection: 5.2.1 Training, validation, and test sets
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_174.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_174.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p174_e03] (body) talk about the use of common-sense baselines to check that your training is going somewhere.
[p174_e04] (heading_subsubsection) Simple hold-out validation
[p174_e05] (body) Set apart some fraction of your data as your test set. Train on the remaining data, and evaluate on the test set. As you saw in the previous sections, to prevent information leaks, you shouldn’t tune your model based on the test set, and therefore you should also reserve a validation set.
[p174_e06] (body) Schematically,
[p174_e07] (body) hold-out
[p174_e08] (body) validation
[p174_e09] (figure_caption) Figure 5.12 Simple hold-out validation split
[p174_e10] (body) looks like figure 5.12. The following listing shows a simple implementation.
[p174_e11] (listing_caption) Listing 5.5 Hold-out validation (note that labels are omitted for simplicity)
[p174_e12] (code)
```
num_validation_samples = 10000
np.random.shuffle(data)
validation_data = data[:num_validation_samples]
training_data = data[num_validation_samples:]
model = get_model()
model.fit(training_data, ...)
validation_score = model.evaluate(validation_data, ...)
...
model = get_model()
model.fit(
    np.concatenate([training_data, validation_data]),
    ...,
)
test_score = model.evaluate(test_data, ...)
```
[p174_e13] (body) This is the simplest evaluation protocol, and it suffers from one flaw: if little data is available, then your validation and test sets may contain too few samples to be statistically representative of the data at hand. This is easy to recognize: if different random shuffling rounds of the data before splitting end up yielding very different measures of model performance, then you’re having this issue. K-fold validation and iterated K-fold validation are two ways to address this, as discussed next.
[p174_e14] (heading_subsubsection) K-fold validation
[p174_e15] (body) With this approach, you split your data into `K` partitions of equal size. For each partition `i`, train a model on the remaining `K - 1` partitions and evaluate it on partition `i`.

## Code annotations printed next to listings (callouts, in order)
- Defines the training set
- Shuffling the data is
- usually appropriate.
- Defines the validation set
- Trains a model on the
- training data and evaluates
- it on the validation data
- At this point, you can tune your model,
- retrain it, evaluate it, tune it again, and so on.
- Once you’ve tuned your
- hyperparameters, it’s
- common to train your final
- model from scratch on all
- non-test data available.

## Listings
- Listing 5.5: Hold-out validation (note that labels are omitted for simplicity) (code elements p174_e12)

## Figures
- Figure 5.12: Simple hold-out validation split  | labels inside the figure: Training set, Total available labeled data, Train on this, Evaluate, Held-out, validation, set

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

## Previous page (PDF 173) text, for continuity
5.2

Evaluating machine-learning models

You can only control what you can observe. Since your goal is to develop models that can successfully generalize to new data, it’s essential to be able to reliably measure the generalization power of your model. In this section, we’ll formally introduce the different ways you can evaluate machine learning models. You’ve already seen most of them in action in the previous chapter.

5.2.1

Training, validation, and test sets

Evaluating a model always boils down to splitting the available data into three sets: training, validation, and test. You train on the training data and evaluate your model on the validation data. Once your model is ready for prime time, you test it one final time on the test data, which is meant to be as similar as possible to production data. Then you can deploy the model in production.

You may ask, why not have two sets: a training set and a test set? You’d train on the training data and evaluate on the test data. Much simpler! The reason is that developing a model always involves tuning its configuration: for example, choosing the number of layers or the size of the layers (called the hyperparameters of the model, to distinguish them from the parameters, which are the network’s weights). You do this tuning by using as a feedback signal the performance of the model on the validation data. In essence, this tuning is a form of learning: a search for a good configuration in some parameter space. As a result, tuning the configuration of the model based on its performance on the validation set can quickly result in overfitting to the validation set, even though your model is never directly trained on it. Central to this phenomenon is the notion of information leaks. Every time you tune a hyperparameter of your  …

## Next page (PDF 175) text, for continuity
Your final score is then the averages of the K scores obtained. This method is helpful when the performance of your model shows significant variance based on your train/ test split. Like hold-out validation, this method doesn’t exempt you from using a distinct validation set for model calibration.

Schematically, K-fold cross-validation looks like figure 5.13. Listing 5.6 shows a simple implementation.

Figure 5.13 Three-fold validation

Listing 5.6 K-fold cross-validation (note that labels are omitted for simplicity)

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.176 — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models
- PDF p.177 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.205 — 6 The universal workflow of machine learning › 6.2 Developing a model
- PDF p.135 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.153 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.144 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.186 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.193 — 5 Fundamentals of machine learning › 5.4 Improving generalization

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.205 (printed 181) — 6 The universal workflow of machine learning › 6.2 Developing a model › 
performance. The goal of your validation protocol is to accurately estimate what your success metric of choice (such as accuracy) will be on actual production data. The reliability of that process is critical to building a useful model.

In chapter 5, we reviewed three common evaluation protocols:

Maintaining a hold-out validation set—The way to go when you have plenty of data

Doing K-fold cross-validation—The right choice when you have too few samples for hold-out validation to be reliable

Doing iterated K-fold validation—For performing highly accurate model evaluation when little data is available

Just pick one of these. In most cases, the first will work well enough. As you’ve learned before, always be mindful of the representativity of your validation set(s) and be careful not to have redundant samples between your training set and your validation set(s).

### PDF p.153 (printed 129) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Validating your approach using K-fold validation
To evaluate your model while you keep adjusting its parameters (such as the number of epochs used for training), you could split the data into a training set and a validation set, as you did in the previous examples. But because you have so few data points, the validation set would end up being very small (for instance, about 100 examples). As a consequence, the validation scores might change a lot depending on which data points you chose to use for validation and which you chose for training: the validation scores might have a high variance with regard to the validation split. This would prevent you from reliably evaluating your model.

The best practice in such situations is to use K-fold cross-validation (see figure 4.9). It consists of splitting the available data into K partitions (typically K = 4 or 5), instantiating K identical models, and training each one on K – 1 partitions while evaluating on the remaining partition. The validation score for the model used is then the average of the K validation scores obtained. In terms of code, this is straightforward.

### PDF p.135 (printed 111) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › Validating your approach
As you learned in chapter 3, a deep learning model should never be evaluated on its training data—it’s standard practice to use a “validation set” to monitor the accuracy of the model during training. Here, you’ll create a validation set by setting apart 10,000 samples from the original training data.

You might ask, why not simply use the test data to evaluate the model? That seems like it would be easier. The reason is that you’re going to want to use the results you get on the validation set to inform your next choices to improve training—for instance, your choice of what model size to use or how many epochs to train for. When you start doing this, your validation scores stop being an accurate reflection of the performance of the model on brand-new data, since the model has been deliberately modified to perform better on the validation data. It’s good to keep around a set of never-before-seen samples that you can use to perform the final evaluation round in a completely unbiased way, and that’s exactly what the test set is. We’ll talk more about this in the next chapter.

Listing 4.6 Setting aside a validation set

### PDF p.177 (printed 153) — 5 Fundamentals of machine learning › 5.3 Improving model fit › 
ordered by their class, taking the first 80% of the array as your training set and the remaining 20% as your test set will result in your training set containing only classes 0–7, whereas your test set contains only classes 8–9. This seems like a ridiculous mistake, but it’s surprisingly common. For this reason, you usually should randomly shuffle your data before splitting it into training and test sets.

The arrow of time—If you’re trying to predict the future given the past (for example, tomorrow’s weather, stock movements, and so on), you should not randomly shuffle your data before splitting it because doing so will create a temporal leak: your model will effectively be trained on data from the future. In such situations, you should always make sure all data in your test set is posterior to the data in the training set.

Redundancy in your data—If some data points in your data appear twice (fairly common with real-world data), then shuffling the data and splitting it into a training set and a validation set will result in redundancy between the training and validation sets. In effect, you’ll be testing on part of your training data, which is the worst thing you can do! Make su

### PDF p.176 (printed 152) — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models › Iterated K-fold validation with shuffling
This one is for situations in which you have relatively little data available and you need to evaluate your model as precisely as possible. I’ve found it to be extremely helpful in Kaggle competitions. It consists of applying K-fold validation multiple times, shuffling the data every time before splitting it `K` ways. The final score is the average of the scores obtained at each run of K-fold validation. Note that you end up training and evaluating `P * K` models (where `P` is the number of iterations you use), which can be very expensive.

### PDF p.213 (printed 189) — 6 The universal workflow of machine learning › 6.3 Deploying your model › Summary
When you take on a new machine learning project, first, define the problem at hand: – Understand the broader context of what you’re setting out to do—what’s the end goal and what are the constraints? – Collect and annotate a dataset; make sure you understand your data in depth. – Choose how you’ll measure success on your problem. What metrics will you monitor on your validation data?

Once you understand the problem and you have an appropriate dataset, develop a model: – Prepare your data. – Pick your evaluation protocol. Hold-out validation? K-fold validation? Which portion of the data should you use for validation? – Achieve statistical power: beat a simple baseline. – Scale up: develop a model that can overfit. – Regularize your model and tune its hyperparameters, based on performance on the validation data. A lot of machine learning research tends to focus only on this step—but keep the big picture in mind.

When your model is ready and yields good performance on the test data, it’s time for deployment: – First, make sure to set appropriate expectations with stakeholders. – Optimize a final model for inference, and ship the model to the deployment environment of choice—web serv

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

### PDF p.144 (printed 120) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › Validating your approach
Let’s set apart 1,000 samples in the training data to use as a validation set.

Listing 4.17 Setting aside a validation set
