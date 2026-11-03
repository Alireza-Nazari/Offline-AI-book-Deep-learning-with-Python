# Generation bundle for page_175  (PDF page 175, printed page 151)

Page kind: content
Chapter: 5 Fundamentals of machine learning
Section: 5.2 Evaluating machine-learning models
Subsection: 5.2.1 Training, validation, and test sets
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_175.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_175.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p175_e03] (body) Your final score is then the averages of the K scores obtained. This method is helpful when the performance of your model shows significant variance based on your train/ test split. Like hold-out validation, this method doesn’t exempt you from using a distinct validation set for model calibration.
[p175_e04] (body) Schematically, K-fold cross-validation looks like figure 5.13. Listing 5.6 shows a simple implementation.
[p175_e05] (figure_caption) Figure 5.13 Three-fold validation
[p175_e06] (listing_caption) Listing 5.6 K-fold cross-validation (note that labels are omitted for simplicity)
[p175_e07] (code)
```
k = 3
num_validation_samples = len(data) // k
np.random.shuffle(data)
validation_scores = []
for fold in range(k):
    validation_data = data[
        num_validation_samples * fold : num_validation_samples * (fold + 1)
    ]
    training_data = np.concatenate(
        data[: num_validation_samples * fold],
        data[num_validation_samples * (fold + 1) :],
    )
    model = get_model()
    model.fit(training_data, ...)
    validation_score = model.evaluate(validation_data, ...)
    validation_scores.append(validation_score)
validation_score = np.average(validation_scores)
model = get_model()
model.fit(data, ...)
test_score = model.evaluate(test_data, ...)
```

## Code annotations printed next to listings (callouts, in order)
- Selects the
- validation-data partition
- Uses the remainder of
- the data as training data
- Creates a brand-
- new instance of the
- model (untrained)
- Trains the final model on all
- non-test data available
- Validation score: average of the
- validation scores of the k folds

## Listings
- Listing 5.6: K-fold cross-validation (note that labels are omitted for simplicity) (code elements p175_e07)

## Figures
- Figure 5.13: Three-fold validation  | labels inside the figure: Data split into 3 partitions, Validation, Training, Training, Validation, score #1, Training, Validation, Training, Validation, score #2, Training, Training, Validation, Validation, score #3

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

## Previous page (PDF 174) text, for continuity
talk about the use of common-sense baselines to check that your training is going somewhere.

Simple hold-out validation

Set apart some fraction of your data as your test set. Train on the remaining data, and evaluate on the test set. As you saw in the previous sections, to prevent information leaks, you shouldn’t tune your model based on the test set, and therefore you should also reserve a validation set.

Schematically,

hold-out

validation

Figure 5.12 Simple hold-out validation split

looks like figure 5.12. The following listing shows a simple implementation.

Listing 5.5 Hold-out validation (note that labels are omitted for simplicity)

This is the simplest evaluation protocol, and it suffers from one flaw: if little data is available, then your validation and test sets may contain too few samples to be statistically representative of the data at hand. This is easy to recognize: if different random shuffling rounds of the data before splitting end up yielding very different measures of model performance, then you’re having this issue. K-fold validation and iterated K-fold validation are two ways to address this, as discussed next.

K-fold validation

With this approach, you split your data into `K` partitions of equal size. For each partition `i`, train a model on the remaining `K - 1` partitions and evaluate it on partition `i`.

## Next page (PDF 176) text, for continuity
Iterated K-fold validation with shuffling

This one is for situations in which you have relatively little data available and you need to evaluate your model as precisely as possible. I’ve found it to be extremely helpful in Kaggle competitions. It consists of applying K-fold validation multiple times, shuffling the data every time before splitting it `K` ways. The final score is the average of the scores obtained at each run of K-fold validation. Note that you end up training and evaluating `P * K` models (where `P` is the number of iterations you use), which can be very expensive.

5.2.2

Beating a common-sense baseline

Besides the different evaluation protocols you have available, one last thing you should know about is the use of common-sense baselines.

Training a deep learning model is a bit like pressing a button that launches a rocket in a parallel world. You can’t hear it or see it. You can’t observe the manifold learning process—it’s happening in a space with thousands of dimensions, and even if you projected it to 3D, you couldn’t interpret it. The only feedback you have is your validation metrics—like an altitude meter on your invisible rocket. A particularly important point is to be able to tell whether you’re getting off the ground at all. What was the altitude you started at? Your model seems to have an accuracy of 15%, is that any good? Before you start working with a dataset, you should always pick a trivial baseline that you’ll try to beat. If you cross that threshold, you’ll know you’re doing something right: your model is actually using the information in the input data to make predictions that generalize—you can keep going. This baseline could be performance of a random classifier, or the performance of the simplest non-machine learning technique yo …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.154 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.155 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.153 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.205 — 6 The universal workflow of machine learning › 6.2 Developing a model
- PDF p.173 — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models
- PDF p.156 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.125 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.135 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.153 (printed 129) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Validating your approach using K-fold validation
To evaluate your model while you keep adjusting its parameters (such as the number of epochs used for training), you could split the data into a training set and a validation set, as you did in the previous examples. But because you have so few data points, the validation set would end up being very small (for instance, about 100 examples). As a consequence, the validation scores might change a lot depending on which data points you chose to use for validation and which you chose for training: the validation scores might have a high variance with regard to the validation split. This would prevent you from reliably evaluating your model.

The best practice in such situations is to use K-fold cross-validation (see figure 4.9). It consists of splitting the available data into K partitions (typically K = 4 or 5), instantiating K identical models, and training each one on K – 1 partitions while evaluating on the remaining partition. The validation score for the model used is then the average of the K validation scores obtained. In terms of code, this is straightforward.

### PDF p.173 (printed 149) — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models › Training, validation, and test sets
But if you repeat this many times— running one experiment, evaluating on the validation set, and modifying your model as a result—then you’ll leak an increasingly significant amount of information about the validation set into the model. At the end of the day, you’ll end up with a model that performs artificially well on the validation data because that’s what you optimized it for. You care about performance on completely new data, not the validation data, so you need to use a completely different, never-before-seen dataset to evaluate the model: the test dataset. Your model shouldn’t have had access to any information about the test set, even indirectly. If anything about the model has been tuned based on test set performance, then your measure of generalization will be flawed. Splitting your data into training, validation, and test sets may seem straightforward, but there are a few advanced ways to do it that can come in handy when little data is available. Let’s review three classic evaluation recipes: simple hold-out validation, K-fold validation, and iterated K-fold validation with shuffling. We’ll also

### PDF p.205 (printed 181) — 6 The universal workflow of machine learning › 6.2 Developing a model › 
performance. The goal of your validation protocol is to accurately estimate what your success metric of choice (such as accuracy) will be on actual production data. The reliability of that process is critical to building a useful model.

In chapter 5, we reviewed three common evaluation protocols:

Maintaining a hold-out validation set—The way to go when you have plenty of data

Doing K-fold cross-validation—The right choice when you have too few samples for hold-out validation to be reliable

Doing iterated K-fold validation—For performing highly accurate model evaluation when little data is available

Just pick one of these. In most cases, the first will work well enough. As you’ve learned before, always be mindful of the representativity of your validation set(s) and be careful not to have redundant samples between your training set and your validation set(s).

### PDF p.173 (printed 149) — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models › Training, validation, and test sets
Evaluating a model always boils down to splitting the available data into three sets: training, validation, and test. You train on the training data and evaluate your model on the validation data. Once your model is ready for prime time, you test it one final time on the test data, which is meant to be as similar as possible to production data. Then you can deploy the model in production.

### PDF p.154 (printed 130) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › 
Figure 4.9 Three-fold cross-validation

Listing 4.28 K-fold validation

### PDF p.155 (printed 131) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › 
The different runs do indeed show meaningfully different validation scores, from 0.232 to 0.349. The average (0.296) is a much more reliable metric than any single score—that’s the entire point of K-fold cross-validation. In this case, you’re off by $29,600 on average, which is significant considering that the prices range from $60,000 to $500,000.

Let’s try training the model a bit longer: 200 epochs. To keep a record of how well the model does at each epoch, you’ll modify the training loop to save the per-epoch validation score log.

Listing 4.29 Saving the validation logs at each fold

### PDF p.135 (printed 111) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › Validating your approach
As you learned in chapter 3, a deep learning model should never be evaluated on its training data—it’s standard practice to use a “validation set” to monitor the accuracy of the model during training. Here, you’ll create a validation set by setting apart 10,000 samples from the original training data.

You might ask, why not simply use the test data to evaluate the model? That seems like it would be easier. The reason is that you’re going to want to use the results you get on the validation set to inform your next choices to improve training—for instance, your choice of what model size to use or how many epochs to train for. When you start doing this, your validation scores stop being an accurate reflection of the performance of the model on brand-new data, since the model has been deliberately modified to perform better on the validation data. It’s good to keep around a set of never-before-seen samples that you can use to perform the final evaluation round in a completely unbiased way, and that’s exactly what the test set is. We’ll talk more about this in the next chapter.

Listing 4.6 Setting aside a validation set

### PDF p.213 (printed 189) — 6 The universal workflow of machine learning › 6.3 Deploying your model › Summary
When you take on a new machine learning project, first, define the problem at hand: – Understand the broader context of what you’re setting out to do—what’s the end goal and what are the constraints? – Collect and annotate a dataset; make sure you understand your data in depth. – Choose how you’ll measure success on your problem. What metrics will you monitor on your validation data?

Once you understand the problem and you have an appropriate dataset, develop a model: – Prepare your data. – Pick your evaluation protocol. Hold-out validation? K-fold validation? Which portion of the data should you use for validation? – Achieve statistical power: beat a simple baseline. – Scale up: develop a model that can overfit. – Regularize your model and tune its hyperparameters, based on performance on the validation data. A lot of machine learning research tends to focus only on this step—but keep the big picture in mind.

When your model is ready and yields good performance on the test data, it’s time for deployment: – First, make sure to set appropriate expectations with stakeholders. – Optimize a final model for inference, and ship the model to the deployment environment of choice—web serv
