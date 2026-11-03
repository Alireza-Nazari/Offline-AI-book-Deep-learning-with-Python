# Generation bundle for page_176  (PDF page 176, printed page 152)

Page kind: content
Chapter: 5 Fundamentals of machine learning
Section: 5.2 Evaluating machine-learning models
Subsection: 5.2.3 Things to keep in mind about model evaluation
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_176.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_176.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p176_e03] (heading_subsubsection) Iterated K-fold validation with shuffling
[p176_e04] (body) This one is for situations in which you have relatively little data available and you need to evaluate your model as precisely as possible. I’ve found it to be extremely helpful in Kaggle competitions. It consists of applying K-fold validation multiple times, shuffling the data every time before splitting it `K` ways. The final score is the average of the scores obtained at each run of K-fold validation. Note that you end up training and evaluating `P * K` models (where `P` is the number of iterations you use), which can be very expensive.
[p176_e05] (heading_subsection) 5.2.2
[p176_e06] (heading_subsection) Beating a common-sense baseline
[p176_e07] (body) Besides the different evaluation protocols you have available, one last thing you should know about is the use of common-sense baselines.
[p176_e08] (body) Training a deep learning model is a bit like pressing a button that launches a rocket in a parallel world. You can’t hear it or see it. You can’t observe the manifold learning process—it’s happening in a space with thousands of dimensions, and even if you projected it to 3D, you couldn’t interpret it. The only feedback you have is your validation metrics—like an altitude meter on your invisible rocket. A particularly important point is to be able to tell whether you’re getting off the ground at all. What was the altitude you started at? Your model seems to have an accuracy of 15%, is that any good? Before you start working with a dataset, you should always pick a trivial baseline that you’ll try to beat. If you cross that threshold, you’ll know you’re doing something right: your model is actually using the information in the input data to make predictions that generalize—you can keep going. This baseline could be performance of a random classifier, or the performance of the simplest non-machine learning technique you can imagine. For instance, in the MNIST digit-classification example, a simple baseline would be a validation accuracy greater than 0.1 (random classifier); in the IMDB example, it would be a validation accuracy greater than 0.5. In the Reuters example, it would be around 0.18–0.19, due to class imbalance. If you have a binary classification problem where 90% of samples belong to class A and 10% belong to class B, then a classifier that always predicts A already achieves 0.9 in validation accuracy, and you’ll need to do better than that. Having a common sense baseline you can refer to is essential when you’re getting started on a problem no one has solved before. If you can’t beat a trivial solution, your model is worthless—perhaps you’re using the wrong model or perhaps the problem you’re tackling can’t even be approached with machine learning in the first place. Time to go back to the drawing board.
[p176_e09] (heading_subsection) 5.2.3
[p176_e10] (heading_subsection) Things to keep in mind about model evaluation
[p176_e11] (body) Keep an eye out for the following when you’re choosing an evaluation protocol:
[p176_e12] (bullet) Data representativeness—You want both your training set and test set to be representative of the data at hand. For instance, if you’re trying to classify images of digits, and you’re starting from an array of samples where the samples are

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

## Previous page (PDF 175) text, for continuity
Your final score is then the averages of the K scores obtained. This method is helpful when the performance of your model shows significant variance based on your train/ test split. Like hold-out validation, this method doesn’t exempt you from using a distinct validation set for model calibration.

Schematically, K-fold cross-validation looks like figure 5.13. Listing 5.6 shows a simple implementation.

Figure 5.13 Three-fold validation

Listing 5.6 K-fold cross-validation (note that labels are omitted for simplicity)

## Next page (PDF 177) text, for continuity
ordered by their class, taking the first 80% of the array as your training set and the remaining 20% as your test set will result in your training set containing only classes 0–7, whereas your test set contains only classes 8–9. This seems like a ridiculous mistake, but it’s surprisingly common. For this reason, you usually should randomly shuffle your data before splitting it into training and test sets.

The arrow of time—If you’re trying to predict the future given the past (for example, tomorrow’s weather, stock movements, and so on), you should not randomly shuffle your data before splitting it because doing so will create a temporal leak: your model will effectively be trained on data from the future. In such situations, you should always make sure all data in your test set is posterior to the data in the training set.

Redundancy in your data—If some data points in your data appear twice (fairly common with real-world data), then shuffling the data and splitting it into a training set and a validation set will result in redundancy between the training and validation sets. In effect, you’ll be testing on part of your training data, which is the worst thing you can do! Make sure your training set and validation set are disjoint.

Having a reliable way to evaluate the performance of your model is how you’ll be able to monitor the tension at the heart of machine learning—between optimization and generalization, underfitting and overfitting.

5.3

Improving model fit

To achieve the perfect fit, you must first overfit. Since you don’t know in advance where the boundary lies, you must cross it to find it. Thus, your initial goal as you start working on a problem is to achieve a model that shows some generalization power, and that is able to overfit. Once you have such  …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.205 — 6 The universal workflow of machine learning › 6.2 Developing a model
- PDF p.174 — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models
- PDF p.179 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.173 — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models
- PDF p.193 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.153 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.135 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.213 — 6 The universal workflow of machine learning › 6.3 Deploying your model

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.205 (printed 181) — 6 The universal workflow of machine learning › 6.2 Developing a model › 
performance. The goal of your validation protocol is to accurately estimate what your success metric of choice (such as accuracy) will be on actual production data. The reliability of that process is critical to building a useful model.

In chapter 5, we reviewed three common evaluation protocols:

Maintaining a hold-out validation set—The way to go when you have plenty of data

Doing K-fold cross-validation—The right choice when you have too few samples for hold-out validation to be reliable

Doing iterated K-fold validation—For performing highly accurate model evaluation when little data is available

Just pick one of these. In most cases, the first will work well enough. As you’ve learned before, always be mindful of the representativity of your validation set(s) and be careful not to have redundant samples between your training set and your validation set(s).

### PDF p.174 (printed 150) — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models › Simple hold-out validation
This is the simplest evaluation protocol, and it suffers from one flaw: if little data is available, then your validation and test sets may contain too few samples to be statistically representative of the data at hand. This is easy to recognize: if different random shuffling rounds of the data before splitting end up yielding very different measures of model performance, then you’re having this issue. K-fold validation and iterated K-fold validation are two ways to address this, as discussed next.

### PDF p.173 (printed 149) — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models › Training, validation, and test sets
But if you repeat this many times— running one experiment, evaluating on the validation set, and modifying your model as a result—then you’ll leak an increasingly significant amount of information about the validation set into the model. At the end of the day, you’ll end up with a model that performs artificially well on the validation data because that’s what you optimized it for. You care about performance on completely new data, not the validation data, so you need to use a completely different, never-before-seen dataset to evaluate the model: the test dataset. Your model shouldn’t have had access to any information about the test set, even indirectly. If anything about the model has been tuned based on test set performance, then your measure of generalization will be flawed. Splitting your data into training, validation, and test sets may seem straightforward, but there are a few advanced ways to do it that can come in handy when little data is available. Let’s review three classic evaluation recipes: simple hold-out validation, K-fold validation, and iterated K-fold validation with shuffling. We’ll also

### PDF p.153 (printed 129) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › Validating your approach using K-fold validation
To evaluate your model while you keep adjusting its parameters (such as the number of epochs used for training), you could split the data into a training set and a validation set, as you did in the previous examples. But because you have so few data points, the validation set would end up being very small (for instance, about 100 examples). As a consequence, the validation scores might change a lot depending on which data points you chose to use for validation and which you chose for training: the validation scores might have a high variance with regard to the validation split. This would prevent you from reliably evaluating your model.

The best practice in such situations is to use K-fold cross-validation (see figure 4.9). It consists of splitting the available data into K partitions (typically K = 4 or 5), instantiating K identical models, and training each one on K – 1 partitions while evaluating on the remaining partition. The validation score for the model used is then the average of the K validation scores obtained. In terms of code, this is straightforward.

### PDF p.213 (printed 189) — 6 The universal workflow of machine learning › 6.3 Deploying your model › Summary
When you take on a new machine learning project, first, define the problem at hand: – Understand the broader context of what you’re setting out to do—what’s the end goal and what are the constraints? – Collect and annotate a dataset; make sure you understand your data in depth. – Choose how you’ll measure success on your problem. What metrics will you monitor on your validation data?

Once you understand the problem and you have an appropriate dataset, develop a model: – Prepare your data. – Pick your evaluation protocol. Hold-out validation? K-fold validation? Which portion of the data should you use for validation? – Achieve statistical power: beat a simple baseline. – Scale up: develop a model that can overfit. – Regularize your model and tune its hyperparameters, based on performance on the validation data. A lot of machine learning research tends to focus only on this step—but keep the big picture in mind.

When your model is ready and yields good performance on the test data, it’s time for deployment: – First, make sure to set appropriate expectations with stakeholders. – Optimize a final model for inference, and ship the model to the deployment environment of choice—web serv

### PDF p.193 (printed 169) — 5 Fundamentals of machine learning › 5.4 Improving generalization › Summary
The purpose of a machine learning model is to generalize: to perform accurately on never-before-seen inputs. It’s harder than it seems.

A deep neural network achieves generalization by learning a parametric model that can successfully interpolate between training samples. Such a model can be said to have learned the latent manifold of the training data. This is why deep learning models can only make sense of inputs that are very close to what they’ve seen during training.

The fundamental problem in machine learning is the tension between optimization and generalization: to attain generalization, you must first achieve a good fit to the training data, but improving your model’s fit to the training data will inevitably start hurting generalization after a while. Every single deep learning best practice deals with managing this tension.

The ability of deep learning models to generalize comes from the fact that they manage to learn to approximate the latent manifold of their data and can thus make sense of new inputs via interpolation.

It’s essential to be able to accurately evaluate the generalization power of your model while you’re developing it. You have at your disposal an arr

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
