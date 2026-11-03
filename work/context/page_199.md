# Generation bundle for page_199  (PDF page 199, printed page 175)

Page kind: content
Chapter: 6 The universal workflow of machine learning
Section: 6.1 Defining the task
Subsection: 6.1.2 Collecting a dataset
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_199.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_199.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p199_e03] (body) access to a separate dataset of tens of thousands of unfiltered social media posts, and manually tag them as spam, offensive, or acceptable.
[p199_e04] (bullet) For the music recommendation engine, you can just use the “likes” of your users. No new data needs to be collected. Likewise, for the click-through rate prediction project, you have an extensive record of click-through rate for your past ads, going back years.
[p199_e05] (bullet) For the cookie-flagging model, you will need to install cameras above the conveyor belts to collect tens of thousands of images, and then someone will need to manually label these images. The people who know how to do this currently work at the cookie factory—but it doesn’t seem too difficult, you should be able to train people to do it.
[p199_e06] (bullet) The satellite imagery project will require a team of archaeologists to collect a database of existing sites of interest, and for each site, you will need to find existing satellite images taken in different weather conditions. To get a good model, you’re going to need thousands of different sites.
[p199_e07] (body) You’ve learned in chapter 5 that a model’s ability to generalize comes almost entirely from the properties of the data it is trained on—the number of data points you have, the reliability of your labels, the quality of your features. A good dataset is an asset worthy of care and investment. If you get an extra 50 hours to spend on a project, chances are that the most effective way to allocate them is to collect more data, rather than search for incremental modeling improvements.
[p199_e08] (body) The point that data matters more than algorithms was most famously made in a 2009 paper by Google researchers titled “The Unreasonable Effectiveness of Data” (the title is a riff on the well-known 1960 book The Unreasonable Effectiveness of Mathematics in the Natural Sciences by Eugene Wigner). This was before deep learning was popular, but remarkably, the rise of deep learning has only increased the importance of data. If you’re doing supervised learning, then once you’ve collected inputs (such as images) you’re going to need annotations for them (such as tags for those images): the targets you will train your model to predict. Sometimes, annotations can be retrieved automatically—for instance, in the case of our music recommendation task or our click-through rate prediction task. But often, you have to annotate your data by hand. This is a labor-heavy process.
[p199_e09] (heading_subsubsection) Investing in data annotation infrastructure
[p199_e10] (body) Your data annotation process will determine the quality of your targets, which, in turn, determines the quality of your model. Carefully consider the options you have available:
[p199_e11] (bullet) Should you annotate the data yourself?
[p199_e12] (bullet) Should you use a crowdsourcing platform like Mechanical Turk to collect labels?
[p199_e13] (bullet) Should you use the services of a specialized data-labeling company?

## Chapter outline
Chapter: 6 The universal workflow of machine learning (PDF pages 195-213)
  - 6.1 Defining the task (p.196)
      - 6.1.1 Framing the problem (p.196)
      - 6.1.2 Collecting a dataset (p.198)
      - 6.1.3 Understanding your data (p.202)
      - 6.1.4 Choosing a measure of success (p.202)
  - 6.2 Developing a model (p.203)
      - 6.2.1 Preparing the data (p.203)
      - 6.2.2 Choosing an evaluation protocol (p.204)
      - 6.2.3 Beating a baseline (p.205)
      - 6.2.4 Scaling up: Developing a model that overfits (p.206)
      - 6.2.5 Regularizing and tuning your model (p.207)
  - 6.3 Deploying your model (p.207)
      - 6.3.1 Explaining your work to stakeholders and setting expectations (p.208)
      - 6.3.2 Shipping an inference model (p.208)
      - 6.3.3 Monitoring your model in the wild (p.212)
      - 6.3.4 Maintaining your model (p.212)

## Previous page (PDF 198) text, for continuity
Once you’ve done your research, you should know what your inputs will be, what your targets will be, and what broad type of machine learning task the problem maps to. Be aware of the hypotheses you’re making at this stage:

You hypothesize that your targets can be predicted given your inputs.

You hypothesize that the data that’s available (or that you will soon collect) is sufficiently informative to learn the relationship between inputs and targets.

Until you have a working model, these are merely hypotheses, waiting to be validated or invalidated. Not all problems can be solved with machine learning; just because you’ve assembled examples of inputs X and targets Y doesn’t mean X contains enough information to predict Y. For instance, if you’re trying to predict the movements of a stock on the stock market given its recent price history, you’re unlikely to succeed, because price history doesn’t contain much predictive information.

Note on ethics

You may sometimes be offered ethically dubious projects, such as “building an AI that rates the trustworthiness of someone from a picture of their face.” First of all, the validity of the project is in doubt: it isn’t clear why trustworthiness would be reflected on someone’s face. Second, such a task opens the door to all kinds of ethical problems. Collecting a dataset for this task would amount to recording the biases and prejudices of the people who label the pictures. The models you would train on such data would be merely encoding these same biases—into a black box algorithm, which would give them a thin veneer of legitimacy. In a largely tech-illiterate society like ours, “The AI algorithm said this person cannot be trusted” strangely appears to carry more weight and objectivity than “John Smith said this person cannot …

## Next page (PDF 200) text, for continuity
Outsourcing can potentially save you time and money but takes away control. Using something like Mechanical Turk is likely to be inexpensive and to scale well, but your annotations may end up being quite noisy.

To pick the best option, consider the constraints you’re working with:

Do the data labelers need to be subject matter experts, or could anyone annotate the data? The labels for a cat-versus-dog image classification problem can be selected by anyone, but those for a dog breed classification task require specialized knowledge. Meanwhile, annotating CT scans of bone fractures pretty much requires a medical degree.

If annotating the data requires specialized knowledge, can you train people to do it? If not, how can you get access to relevant experts?

Do you, yourself, understand the way experts come up with the annotations? If you don’t, you will have to treat your dataset as a black box, and you won’t be able to perform manual feature engineering—this isn’t critical, but it can be limiting.

If you decide to label your data in-house, ask yourself what software you will use to record annotations. You may well need to develop that software yourself. Productive data annotation software will save you a lot of time, so it’s something worth investing in early in a project.

Beware of nonrepresentative data

Machine learning models can only make sense of inputs that are similar to what they’ve seen before. As such, it’s critical that the data used for training should be representative of the production data. This concern should be the foundation of all of your data collection work.

Suppose you’re developing an app where users can take pictures of a dish to find out its name. You train a model using pictures from an image-sharing social network that’s popular with food …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.196 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.197 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.212 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.623 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.202 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.207 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.213 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.183 — 5 Fundamentals of machine learning › 5.4 Improving generalization

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.197 (printed 173) — 6 The universal workflow of machine learning › 6.1 Defining the task › 
What will your input data be? What are you trying to predict? You can only learn to predict something if you have available training data: for example, you can only learn to classify the sentiment of movie reviews if you have both movie reviews and sentiment annotations available. As such, data availability is usually the limiting factor at this stage. In many cases, you will have to resort to collecting and annotating new datasets yourself (which we cover in the next section).

What type of machine learning task are you facing? Is it binary classification? Multiclass classification? Scalar regression? Vector regression? Multiclass, multilabel classification? Image segmentation? Ranking? Something else, like clustering, generation, or reinforcement learning? In some cases, it may be that machine learning isn’t even the best way to make sense of your data, and you should use something else, such as plain old-school statistical analysis: – The photo search engine project is a multiclass, multilabel classification task. – The spam detection project is a binary classification task. If you set “offensive content” as a separate class, it’s a three-way classification task. – The music rec

### PDF p.196 (printed 172) — 6 The universal workflow of machine learning › 6.1 Defining the task › 
Building a music recommendation system for users of an online radio.

Detecting credit card fraud for an e-commerce website.

Predicting display ad click-through rate to decide which ad to serve to a given user at a given time.

Flagging anomalous cookies on the conveyor belt of a cookie-manufacturing line.

Using satellite images to predict the location of as-yet unknown archaeological sites.

It would be very convenient if you could import the correct dataset from `keras
` `.datasets` and start fitting some deep learning models. Unfortunately, in the real world, you’ll have to start from scratch.

In this chapter, you’ll learn about the universal step-by-step blueprint that you can use to approach and solve any machine learning problem, like those previously listed. This template will bring together and consolidate everything you’ve learned in chapters 4 and 5 and give you the wider context that should anchor what you will learn in the next chapters. The universal workflow of machine learning is broadly structured in three parts:

Define the task—Understand the problem domain and the business logic underlying what the customer asked. Collect a dataset, understand what the data re

### PDF p.212 (printed 188) — 6 The universal workflow of machine learning › 6.3 Deploying your model › Monitoring your model in the wild
You’ve exported an inference model, you’ve integrated it into your application, and you’ve done a dry run on production data—the model behaved exactly as you expected. You’ve written unit tests as well as logging and status-monitoring code— perfect. Now it’s time to press the big red button and deploy to production.

Even this is not the end. Once you’ve deployed a model, you need to keep monitoring its behavior, its performance on new data, its interaction with the rest of the application, and its eventual impact on business metrics:

Is user engagement in your online radio up or down after deploying the new music recommender system? Has average ad click-through rate increased after switching to the new click-through rate prediction model? Consider using randomized A/B testing to isolate the impact of the model itself from other changes: a subset of cases should go through the new model, while another control subset should stick to the old process. Once sufficiently many cases have been processed, the difference in outcomes between the two is likely attributable to the model.

If possible, do a regular manual audit of the model’s predictions on production data. It’s generally poss

### PDF p.197 (printed 173) — 6 The universal workflow of machine learning › 6.1 Defining the task › 
What do existing solutions look like? Perhaps your customer already has a hand-crafted algorithm that handles spam filtering or credit card fraud detection— with lots of nested `if` statements. Perhaps a human is currently in charge of manually handling the process considered—monitoring the conveyor belt at the cookie plant and manually removing the bad cookies, or crafting playlists of song recommendations to be sent out to users who liked a specific artist. You should make sure to understand what systems are already in place, and how they work.

Are there particular constraints you will need to deal with? For example, you could find out that the app for which you’re building a spam detection system is strictly end-to-end encrypted, so that the spam detection model will have to live on the end-user’s phone, and must be trained on an external dataset. Perhaps the cookie-filtering model has such latency constraints that it will need to run on an embedded device at the factory rather than on a remote server. You should understand the full context in which your work will fit.

### PDF p.183 (printed 159) — 5 Fundamentals of machine learning › 5.4 Improving generalization › Dataset curation
You’ve already learned that generalization in deep learning originates from the latent structure of your data. If your data makes it possible to smoothly interpolate between samples, then you will be able to train a deep learning model that generalizes. If your problem is overly noisy or fundamentally discrete, like, say, list sorting, deep learning will not help you. Deep learning is curve fitting, not magic.

As such, it is essential that you make sure that you’re working with an appropriate dataset. Spending more effort and money on data collection almost always yields a much greater return on investment than spending the same on developing a better model:

Make sure you have enough data. Remember that you need a dense sampling of the input-cross-output space. More data will yield a better model. Sometimes, problems that seem impossible at first become solvable with a larger dataset.

Minimize labeling errors—visualize your inputs to check for anomalies, and proofread your labels.

Clean your data and deal with missing values (we cover this in the next chapter).

If you have many features and you aren’t sure which ones are actually useful, do feature selection.

A particularly i

### PDF p.623 (printed 599) — 20 Conclusions › 20.1 Key concepts in review › 20.1.5 The universal machine learning workflow
Having access to an extremely powerful tool for creating models that map any input space to any target space is great, but the difficult part of the machine learning workflow is often everything that comes before designing and training such models (and, for production models, what comes after, as well). Understanding the problem domain to be able to determine what to attempt to predict, given what data, and how to measure success is a prerequisite for any successful application of machine learning, and it isn’t something that advanced tools like Keras and TensorFlow can help you with. As a reminder, here’s a quick summary of the typical machine learning workflow as described in chapter 6:

Define the problem. What data is available, and what are you trying to predict? Will you need to collect more data or hire people to manually label a dataset?

Identify a way to reliably measure success on your goal. For simple tasks, this may be prediction accuracy, but in many cases, it will require sophisticated, domain-specific metrics.

Prepare the validation process that you’ll use to evaluate your models. In particular, you should define a training set, a validation set, and a test set. Th

### PDF p.202 (printed 178) — 6 The universal workflow of machine learning › 6.1 Defining the task › Understanding your data
It’s bad practice to treat a dataset as a black box. Before you start training models, you should explore and visualize your data to gain insights about what makes it predictive— which will inform feature engineering—and screen for potential issues:

If your data includes images or natural language text, take a look at a few samples (and their labels) directly.

If your data contains numerical features, it’s a good idea to plot the histogram of feature values to get a feel for the range of values taken and the frequency of different values.

If your data includes location information, plot it on a map. Do any clear patterns emerge?

Are some samples missing values for some features? If so, you’ll need to deal with this when you prepare the data (we cover how to do this in the next section).

If your task is a classification problem, print the number of instances of each class in your data. Are the classes roughly equally represented? If not, you will need to account for this imbalance.

Check for target leaking—the presence of features in your data that provide information about the targets that may not be available in production. If you’re training a model on medical records to pr

### PDF p.213 (printed 189) — 6 The universal workflow of machine learning › 6.3 Deploying your model › Summary
When you take on a new machine learning project, first, define the problem at hand: – Understand the broader context of what you’re setting out to do—what’s the end goal and what are the constraints? – Collect and annotate a dataset; make sure you understand your data in depth. – Choose how you’ll measure success on your problem. What metrics will you monitor on your validation data?

Once you understand the problem and you have an appropriate dataset, develop a model: – Prepare your data. – Pick your evaluation protocol. Hold-out validation? K-fold validation? Which portion of the data should you use for validation? – Achieve statistical power: beat a simple baseline. – Scale up: develop a model that can overfit. – Regularize your model and tune its hyperparameters, based on performance on the validation data. A lot of machine learning research tends to focus only on this step—but keep the big picture in mind.

When your model is ready and yields good performance on the test data, it’s time for deployment: – First, make sure to set appropriate expectations with stakeholders. – Optimize a final model for inference, and ship the model to the deployment environment of choice—web serv
