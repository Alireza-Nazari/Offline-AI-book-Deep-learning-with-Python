# Generation bundle for page_200  (PDF page 200, printed page 176)

Page kind: content
Chapter: 6 The universal workflow of machine learning
Section: 6.1 Defining the task
Subsection: 6.1.2 Collecting a dataset
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_200.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_200.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p200_e03] (body) Outsourcing can potentially save you time and money but takes away control. Using something like Mechanical Turk is likely to be inexpensive and to scale well, but your annotations may end up being quite noisy.
[p200_e04] (body) To pick the best option, consider the constraints you’re working with:
[p200_e05] (bullet) Do the data labelers need to be subject matter experts, or could anyone annotate the data? The labels for a cat-versus-dog image classification problem can be selected by anyone, but those for a dog breed classification task require specialized knowledge. Meanwhile, annotating CT scans of bone fractures pretty much requires a medical degree.
[p200_e06] (bullet) If annotating the data requires specialized knowledge, can you train people to do it? If not, how can you get access to relevant experts?
[p200_e07] (bullet) Do you, yourself, understand the way experts come up with the annotations? If you don’t, you will have to treat your dataset as a black box, and you won’t be able to perform manual feature engineering—this isn’t critical, but it can be limiting.
[p200_e08] (body) If you decide to label your data in-house, ask yourself what software you will use to record annotations. You may well need to develop that software yourself. Productive data annotation software will save you a lot of time, so it’s something worth investing in early in a project.
[p200_e09] (heading_subsubsection) Beware of nonrepresentative data
[p200_e10] (body) Machine learning models can only make sense of inputs that are similar to what they’ve seen before. As such, it’s critical that the data used for training should be representative of the production data. This concern should be the foundation of all of your data collection work.
[p200_e11] (body) Suppose you’re developing an app where users can take pictures of a dish to find out its name. You train a model using pictures from an image-sharing social network that’s popular with foodies. Come deployment time, and feedback from angry users starts rolling in: your app gets the answer wrong 8 times out of 10. What’s going on? Your accuracy on the test set was well over 90%! A quick look at user-uploaded data reveals that mobile picture uploads of random dishes from random restaurants taken with random smartphones look nothing like the professional-quality, well-lit, appetizing pictures you trained the model on: your training data wasn’t representative of the production data. That’s a cardinal sin—welcome to machine learning hell. If possible, collect data directly from the environment where your model will be used. A movie review sentiment classification model should be used on new IMDB reviews, not on Yelp restaurant reviews, nor on Twitter status updates. If you want to rate the sentiment of a tweet, start by collecting and annotating actual tweets—from a similar set of users as those you’re expecting in production. If it’s not possible to train on production data, then make sure you fully understand how your training and production data differ, and that you are actively correcting these differences. A related phenomenon you should be aware of is concept drift. You’ll encounter concept drift in almost all real-world problems, especially those that deal with

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

## Previous page (PDF 199) text, for continuity
access to a separate dataset of tens of thousands of unfiltered social media posts, and manually tag them as spam, offensive, or acceptable.

For the music recommendation engine, you can just use the “likes” of your users. No new data needs to be collected. Likewise, for the click-through rate prediction project, you have an extensive record of click-through rate for your past ads, going back years.

For the cookie-flagging model, you will need to install cameras above the conveyor belts to collect tens of thousands of images, and then someone will need to manually label these images. The people who know how to do this currently work at the cookie factory—but it doesn’t seem too difficult, you should be able to train people to do it.

The satellite imagery project will require a team of archaeologists to collect a database of existing sites of interest, and for each site, you will need to find existing satellite images taken in different weather conditions. To get a good model, you’re going to need thousands of different sites.

You’ve learned in chapter 5 that a model’s ability to generalize comes almost entirely from the properties of the data it is trained on—the number of data points you have, the reliability of your labels, the quality of your features. A good dataset is an asset worthy of care and investment. If you get an extra 50 hours to spend on a project, chances are that the most effective way to allocate them is to collect more data, rather than search for incremental modeling improvements.

The point that data matters more than algorithms was most famously made in a 2009 paper by Google researchers titled “The Unreasonable Effectiveness of Data” (the title is a riff on the well-known 1960 book The Unreasonable Effectiveness of Mathematics in the Natural Sc …

## Next page (PDF 201) text, for continuity
user-generated data. Concept drift occurs when the properties of the production data change over time, causing model accuracy to gradually decay. A music recommendation engine trained in the year 2013 may not be very effective today. Likewise, the IMDB dataset you worked with was collected in 2011, and a model trained on it would likely not perform as well on reviews from 2020 compared to reviews from 2012, as vocabulary, expressions, and movie genres evolve over time. Concept drift is particularly acute in adversarial contexts like credit card fraud detection, where fraud patterns change practically every day. Dealing with fast concept drift requires constant data collection, annotation, and model retraining.

Keep in mind that machine learning can only be used to memorize patterns that are present in your training data. You can only recognize what you’ve seen before. Using machine learning trained on past data to predict the future is making the assumption that the future will behave like the past. That often isn’t the case.

The problem of sampling bias

A particularly insidious and common case of nonrepresentative data is sampling bias. Sampling bias occurs when your data collection process interacts with what you are trying to predict, resulting in biased measurements. A famous historical example occurred in the 1948 US presidential election. On election night, the Chicago Tribune printed the headline “DEWEY DEFEATS TRUMAN.” The next morning, Truman emerged as the winner. The editor of the Tribune had trusted the results of a phone survey—but phone users in 1948 were not a random, representative sample of the voting population. They were more likely to be richer, conservative, and to vote for Dewey, the Republican candidate. Nowadays, every phone survey takes sampl …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.213 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.198 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.197 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.202 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.212 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.208 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.195 — 6 The universal workflow of machine learning › 
- PDF p.623 — 20 Conclusions › 20.1 Key concepts in review

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.198 (printed 174) — 6 The universal workflow of machine learning › 6.1 Defining the task › Collecting a dataset
Once you understand the nature of the task and you know what your inputs and targets are going to be, it’s time for data collection—the most arduous, time-consuming, and costly part of most machine learning projects:

The photo search engine project requires you to first select the set of labels you want to classify—you settle on 10,000 common image categories. Then, you need to manually tag hundreds of thousands of your past user-uploaded images with labels from this set.

For the chat app spam detection project, because user chats are end-to-end encrypted, you cannot use their contents for training a model. You need to gain

### PDF p.197 (printed 173) — 6 The universal workflow of machine learning › 6.1 Defining the task › 
What will your input data be? What are you trying to predict? You can only learn to predict something if you have available training data: for example, you can only learn to classify the sentiment of movie reviews if you have both movie reviews and sentiment annotations available. As such, data availability is usually the limiting factor at this stage. In many cases, you will have to resort to collecting and annotating new datasets yourself (which we cover in the next section).

What type of machine learning task are you facing? Is it binary classification? Multiclass classification? Scalar regression? Vector regression? Multiclass, multilabel classification? Image segmentation? Ranking? Something else, like clustering, generation, or reinforcement learning? In some cases, it may be that machine learning isn’t even the best way to make sense of your data, and you should use something else, such as plain old-school statistical analysis: – The photo search engine project is a multiclass, multilabel classification task. – The spam detection project is a binary classification task. If you set “offensive content” as a separate class, it’s a three-way classification task. – The music rec

### PDF p.213 (printed 189) — 6 The universal workflow of machine learning › 6.3 Deploying your model › Summary
When you take on a new machine learning project, first, define the problem at hand: – Understand the broader context of what you’re setting out to do—what’s the end goal and what are the constraints? – Collect and annotate a dataset; make sure you understand your data in depth. – Choose how you’ll measure success on your problem. What metrics will you monitor on your validation data?

Once you understand the problem and you have an appropriate dataset, develop a model: – Prepare your data. – Pick your evaluation protocol. Hold-out validation? K-fold validation? Which portion of the data should you use for validation? – Achieve statistical power: beat a simple baseline. – Scale up: develop a model that can overfit. – Regularize your model and tune its hyperparameters, based on performance on the validation data. A lot of machine learning research tends to focus only on this step—but keep the big picture in mind.

When your model is ready and yields good performance on the test data, it’s time for deployment: – First, make sure to set appropriate expectations with stakeholders. – Optimize a final model for inference, and ship the model to the deployment environment of choice—web serv

### PDF p.202 (printed 178) — 6 The universal workflow of machine learning › 6.1 Defining the task › Understanding your data
It’s bad practice to treat a dataset as a black box. Before you start training models, you should explore and visualize your data to gain insights about what makes it predictive— which will inform feature engineering—and screen for potential issues:

If your data includes images or natural language text, take a look at a few samples (and their labels) directly.

If your data contains numerical features, it’s a good idea to plot the histogram of feature values to get a feel for the range of values taken and the frequency of different values.

If your data includes location information, plot it on a map. Do any clear patterns emerge?

Are some samples missing values for some features? If so, you’ll need to deal with this when you prepare the data (we cover how to do this in the next section).

If your task is a classification problem, print the number of instances of each class in your data. Are the classes roughly equally represented? If not, you will need to account for this imbalance.

Check for target leaking—the presence of features in your data that provide information about the targets that may not be available in production. If you’re training a model on medical records to pr

### PDF p.213 (printed 189) — 6 The universal workflow of machine learning › 6.3 Deploying your model › 
As soon as your model has launched, you should be getting ready to train the next generation that will replace it:

Watch out for changes in the production data. Are new features becoming available? Should you expand or otherwise edit the label set?

Keep collecting and annotating data, and keep improving your annotation pipeline over time. In particular, you should pay special attention to collecting samples that seem to be difficult to classify for your current model—such samples are the most likely to help improve performance.

This concludes the universal workflow of machine learning—that’s a lot of things to keep in mind. It takes time and experience to become an expert, but don’t worry, you’re already a lot wiser than you were a few chapters ago. You are now familiar with the big picture—the entire spectrum of what machine learning projects entail. While most of this book will focus on the model development part, you’re now aware that it’s only one part of the entire workflow. Always keep in mind the big picture!

### PDF p.623 (printed 599) — 20 Conclusions › 20.1 Key concepts in review › 20.1.5 The universal machine learning workflow
Having access to an extremely powerful tool for creating models that map any input space to any target space is great, but the difficult part of the machine learning workflow is often everything that comes before designing and training such models (and, for production models, what comes after, as well). Understanding the problem domain to be able to determine what to attempt to predict, given what data, and how to measure success is a prerequisite for any successful application of machine learning, and it isn’t something that advanced tools like Keras and TensorFlow can help you with. As a reminder, here’s a quick summary of the typical machine learning workflow as described in chapter 6:

Define the problem. What data is available, and what are you trying to predict? Will you need to collect more data or hire people to manually label a dataset?

Identify a way to reliably measure success on your goal. For simple tasks, this may be prediction accuracy, but in many cases, it will require sophisticated, domain-specific metrics.

Prepare the validation process that you’ll use to evaluate your models. In particular, you should define a training set, a validation set, and a test set. Th

### PDF p.195 (printed 171) — 6 The universal workflow of machine learning ›  › This chapter covers
Framing a machine learning problem

Developing a working model

Deploying your model in production and

maintaining it

Our previous examples have assumed that we already had a labeled dataset to start from, and that we could immediately start training a model. In the real world, this is often not the case. You don’t start from a dataset; you start from a problem.

Imagine that you’re launching your own machine learning consulting shop. You incorporate, you put up a fancy website, you notify your network. The projects start rolling in:

A personalized photo search engine for a picture-sharing social network— type in “wedding” and retrieve all the pictures you took at weddings, without any manual tagging needed.

Flagging spam and offensive text content among the posts of a budding chat app.

### PDF p.129 (printed 105) — 4 Classification and regression ›  › Classification and regression glossary
Classification and regression involve many specialized terms. You’ve come across some of them in earlier examples, and you’ll see more of them in future chapters. They have precise, machine learning–specific definitions, and you should be familiar with them:

Sample or input—One data point that goes into your model.

Prediction or output—What comes out of your model.

Target—The truth. What your model should ideally have predicted, according to

an external source of data.

Prediction error or loss value—A measure of the distance between your model’s

prediction and the target.

Classes—A set of possible labels to choose from in a classification problem.

For example, when classifying cat and dog pictures, “dog” and “cat” are the two classes.

Label—A specific instance of a class annotation in a classification problem.

For instance, if picture #1234 is annotated as containing the class “dog,” then “dog” is a label of picture #1234.

Ground-truth or annotations—All targets for a dataset, typically collected by

humans.

Binary classification—A classification task where each input sample should be

categorized into two exclusive categories.

Categorical classification or multiclass 
