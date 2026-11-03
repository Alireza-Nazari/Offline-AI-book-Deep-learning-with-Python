# Generation bundle for page_195  (PDF page 195, printed page 171)

Page kind: chapter_opener
Chapter: 6 The universal workflow of machine learning
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_195.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_195.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p195_e01] (chapter_number) 6
[p195_e02] (chapter_title) The universal
[p195_e03] (chapter_title) workflow of
[p195_e04] (chapter_title) machine learning
[p195_e05] (chapter_covers) This chapter covers
[p195_e06] (bullet) Framing a machine learning problem
[p195_e07] (bullet) Developing a working model
[p195_e08] (bullet) Deploying your model in production and
[p195_e09] (bullet) maintaining it
[p195_e10] (body) Our previous examples have assumed that we already had a labeled dataset to start from, and that we could immediately start training a model. In the real world, this is often not the case. You don’t start from a dataset; you start from a problem.
[p195_e11] (body) Imagine that you’re launching your own machine learning consulting shop. You incorporate, you put up a fancy website, you notify your network. The projects start rolling in:
[p195_e12] (bullet) A personalized photo search engine for a picture-sharing social network— type in “wedding” and retrieve all the pictures you took at weddings, without any manual tagging needed.
[p195_e13] (bullet) Flagging spam and offensive text content among the posts of a budding chat app.

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

## Previous page (PDF 194) text, for continuity
separate test set for final model evaluation, since information leaks from your validation data to your model may have occurred.

When you start working on a model, your goal is first to achieve a model that has some generalization power and that can overfit. Best practices to do this include tuning your learning rate and batch size, using better architecture priors, increasing model capacity, or simply training longer.

As your model starts overfitting, your goal switches to improving generalization through model regularization. You can reduce your model’s capacity, add dropout or weight regularization, and use early stopping. And naturally, a larger or better dataset is always the number one way to help a model generalize.

## Next page (PDF 196) text, for continuity
Building a music recommendation system for users of an online radio.

Detecting credit card fraud for an e-commerce website.

Predicting display ad click-through rate to decide which ad to serve to a given user at a given time.

Flagging anomalous cookies on the conveyor belt of a cookie-manufacturing line.

Using satellite images to predict the location of as-yet unknown archaeological sites.

It would be very convenient if you could import the correct dataset from `keras
` `.datasets` and start fitting some deep learning models. Unfortunately, in the real world, you’ll have to start from scratch.

In this chapter, you’ll learn about the universal step-by-step blueprint that you can use to approach and solve any machine learning problem, like those previously listed. This template will bring together and consolidate everything you’ve learned in chapters 4 and 5 and give you the wider context that should anchor what you will learn in the next chapters. The universal workflow of machine learning is broadly structured in three parts:

Define the task—Understand the problem domain and the business logic underlying what the customer asked. Collect a dataset, understand what the data represents, and choose how you will measure success on the task.

Develop a model—Prepare your data so that it can be processed by a machine learning model, select a model evaluation protocol and a simple baseline to beat, train a first model that has generalization power that can overfit, and then regularize and tune your model until you achieve the best possible generalization performance.

Deploy the model—Present your work to stakeholders, ship the model to a web server, a mobile app, a web page, or an embedded device, monitor the model’s performance in the wild, and start collecting the dat …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.200 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.198 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.203 — 6 The universal workflow of machine learning › 6.2 Developing a model
- PDF p.212 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.205 — 6 The universal workflow of machine learning › 6.2 Developing a model
- PDF p.41 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network
- PDF p.28 — 1 What is deep learning? › 1.4 Learning rules and representations from data
- PDF p.213 — 6 The universal workflow of machine learning › 6.3 Deploying your model

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.203 (printed 179) — 6 The universal workflow of machine learning › 6.2 Developing a model › Developing a model
Once you know how you will measure your progress, you can get started with model development. Most tutorials and research projects assume that this is the only step— skipping problem definition and dataset collection, which are assumed to be already done, and skipping model deployment and maintenance, which is assumed to be handled by someone else. In fact, model development is only one step in the machine learning workflow, and if you ask us, it’s not the most difficult one. The hardest things in machine learning are framing problems and collecting, annotating, and cleaning data. So cheer up, what comes next will be easy in comparison!

### PDF p.198 (printed 174) — 6 The universal workflow of machine learning › 6.1 Defining the task › Collecting a dataset
Once you understand the nature of the task and you know what your inputs and targets are going to be, it’s time for data collection—the most arduous, time-consuming, and costly part of most machine learning projects:

The photo search engine project requires you to first select the set of labels you want to classify—you settle on 10,000 common image categories. Then, you need to manually tag hundreds of thousands of your past user-uploaded images with labels from this set.

For the chat app spam detection project, because user chats are end-to-end encrypted, you cannot use their contents for training a model. You need to gain

### PDF p.200 (printed 176) — 6 The universal workflow of machine learning › 6.1 Defining the task › Beware of nonrepresentative data
Machine learning models can only make sense of inputs that are similar to what they’ve seen before. As such, it’s critical that the data used for training should be representative of the production data. This concern should be the foundation of all of your data collection work.

Suppose you’re developing an app where users can take pictures of a dish to find out its name. You train a model using pictures from an image-sharing social network that’s popular with foodies. Come deployment time, and feedback from angry users starts rolling in: your app gets the answer wrong 8 times out of 10. What’s going on? Your accuracy on the test set was well over 90%! A quick look at user-uploaded data reveals that mobile picture uploads of random dishes from random restaurants taken with random smartphones look nothing like the professional-quality, well-lit, appetizing pictures you trained the model on: your training data wasn’t representative of the production data. That’s a cardinal sin—welcome to machine learning hell. If possible, collect data directly from the environment where your model will be used. A movie review sentiment classification model should be used on new IMDB reviews, not on 

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

### PDF p.205 (printed 181) — 6 The universal workflow of machine learning › 6.2 Developing a model › Beating a baseline
As you start working on the model itself, your initial goal is to achieve statistical power, as you saw in chapter 5—that is, to develop a small model that is capable of beating a simple baseline.

At this stage, these are the three most important things you should focus on:

Feature engineering—Filter out uninformative features (feature selection) and use your knowledge of the problem to develop new features that are likely to be useful.

Selecting the correct architecture priors—What type of model architecture will you use? A densely connected network, a ConvNet, a recurrent neural network, a Transformer? Is deep learning even a good approach for the task, or should you use something else?

Selecting a good enough training configuration—What loss function should you use? What batch size and learning rate?

### PDF p.11 (printed xi) — contents ›  › 
5.4

Improving generalization 158

Dataset curation 159 ■ Feature engineering 159 ■ Using early stopping 161 ■ Regularizing your model 161

6

The universal workflow of machine learning 171 6.1

Defining the task 172

Framing the problem 172 ■ Collecting a dataset 174 Understanding your data 178 ■ Choosing a measure of success 178

6.2

Developing a model 179

Preparing the data 179 ■ Choosing an evaluation protocol 180 Beating a baseline 181 ■ Scaling up: Developing a model that overfits 182 ■ Regularizing and tuning your model 183

6.3

Deploying your model 183

Explaining your work to stakeholders and setting expectations 184 Shipping an inference model 184 ■ Monitoring your model in the wild 188 ■ Maintaining your model 188

7

A deep dive on Keras 190 7.1

A spectrum of workflows 191 7.2 Different ways to build Keras models 192

The Sequential model 192 ■ The Functional API 195 Subclassing the Model class 202 ■ Mixing and matching different components 204 ■ Remember: Use the right tool for the job 205

7.3

Using built-in training and evaluation loops 206

Writing your own metrics 207 ■ Using callbacks 208 Writing your own callbacks 210 ■ Monitoring and visualization with Tens

### PDF p.623 (printed 599) — 20 Conclusions › 20.1 Key concepts in review › 20.1.5 The universal machine learning workflow
easily approachable by a neural network (normalization and so on).
