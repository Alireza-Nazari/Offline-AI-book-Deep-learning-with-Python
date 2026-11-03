# Generation bundle for page_201  (PDF page 201, printed page 177)

Page kind: content
Chapter: 6 The universal workflow of machine learning
Section: 6.1 Defining the task
Subsection: 6.1.2 Collecting a dataset
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_201.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_201.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p201_e03] (body) user-generated data. Concept drift occurs when the properties of the production data change over time, causing model accuracy to gradually decay. A music recommendation engine trained in the year 2013 may not be very effective today. Likewise, the IMDB dataset you worked with was collected in 2011, and a model trained on it would likely not perform as well on reviews from 2020 compared to reviews from 2012, as vocabulary, expressions, and movie genres evolve over time. Concept drift is particularly acute in adversarial contexts like credit card fraud detection, where fraud patterns change practically every day. Dealing with fast concept drift requires constant data collection, annotation, and model retraining.
[p201_e04] (body) Keep in mind that machine learning can only be used to memorize patterns that are present in your training data. You can only recognize what you’ve seen before. Using machine learning trained on past data to predict the future is making the assumption that the future will behave like the past. That often isn’t the case.
[p201_e05] (heading_subsection) The problem of sampling bias
[p201_e06] (sidebar) A particularly insidious and common case of nonrepresentative data is sampling bias. Sampling bias occurs when your data collection process interacts with what you are trying to predict, resulting in biased measurements. A famous historical example occurred in the 1948 US presidential election. On election night, the Chicago Tribune printed the headline “DEWEY DEFEATS TRUMAN.” The next morning, Truman emerged as the winner. The editor of the Tribune had trusted the results of a phone survey—but phone users in 1948 were not a random, representative sample of the voting population. They were more likely to be richer, conservative, and to vote for Dewey, the Republican candidate. Nowadays, every phone survey takes sampling bias into account. That doesn’t mean that sampling bias is a thing of the past in political polling—far from it. But unlike in 1948, pollsters are aware of it and take steps to correct it.
[p201_e07] (figure_caption) “DEWEY DEFEATS TRUMAN”: a famous example of sampling bias

## Figures
- Figure None: “DEWEY DEFEATS TRUMAN”: a famous example of sampling bias  | labels inside the figure: (none - image)

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

## Previous page (PDF 200) text, for continuity
Outsourcing can potentially save you time and money but takes away control. Using something like Mechanical Turk is likely to be inexpensive and to scale well, but your annotations may end up being quite noisy.

To pick the best option, consider the constraints you’re working with:

Do the data labelers need to be subject matter experts, or could anyone annotate the data? The labels for a cat-versus-dog image classification problem can be selected by anyone, but those for a dog breed classification task require specialized knowledge. Meanwhile, annotating CT scans of bone fractures pretty much requires a medical degree.

If annotating the data requires specialized knowledge, can you train people to do it? If not, how can you get access to relevant experts?

Do you, yourself, understand the way experts come up with the annotations? If you don’t, you will have to treat your dataset as a black box, and you won’t be able to perform manual feature engineering—this isn’t critical, but it can be limiting.

If you decide to label your data in-house, ask yourself what software you will use to record annotations. You may well need to develop that software yourself. Productive data annotation software will save you a lot of time, so it’s something worth investing in early in a project.

Beware of nonrepresentative data

Machine learning models can only make sense of inputs that are similar to what they’ve seen before. As such, it’s critical that the data used for training should be representative of the production data. This concern should be the foundation of all of your data collection work.

Suppose you’re developing an app where users can take pictures of a dish to find out its name. You train a model using pictures from an image-sharing social network that’s popular with food …

## Next page (PDF 202) text, for continuity
6.1.3

Understanding your data

It’s bad practice to treat a dataset as a black box. Before you start training models, you should explore and visualize your data to gain insights about what makes it predictive— which will inform feature engineering—and screen for potential issues:

If your data includes images or natural language text, take a look at a few samples (and their labels) directly.

If your data contains numerical features, it’s a good idea to plot the histogram of feature values to get a feel for the range of values taken and the frequency of different values.

If your data includes location information, plot it on a map. Do any clear patterns emerge?

Are some samples missing values for some features? If so, you’ll need to deal with this when you prepare the data (we cover how to do this in the next section).

If your task is a classification problem, print the number of instances of each class in your data. Are the classes roughly equally represented? If not, you will need to account for this imbalance.

Check for target leaking—the presence of features in your data that provide information about the targets that may not be available in production. If you’re training a model on medical records to predict whether someone will be treated for cancer in the future, and the records include the feature “This person has been diagnosed with cancer,” then your targets are being artificially leaked into your data. Always ask yourself, is every feature in your data something that will be available in the same form in production?

6.1.4

Choosing a measure of success

To control something, you need to be able to observe it. To achieve success on a project, you must first define what you mean by success. Accuracy? Precision and recall? Customer retention rate? Your met …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.212 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.198 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.197 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.177 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.199 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.162 — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning
- PDF p.403 — 13 Timeseries forecasting › 13.4 Going even further
- PDF p.164 — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.212 (printed 188) — 6 The universal workflow of machine learning › 6.3 Deploying your model › Maintaining your model
Lastly, no model lasts forever. You’ve already learned about concept drift: over time, the characteristics of your production data will change, gradually degrading the performance and relevance of your model. The lifespan of your music recommender system will be counted in weeks. For the credit card fraud detection system, it would be days; a couple of years in the best case for the image search engine.

### PDF p.197 (printed 173) — 6 The universal workflow of machine learning › 6.1 Defining the task › 
What will your input data be? What are you trying to predict? You can only learn to predict something if you have available training data: for example, you can only learn to classify the sentiment of movie reviews if you have both movie reviews and sentiment annotations available. As such, data availability is usually the limiting factor at this stage. In many cases, you will have to resort to collecting and annotating new datasets yourself (which we cover in the next section).

What type of machine learning task are you facing? Is it binary classification? Multiclass classification? Scalar regression? Vector regression? Multiclass, multilabel classification? Image segmentation? Ranking? Something else, like clustering, generation, or reinforcement learning? In some cases, it may be that machine learning isn’t even the best way to make sense of your data, and you should use something else, such as plain old-school statistical analysis: – The photo search engine project is a multiclass, multilabel classification task. – The spam detection project is a binary classification task. If you set “offensive content” as a separate class, it’s a three-way classification task. – The music rec

### PDF p.198 (printed 174) — 6 The universal workflow of machine learning › 6.1 Defining the task › Collecting a dataset
Once you understand the nature of the task and you know what your inputs and targets are going to be, it’s time for data collection—the most arduous, time-consuming, and costly part of most machine learning projects:

The photo search engine project requires you to first select the set of labels you want to classify—you settle on 10,000 common image categories. Then, you need to manually tag hundreds of thousands of your past user-uploaded images with labels from this set.

For the chat app spam detection project, because user chats are end-to-end encrypted, you cannot use their contents for training a model. You need to gain

### PDF p.403 (printed 379) — 13 Timeseries forecasting › 13.4 Going even further › Markets and machine learning
Some readers are bound to want to take the techniques we’ve introduced here and try them on the problem of forecasting the future price of securities on the stock market (or currency exchange rates, and so on). However, markets have very different statistical characteristics than natural phenomena such as weather patterns. When it comes to markets, past performance is not a good predictor of future returns—looking in the rear-view mirror is a bad way to drive. Machine learning, on the other hand, is applicable to datasets where the past is a good predictor of the future, like weather, electricity consumption, or foot traffic at a store. Always remember that all trading is fundamentally information arbitrage: gaining an advantage by using data or insights that other market participants are missing. Trying to use well-known machine learning techniques and publicly available data to beat the markets is effectively a dead end, since you won’t have any information advantage compared to everyone else. You’re likely to waste your time and resources with nothing to show for it.

### PDF p.420 (printed 396) — 14 Text classification › 14.3 Sets vs. sequences › 
Historically, most early applications of machine learning to NLP just involved bag-of-words models that discarded sequence data. Interest in sequence models only started rising in 2015, with the rebirth of RNNs. Today, both approaches remain relevant. Let’s see how they work and when to use which. We will demonstrate each approach on a well-known text classification benchmark: the IMDb movie review sentiment-classification dataset. In chapters 4 and 5, you worked with a prevectorized version of the IMDb dataset; now let’s process the raw IMDb text data, just like you would do when approaching a new text-classification problem in the real world.

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

### PDF p.505 (printed 481) — 16 Text generation › 16.2 Training a mini-GPT › 
This isn’t so much a bug as it’s a direct consequence of our training objective. Our model is trying to predict the most likely next token in a sequence across about a billion words on many, many topics. If there’s no obvious choice for where a sequence of text should head next, an effective strategy is to guess common words or repeated patterns of words. Unsurprisingly, our model learns to do this during training almost immediately. If you were to stop training our model very early on, it would likely generate the word `"the"` incessantly, as `"the"` is the most common word in the English language. During our generative loop, we have always chosen the most likely predicted token in our model’s output. But our output is not just a single predicted token; it is a probability distribution across all 32,000 tokens in our vocabulary. Using the most likely output at each generation step is called greedy search. It’s the most straightforward approach to using model predictions, but it is hardly the only one. If we instead add some randomness to the process, we can explore the probability distribution learned by the model more broadly. This can keep us from getting stuck in loops of high-

### PDF p.199 (printed 175) — 6 The universal workflow of machine learning › 6.1 Defining the task › 
The point that data matters more than algorithms was most famously made in a 2009 paper by Google researchers titled “The Unreasonable Effectiveness of Data” (the title is a riff on the well-known 1960 book The Unreasonable Effectiveness of Mathematics in the Natural Sciences by Eugene Wigner). This was before deep learning was popular, but remarkably, the rise of deep learning has only increased the importance of data. If you’re doing supervised learning, then once you’ve collected inputs (such as images) you’re going to need annotations for them (such as tags for those images): the targets you will train your model to predict. Sometimes, annotations can be retrieved automatically—for instance, in the case of our music recommendation task or our click-through rate prediction task. But often, you have to annotate your data by hand. This is a labor-heavy process.
