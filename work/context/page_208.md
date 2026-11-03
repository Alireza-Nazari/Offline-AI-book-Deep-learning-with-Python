# Generation bundle for page_208  (PDF page 208, printed page 184)

Page kind: content
Chapter: 6 The universal workflow of machine learning
Section: 6.3 Deploying your model
Subsection: 6.3.2 Shipping an inference model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_208.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_208.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p208_e03] (heading_subsection) 6.3.1
[p208_e04] (heading_subsection) Explaining your work to stakeholders and setting expectations
[p208_e05] (body) Success and customer trust are about consistently meeting or exceeding people’s expectations; the actual system you deliver is only half of that picture. The other half is setting appropriate expectations before launch.
[p208_e06] (body) The expectations of nonspecialists toward AI systems are often unrealistic. For example, they might expect that the system “understands” its task and is capable of exercising human-like common sense in the context of the task. To address this, you should consider showing some examples of the failure modes of your model (for instance, show what incorrectly classified samples look like, especially those for which the misclassification seems surprising). They might also expect human-level performance, especially for processes that were previously handled by people. Most machine learning models, because they are (imperfectly) trained to approximate human-generated labels, do not nearly get there. You should clearly convey model performance expectations. Avoid using abstract statements like “The model has 98% accuracy” (which most people mentally round up to 100%), and prefer talking, for instance, about false-negative rates and false-positive rates. You could say, “With these settings, the fraud detection model would have a 5% false-negative rate and a 2.5% false-positive rate. Every day, an average of 200 valid transactions would be flagged as fraudulent and sent for manual review, and an average of 14 fraudulent transactions would be missed. An average of 266 fraudulent transactions would be correctly caught.” Clearly relate the model’s performance metrics to business goals. You should also make sure to discuss with stakeholders the choice of key launch parameters—for instance, the probability threshold at which a transaction should be flagged (different thresholds will produce different false-negative and false-positive rates). Such decisions involve tradeoffs that can only be handled with a deep understanding of the business context.
[p208_e07] (heading_subsection) 6.3.2
[p208_e08] (heading_subsection) Shipping an inference model
[p208_e09] (body) A machine learning project doesn’t end when you arrive at a Colab notebook that can save a trained model. You rarely put into production the exact same Python model object that you manipulated during training.
[p208_e10] (body) First, you may want to export your model to something other than Python:
[p208_e11] (bullet) Your production environment may not support Python at all—for instance, if it’s a mobile app or an embedded system.
[p208_e12] (bullet) If the rest of the app isn’t in Python (it could be in JavaScript, C++, etc.), the use of Python to serve a model may induce significant overhead.
[p208_e13] (body) Second, since your production model will only be used to output predictions (a phase called inference), rather than for training, you have room to perform various optimizations that can make the model faster and reduce its memory footprint.
[p208_e14] (body) Let’s take a quick look at the different model deployment options you have available.

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

## Previous page (PDF 207) text, for continuity
Always monitor the training loss and validation loss, as well as the training and validation values for any metrics you care about. When you see that the model’s performance on the validation data begins to degrade, you’ve achieved overfitting.

6.2.5

Regularizing and tuning your model

Once you’ve achieved statistical power and you’re able to overfit, you know you’re on the right path. At this point, your goal becomes to maximize generalization performance.

This phase will take the most time: you’ll repeatedly modify your model, train it, evaluate on your validation data (not the test data, at this point), modify it again, and repeat, until the model is as good as it can get. Here are some things you should try:

Try different architectures; add or remove layers.

Add dropout.

If your model is small, add L1 or L2 regularization.

Try different hyperparameters (such as the number of units per layer or the learning rate of the optimizer) to find the optimal configuration.

Optionally, iterate on data curation or feature engineering: collect and annotate more data, develop better features, or remove features that don’t seem to be informative.

It’s possible to automate a large chunk of this work by using automated hyperparameter-tuning software, such as KerasTuner. We’ll cover this in chapter 18.

Be mindful of the following: every time you use feedback from your validation process to tune your model, you leak information about the validation process into the model. Repeated just a few times, this is innocuous; however, done systematically over many iterations, it will eventually cause your model to overfit to the validation process (even though no model is directly trained on any of the validation data). This makes the evaluation process less reliable. Once you’ve dev …

## Next page (PDF 209) text, for continuity
Deploying a model as a REST API

Perhaps the easiest way to turn a model into a product is to serve it online via a REST API. There are a number of libraries out there for making this happen. Keras supports two of the most popular approaches out of the box—TensorFlow Serving and ONNX (short for Open Neural Network Exchange). Both libraries operate by lifting all model weights and a computation graph outside of the Python program, so you can serve it from a number of different environments (for example, a C++ server). If this sounds a lot like the compilation mechanism discussed in chapter 3 you are spot-on. TensorFlow Serving is essentially a library for serving `tf.function` computation graphs with a specific set of saved weights.

Keras allows access to both TensorFlow Serving and ONNX via an easy-to-use `export()` method available on all Keras models. Here’s a code snippet showing how this works for TensorFlow Serving:

A similar flow exists for ONNX:

You should use this deployment setup when

The application that will consume the model’s prediction will have reliable access to the internet (obviously). For instance, if your application is a mobile app, serving predictions from a remote API means that the application won’t be usable in airplane mode or in a low-connectivity environment.

The application does not have strict latency requirements: the request, inference, and answer round trip will typically take around 500 ms.

The input data sent for inference is not highly sensitive: the data will need to be available on the server in a decrypted form, since it will need to be seen by the model (but note that you should use SSL encryption for the HTTP request and answer).

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.213 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.212 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.200 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.196 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.202 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.198 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.197 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.211 — 6 The universal workflow of machine learning › 6.3 Deploying your model

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.213 (printed 189) — 6 The universal workflow of machine learning › 6.3 Deploying your model › Summary
When you take on a new machine learning project, first, define the problem at hand: – Understand the broader context of what you’re setting out to do—what’s the end goal and what are the constraints? – Collect and annotate a dataset; make sure you understand your data in depth. – Choose how you’ll measure success on your problem. What metrics will you monitor on your validation data?

Once you understand the problem and you have an appropriate dataset, develop a model: – Prepare your data. – Pick your evaluation protocol. Hold-out validation? K-fold validation? Which portion of the data should you use for validation? – Achieve statistical power: beat a simple baseline. – Scale up: develop a model that can overfit. – Regularize your model and tune its hyperparameters, based on performance on the validation data. A lot of machine learning research tends to focus only on this step—but keep the big picture in mind.

When your model is ready and yields good performance on the test data, it’s time for deployment: – First, make sure to set appropriate expectations with stakeholders. – Optimize a final model for inference, and ship the model to the deployment environment of choice—web serv

### PDF p.212 (printed 188) — 6 The universal workflow of machine learning › 6.3 Deploying your model › Monitoring your model in the wild
You’ve exported an inference model, you’ve integrated it into your application, and you’ve done a dry run on production data—the model behaved exactly as you expected. You’ve written unit tests as well as logging and status-monitoring code— perfect. Now it’s time to press the big red button and deploy to production.

Even this is not the end. Once you’ve deployed a model, you need to keep monitoring its behavior, its performance on new data, its interaction with the rest of the application, and its eventual impact on business metrics:

Is user engagement in your online radio up or down after deploying the new music recommender system? Has average ad click-through rate increased after switching to the new click-through rate prediction model? Consider using randomized A/B testing to isolate the impact of the model itself from other changes: a subset of cases should go through the new model, while another control subset should stick to the old process. Once sufficiently many cases have been processed, the difference in outcomes between the two is likely attributable to the model.

If possible, do a regular manual audit of the model’s predictions on production data. It’s generally poss

### PDF p.200 (printed 176) — 6 The universal workflow of machine learning › 6.1 Defining the task › Beware of nonrepresentative data
Machine learning models can only make sense of inputs that are similar to what they’ve seen before. As such, it’s critical that the data used for training should be representative of the production data. This concern should be the foundation of all of your data collection work.

Suppose you’re developing an app where users can take pictures of a dish to find out its name. You train a model using pictures from an image-sharing social network that’s popular with foodies. Come deployment time, and feedback from angry users starts rolling in: your app gets the answer wrong 8 times out of 10. What’s going on? Your accuracy on the test set was well over 90%! A quick look at user-uploaded data reveals that mobile picture uploads of random dishes from random restaurants taken with random smartphones look nothing like the professional-quality, well-lit, appetizing pictures you trained the model on: your training data wasn’t representative of the production data. That’s a cardinal sin—welcome to machine learning hell. If possible, collect data directly from the environment where your model will be used. A movie review sentiment classification model should be used on new IMDB reviews, not on 

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

### PDF p.197 (printed 173) — 6 The universal workflow of machine learning › 6.1 Defining the task › 
What do existing solutions look like? Perhaps your customer already has a hand-crafted algorithm that handles spam filtering or credit card fraud detection— with lots of nested `if` statements. Perhaps a human is currently in charge of manually handling the process considered—monitoring the conveyor belt at the cookie plant and manually removing the bad cookies, or crafting playlists of song recommendations to be sent out to users who liked a specific artist. You should make sure to understand what systems are already in place, and how they work.

Are there particular constraints you will need to deal with? For example, you could find out that the app for which you’re building a spam detection system is strictly end-to-end encrypted, so that the spam detection model will have to live on the end-user’s phone, and must be trained on an external dataset. Perhaps the cookie-filtering model has such latency constraints that it will need to run on an embedded device at the factory rather than on a remote server. You should understand the full context in which your work will fit.

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

### PDF p.459 (printed 435) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 
We picked accuracy as a crude way to monitor validation set performance during training. We get to 65% accuracy: on average, the model correctly predicts the next word in the Spanish sentence 65% of the time. However, in practice, next-token accuracy isn’t a great metric for machine translation models, in particular because it makes the assumption that the correct target tokens from `0` to `N` are already known when predicting token `N + 1`. In reality, during inference, you’re generating the target sentence from scratch, and you can’t rely on previously generated tokens being 100% correct. When working on a real-world machine translation system, metrics must be more carefully designed. There are standard metrics, such as a BLEU score, that measure the similarity of the machine-translated text to a set of high-quality reference translations and can tolerate slightly misaligned sequences.

At last, let’s use our model for inference. We’ll pick a few sentences in the test set and check how our model translates them. We’ll start from the seed token `"[start]"` and feed it into the decoder model, together with the encoded English source sentence. We’ll retrieve a next-token prediction,

### PDF p.624 (printed 600) — 20 Conclusions › 20.1 Key concepts in review › 
Develop a first model that beats a trivial common-sense baseline, thus demonstrating that

machine learning can work on your problem. This may not always be the case!

Gradually refine your model architecture by tuning hyperparameters and adding regularization. Make changes based on performance on the validation data only, not the test data or the training data. Remember that you should get your model to overfit (thus identifying a model capacity level that’s greater than you need) and only then begin to add regularization or downsize your model. Beware of validation-set overfitting when tuning hyperparameters—the fact that your hyperparameters may end up being overspecialized to the validation set. Avoiding this is the purpose of having a separate test set!

Deploy your final model in production—as a web API, as part of a JavaScript or C++ application, on an embedded device, etc. Keep monitoring its performance on real-world data and use your findings to refine the next iteration of the model!
