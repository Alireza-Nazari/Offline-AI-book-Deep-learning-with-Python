# Generation bundle for page_210  (PDF page 210, printed page 186)

Page kind: content
Chapter: 6 The universal workflow of machine learning
Section: 6.3 Deploying your model
Subsection: 6.3.2 Shipping an inference model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_210.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_210.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p210_e03] (body) For instance, the image search engine project, the music recommender system, the credit card fraud detection project, and the satellite imagery project are all a good fit for serving via a REST API.
[p210_e04] (body) An important question when deploying a model as a REST API is whether you want to host the code on your own or whether you want to use a fully managed third-party cloud service. For instance, Cloud AI Platform, a Google product, lets you simply upload your TensorFlow model to Google Cloud Storage (GCS) and gives you an API endpoint to query it. It takes care of many practical details such as batching predictions, load balancing, and scaling.
[p210_e05] (heading_subsubsection) Deploying a model on a device
[p210_e06] (body) Sometimes, you may need your model to live on the same device that runs the application that uses it—maybe a smartphone, an embedded ARM CPU on a robot, or a microcontroller on a tiny device. For instance, perhaps you’ve already seen a camera capable of automatically detecting people and faces in the scenes you pointed it at: that was probably a small deep learning model running directly on the camera.
[p210_e07] (body) You should use this setup when
[p210_e08] (bullet) Your model has strict latency constraints or needs to run in a low-connectivity environment. If you’re building an immersive augmented-reality application, querying a remote server is not a viable option.
[p210_e09] (bullet) Your model can be made sufficiently small that it can run under the memory and power constraints of the target device.
[p210_e10] (bullet) Getting the highest possible accuracy isn’t mission critical for your task: there is always a tradeoff between runtime efficiency and accuracy, so memory and power constraints often require you to ship a model that isn’t quite as good as the best model you could run on a large GPU.
[p210_e11] (bullet) The input data is strictly sensitive and thus shouldn’t be decryptable on a remote server.
[p210_e12] (body) For instance, our spam detection model will need to run on the end user’s smartphone as part of the chat app, because messages are end-to-end encrypted and thus cannot be read by a remotely hosted model at all. Likewise, the bad-cookie-detection model has strict latency constraints and will need to run at the factory. Thankfully, in this case, we don’t have any power or space constraints, so we can actually run the model on a GPU.
[p210_e13] (body) To deploy a Keras model on a smartphone or embedded device, you can again use the `export()` method to create a TensorFlow or ONNX save of your model including the computation graph. TensorFlow Lite (https://www.tensorflow.org/lite) is a framework for efficient on-device deep learning inference that runs on Android and iOS smartphones, as well as ARM CPUs, Raspberry Pi, or certain microcontrollers. It uses the same TensorFlow save model format as TensorFlow Serving. The ONNX runtime can also run on mobile devices.

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

## Previous page (PDF 209) text, for continuity
Deploying a model as a REST API

Perhaps the easiest way to turn a model into a product is to serve it online via a REST API. There are a number of libraries out there for making this happen. Keras supports two of the most popular approaches out of the box—TensorFlow Serving and ONNX (short for Open Neural Network Exchange). Both libraries operate by lifting all model weights and a computation graph outside of the Python program, so you can serve it from a number of different environments (for example, a C++ server). If this sounds a lot like the compilation mechanism discussed in chapter 3 you are spot-on. TensorFlow Serving is essentially a library for serving `tf.function` computation graphs with a specific set of saved weights.

Keras allows access to both TensorFlow Serving and ONNX via an easy-to-use `export()` method available on all Keras models. Here’s a code snippet showing how this works for TensorFlow Serving:

A similar flow exists for ONNX:

You should use this deployment setup when

The application that will consume the model’s prediction will have reliable access to the internet (obviously). For instance, if your application is a mobile app, serving predictions from a remote API means that the application won’t be usable in airplane mode or in a low-connectivity environment.

The application does not have strict latency requirements: the request, inference, and answer round trip will typically take around 500 ms.

The input data sent for inference is not highly sensitive: the data will need to be available on the server in a decrypted form, since it will need to be seen by the model (but note that you should use SSL encryption for the HTTP request and answer).

## Next page (PDF 211) text, for continuity
Deploying a model in the browser

Deep learning is often used in browser-based or desktop-based JavaScript applications. While it is usually possible to have the application query a remote model via a REST API, there can be key advantages in instead having the model run directly in the browser, on the user’s computer (utilizing GPU resources if available).

Use this setup when

You want to offload compute to the end user, which can dramatically reduce server costs.

The input data needs to stay on the end user’s computer or phone. For instance, in our spam detection project, the web version and the desktop version of the chat app (implemented as a cross-platform app written in JavaScript) should use a locally run model.

Your application has strict latency constraints: while a model running on the end user’s laptop or smartphone is likely to be slower than one running on a large GPU on your own server, you don’t have the extra 100 ms of network round trip.

You need your app to keep working without connectivity, after the model has been downloaded and cached.

Of course, you should only go with this option if your model is small enough that it won’t hog the CPU, GPU, or RAM of your user’s laptop or smartphone. In addition, since the entire model will be downloaded to the user’s device, you should make sure that nothing about the model needs to stay confidential. Be mindful of the fact that, given a trained deep learning model, it is usually possible to recover some information about the training data: better not to make your trained model public if it was trained on sensitive data.

To deploy a model in JavaScript, the TensorFlow ecosystem includes TensorFlow .js (https://www.tensorflow.org/js), and ONNX supports a native JavaScript runtime. TensorFlow.js even implement …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.208 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.212 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.197 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.198 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.115 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.196 — 6 The universal workflow of machine learning › 6.1 Defining the task
- PDF p.213 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.576 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.197 (printed 173) — 6 The universal workflow of machine learning › 6.1 Defining the task › 
What do existing solutions look like? Perhaps your customer already has a hand-crafted algorithm that handles spam filtering or credit card fraud detection— with lots of nested `if` statements. Perhaps a human is currently in charge of manually handling the process considered—monitoring the conveyor belt at the cookie plant and manually removing the bad cookies, or crafting playlists of song recommendations to be sent out to users who liked a specific artist. You should make sure to understand what systems are already in place, and how they work.

Are there particular constraints you will need to deal with? For example, you could find out that the app for which you’re building a spam detection system is strictly end-to-end encrypted, so that the spam detection model will have to live on the end-user’s phone, and must be trained on an external dataset. Perhaps the cookie-filtering model has such latency constraints that it will need to run on an embedded device at the factory rather than on a remote server. You should understand the full context in which your work will fit.

### PDF p.208 (printed 184) — 6 The universal workflow of machine learning › 6.3 Deploying your model › Shipping an inference model
A machine learning project doesn’t end when you arrive at a Colab notebook that can save a trained model. You rarely put into production the exact same Python model object that you manipulated during training.

First, you may want to export your model to something other than Python:

Your production environment may not support Python at all—for instance, if it’s a mobile app or an embedded system.

If the rest of the app isn’t in Python (it could be in JavaScript, C++, etc.), the use of Python to serve a model may induce significant overhead.

Second, since your production model will only be used to output predictions (a phase called inference), rather than for training, you have room to perform various optimizations that can make the model faster and reduce its memory footprint.

Let’s take a quick look at the different model deployment options you have available.

### PDF p.212 (printed 188) — 6 The universal workflow of machine learning › 6.3 Deploying your model › Monitoring your model in the wild
You’ve exported an inference model, you’ve integrated it into your application, and you’ve done a dry run on production data—the model behaved exactly as you expected. You’ve written unit tests as well as logging and status-monitoring code— perfect. Now it’s time to press the big red button and deploy to production.

Even this is not the end. Once you’ve deployed a model, you need to keep monitoring its behavior, its performance on new data, its interaction with the rest of the application, and its eventual impact on business metrics:

Is user engagement in your online radio up or down after deploying the new music recommender system? Has average ad click-through rate increased after switching to the new click-through rate prediction model? Consider using randomized A/B testing to isolate the impact of the model itself from other changes: a subset of cases should go through the new model, while another control subset should stick to the old process. Once sufficiently many cases have been processed, the difference in outcomes between the two is likely attributable to the model.

If possible, do a regular manual audit of the model’s predictions on production data. It’s generally poss

### PDF p.580 (printed 556) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 
TPU runtime in addition to the GPU runtime. For more serious training runs, Google Cloud also makes available TPU v3 through v5, which are even faster.

When running Keras code with the JAX backend on a TPU-enabled notebook, you don’t need anything more than calling `keras.distribution.set_distribution
` `(distribution)` with a `DataParallel` or `ModelParallel` distribution instance to start using your TPU cores. Make sure to call it before creating your model! NOTE Because TPUs can process batches of data extremely quickly, the speed at which you can read data from Google Cloud Storage (GCS) can easily become a bottleneck. If your dataset is small enough, you should keep it in the memory of the virtual machine. You can do so by calling `dataset.cache()` on your `tf.data.Dataset` instance. That way, the data will only be read from GCS once.

### PDF p.213 (printed 189) — 6 The universal workflow of machine learning › 6.3 Deploying your model › Summary
When you take on a new machine learning project, first, define the problem at hand: – Understand the broader context of what you’re setting out to do—what’s the end goal and what are the constraints? – Collect and annotate a dataset; make sure you understand your data in depth. – Choose how you’ll measure success on your problem. What metrics will you monitor on your validation data?

Once you understand the problem and you have an appropriate dataset, develop a model: – Prepare your data. – Pick your evaluation protocol. Hold-out validation? K-fold validation? Which portion of the data should you use for validation? – Achieve statistical power: beat a simple baseline. – Scale up: develop a model that can overfit. – Regularize your model and tune its hyperparameters, based on performance on the validation data. A lot of machine learning research tends to focus only on this step—but keep the big picture in mind.

When your model is ready and yields good performance on the test data, it’s time for deployment: – First, make sure to set appropriate expectations with stakeholders. – Optimize a final model for inference, and ship the model to the deployment environment of choice—web serv

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

### PDF p.199 (printed 175) — 6 The universal workflow of machine learning › 6.1 Defining the task › 
access to a separate dataset of tens of thousands of unfiltered social media posts, and manually tag them as spam, offensive, or acceptable.

For the music recommendation engine, you can just use the “likes” of your users. No new data needs to be collected. Likewise, for the click-through rate prediction project, you have an extensive record of click-through rate for your past ads, going back years.

For the cookie-flagging model, you will need to install cameras above the conveyor belts to collect tens of thousands of images, and then someone will need to manually label these images. The people who know how to do this currently work at the cookie factory—but it doesn’t seem too difficult, you should be able to train people to do it.

The satellite imagery project will require a team of archaeologists to collect a database of existing sites of interest, and for each site, you will need to find existing satellite images taken in different weather conditions. To get a good model, you’re going to need thousands of different sites.

You’ve learned in chapter 5 that a model’s ability to generalize comes almost entirely from the properties of the data it is trained on—the number of data 

### PDF p.115 (printed 91) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Picking a backend framework
Keras can be used together with JAX, TensorFlow, or PyTorch. They’re the “backend frameworks” of Keras. Through these backend frameworks, Keras can run on top of different types of hardware (see figure 3.4)—GPU, TPU, or plain CPU—can be seamlessly scaled to thousands of machines, and can be deployed to a variety of platforms.

Backend frameworks are pluggable: you can switch to a different backend framework after you’ve written some Keras code. You aren’t locked into a single framework and a single ecosystem—you can move your models from JAX to TensorFlow to PyTorch depending on your current needs. For instance, when you develop a Keras model, you could debug it with PyTorch, train it on TPU with JAX for maximum efficiency, and finally run inference with the excellent tooling from the TensorFlow ecosystem.

Figure 3.4 Keras and its backends. A backend is a low-level tensor-computing platform; Keras is a high-level deep learning API.

The default backend for Keras right now is TensorFlow, so if you run `import` `keras` in a fresh environment, without having configured anything, you will be running on top of TensorFlow. There are two ways to pick a different backend:

Set the environ
