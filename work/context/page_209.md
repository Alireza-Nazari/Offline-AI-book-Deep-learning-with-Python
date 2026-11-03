# Generation bundle for page_209  (PDF page 209, printed page 185)

Page kind: content
Chapter: 6 The universal workflow of machine learning
Section: 6.3 Deploying your model
Subsection: 6.3.2 Shipping an inference model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_209.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_209.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p209_e03] (heading_subsubsection) Deploying a model as a REST API
[p209_e04] (body) Perhaps the easiest way to turn a model into a product is to serve it online via a REST API. There are a number of libraries out there for making this happen. Keras supports two of the most popular approaches out of the box—TensorFlow Serving and ONNX (short for Open Neural Network Exchange). Both libraries operate by lifting all model weights and a computation graph outside of the Python program, so you can serve it from a number of different environments (for example, a C++ server). If this sounds a lot like the compilation mechanism discussed in chapter 3 you are spot-on. TensorFlow Serving is essentially a library for serving `tf.function` computation graphs with a specific set of saved weights.
[p209_e05] (body) Keras allows access to both TensorFlow Serving and ONNX via an easy-to-use `export()` method available on all Keras models. Here’s a code snippet showing how this works for TensorFlow Serving:
[p209_e06] (code)
```
model.export("path/to/location", format="tf_saved_model")
reloaded_artifact = tf.saved_model.load("path/to/location")
predictions = reloaded_artifact.serve(input_data)
```
[p209_e07] (body) A similar flow exists for ONNX:
[p209_e08] (code)
```
model.export("path/to/location", format="onnx")
ort_session = onnxruntime.InferenceSession("path/to/location")
predictions = ort_session.run(None, input_data)
```
[p209_e09] (body) You should use this deployment setup when
[p209_e10] (bullet) The application that will consume the model’s prediction will have reliable access to the internet (obviously). For instance, if your application is a mobile app, serving predictions from a remote API means that the application won’t be usable in airplane mode or in a low-connectivity environment.
[p209_e11] (bullet) The application does not have strict latency requirements: the request, inference, and answer round trip will typically take around 500 ms.
[p209_e12] (bullet) The input data sent for inference is not highly sensitive: the data will need to be available on the server in a decrypted form, since it will need to be seen by the model (but note that you should use SSL encryption for the HTTP request and answer).

## Code annotations printed next to listings (callouts, in order)
- Exports the model
- as a TensorFlow
- SavedModel artifact
- ads the artifact in a different process,
- environment, or programming language

## Equation candidates (formulas written inline in the text)
- [p209_e08] model.export("path/to/location", format="onnx")
ort_session = onnxruntime.InferenceSession("path/to/location")
predictions = ort_session.run(None, input_data)

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

## Previous page (PDF 208) text, for continuity
6.3.1

Explaining your work to stakeholders and setting expectations

Success and customer trust are about consistently meeting or exceeding people’s expectations; the actual system you deliver is only half of that picture. The other half is setting appropriate expectations before launch.

The expectations of nonspecialists toward AI systems are often unrealistic. For example, they might expect that the system “understands” its task and is capable of exercising human-like common sense in the context of the task. To address this, you should consider showing some examples of the failure modes of your model (for instance, show what incorrectly classified samples look like, especially those for which the misclassification seems surprising). They might also expect human-level performance, especially for processes that were previously handled by people. Most machine learning models, because they are (imperfectly) trained to approximate human-generated labels, do not nearly get there. You should clearly convey model performance expectations. Avoid using abstract statements like “The model has 98% accuracy” (which most people mentally round up to 100%), and prefer talking, for instance, about false-negative rates and false-positive rates. You could say, “With these settings, the fraud detection model would have a 5% false-negative rate and a 2.5% false-positive rate. Every day, an average of 200 valid transactions would be flagged as fraudulent and sent for manual review, and an average of 14 fraudulent transactions would be missed. An average of 266 fraudulent transactions would be correctly caught.” Clearly relate the model’s performance metrics to business goals. You should also make sure to discuss with stakeholders the choice of key launch parameters—for instance, the prob …

## Next page (PDF 210) text, for continuity
For instance, the image search engine project, the music recommender system, the credit card fraud detection project, and the satellite imagery project are all a good fit for serving via a REST API.

An important question when deploying a model as a REST API is whether you want to host the code on your own or whether you want to use a fully managed third-party cloud service. For instance, Cloud AI Platform, a Google product, lets you simply upload your TensorFlow model to Google Cloud Storage (GCS) and gives you an API endpoint to query it. It takes care of many practical details such as batching predictions, load balancing, and scaling.

Deploying a model on a device

Sometimes, you may need your model to live on the same device that runs the application that uses it—maybe a smartphone, an embedded ARM CPU on a robot, or a microcontroller on a tiny device. For instance, perhaps you’ve already seen a camera capable of automatically detecting people and faces in the scenes you pointed it at: that was probably a small deep learning model running directly on the camera.

You should use this setup when

Your model has strict latency constraints or needs to run in a low-connectivity environment. If you’re building an immersive augmented-reality application, querying a remote server is not a viable option.

Your model can be made sufficiently small that it can run under the memory and power constraints of the target device.

Getting the highest possible accuracy isn’t mission critical for your task: there is always a tradeoff between runtime efficiency and accuracy, so memory and power constraints often require you to ship a model that isn’t quite as good as the best model you could run on a large GPU.

The input data is strictly sensitive and thus shouldn’t be decryptable on …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.211 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.126 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.623 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.114 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.212 — 6 The universal workflow of machine learning › 6.3 Deploying your model
- PDF p.115 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.127 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.215 — 7 A deep dive on Keras › 7.1 A spectrum of workflows

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.211 (printed 187) — 6 The universal workflow of machine learning › 6.3 Deploying your model › Deploying a model in the browser
To deploy a model in JavaScript, the TensorFlow ecosystem includes TensorFlow .js (https://www.tensorflow.org/js), and ONNX supports a native JavaScript runtime. TensorFlow.js even implements almost all of the Keras API (it was originally developed under the working name WebKeras) as well as many lower-level TensorFlow APIs. You can easily import a saved Keras model into TensorFlow.js to query it as part of your browser-based JavaScript app or your desktop Electron app.

### PDF p.211 (printed 187) — 6 The universal workflow of machine learning › 6.3 Deploying your model › Deploying a model in the browser
Deep learning is often used in browser-based or desktop-based JavaScript applications. While it is usually possible to have the application query a remote model via a REST API, there can be key advantages in instead having the model run directly in the browser, on the user’s computer (utilizing GPU resources if available).

Use this setup when

You want to offload compute to the end user, which can dramatically reduce server costs.

The input data needs to stay on the end user’s computer or phone. For instance, in our spam detection project, the web version and the desktop version of the chat app (implemented as a cross-platform app written in JavaScript) should use a locally run model.

Your application has strict latency constraints: while a model running on the end user’s laptop or smartphone is likely to be slower than one running on a large GPU on your own server, you don’t have the extra 100 ms of network round trip.

You need your app to keep working without connectivity, after the model has been downloaded and cached.

Of course, you should only go with this option if your model is small enough that it won’t hog the CPU, GPU, or RAM of your user’s laptop or smartphone. In a

### PDF p.115 (printed 91) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Picking a backend framework
Keras can be used together with JAX, TensorFlow, or PyTorch. They’re the “backend frameworks” of Keras. Through these backend frameworks, Keras can run on top of different types of hardware (see figure 3.4)—GPU, TPU, or plain CPU—can be seamlessly scaled to thousands of machines, and can be deployed to a variety of platforms.

Backend frameworks are pluggable: you can switch to a different backend framework after you’ve written some Keras code. You aren’t locked into a single framework and a single ecosystem—you can move your models from JAX to TensorFlow to PyTorch depending on your current needs. For instance, when you develop a Keras model, you could debug it with PyTorch, train it on TPU with JAX for maximum efficiency, and finally run inference with the excellent tooling from the TensorFlow ecosystem.

Figure 3.4 Keras and its backends. A backend is a low-level tensor-computing platform; Keras is a high-level deep learning API.

The default backend for Keras right now is TensorFlow, so if you run `import` `keras` in a fresh environment, without having configured anything, you will be running on top of TensorFlow. There are two ways to pick a different backend:

Set the environ

### PDF p.127 (printed 103) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Summary
TensorFlow, PyTorch, and JAX are three popular low-level frameworks for numerical computation and autodifferentiation. They all have their own way of doing things and their own strengths and weaknesses.

Keras is a high-level API for building and training neural networks. It can be used with either TensorFlow, PyTorch, or JAX—just pick the backend you like best.

The central class of Keras is the `Layer`. A layer encapsulates some weights and some computation. Layers are assembled into models.

Before you start training a model, you need to pick an optimizer, a loss, and some metrics, which you specify via the `model.compile()` method.

To train a model, you can use the `fit()` method, which runs mini-batch gradient descent for you. You can also use it to monitor your loss and metrics on validation data, a set of inputs that the model doesn’t see during training.

Once your model is trained, you can use the `model.predict()` method to generate predictions on new inputs.

### PDF p.126 (printed 102) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Inference: Using a model after training
However, this will process all inputs in `new_inputs` at once, which may not be feasible if you’re looking at a lot of data (in particular, it may require more memory than your GPU has).

A better way to do inference is to use the `predict()` method. It will iterate over the data in small batches and return a NumPy array of predictions. And unlike `__call__()`, it can also process TensorFlow `Dataset` objects:

### PDF p.88 (printed 64) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › 
It’s important to keep in mind that TensorFlow is more than a single library. It’s really a platform, home to a vast ecosystem of components, some developed by Google, some developed by third parties. For instance, there’s TFX for industry-strength machine learning workflow management, TF-Serving for production deployment, the TF Optimization Toolkit for model quantization and pruning, and TFLite and MediaPipe for mobile application deployment. Together, these components cover a very wide range of use cases, from cutting-edge research to large-scale production applications.

### PDF p.216 (printed 192) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › Different ways to build Keras models
There are three APIs for building models in Keras, as shown in figure 7.1:

The Sequential model is the most approachable API—it’s basically a Python list. As such, it’s limited to simple stacks of layers.

The Functional API, which focuses on graph-like model architectures. It represents a nice mid-point between usability and flexibility, and as such, it’s the most commonly used model-building API.

Model subclassing, a low-level option where you write everything yourself from scratch. This is ideal if you want full control over every little thing. However, you won’t get access to many built-in Keras features, and you will be more at risk of making mistakes.

Figure 7.1 Progressive disclosure of complexity for model building

### PDF p.237 (printed 213) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › 
Once the model starts running, it will write logs at the target location. If you are running your Python script on a local machine, you can then launch the local TensorBoard server using the following command (note that the `tensorboard` executable should already be available if you have installed TensorFlow via `pip`; if not, you can install TensorBoard manually via `pip install tensorboard`):
