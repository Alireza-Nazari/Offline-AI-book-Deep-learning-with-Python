# Generation bundle for page_272  (PDF page 272, printed page 248)

Page kind: content
Chapter: 8 Image classification
Section: 8.2 Training a convnet from scratch on a small dataset
Subsection: 8.2.4 Data preprocessing
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_272.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_272.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p272_e03] (code)
```
image_size = (180, 180)
train_dataset = image_dataset_from_directory(
    new_base_dir / "train", image_size=image_size, batch_size=batch_size
)
validation_dataset = image_dataset_from_directory(
    new_base_dir / "validation", image_size=image_size, batch_size=batch_size
)
test_dataset = image_dataset_from_directory(
    new_base_dir / "test", image_size=image_size, batch_size=batch_size
)
```
[p272_e04] (heading_subsubsection) Understanding TensorFlow Dataset objects
[p272_e05] (body) TensorFlow makes available the `tf.data` API to create efficient input pipelines for machine learning models. Its core class is `tf.data.Dataset`.
[p272_e06] (body) The `Dataset` class can be used for data loading and preprocessing in any framework—not just TensorFlow. You can use it together with JAX or PyTorch. When you use it with a Keras model, it works the same, independently of the backend you’re currently using. A `Dataset` object is an iterator: you can use it in a `for` loop. It will typically return batches of input data and labels. You can pass a `Dataset` object directly to the `fit()` method of a Keras model. The `Dataset` class handles many key features that would otherwise be cumbersome to implement yourself, in particular parallelization of the preprocessing logic across multiple CPU cores, as well as asynchronous data prefetching (preprocessing the next batch of data while the previous one is being handled by the model, which keeps execution flowing without interruptions). The `Dataset` class also exposes a functional-style API for modifying datasets. Here’s a quick example: let’s create a `Dataset` instance from a NumPy array of random numbers. We’ll consider 1,000 samples, where each sample is a vector of size 16.
[p272_e07] (listing_caption) Listing 8.10 Instantiating a `Dataset` from a NumPy array
[p272_e08] (code)
```
import numpy as np
import tensorflow as tf
random_numbers = np.random.normal(size=(1000, 16))
dataset = tf.data.Dataset.from_tensor_slices(random_numbers)
```
[p272_e09] (body) At first, our dataset just yields single samples.

## Code annotations printed next to listings (callouts, in order)
- The from_tensor_slices() class method can
- be used to create a Dataset from a NumPy
- array or a tuple or dict of NumPy arrays.

## Listings
- Listing 8.10: Instantiating a `Dataset` from a NumPy array (code elements p272_e08)

## Chapter outline
Chapter: 8 Image classification (PDF pages 255-291)
  - 8.1 Introduction to convnets (p.256)
      - 8.1.1 The convolution operation (p.258)
      - 8.1.2 The max-pooling operation (p.263)
  - 8.2 Training a convnet from scratch on a small dataset (p.265)
      - 8.2.1 The relevance of deep learning for small-data problems (p.266)
      - 8.2.2 Downloading the data (p.266)
      - 8.2.3 Building your model (p.269)
      - 8.2.4 Data preprocessing (p.271)
      - 8.2.5 Using data augmentation (p.276)
  - 8.3 Using a pretrained model (p.280)
      - 8.3.1 Feature extraction with a pretrained model (p.280)
      - 8.3.2 Fine-tuning a pretrained model (p.288)

## Previous page (PDF 271) text, for continuity
For the compilation step, you’ll go with the `adam` optimizer, as usual. Because you ended the model with a single sigmoid unit, you’ll use binary crossentropy as the loss (as a reminder, check out table 6.1 in chapter 6 for a cheat sheet on what loss function to use in various situations).

Listing 8.8 Configuring the model for training

8.2.4

Data preprocessing

As you know by now, data should be formatted into appropriately preprocessed floating-point tensors before being fed into the model. Currently, the data sits on a drive as JPEG files, so the steps for getting it into the model are roughly as follows:

1 Read the picture files. 2 Decode the JPEG content to RGB grids of pixels. 3 Convert these into floating-point tensors. 4 Resize them to a shared size (we’ll use 180 x 180). 5 Pack them into batches (we’ll use batches of 32 images). This may seem a bit daunting, but fortunately Keras has utilities to take care of these steps automatically. In particular, Keras features the utility function `image_dataset_` `from_directory`, which lets you quickly set up a data pipeline that can automatically turn image files on disk into batches of preprocessed tensors. This is what you’ll use here. Calling `image_dataset_from_directory(directory)` will first list the subdirectories of `directory` and assume each one contains images from one of your classes. It will then index the image files in each subdirectory. Finally, it will create and return a `tf.data
` `.Dataset` object configured to read these files, shuffle them, decode them to tensors, resize them to a shared size, and pack them into batches.

Listing 8.9 Using `image_dataset_from_directory` to read images from directories

## Next page (PDF 273) text, for continuity
Listing 8.11 Iterating on a dataset

You can use the `.batch()` method to batch the data.

Listing 8.12 Batching a dataset

More broadly, you have access to a range of useful dataset methods, such as these:

`.shuffle(buffer_size)` will shuffle elements within a buffer.

`.prefetch(buffer_size)` will prefetch a buffer of elements in GPU memory to achieve better device utilization.

`.map(callable)` will apply an arbitrary transformation to each element of the dataset (the function `callable`, expected to take as input a single element yielded by the dataset).

The method `.map(function, num_parallel_calls)` in particular is one that you will use often. Here’s an example: let’s use it to reshape the elements in our toy dataset from shape `(16,)` to shape `(4,` `4)`.

Listing 8.13 Applying a transformation to `Dataset` elements using `map()`

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.366 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.283 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.124 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.545 — 17 Image generation › 17.2 Diffusion models
- PDF p.274 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.42 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network
- PDF p.284 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.126 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.579 (printed 555) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › tf.data performance tips
When doing distributed training, always provide your data as a `tf.data.Dataset` object to guarantee best performance (passing your data as NumPy arrays also works since those get converted to `Dataset` objects by `fit()`). You should also make sure to use data prefetching: before passing the dataset to `fit()`, call `dataset
` `.prefetch(buffer_size)`. If you aren’t sure what buffer size to pick, try the `dataset
` `.prefetch(tf.data.AUTOTUNE)` option, which will pick a buffer size for you.

### PDF p.426 (printed 402) — 14 Text classification › 14.4 Set models › Preprocessing text efficiently
use it throughout this chapter—`keras.utils.text_dataset_from_directory()` will

load a `tf.data.Dataset`, and `map()` will transform our input data, for example, by applying a `TextVectorization` layer. `tf.data` works by running text preprocessing in parallel multiple CPU cores, which is generally sufficient to avoid bottlenecking accelerators during a training run. It is important to note that the code in this chapter is still multi-backend (in fact, we generated the outputs for this chapter using Jax). You can use `tf.data` with PyTorch, JAX, or TensorFlow itself—Keras will automatically convert input Tensors to the correct format for a given backend.

### PDF p.126 (printed 102) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Inference: Using a model after training
However, this will process all inputs in `new_inputs` at once, which may not be feasible if you’re looking at a lot of data (in particular, it may require more memory than your GPU has).

A better way to do inference is to use the `predict()` method. It will iterate over the data in small batches and return a NumPy array of predictions. And unlike `__call__()`, it can also process TensorFlow `Dataset` objects:

### PDF p.124 (printed 100) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Understanding the fit method
After `compile()` comes `fit()`. The `fit` method implements the training loop itself. Its key arguments are

The data (inputs and targets) to train on. It will typically be passed either in the form of NumPy arrays or a TensorFlow `Dataset` object. You’ll learn more about the `Dataset` API in the next chapters.

The number of epochs to train for: how many times the training loop should iterate over the data passed.

The batch size to use within each epoch of mini-batch gradient descent: the number of training examples considered to compute the gradients for one weight update step.

Listing 3.34 Calling `fit` with NumPy data

### PDF p.366 (printed 342) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Lastly, let’s use `tf.data` to load our image data. We will load our images from disk, apply our preprocessing, and batch them. We should also split a validation set to monitor training.

Listing 12.9 Creating a dataset to train on

### PDF p.366 (printed 342) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Listing 12.9 Creating a dataset to train on
import tensorflow as tf
def load_image(path):
    x = tf.io.read_file(path)
    x = tf.image.decode_jpeg(x, channels=3)
    return preprocessor(x)
images = tf.data.Dataset.from_tensor_slices([x["path"] for x in metadata])
images = images.map(load_image, num_parallel_calls=8)
labels = {"box": box_array, "class": class_array}
labels = tf.data.Dataset.from_tensor_slices(labels)
dataset = tf.data.Dataset.zip(images, labels).batch(16).prefetch(2)
val_dataset, train_dataset = dataset.take(500), dataset.skip(500)

### PDF p.98 (printed 74) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.4 Introduction to PyTorch › What makes the TensorFlow approach unique
You’re now familiar with all the basic APIs that underlie TensorFlow-based workflows, and you’re about to dive into more frameworks—in particular, PyTorch and JAX. What makes working with TensorFlow different from working with any other framework? When should you use TensorFlow, and when could you use something else?

If you ask us, here are the main benefits of TensorFlow:

Thanks to graph mode and XLA compilation, it’s fast. It’s usually significantly faster than PyTorch and NumPy, though JAX is often even faster.

It is extremely feature complete. Unique among all frameworks, it has support for string tensors as well as “ragged tensors” (tensors where different entries may have different dimensions—very useful for handling sequences without requiring to pad them to a shared length). It also has outstanding support for data preprocessing, via the highly performant `tf.data` API. `tf.data` is so good that even JAX recommends it for data preprocessing. Whatever you need to do, TensorFlow has a solution for it.

Its ecosystem for production deployment is the most mature among all frameworks, especially when it comes to deploying on mobile or in the browser.

However, TensorFlow also

### PDF p.643 (printed 619) — index ›  › T
TextVectorization layer 399, 402, 403, 405, 415, tanh activation 116

424, 430, 431 target 105, 214 tf.data 342 target boxes, prompting model with 327 tf.data API 248 target leaking 178 tf.data.Dataset class 248 target points, prompting model with 323–327 tf.data.Dataset instance 556 tar shell utility 311 tf.data pipeline 432, 473 task API 300 tf.GradientTape() function 302 TensorBoard 212 tf module 302 TensorFlow 60 theory of mind 570 class activation heatmap 302 timeseries_dataset_from_array() function 356, 357 customizing fit() with 223 timeseries forecasting 351 Dataset objects 248 recurrent neural networks 364–378 gradient ascent in 294 tasks 352 relationship between frameworks 63 temperature forecasting example 352–364 training step functions 216 tokenization 387 train_step() metrics handling with 226 tokens 384 tensorflow module 302 Top-K sampling 482
