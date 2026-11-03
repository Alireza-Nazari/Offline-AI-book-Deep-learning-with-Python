# Generation bundle for page_273  (PDF page 273, printed page 249)

Page kind: content
Chapter: 8 Image classification
Section: 8.2 Training a convnet from scratch on a small dataset
Subsection: 8.2.4 Data preprocessing
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_273.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_273.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p273_e03] (listing_caption) Listing 8.11 Iterating on a dataset
[p273_e04] (code)
```
>>> for i, element in enumerate(dataset):
>>>     print(element.shape)
>>>     if i >= 2:
>>>         break
(16,)
(16,)
(16,)
```
[p273_e05] (body) You can use the `.batch()` method to batch the data.
[p273_e06] (listing_caption) Listing 8.12 Batching a dataset
[p273_e07] (code)
```
>>> batched_dataset = dataset.batch(32)
>>> for i, element in enumerate(batched_dataset):
>>>     print(element.shape)
>>>     if i >= 2:
>>>         break
(32, 16)
(32, 16)
(32, 16)
```
[p273_e08] (body) More broadly, you have access to a range of useful dataset methods, such as these:
[p273_e09] (bullet) `.shuffle(buffer_size)` will shuffle elements within a buffer.
[p273_e10] (bullet) `.prefetch(buffer_size)` will prefetch a buffer of elements in GPU memory to achieve better device utilization.
[p273_e11] (bullet) `.map(callable)` will apply an arbitrary transformation to each element of the dataset (the function `callable`, expected to take as input a single element yielded by the dataset).
[p273_e12] (body) The method `.map(function, num_parallel_calls)` in particular is one that you will use often. Here’s an example: let’s use it to reshape the elements in our toy dataset from shape `(16,)` to shape `(4,` `4)`.
[p273_e13] (listing_caption) Listing 8.13 Applying a transformation to `Dataset` elements using `map()`
[p273_e14] (code)
```
>>> reshaped_dataset = dataset.map(
...     lambda x: tf.reshape(x, (4, 4)),
...     num_parallel_calls=8)
>>> for i, element in enumerate(reshaped_dataset):
...     print(element.shape)
...     if i >= 2:
...         break
(4, 4)
(4, 4)
(4, 4)
```

## Listings
- Listing 8.11: Iterating on a dataset (code elements p273_e04)
- Listing 8.12: Batching a dataset (code elements p273_e07)
- Listing 8.13: Applying a transformation to `Dataset` elements using `map()` (code elements p273_e14)

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

## Previous page (PDF 272) text, for continuity
Understanding TensorFlow Dataset objects

TensorFlow makes available the `tf.data` API to create efficient input pipelines for machine learning models. Its core class is `tf.data.Dataset`.

The `Dataset` class can be used for data loading and preprocessing in any framework—not just TensorFlow. You can use it together with JAX or PyTorch. When you use it with a Keras model, it works the same, independently of the backend you’re currently using. A `Dataset` object is an iterator: you can use it in a `for` loop. It will typically return batches of input data and labels. You can pass a `Dataset` object directly to the `fit()` method of a Keras model. The `Dataset` class handles many key features that would otherwise be cumbersome to implement yourself, in particular parallelization of the preprocessing logic across multiple CPU cores, as well as asynchronous data prefetching (preprocessing the next batch of data while the previous one is being handled by the model, which keeps execution flowing without interruptions). The `Dataset` class also exposes a functional-style API for modifying datasets. Here’s a quick example: let’s create a `Dataset` instance from a NumPy array of random numbers. We’ll consider 1,000 samples, where each sample is a vector of size 16.

Listing 8.10 Instantiating a `Dataset` from a NumPy array

At first, our dataset just yields single samples.

## Next page (PDF 274) text, for continuity
You’re about to see more `map()` action over the next chapters.

Fitting the model

Let’s look at the output of one of these `Dataset` objects: it yields batches of 180 × 180 RGB images (shape `(32, 180, 180, 3)`) and integer labels (shape `(32,)`). There are 32 samples in each batch (the batch size).

Listing 8.14 Displaying the shapes yielded by the `Dataset`

Let’s fit the model on our dataset. We use the `validation_data` argument in `fit()` to monitor validation metrics on a separate `Dataset` object.

Note that we also use a `ModelCheckpoint` callback to save the model after each epoch. We configure it with the path where to save the file, as well as the arguments `save_best_` `only=True` and `monitor="val_loss"`: they tell the callback to only save a new file (overwriting any previous one) when the current value of the `val_loss` metric is lower than at any previous time during training. This guarantees that your saved file will always contain the state of the model corresponding to its best-performing training epoch, in terms of its performance on the validation data. As a result, we won’t have to retrain a new model for a lower number of epochs if we start overfitting: we can just reload our saved file.

Listing 8.15 Fitting the model using a `Dataset`

Let’s plot the loss and accuracy of the model over the training and validation data during training (see figure 8.9).

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.545 — 17 Image generation › 17.2 Diffusion models
- PDF p.366 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.513 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.283 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.441 — 14 Text classification › 14.5 Sequence models
- PDF p.383 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.284 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.271 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.579 (printed 555) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › tf.data performance tips
When doing distributed training, always provide your data as a `tf.data.Dataset` object to guarantee best performance (passing your data as NumPy arrays also works since those get converted to `Dataset` objects by `fit()`). You should also make sure to use data prefetching: before passing the dataset to `fit()`, call `dataset
` `.prefetch(buffer_size)`. If you aren’t sure what buffer size to pick, try the `dataset
` `.prefetch(tf.data.AUTOTUNE)` option, which will pick a buffer size for you.

### PDF p.366 (printed 342) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
With that, our data is ready for training.

NOTE This training example shows clearly why a streaming library like `tf.data` is helpful. Loading all the images in this large dataset in one go would overwhelm our system memory (remember an image tensor is much larger than a compressed JPEG file). With `tf.data`, we can load our image data in batch by batch and release the memory when we are done, only mapping in the particular parts of the dataset we need at a given moment. The `prefetch(2)` call will cause `tf.data` to keep two batches buffered and ready before they are used so we don’t interrupt training each batch to load and resize more images.

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

### PDF p.497 (printed 473) — 16 Text generation › 16.2 Training a mini-GPT › 
As we first did in chapter 8, we will end our `tf.data` pipeline with a call to `prefetch()`. This will make sure we always have some batches loaded onto our GPU and ready for the model.

We have 29,373 batches. You could count this yourself if you would like—the line `ds.reduce(0, lambda c, _: c + 1)` will iterate over the entire dataset and increment a counter. But simply tokenizing a dataset of this size will take a few minutes on a decently fast CPU. At 128 samples per batch and 256 tokens per sample, this is just under a billion tokens of data. Let’s split off 500 batches as a quick validation set, and we are ready to start pretraining:

### PDF p.126 (printed 102) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Inference: Using a model after training
However, this will process all inputs in `new_inputs` at once, which may not be feasible if you’re looking at a lot of data (in particular, it may require more memory than your GPU has).

A better way to do inference is to use the `predict()` method. It will iterate over the data in small batches and return a NumPy array of predictions. And unlike `__call__()`, it can also process TensorFlow `Dataset` objects:

### PDF p.448 (printed 424) — 15 Language models and the Transformer › 15.1 The language model › 
dataset = dataset.map(
    lambda features, labels: (tokenizer(features), tokenizer(labels)),
    num_parallel_calls=8,
)
training_data = dataset.shuffle(10_000).batch(64).cache()

### PDF p.441 (printed 417) — 14 Text classification › 14.5 Sequence models › 
Because our model is so simple, we can use a large batch size to speed up training without worrying about memory constraints.

We will also call `cache()` on this batched dataset so that we store the entire preprocessed dataset in memory rather than recomputing it each epoch. This is because for this very simple model, we are bottlenecked on preprocessing rather than training. That is, it is slower to tokenize our text and compute sliding windows on the CPU than to update our model parameters on the GPU. In such cases, saving your preprocessed outputs in memory or on disk is usually a good idea. You will notice how our later epochs are more than three times faster than the first. This is thanks to the cache of preprocessed training data.

Listing 14.29 Training the CBOW model

### PDF p.124 (printed 100) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › Understanding the fit method
After `compile()` comes `fit()`. The `fit` method implements the training loop itself. Its key arguments are

The data (inputs and targets) to train on. It will typically be passed either in the form of NumPy arrays or a TensorFlow `Dataset` object. You’ll learn more about the `Dataset` API in the next chapters.

The number of epochs to train for: how many times the training loop should iterate over the data passed.

The batch size to use within each epoch of mini-batch gradient descent: the number of training examples considered to compute the gradients for one weight update step.

Listing 3.34 Calling `fit` with NumPy data
