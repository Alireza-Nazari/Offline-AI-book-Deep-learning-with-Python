# Generation bundle for page_124  (PDF page 124, printed page 100)

Page kind: content
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras
Section: 3.6 Introduction to Keras
Subsection: 3.6.6 Understanding the fit method
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_124.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_124.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p124_e03] (body) correct loss. For instance, you’ll use binary crossentropy for a two-class classification problem, categorical crossentropy for a many-class classification problem, and so on. Only when you’re working on truly new research problems will you have to develop your own loss functions. In the next few chapters, we’ll detail explicitly which loss functions to choose for a wide range of common tasks.
[p124_e04] (heading_subsection) 3.6.6
[p124_e05] (heading_subsection) Understanding the fit method
[p124_e06] (body) After `compile()` comes `fit()`. The `fit` method implements the training loop itself. Its key arguments are
[p124_e07] (bullet) The data (inputs and targets) to train on. It will typically be passed either in the form of NumPy arrays or a TensorFlow `Dataset` object. You’ll learn more about the `Dataset` API in the next chapters.
[p124_e08] (bullet) The number of epochs to train for: how many times the training loop should iterate over the data passed.
[p124_e09] (bullet) The batch size to use within each epoch of mini-batch gradient descent: the number of training examples considered to compute the gradients for one weight update step.
[p124_e10] (listing_caption) Listing 3.34 Calling `fit` with NumPy data
[p124_e11] (code)
```
history = model.fit(
    inputs,
    targets,
    epochs=5,
    batch_size=128,
)
```
[p124_e12] (body) The call to `fit` returns a `History` object. This object contains a `history` field, which is a dict mapping key, such as `"loss"` or specific metric names to the list of their per-epoch values:
[p124_e13] (code)
```
>>> history.history
{"binary_accuracy": [0.855, 0.9565, 0.9555, 0.95, 0.951],
 "loss": [0.6573270302042366,
  0.07434618508815766,
  0.07687718723714351,
  0.07412414988875389,
  0.07617757616937161]}
```

## Code annotations printed next to listings (callouts, in order)
- The input examples, as a NumPy array
- The corresponding training
- targets, as a NumPy array
- The training loop will iterate
- over the data 5 times.
- The training loop will iterate over the
- data in batches of 128 examples.

## Listings
- Listing 3.34: Calling `fit` with NumPy data (code elements p124_e11)

## Chapter outline
Chapter: 3 Introduction to TensorFlow, PyTorch, JAX, and Keras (PDF pages 84-127)
  - 3.1 A brief history of deep learning frameworks (p.85)
  - 3.2 How these frameworks relate to each other (p.87)
  - 3.3 Introduction to TensorFlow (p.87)
      - 3.3.1 First steps with TensorFlow (p.88)
      - 3.3.2 An end-to-end example: A linear classifier in pure TensorFlow (p.93)
      - 3.3.3 What makes the TensorFlow approach unique (p.98)
  - 3.4 Introduction to PyTorch (p.98)
      - 3.4.1 First steps with PyTorch (p.99)
      - 3.4.2 An end-to-end example: A linear classifier in pure PyTorch (p.102)
      - 3.4.3 What makes the PyTorch approach unique (p.105)
  - 3.5 Introduction to JAX (p.106)
      - 3.5.1 First steps with JAX (p.106)
      - 3.5.2 Tensors in JAX (p.107)
      - 3.5.3 Random number generation in JAX (p.107)
      - 3.5.4 An end-to-end example: A linear classifier in pure JAX (p.112)
      - 3.5.5 What makes the JAX approach unique (p.114)
  - 3.6 Introduction to Keras (p.114)
      - 3.6.1 First steps with Keras (p.115)
      - 3.6.2 Layers: The building blocks of deep learning (p.116)
      - 3.6.3 From layers to models (p.120)
      - 3.6.4 The “compile” step: Configuring the learning process (p.121)
      - 3.6.5 Picking a loss function (p.123)
      - 3.6.6 Understanding the fit method (p.124)
      - 3.6.7 Monitoring loss and metrics on validation data (p.125)
      - 3.6.8 Inference: Using a model after training (p.126)

## Previous page (PDF 123) text, for continuity
In chapter 7, we cover how to create custom losses and metrics. In general, you won’t have to create your own losses, metrics, or optimizers from scratch because Keras offers a wide range of built-in options that is likely to include what you need:

Optimizers – `SGD()` (with or without momentum) – `RMSprop()` – `Adam()` – Etc.

Losses – `CategoricalCrossentropy()` – `SparseCategoricalCrossentropy()` – `BinaryCrossentropy()` – `MeanSquaredError()` – `KLDivergence()` – `CosineSimilarity()` – `Etc.`

Metrics – `CategoricalAccuracy()` – `SparseCategoricalAccuracy()` – `BinaryAccuracy()` – `AUC()` – `Precision()` – `Recall()` – Etc.

Throughout this book, you’ll see concrete applications of many of these options.

3.6.5

Picking a loss function

Choosing the right loss function for the right problem is extremely important: your network will take any shortcut it can to minimize the loss. So if the objective doesn’t fully correlate with success for the task at hand, your network will end up doing things you may not have wanted. Imagine a stupid, omnipotent AI trained via SGD, with this poorly chosen objective function: “Maximize the average well-being of all humans alive.” To make its job easier, this AI might choose to kill all humans except a few and focus on the well-being of the remaining ones because average well-being isn’t affected by how many humans are left. That might not be what you intended! Just remember that all neural networks you build will be just as ruthless in lowering their loss function, so choose the objective wisely, or you’ll have to face unintended side effects.

Fortunately, when it comes to common problems such as classification, regression, and sequence prediction, there are simple guidelines you can follow to choose the

## Next page (PDF 125) text, for continuity
3.6.7

Monitoring loss and metrics on validation data

The goal of machine learning is not to obtain models that perform well on the training data, which is easy—all you have to do is follow the gradient. The goal is to obtain models that perform well in general, particularly on data points that the model has never encountered before. Just because a model performs well on its training data doesn’t mean it will perform well on data it has never seen! For instance, it’s possible that your model could end up merely memorizing a mapping between your training samples and their targets, which would be useless for the task of predicting targets for data the model has never seen before. We’ll go over this point in much more detail in the chapter 5.

To keep an eye on how the model does on new data, it’s standard practice to reserve a subset of the training data as “validation data”: you won’t be training the model on this data, but you will use it to compute a loss value and metrics value. You do this by using the `validation_data` argument in `fit()`. Like the training data, the validation data could be passed as NumPy arrays or as a TensorFlow `Dataset` object.

Listing 3.35 Using the validation data argument

The value of the loss on the validation data is called the validation loss, to distinguish it from the training loss. Note that it’s essential to keep the training data and validation

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.96 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow
- PDF p.247 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.122 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.239 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.127 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras
- PDF p.248 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.249 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.76 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.239 (printed 215) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › 
Whenever you find yourself in a situation where the built-in `fit()` is not enough, you will need to write your own custom training logic. You’ve already seen simple examples of low-level training loops in chapters 2 and 3. As a reminder, the contents of a typical training loop look like this: 1 Run the “forward pass” (compute the model’s output) to obtain a loss value for

the current batch of data. 2 Retrieve the gradients of the loss with regard to the model’s weights. 3 Update the model’s weights so as to lower the loss value on the current batch of data. These steps are repeated for as many batches as necessary. This is essentially what `fit()` does under the hood. In this section, you will learn to reimplement `fit()` from scratch, which will give you all the knowledge you need to write any training algorithm you may come up with. Let’s go over the details. Throughout the next few sections, you’ll work your way up to writing a fully featured custom training loop in TensorFlow, PyTorch, and JAX.

### PDF p.246 (printed 222) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Using fit() with a custom training loop
In the previous sections, we were writing our own training logic entirely from scratch. Doing so provides you with the most flexibility, but you end up writing a lot of code, while simultaneously missing out on many convenient features of `fit()`, such as callbacks, performance optimizations, or built-in support for distributed training.

What if you need a custom training algorithm, but you still want to use the power of the built-in Keras training loop? There’s actually a middle ground between `fit()` and a training loop written from scratch: you can provide a custom training step function and let the framework do the rest. You can do this by overriding the `train_step()` method of the `Model` class. This is the function that is called by `fit()` for every batch of data. You will then be able to call `fit()` as usual—and it will be running your own learning algorithm under the hood. Here’s how it works:

Create a new class that subclasses `keras.Model`.

### PDF p.76 (printed 52) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › 
Now you understand what happens when you call `fit`: the model will start to iterate on the training data in mini-batches of 128 samples, 5 times over (each iteration over all the training data is called an epoch). For each batch, the model will compute the gradient of the loss with regard to the weights (using the Backpropagation algorithm, which derives from the chain rule in calculus) and move the weights in the direction that will reduce the value of the loss for this batch.

After these 5 epochs, the model will have performed 2,345 gradient updates (469 per epoch), and the loss of the model will be sufficiently low that the model will be capable of classifying handwritten digits with high accuracy.

### PDF p.247 (printed 223) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Customizing fit() with TensorFlow
Listing 7.21 Customizing `fit()`: TensorFlow version
import keras
from keras import layers
loss_fn = keras.losses.SparseCategoricalCrossentropy()
loss_tracker = keras.metrics.Mean(name="loss")
class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply(gradients, self.trainable_weights)
        loss_tracker.update_state(loss)
        return {"loss": loss_tracker.result()}
    @property
    def metrics(self):
        return [loss_tracker]

### PDF p.81 (printed 57) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › The full training loop
def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter}: {loss:.2f}")

### PDF p.122 (printed 98) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.6 Introduction to Keras › 
Metrics—The measures of success you want to monitor during training and validation, such as classification accuracy. Unlike the loss, training will not optimize directly for these metrics. As such, metrics don’t need to be differentiable.

Once you’ve picked your loss, optimizer, and metrics, you can use the built-in `compile()` and `fit()` methods to start training your model. Alternatively, you can write your own custom training loops—we cover how to do this in chapter 7. It’s a lot more work! For now, let’s take a look at `compile()` and `fit()`.

The `compile()` method configures the training process—you’ve already been introduced to it in your very first neural network example in chapter 2. It takes the arguments `optimizer`, `loss`, and `metrics` (a list):

### PDF p.248 (printed 224) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Customizing fit() with PyTorch
import keras
from keras import layers
loss_fn = keras.losses.SparseCategoricalCrossentropy()
loss_tracker = keras.metrics.Mean(name="loss")
class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        predictions = self(inputs, training=True)
        loss = loss_fn(targets, predictions)
        loss.backward()
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)
        loss_tracker.update_state(loss)
        return {"loss": loss_tracker.result()}
    @property
    def metrics(self):
        return [loss_tracker]

### PDF p.44 (printed 20) — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network › 
We’re now ready to train the model, which in Keras is done via a call to the model’s `fit()` method—we fit the model to its training data.

Listing 2.5 “Fitting” the model
