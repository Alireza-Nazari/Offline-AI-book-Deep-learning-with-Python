# Generation bundle for page_540  (PDF page 540, printed page 516)

Page kind: content
Chapter: 17 Image generation
Section: 17.1 Deep learning for image generation
Subsection: 17.1.3 Implementing a VAE with Keras
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_540.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_540.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p540_e03] (body) Now, let’s create the VAE model itself. This is your first example of a model that isn’t doing supervised learning (an autencoder is an example of self-supervised learning because it uses its inputs as targets). Whenever you depart from classic supervised learning, it’s common to subclass the `Model` class and implement a custom `train_` `step()` to specify the new training logic, a workflow you’ve learned about in chapter 7. We could easily do that here, but a downside of this technique is that the `train_step()` contents must be backend specific—you’d use `GradientTape` with TensorFlow, you’d use `loss.backward()` with PyTorch, and so on. A simpler way to customize your training logic is to just implement the `compute_loss()` method instead and keep the default `train_step()`. `compute_loss()` is the key bit of differentiable logic called by the built-in `train_step()`. Since it doesn’t involve direct manipulation of gradients, it’s easy to keep it backend agnostic.
[p540_e04] (body) Its signature is as follows:
[p540_e05] (code)
```
compute_loss(x, y, y_pred, sample_weight=None, training=True)
```
[p540_e06] (body) where `x` is the model’s input; `y` is the model’s target (in our case, it is `None` since the dataset we use only has inputs, no targets); and `y_pred` is the output of `call()`—the model’s predictions. In any supervised training workflow, you’d compute the loss based on `y` and `y_pred`. In our case, since `y` is `None` and `y_pred` contains the latent parameters, we’ll compute the loss using `x` (the original input) and the `reconstruction` derived from `y_pred`.
[p540_e07] (body) The method must return a scalar, the loss value to be minimized. You can also use `compute_loss()` to update the state of your metrics, which is something we’ll want to do in our case. Now, let’s write our VAE with a custom `compute_loss()` method. It works with all backends with no code changes!
[p540_e08] (listing_caption) Listing 17.4 VAE model with custom `compute_loss()` method
[p540_e09] (code)
```
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampler()
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    def call(self, inputs):
        return self.encoder(inputs)
    def compute_loss(self, x, y, y_pred, sample_weight=None, training=True):
```

## Code annotations printed next to listings (callouts, in order)
- We’ll use these metrics to
- keep track of the loss
- averages over each epoch.

## Listings
- Listing 17.4: VAE model with custom `compute_loss()` method (code elements p540_e09)

## Equation candidates (formulas written inline in the text)
- [p540_e05] compute_loss(x, y, y_pred, sample_weight=None, training=True)

## Chapter outline
Chapter: 17 Image generation (PDF pages 532-561)
  - 17.1 Deep learning for image generation (p.532)
      - 17.1.1 Sampling from latent spaces of images (p.533)
      - 17.1.2 Variational autoencoders (p.534)
      - 17.1.3 Implementing a VAE with Keras (p.537)
  - 17.2 Diffusion models (p.542)
      - 17.2.1 The Oxford Flowers dataset (p.544)
      - 17.2.2 A U-Net denoising autoencoder (p.545)
      - 17.2.3 The concepts of diffusion time and diffusion schedule (p.547)
      - 17.2.4 The training process (p.549)
      - 17.2.5 The generation process (p.551)
      - 17.2.6 Visualizing results with a custom callback (p.552)
      - 17.2.7 It’s go time! (p.553)
  - 17.3 Text-to-image models (p.555)
      - 17.3.1 Exploring the latent space of a text-to-image model (p.557)

## Previous page (PDF 539) text, for continuity
Listing 17.3 VAE decoder network, mapping latent space points to images

Its summary looks like this:

## Next page (PDF 541) text, for continuity
Finally, you’re ready to instantiate and train the model on MNIST digits. Because `compute_loss()` already takes care of the loss, you don’t specify an external loss at compile time (`loss=None`), which, in turn, means you won’t pass target data during training (as you can see, you only pass `x_train` to the model in `fit`).

Listing 17.5 Training the VAE

Once the model is trained, you can use the `decoder` network to turn arbitrary latent space vectors into images.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.550 — 17 Image generation › 17.2 Diffusion models
- PDF p.251 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.250 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.113 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.5 Introduction to JAX
- PDF p.551 — 17 Image generation › 17.2 Diffusion models
- PDF p.252 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.253 — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops
- PDF p.538 — 17 Image generation › 17.1 Deep learning for image generation

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.550 (printed 526) — 17 Image generation › 17.2 Diffusion models › 
Next comes the training logic. This is the most important part! Like in the VAE example, we’re going to implement a custom `compute_loss()` method to keep our model backend agnostic. Of course, if you are set on using one specific backend, you could also write a custom `train_step()` with the exact same logic in it, plus the backend-specific logic for gradient computation and weight updates.

Since `compute_loss()` receives as input the output of `call()`, we’re going to put the denoising forward pass in `call()`. Our `call()` takes a batch of clean input images and applies the following steps: 1 Normalizes the images 2 Samples random diffusion times (the denoising model needs to be trained on

the full spectrum of diffusion times) 3 Computes corresponding noise rates and signal rates (using the diffusion schedule) 4 Adds random noise to the clean images (based on the computed noise rates and signal rates) 5 Denoises the images

### PDF p.251 (printed 227) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › train_step() metrics handling with PyTorch
import keras
from keras import layers
class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        predictions = self(inputs, training=True)
        loss = self.compute_loss(y=targets, y_pred=predictions)
        loss.backward()
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

### PDF p.250 (printed 226) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › train_step() metrics handling with TensorFlow
import keras
from keras import layers
class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.compute_loss(y=targets, y_pred=predictions)

### PDF p.240 (printed 216) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Writing custom training step functions
def train_step(inputs, targets):
    predictions = model(inputs, training=True)
    loss = loss_fn(targets, predictions)
    gradients = get_gradients_of(loss, wrt=model.trainable_weights)
    optimizer.apply(gradients, model.trainable_weights)

### PDF p.253 (printed 229) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › 
            inputs,
            training=training,
        )
        loss = self.compute_loss(y=targets, y_pred=predictions)
        return loss, (predictions, non_trainable_variables)

### PDF p.238 (printed 214) — 7 A deep dive on Keras › 7.4 Writing your own training and evaluation loops › Writing your own training and evaluation loops
The `fit()` workflow strikes a nice balance between ease of use and flexibility. It’s what you will use most of the time. However, it isn’t meant to support everything a deep learning researcher may want to do—even with custom metrics, custom losses, and custom callbacks.

After all, the built-in `fit()` workflow is solely focused on supervised learning: a setup where there are known targets (also called labels or annotations) associated with your input data and where you compute your loss as a function of these targets and the model’s predictions. However, not every form of machine learning falls into this category. There are other setups where no explicit targets are present, such as generative learning (which we will introduce in chapter 16), self-supervised learning (where targets are obtained from the inputs), or reinforcement learning (where learning is driven by occasional “rewards”—much like training a dog). And even if you’re doing regular supervised learning, as a researcher, you may want to add some novel bells and whistles that require low-level flexibility.

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

### PDF p.68 (printed 44) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
Instead, you can use the four-step algorithm outlined at the beginning of this section: modify the parameters little by little based on the current loss value on a random batch of data. Because you’re dealing with a differentiable function, you can compute its gradient, which gives you an efficient way to implement step 4. If you update the weights in the opposite direction from the gradient, the loss will be a little less every time: 1 Draw a batch of training samples `x` and corresponding targets `y_true`. 2 Run the model on `x` to obtain predictions `y_pred` (this is called the forward pass). 3 Compute the loss of the model on the batch, a measure of the mismatch between

`y_pred` and `y_true`. 4 Compute the gradient of the loss with regard to the model’s parameters (this is

called the backward pass).

5 Move the parameters a little in the opposite direction from the gradient—for

example, `W -= learning_rate * gradient`—thus reducing the loss on the batch a bit. The learning rate (`learning_rate` here) would be a scalar factor modulating the “speed” of the gradient descent process. Easy enough! What we just described is called mini-batch stochastic gradient descent (mini-batch
