# Generation bundle for page_133  (PDF page 133, printed page 109)

Page kind: content
Chapter: 4 Classification and regression
Section: 4.1 Classifying movie reviews: A binary classification example
Subsection: 4.1.3 Building your model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_133.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_133.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p133_e03] (body) Figure 4.1 shows what the model looks like. Here’s the Keras implementation, similar to the MNIST example you saw previously.
[p133_e04] (code)
```
Dense (units=16)
```
[p133_e05] (listing_caption) Listing 4.4 Model definition
[p133_e06] (code)
```
Dense (units=16)
import keras
from keras import layers
Dense (units=1)
model = keras.Sequential(
    [
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
```
[p133_e07] (figure_caption) Figure 4.1 The three-
[p133_e08] (code)
```
    ]
)
```
[p133_e09] (figure_caption) layer model
[p133_e10] (body) The first argument being passed to each `Dense` layer is the number of units in the layer: the dimensionality of representation space of the layer. You remember from chapters 2 and 3 that each such `Dense` layer with a `relu` activation implements the following chain of tensor operations:
[p133_e11] (code)
```
output = relu(dot(input, W) + b)
```
[p133_e12] (body) Having 16 units means the weight matrix `W` will have shape `(input_dimension,` `16)`: the dot product with `W` will project the input data onto a 16-dimensional representation space (and then you’ll add the bias vector `b` and apply the `relu` operation). You can intuitively understand the dimensionality of your representation space as “how much freedom you’re allowing the model to have when learning internal representations.” Having more units (a higher-dimensional representation space) allows your model to learn more complex representations, but it makes the model more computationally expensive and may lead to learning unwanted patterns (patterns that will improve performance on the training data but not on the test data).
[p133_e13] (body) The intermediate layers use `relu` as their activation function, and the final layer uses a sigmoid activation to output a probability (a score between 0 and 1, indicating how likely the review is to be positive). A `relu` (rectified linear unit) is a function meant to zero-out negative values (see figure 4.2), whereas a sigmoid “squashes” arbitrary values into the `[0, 1]` interval (see figure 4.3), outputting something that can be interpreted as a probability.
[p133_e14] (heading_subsection) What are activation functions, and why are they necessary?
[p133_e15] (sidebar) Without an activation function like `relu` (also called a non-linearity), the `Dense` layer would consist of two linear operations—a dot product and an addition:

## Listings
- Listing 4.4: Model definition (code elements p133_e06)

## Figures
- Figure 4.1: The three-  | labels inside the figure: Input, (vectorized text), Output, (probability)
- Figure None: layer model  | labels inside the figure: (none - image)

## Equation candidates (formulas written inline in the text)
- [p133_e04] Dense (units=16)
- [p133_e11] output = relu(dot(input, W) + b)

## Chapter outline
Chapter: 4 Classification and regression (PDF pages 128-159)
  - 4.1 Classifying movie reviews: A binary classification example (p.130)
      - 4.1.1 The IMDb dataset (p.130)
      - 4.1.2 Preparing the data (p.131)
      - 4.1.3 Building your model (p.132)
      - 4.1.4 Validating your approach (p.135)
      - 4.1.5 Using a trained model to generate predictions on new data (p.139)
      - 4.1.6 Further experiments (p.139)
      - 4.1.7 Wrapping up (p.140)
  - 4.2 Classifying newswires: A multiclass classification example (p.140)
      - 4.2.1 The Reuters dataset (p.140)
      - 4.2.2 Preparing the data (p.142)
      - 4.2.3 Building your model (p.142)
      - 4.2.4 Validating your approach (p.144)
      - 4.2.5 Generating predictions on new data (p.148)
      - 4.2.6 A different way to handle the labels and the loss (p.148)
      - 4.2.7 The importance of having sufficiently large intermediate layers (p.149)
      - 4.2.8 Further experiments (p.149)
      - 4.2.9 Wrapping up (p.150)
  - 4.3 Predicting house prices: A regression example (p.150)
      - 4.3.1 The California Housing Price dataset (p.150)
      - 4.3.2 Preparing the data (p.152)
      - 4.3.3 Building your model (p.152)
      - 4.3.4 Validating your approach using K-fold validation (p.153)
      - 4.3.5 Generating predictions on new data (p.158)
      - 4.3.6 Wrapping up (p.158)

## Previous page (PDF 132) text, for continuity
Listing 4.3 Encoding the integer sequences via multi-hot encoding

Here’s what the samples look like now:

In addition to vectorizing the input sequences, you should also vectorize their labels, which is straightforward. Our labels are already NumPy arrays, so just convert the type from ints to floats:

Now the data is ready to be fed into a neural network.

4.1.3

Building your model

The input data is vectors, and the labels are scalars (1s and 0s): this is one of the simplest problem setups you’ll ever encounter. A type of model that performs well on such a problem is a plain stack of densely connected (`Dense`) layers with `relu` activations.

There are two key architecture decisions to be made about such a stack of `Dense` layers:

How many layers to use

How many units to choose for each layer

In chapter 5, you’ll learn formal principles to guide you in making these choices. For the time being, you’ll have to trust us with the following architecture choice:

Two intermediate layers with 16 units each

A third layer that will output the scalar prediction regarding the sentiment of the current review

## Next page (PDF 134) text, for continuity
(continued)

So the layer could only learn linear transformations (affine transformations) of the input data: the hypothesis space of the layer would be the set of all possible linear transformations of the input data into a 16-dimensional space. Such a hypothesis space is too restricted and wouldn’t benefit from multiple layers of representations because a deep stack of linear layers would still implement a linear operation: adding more layers wouldn’t extend the hypothesis space (as you saw in chapter 2). To get access to a much richer hypothesis space that would benefit from deep representations, you need a non-linearity or activation function. `relu` is the most popular activation function in deep learning, but there are many other candidates, which all come with similarly strange names: `prelu`, `elu`, and so on.

Figure 4.2 The rectified linear unit function Figure 4.3 The sigmoid function

Finally, you need to choose a loss function and an optimizer. Because you’re facing a binary classification problem and the output of your model is a probability (you end your model with a single-unit layer with a sigmoid activation), it’s best to use the `binary_crossentropy` loss. It isn’t the only viable choice: you could use, for instance, `mean_squared_error`. But crossentropy is usually the best choice when you’re dealing with models that output probabilities. Crossentropy is a quantity from the field of information theory that measures the distance between probability distributions or, in this case, between the ground-truth distribution and your predictions.

As for the choice of the optimizer, we’ll go with `adam`, which is usually a good default choice for virtually any problem. Here’s the step where you configure the model with the `adam` optimizer and the `binary_` ` …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.143 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.625 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.140 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.149 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.77 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.147 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.139 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.153 — 4 Classification and regression › 4.3 Predicting house prices: A regression example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.77 (printed 53) — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example › A simple Dense class
You’ve learned earlier that the `Dense` layer implements the following input transformation, where `W` and `b` are model parameters, and `activation` is an element-wise function (usually `relu`):

### PDF p.140 (printed 116) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › Wrapping up
Here’s what you should take away from this example:

You usually need to do quite a bit of preprocessing on your raw data to be able to feed it—as tensors—into a neural network. Sequences of words can be encoded as binary vectors, but there are other encoding options, too.

Stacks of `Dense` layers with `relu` activations can solve a wide range of problems (including sentiment classification), and you’ll use them frequently.

In a binary classification problem (two output classes), your model should end with a `Dense` layer with one unit and a `sigmoid` activation: the output of your model should be a scalar between 0 and 1, encoding a probability.

With such a scalar sigmoid output on a binary classification problem, the loss function you should use is `binary_crossentropy`.

The `adam` optimizer is generally a good enough choice, whatever your problem. That’s one less thing for you to worry about.

As they get better on their training data, neural networks eventually start overfitting and end up obtaining increasingly worse results on data they’ve never seen before. Be sure to always monitor performance on data that is outside of the training set!

### PDF p.625 (printed 601) — 20 Conclusions › 20.1 Key concepts in review › 
assume no specific structure in the input features: they’re called densely connected because the units of a `Dense` layer are connected to every other unit. The layer attempts to map relationships between any two input features; this is unlike a 2D convolution layer, for instance, which only looks at local relationships.

Densely connected networks are most commonly used for categorical data (for example, where the input features are lists of attributes), such as the Boston Housing Price dataset used in chapter 4. They’re also used as the final classification or regression stage of most networks. For instance, the ConvNets covered in chapter 8 typically end with one or two `Dense` layers, and so do the recurrent networks in chapter 13. Remember, to perform binary classification, end your stack of layers with a `Dense` layer with a single unit and a `sigmoid` activation, and use `binary_crossentropy` as the loss. Your targets should be either 0 or 1:

### PDF p.143 (printed 119) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › 
Listing 4.15 Model definition
model = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(46, activation="softmax"),
    ]
)

### PDF p.219 (printed 195) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › A simple example
Listing 7.8 A simple Functional model with two `Dense` layers
inputs = keras.Input(shape=(3,), name="my_input")
features = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs, name="my_functional_model")

### PDF p.143 (printed 119) — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example › 
There are two other things you should note about this architecture:

You end the model with a `Dense` layer of size 46. This means for each input sample, the network will output a 46-dimensional vector. Each entry in this vector (each dimension) will encode a different output class.

The last layer uses a `softmax` activation. You saw this pattern in the MNIST example. It means the model will output a probability distribution over the 46 different output classes—for every input sample, the model will produce a 46-dimensional output vector, where `output[i]` is the probability that the sample belongs to class `i`. The 46 scores will sum to 1.

The best loss function to use in this case is `categorical_crossentropy`. It measures the distance between two probability distributions—here, between the probability distribution outputted by the model and the true distribution of the labels. By minimizing the distance between these two distributions, you train the model to output something as close as possible to the true labels.

Like last time, we’ll also monitor accuracy. However, accuracy is a bit of a crude metric in this case: if the model has the correct class as its second choice for

### PDF p.62 (printed 38) — 2 The mathematical building blocks of neural networks › 2.3 The gears of neural networks: Tensor operations › 
Figure 2.12 Affine transform in the plane

`Dense` layer with `relu` activation—An important observation about affine transforms is that if you apply many of them repeatedly, you still end up with an affine transform (so you could just have applied that one affine transform in the first place). Let’s try it with two: `affine2(affine1(x)) = W2 @ (W1 @ x + b1) + b2 = (W2 @ W1) 
` `@ x + (W2 @ b1 + b2)`. That’s an affine transform where the linear part is the matrix `W2 @ W1` and the translation part is the vector `W2 @ b1 + b2`. As a consequence, a multilayer neural network made entirely of `Dense` layers without activations would be equivalent to a single `Dense` layer. This “deep” neural network would just be a linear model in disguise! This is why we need activation functions, like `relu` (seen in action in figure 2.13). Thanks to activation functions, a chain of `Dense` layers can be made to implement very complex, nonlinear geometric transformation, resulting in very rich hypothesis spaces for your deep neural networks. We cover this idea in more detail in the next chapter.

Figure 2.13 Affine transform followed by `relu` activation

### PDF p.625 (printed 601) — 20 Conclusions › 20.1 Key concepts in review › 
import keras
from keras import layers
inputs = keras.Input(shape=(num_input_features,))
x = layers.Dense(32, activation="relu")(inputs)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy")
