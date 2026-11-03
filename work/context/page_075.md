# Generation bundle for page_075  (PDF page 75, printed page 51)

Page kind: content
Chapter: 2 The mathematical building blocks of neural networks
Section: 2.5 Looking back at our first example
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_075.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_075.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p075_e03] (heading_section) 2.5
[p075_e04] (heading_section) Looking back at our first example
[p075_e05] (body) You’re nearing the end of this chapter, and you should now have a general understanding of what’s going on behind the scenes in a neural network. What was a magical black box at the start of the chapter has turned into a clearer picture, as illustrated in figure 2.26: the model, composed of layers that are chained together, maps the input data to predictions. The loss function then compares these predictions to the targets, producing a loss value: a measure of how well the model’s predictions match what was expected. The optimizer uses this loss value to update the model’s weights.
[p075_e06] (figure_caption) Figure 2.26 Relationship between the network, layers, loss function, and optimizer
[p075_e07] (body) Let’s go back to the first example and review each piece of it in the light of what you’ve learned in the previous sections.
[p075_e08] (body) This was the input data:
[p075_e09] (code)
```
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
```
[p075_e10] (body) Now you understand that the input images are stored in NumPy tensors, which are here formatted as `float32` tensors of shape `(60000,` `784)` (training data) and `(10000, ` `784)` (test data), respectively.
[p075_e11] (body) This was our model:

## Figures
- Figure 2.26: Relationship between the network, layers, loss function, and optimizer  | labels inside the figure: Layer, (data transformation), Weights, Layer, (data transformation), Predictions, Y', Weight, update, True targets, Y, Weights, Loss function, Optimizer, Loss score

## Chapter outline
Chapter: 2 The mathematical building blocks of neural networks (PDF pages 40-83)
  - 2.1 A first look at a neural network (p.41)
  - 2.2 Data representations for neural networks (p.45)
      - 2.2.1 Scalars (rank-0 tensors) (p.46)
      - 2.2.2 Vectors (rank-1 tensors) (p.46)
      - 2.2.3 Matrices (rank-2 tensors) (p.46)
      - 2.2.4 Rank-3 tensors and higher-rank tensors (p.47)
      - 2.2.5 Key attributes (p.47)
      - 2.2.6 Manipulating tensors in NumPy (p.49)
      - 2.2.7 The notion of data batches (p.49)
      - 2.2.8 Real-world examples of data tensors (p.50)
  - 2.3 The gears of neural networks: Tensor operations (p.52)
      - 2.3.1 Element-wise operations (p.53)
      - 2.3.2 Broadcasting (p.54)
      - 2.3.3 Tensor product (p.56)
      - 2.3.4 Tensor reshaping (p.58)
      - 2.3.5 Geometric interpretation of tensor operations (p.59)
      - 2.3.6 A geometric interpretation of deep learning (p.62)
  - 2.4 The engine of neural networks: Gradient-based optimization (p.63)
      - 2.4.1 What’s a derivative? (p.65)
      - 2.4.2 Derivative of a tensor operation: The gradient (p.66)
      - 2.4.3 Stochastic gradient descent (p.67)
      - 2.4.4 Chaining derivatives: The Backpropagation algorithm (p.70)
  - 2.5 Looking back at our first example (p.75)
      - 2.5.1 Reimplementing our first example from scratch (p.77)
      - 2.5.2 Running one training step (p.79)
      - 2.5.3 The full training loop (p.81)
      - 2.5.4 Evaluating the model (p.82)

## Previous page (PDF 74) text, for continuity
Figure 2.25 Path from `loss_val` to `w` in the backward graph

NOTE If there are multiple paths linking the two nodes of interest `a`, `b` in the backward graph, we would obtain `grad(b, a)` by summing the contributions of all the paths.

And with that, you just saw backpropagation in action! Backpropagation is simply the application of the chain rule to a computation graph. There’s nothing more to it. Backpropagation starts with the final loss value and works backward from the top layers to the bottom layers, computing the contribution that each parameter had in the loss value. That’s where the name “backpropagation” comes from: we “back propagate” the loss contributions of different nodes in a computation graph.

Nowadays, people implement neural networks in modern frameworks that are capable of automatic differentiation, such as JAX, TensorFlow, and PyTorch. Automatic differentiation is implemented with the kind of computation graph previously presented. Automatic differentiation makes it possible to retrieve the gradients of arbitrary compositions of differentiable tensor operations without doing any extra work besides writing down the forward pass. When I wrote my first neural networks in C in the 2000s, I had to write my gradients by hand. Now, thanks to modern automatic differentiation tools, you’ll never have to implement backpropagation yourself. Consider yourself lucky!

## Next page (PDF 76) text, for continuity
Now you understand that this model consists of a chain of two `Dense` layers, that each layer applies a few simple tensor operations to the input data, and that these operations involve weight tensors. Weight tensors, which are attributes of the layers, are where the knowledge of the model persists.

This was the model-compilation step:

Now you understand that `"sparse_categorical_crossentropy"` is the loss function that’s used as a feedback signal for learning the weight tensors, which the training phase will attempt to minimize. You also know that this reduction of the loss happens via mini-batch stochastic gradient descent. The exact rules governing a specific use of gradient descent are defined by the `"adam"` optimizer passed as the first argument.

Finally, this was the training loop:

Now you understand what happens when you call `fit`: the model will start to iterate on the training data in mini-batches of 128 samples, 5 times over (each iteration over all the training data is called an epoch). For each batch, the model will compute the gradient of the loss with regard to the weights (using the Backpropagation algorithm, which derives from the chain rule in calculus) and move the weights in the direction that will reduce the value of the loss for this batch.

After these 5 epochs, the model will have performed 2,345 gradient updates (469 per epoch), and the loss of the model will be sufficiently low that the model will be capable of classifying handwritten digits with high accuracy.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.44 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network
- PDF p.82 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.43 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network
- PDF p.81 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example
- PDF p.42 — 2 The mathematical building blocks of neural networks › 2.1 A first look at a neural network
- PDF p.45 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.48 — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks
- PDF p.79 — 2 The mathematical building blocks of neural networks › 2.5 Looking back at our first example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.33 (printed 9) — 1 What is deep learning? › 1.6 Understanding how deep learning works, in three figures › 
value for all of them may seem like a daunting task, especially given that modifying the value of one parameter will affect the behavior of all the others!

Figure 1.7 A neural network is parameterized by its weights.

To control something, first you need to be able to observe it. To control the output of a neural network, you need to be able to measure how far this output is from what you expected. This is the job of the loss function of the network, also sometimes called the objective function or cost function. The loss function takes the predictions of the network and the true target (what you wanted the network to output) and computes a distance score, capturing how well the network has done on this specific example (see figure 1.8).

The fundamental trick in deep learning is to use this score as a feedback signal to adjust the value of the weights a little, in a direction that will lower the loss score for the current example (see figure 1.9). This adjustment is the job of the optimizer, which implements what’s called the Backpropagation algorithm: the central algorithm

Figure 1.8 A loss function measures the quality of the

in deep learning. The next chap

network’s output.


### PDF p.9 (printed ix) — contents ›  › 
2

The mathematical building blocks of neural networks 16 2.1

A first look at a neural network 17 2.2 Data representations for neural networks 21

Scalars (rank-0 tensors) 22 ■ Vectors (rank-1 tensors) 22 Matrices (rank-2 tensors) 22 ■ Rank-3 tensors and higher-rank tensors 23 ■ Key attributes 23 ■ Manipulating tensors in NumPy 25 ■ The notion of data batches 25 Real-world examples of data tensors 26

2.3

The gears of neural networks: Tensor operations 28

Element-wise operations 29 ■ Broadcasting 30 Tensor product 32 ■ Tensor reshaping 34 ■ Geometric interpretation of tensor operations 35 ■ A geometric interpretation of deep learning 38

2.4

The engine of neural networks: Gradient-based optimization 39

What’s a derivative? 41 ■ Derivative of a tensor operation: The gradient 42 ■ Stochastic gradient descent 43 ■ Chaining derivatives: The Backpropagation algorithm 46

2.5

Looking back at our first example 51

Reimplementing our first example from scratch 53 Running one training step 55 ■ The full training loop 57 Evaluating the model 58

3

Introduction to TensorFlow, PyTorch, JAX, and Keras 60 3.1

A brief history of deep learning frameworks 61 3.2 How these frameworks relate 

### PDF p.64 (printed 40) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › 
`y_pred` and `y_true`. 4 Update all weights of the model in a way that slightly reduces the loss on this batch. You’ll eventually end up with a model that has a very low loss on its training data: a low mismatch between predictions `y_pred` and expected targets `y_true`. The model has “learned” to map its inputs to correct targets. From afar, it may look like magic, but when you reduce it to elementary steps, it turns out to be simple. Step 1 sounds easy enough—it’s just I/O code. Steps 2 and 3 are merely the application of a handful of tensor operations, so you could implement these steps purely from what you learned in the previous section. The difficult part is step 4: updating the model’s weights. Given an individual weight coefficient in the model, how can you compute whether the coefficient should be increased or decreased, and by how much? One naive solution would be to freeze all weights in the model except the one scalar coefficient being considered and try different values for this coefficient. Let’s say the initial value of the coefficient is 0.3. After the forward pass on a batch of data, the loss of the model on the batch is 0.5. If you change the coefficient’s value t

### PDF p.40 (printed 16) — 2 The mathematical building blocks of neural networks ›  › This chapter covers
A first example of a neural network

Tensors and tensor operations

How neural networks learn via backpropagation

and gradient descent

Understanding deep learning requires familiarity with many simple mathematical

concepts: tensors, tensor operations, differentiation, gradient descent, and so on. Our goal

in this chapter will be to build up your intuition about these notions without getting overly technical. In particular, we’ll steer away from mathematical notation, which can introduce unnecessary barriers for those without any mathematics background, and isn’t necessary to explain things well. The most precise, unambiguous description of a mathematical operation is its executable code.

To provide sufficient context for introducing tensors and gradient descent, we’ll begin the chapter with a practical example of a neural network. Then we’ll go over every new concept that’s been introduced, point by point. Keep in mind that these concepts will be essential for you to understand the practical examples that will come in the following chapters!

### PDF p.32 (printed 8) — 1 What is deep learning? › 1.6 Understanding how deep learning works, in three figures › Understanding how deep learning works, in three figures
At this point, you know that machine learning is about mapping inputs (such as images) to targets (such as the label “cat”), which is done by observing many examples of inputs and targets. You also know that deep neural networks do this input-to-target mapping via a deep sequence of simple data transformations (layers) and that these data transformations are learned by exposure to examples. Now let’s look at how this learning happens, concretely.

The specification of what a layer does to its input data is stored in the layer’s weights, which in essence are a bunch of numbers. In technical terms, we’d say that the transformation implemented by a layer is parameterized by its weights (see figure 1.7). (Weights are also sometimes called the parameters of a layer.) In this context, learning means finding a set of values for the weights of all layers in a network, such that the network will correctly map example inputs to their associated targets. But here’s the thing: a deep neural network can contain tens of millions of parameters. Finding the correct

### PDF p.63 (printed 39) — 2 The mathematical building blocks of neural networks › 2.4 The engine of neural networks: Gradient-based optimization › The engine of neural networks: Gradient-based optimization
As you saw in the previous section, each neural layer from our first model example transforms its input data as follows:

### PDF p.45 (printed 21) — 2 The mathematical building blocks of neural networks › 2.2 Data representations for neural networks › 
The test set accuracy turns out to be 97.8%—that’s almost double the error rate of the training set (at 98.9% accuracy). This gap between training accuracy and test accuracy is an example of overfitting: the fact that machine learning models tend to perform worse on new data than on their training data. Overfitting is a central topic in chapter 5.

This concludes our first example. You just saw how you can build and train a neural network to classify handwritten digits in less than 15 lines of Python code. In this chapter and the next, we’ll go into detail about every moving piece we just previewed and clarify what’s going on behind the scenes. You’ll learn about tensors, the data-storing objects going into the model; tensor operations, which layers are made of; and gradient descent, which allows your model to learn from its training examples.

### PDF p.34 (printed 10) — 1 What is deep learning? › 1.7 What makes deep learning different › 
Figure 1.9 The loss score is used as a feedback signal to adjust the weights.

number of times (typically tens of passes over thousands of examples), yields weight values that minimize the loss function. A network with a minimal loss is one for which the outputs are as close as they can be to the targets: a trained network. Once again, it’s a simple mechanism that, once scaled, ends up looking like magic.
