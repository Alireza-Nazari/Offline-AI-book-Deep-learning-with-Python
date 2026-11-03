# Generation bundle for page_299  (PDF page 299, printed page 275)

Page kind: content
Chapter: 9 ConvNet architecture patterns
Section: 9.2 Residual connections
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_299.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_299.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p299_e03] (code)
```
    elif filters != residual.shape[-1]:
        residual = layers.Conv2D(filters, 1)(residual)
    x = layers.add([x, residual])
    return x
x = residual_block(x, filters=32, pooling=True)
x = residual_block(x, filters=64, pooling=True)
x = residual_block(x, filters=128, pooling=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```
[p299_e04] (body) Let’s take a look at the model summary:
[p299_e05] (code)
```
>>> model.summary()
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)         ┃ Output Shape       ┃    Param # ┃ Connected to        ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer_2        │ (None, 32, 32, 3)  │          0 │ -                   │
│ (InputLayer)         │                    │            │                     │
├──────────────────────┼────────────────────┼────────────┼─────────────────────┤
│ rescaling (Rescaling)│ (None, 32, 32, 3)  │          0 │ input_layer_2[0][0] │
├──────────────────────┼────────────────────┼────────────┼─────────────────────┤
│ conv2d_6 (Conv2D)    │ (None, 32, 32, 32) │        896 │ rescaling[0][0]     │
├──────────────────────┼────────────────────┼────────────┼─────────────────────┤
│ conv2d_7 (Conv2D)    │ (None, 32, 32, 32) │      9,248 │ conv2d_6[0][0]      │
├──────────────────────┼────────────────────┼────────────┼─────────────────────┤
│ max_pooling2d_1      │ (None, 16, 16, 32) │          0 │ conv2d_7[0][0]      │
│ (MaxPooling2D)       │                    │            │                     │
├──────────────────────┼────────────────────┼────────────┼─────────────────────┤
│ conv2d_8 (Conv2D)    │ (None, 16, 16, 32) │        128 │ rescaling[0][0]     │
├──────────────────────┼────────────────────┼────────────┼─────────────────────┤
│ add_2 (Add)          │ (None, 16, 16, 32) │          0 │ max_pooling2d_1[0]… │
│                      │                    │            │ conv2d_8[0][0]      │
├──────────────────────┼────────────────────┼────────────┼─────────────────────┤
│ conv2d_9 (Conv2D)    │ (None, 16, 16, 64) │     18,496 │ add_2[0][0]         │
├──────────────────────┼────────────────────┼────────────┼─────────────────────┤
│ conv2d_10 (Conv2D)   │ (None, 16, 16, 64) │     36,928 │ conv2d_9[0][0]      │
├──────────────────────┼────────────────────┼────────────┼─────────────────────┤
│ max_pooling2d_2      │ (None, 8, 8, 64)   │          0 │ conv2d_10[0][0]     │
│ (MaxPooling2D)       │                    │            │                     │
```

## Code annotations printed next to listings (callouts, in order)
- If we don’t use max pooling, we only
- project the residual if the number
- of channels has changed.
- First block
- Second block.
- Note the
- increasing
- filter count in
- each block.
- The last block doesn’t need a max
- pooling layer, since we will apply global
- average pooling right after it.

## Chapter outline
Chapter: 9 ConvNet architecture patterns (PDF pages 292-307)
  - 9.1 Modularity, hierarchy, and reuse (p.293)
  - 9.2 Residual connections (p.296)
  - 9.3 Batch normalization (p.300)
  - 9.4 Depthwise separable convolutions (p.302)
  - 9.5 Putting it together: A mini Xception-like model (p.304)
  - 9.6 Beyond convolution: Vision Transformers (p.306)

## Previous page (PDF 298) text, for continuity
Listing 9.3 The target block including a max pooling layer

To make these ideas more concrete, here’s an example of a simple ConvNet structured into a series of blocks, each made of two convolution layers and one optional max pooling layer, with a residual connection around each block:

## Next page (PDF 300) text, for continuity
With residual connections, you can build networks of arbitrary depth, without having to worry about vanishing gradients. Now, let’s move on to the next essential ConvNet architecture pattern: batch normalization.

9.3

Batch normalization

Normalization in machine learning is a broad category of methods that seek to make different samples seen by a machine learning model more similar to each other, which helps the model learn and generalize well to new data. The most common form of data normalization is one you’ve seen several times in this book already: centering the data on zero by subtracting the mean from the data and giving the data a unit standard deviation by dividing the data by its standard deviation. In effect, this makes the assumption that the data follows a normal (or Gaussian) distribution and makes sure this distribution is centered and scaled to unit variance:

Previous examples you saw in this book normalized data before feeding it into models. But data normalization may be a concern after every transformation performed by the network: even if the data entering a `Dense` or `Conv2D` network has a 0 mean and unit variance, there’s no reason to expect a priori that this will be the case for the data coming out. Could normalizing intermediate activations help?

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.270 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.257 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.264 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.309 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.297 — 9 ConvNet architecture patterns › 9.2 Residual connections
- PDF p.305 — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model
- PDF p.310 — 10 Interpreting what ConvNets learn › 10.1 Visualizing intermediate activations
- PDF p.256 — 8 Image classification › 8.1 Introduction to convnets

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.296 (printed 272) — 9 ConvNet architecture patterns › 9.2 Residual connections › Residual connections
y = f4(f3(f2(f1(x))))

### PDF p.449 (printed 425) — 15 Language models and the Transformer › 15.1 The language model › 
Let’s take a look at our model summary:

### PDF p.473 (printed 449) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
Let’s take a look at the summary of our Transformer model:

### PDF p.297 (printed 273) — 9 ConvNet architecture patterns › 9.2 Residual connections › 
Listing 9.1 A residual connection in pseudocode
x = ...
residual = x
x = block(x)
x = add([x, residual])

### PDF p.220 (printed 196) — 7 A deep dive on Keras › 7.2 Different ways to build Keras models › 
Here’s the summary of our model:

### PDF p.424 (printed 400) — 14 Text classification › 14.4 Set models › 
Let’s take a look at our model’s summary:

### PDF p.297 (printed 273) — 9 ConvNet architecture patterns › 9.2 Residual connections › 
The fix is simple: just force each function in the chain to be nondestructive—to retain a noiseless version of the information contained in the previous input. The easiest way to implement this is called a residual connection. It’s dead easy: just add the input of a layer or block of layers back to its output (see figure 9.3). The residual connection acts as an information shortcut around destructive or noisy blocks (such as blocks that contain ReLU activations or dropout layers), enabling error gradient information from early

Figure 9.3 A residual connection

layers to propagate noiselessly through a deep net

around a processing block

work. This technique was introduced in 2015 with the ResNet family of models (developed by He et al. at Microsoft).1

In practice, you’d implement a residual connection like the following listing.

Listing 9.1 A residual connection in pseudocode

### PDF p.458 (printed 434) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 
Let’s take a look at the seq2seq model in full:
