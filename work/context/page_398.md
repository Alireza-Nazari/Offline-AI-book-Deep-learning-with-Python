# Generation bundle for page_398  (PDF page 398, printed page 374)

Page kind: content
Chapter: 13 Timeseries forecasting
Section: 13.3 Recurrent neural networks
Subsection: 13.3.4 Using recurrent dropout to fight overfitting
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_398.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_398.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p398_e03] (code)
```
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    callbacks=callbacks,
)
```
[p398_e04] (body) Figure 13.11 shows the results. Success! We’re no longer overfitting during the first 20 epochs. We achieve a validation MAE as low as 2.27 degrees (7% improvement over the no-learning baseline) and a test MAE of 2.45 degrees (6.5% improvement over the baseline). Not too bad.
[p398_e05] (figure_caption) Figure 13.11 Training and validation loss on the Jena temperature forecasting task with a dropout-regularized LSTM
[p398_e06] (heading_subsection) On RNN runtime performance
[p398_e07] (sidebar) Recurrent models with very few parameters, like the ones in this chapter, tend to be significantly faster on a multicore CPU than on GPU because they only involve small matrix multiplications, and the chain of multiplications is not well parallelizable due to the presence of a `for` loop. But larger RNNs can greatly benefit from a GPU runtime. When using a Keras `LSTM` or `GRU` layer on GPU with default keyword arguments, your layer will be using a cuDNN kernel, a highly optimized, low-level, NVIDIA-provided implementation of the underlying algorithm (we’ve mentioned those in the previous chapter). As usual, cuDNN kernels are a mixed blessing: they’re fast, but inflexible—if you try doing anything not supported by the default kernel, you will suffer a dramatic

## Figures
- Figure 13.11: Training and validation loss on the Jena temperature forecasting task with a dropout-regularized LSTM  | labels inside the figure: (none - image)

## Chapter outline
Chapter: 13 Timeseries forecasting (PDF pages 375-404)
  - 13.1 Different kinds of timeseries tasks (p.375)
  - 13.2 A temperature forecasting example (p.376)
      - 13.2.1 Preparing the data (p.380)
      - 13.2.2 A commonsense, non-machine-learning baseline (p.383)
      - 13.2.3 Let’s try a basic machine learning model (p.384)
      - 13.2.4 Let’s try a 1D convolutional model (p.386)
  - 13.3 Recurrent neural networks (p.388)
      - 13.3.1 Understanding recurrent neural networks (p.389)
      - 13.3.2 A recurrent layer in Keras (p.392)
      - 13.3.3 Getting the most out of recurrent neural networks (p.396)
      - 13.3.4 Using recurrent dropout to fight overfitting (p.396)
      - 13.3.5 Stacking recurrent layers (p.399)
      - 13.3.6 Using bidirectional RNNs (p.401)
  - 13.4 Going even further (p.403)

## Previous page (PDF 397) text, for continuity
curves, it’s evident that the model is quickly overfitting, despite only having very few units: the training and validation losses start to diverge considerably after a few epochs. You’re already familiar with a classic technique for fighting this phenomenon: dropout, which randomly zeros out input units of a layer to break happenstance correlations in the training data that the layer is exposed to. But how to correctly apply dropout in recurrent networks isn’t a trivial question.

It has long been known that applying dropout before a recurrent layer hinders learning rather than helping with regularization. In 2015, Yarin Gal, as part of his PhD thesis on Bayesian deep learning,5 determined the proper way to use dropout with a recurrent network: the same dropout mask (the same pattern of dropped units) should be applied at every timestep, instead of a dropout mask that varies randomly from timestep to timestep. What’s more, to regularize the representations formed by the recurrent gates of layers such as `GRU` and `LSTM`, a temporally constant dropout mask should be applied to the inner recurrent activations of the layer (a recurrent dropout mask). Using the same dropout mask at every timestep allows the network to properly propagate its learning error through time; a temporally random dropout mask would disrupt this error signal and be harmful to the learning process. Yarin Gal did his research using Keras and helped build this mechanism directly into Keras recurrent layers. Every recurrent layer in Keras has two dropout-related arguments: `dropout`, a float specifying the dropout rate for input units of the layer, and `recurrent_dropout`, specifying the dropout rate of the recurrent units. Let’s add recurrent dropout to the `LSTM` layer of our first LSTM example and s …

## Next page (PDF 399) text, for continuity
slowdown, which more or less forces you stick to what NVIDIA happens to provide. For instance, recurrent dropout isn’t supported by the LSTM and GRU cuDNN kernels, so adding it to your layers forces the runtime to fall back to the regular TensorFlow implementation, which is generally two to five times slower on GPU (even though its computational cost is the same). As a way to speed up your RNN layer when you can’t use cuDNN, you can try unrolling it. Unrolling a `for` loop consists of removing the loop and simply in-lining its content N times. In the case of the `for` loop of an RNN, unrolling can help TensorFlow optimize the underlying computation graph. However, it will also considerably increase the memory consumption of your RNN—as such, it’s only viable for relatively small sequences (around 100 steps or fewer). Also, you can only do this if the number of timesteps in the data is known in advance by the model (that is, if you pass a `shape` without any `None` entries to your initial `Input()`). It works like this:

13.3.5 Stacking recurrent layers

Because you’re no longer overfitting, but seem to have hit a performance bottleneck, you should consider increasing the capacity and expressive power of the network. Recall the description of the universal machine learning workflow: it’s generally a good idea to increase the capacity of your model until overfitting becomes the primary obstacle (assuming you’re already taking basic steps to mitigate overfitting, such as using dropout). As long as you aren’t overfitting too badly, you’re likely under capacity.

Increasing network capacity is typically done by increasing the number of units in the layers or adding more layers. Recurrent layer stacking is a classic way to build more powerful recurrent networks: for instance, …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.385 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.400 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.384 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.387 — 13 Timeseries forecasting › 13.2 A temperature forecasting example
- PDF p.388 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.402 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.396 — 13 Timeseries forecasting › 13.3 Recurrent neural networks
- PDF p.389 — 13 Timeseries forecasting › 13.3 Recurrent neural networks

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.389 (printed 365) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
Figure 13.5 Training and validation MAE on the Jena temperature forecasting task with an LSTM-based model. (Note that we omit epoch 1 on this graph because the high training MAE (7.75) at epoch 1 would distort the scale.)

But why did the LSTM model perform markedly better than the densely connected one or the ConvNet? And how can we further refine the model? To answer this, let’s take a closer look at recurrent neural networks.

### PDF p.388 (printed 364) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3 Recurrent neural networks
Figure 13.5 shows the results. Much better! We achieve a validation MAE as low as 2.39 degrees and a test MAE of 2.55 degrees. The LSTM-based model can finally beat the commonsense baseline (albeit just by a bit, for now), demonstrating the value of machine learning on this task.

### PDF p.396 (printed 372) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3.3 Getting the most out of recurrent neural networks
By this point, you’ve learned

What RNNs are and how they work

What an LSTM is and why it works better on long sequences than a naive RNN

How to use Keras RNN layers to process sequence data

Next, we’ll review a number of more advanced features of RNNs, which can help you get the most out of your deep learning sequence models. By the end of the section, you’ll know most of what there is to know about using recurrent networks with Keras.

We’ll cover the following:

Recurrent dropout—This is a variant of dropout, used to fight overfitting in recurrent layers.

Stacking recurrent layers—This increases the representational power of the model (at the cost of higher computational loads).

Bidirectional recurrent layers—These present the same information to a recurrent network in different ways, increasing accuracy and mitigating forgetting issues.

We’ll use these techniques to refine our temperature forecasting RNN.

### PDF p.396 (printed 372) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3.4 Using recurrent dropout to fight overfitting
Let’s go back to the LSTM-based model we used earlier in the chapter—our first model able to beat the commonsense baseline. If you look at the training and validation

### PDF p.404 (printed 380) — 13 Timeseries forecasting › 13.4 Going even further › Summary
As you first learned in chapter 6, when approaching a new problem, it’s good to first establish commonsense baselines for your metric of choice. If you don’t have a baseline to beat, you can’t tell whether you’re making real progress.

Try simple models before expensive ones, to justify the additional expense. Sometimes a simple model will turn out to be your best option.

When you have data where ordering matters—in particular, for timeseries data— recurrent networks are a great fit and easily outperform models that first flatten the temporal data. The two essential RNN layers available in Keras are the `LSTM` layer and the `GRU` layer.

To use dropout with recurrent networks, you should use a time-constant dropout mask and recurrent dropout mask. These are built into Keras recurrent layers, so all you have to do is use the `recurrent_dropout` arguments of recurrent layers.

Stacked RNNs provide more representational power than a single RNN layer. They’re also much more expensive and thus not always worth it. Although they offer clear gains on complex problems (such as machine translation), they may not always be relevant to smaller, simpler problems.

### PDF p.400 (printed 376) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
Figure 13.12 shows the results. We achieve a test MAE of 2.39 degrees (an 8.8% improvement over the baseline). You can see that the added layer does improve the results a bit, though not dramatically. You may be seeing diminishing returns from increasing network capacity at this point.

Figure 13.12 Training and validation loss on the Jena temperature forecasting task with a stacked GRU network

### PDF p.388 (printed 364) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 13.3 Recurrent neural networks
Listing 13.12 A simple LSTM-based model
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.LSTM(16)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
callbacks = [
    keras.callbacks.ModelCheckpoint("jena_lstm.keras", save_best_only=True)
]
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=callbacks,
)
model = keras.models.load_model("jena_lstm.keras")
print("Test MAE: {model.evaluate(test_dataset)[1]:.2f}")

### PDF p.387 (printed 363) — 13 Timeseries forecasting › 13.2 A temperature forecasting example › 
Figure 13.4 Training and validation MAE on the Jena temperature forecasting task with a 1D ConvNet

As it turns out, this model performs even worse than the densely connected one, only achieving a validation MAE of about 2.9 degrees, far from the commonsense baseline. What went wrong here? Two things:

First, weather data doesn’t quite respect the translation invariance assumption. While the data does feature daily cycles, data from a morning follows different
