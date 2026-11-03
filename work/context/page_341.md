# Generation bundle for page_341  (PDF page 341, printed page 317)

Page kind: content
Chapter: 11 Image segmentation
Section: 11.2 Training a segmentation model from scratch
Subsection: 11.2.2 Building and training the segmentation model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_341.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_341.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p341_e03] (code)
```
    batch_size=64,
    validation_data=(val_input_imgs, val_targets),
)
```
[p341_e04] (body) Let’s display our training and validation loss (see figure 11.5):
[p341_e05] (code)
```
epochs = range(1, len(history.history["loss"]) + 1)
loss = history.history["loss"]
val_loss = history.history["val_loss"]
plt.figure()
plt.plot(epochs, loss, "r--", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
```
[p341_e06] (figure_caption) Figure 11.5 Displaying training and validation loss curves
[p341_e07] (body) You can see that we start overfitting midway, around epoch 25. Let’s reload our best-performing model according to validation loss and demonstrate how to use it to predict a segmentation mask (see figure 11.6):
[p341_e08] (code)
```
model = keras.models.load_model("oxford_segmentation.keras")
i = 4
test_image = val_input_imgs[i]
plt.axis("off")
plt.imshow(array_to_img(test_image))
```

## Figures
- Figure 11.5: Displaying training and validation loss curves  | labels inside the figure: (none - image)

## Equation candidates (formulas written inline in the text)
- [p341_e03]     batch_size=64,
    validation_data=(val_input_imgs, val_targets),
)
- [p341_e08] model = keras.models.load_model("oxford_segmentation.keras")
i = 4
test_image = val_input_imgs[i]
plt.axis("off")
plt.imshow(array_to_img(test_image))

## Chapter outline
Chapter: 11 Image segmentation (PDF pages 332-352)
  - 11.1 Computer vision tasks (p.332)
      - 11.1.1 Types of image segmentation (p.334)
  - 11.2 Training a segmentation model from scratch (p.335)
      - 11.2.1 Downloading a segmentation dataset (p.335)
      - 11.2.2 Building and training the segmentation model (p.338)
  - 11.3 Using a pretrained segmentation model (p.342)
      - 11.3.1 Downloading the Segment Anything Model (p.343)
      - 11.3.2 How Segment Anything works (p.343)
      - 11.3.3 Preparing a test image (p.345)
      - 11.3.4 Prompting the model with a target point (p.347)
      - 11.3.5 Prompting the model with a target box (p.351)

## Previous page (PDF 340) text, for continuity
simply apply the corresponding sequence of `Conv2DTranspose` layers followed by a final `Conv2D` layer to produce outputs of shape `(200, 200, 3)`.

To evaluate the model, we’ll use a metric named Intersection over Union (IoU). It’s a measure of the match between the ground truth segmentation masks and the predicted masks. It can be computed separately for each class or averaged over multiple classes. Here’s how it works: 1 Compute the intersection between the masks, the area where the prediction and

ground truth overlap. 2 Compute the union of the masks, the total area covered by both masks combined. This is the whole space we’re interested in—the target object and any extra bits your model might have included by mistake. 3 Divide the intersection area by the union area to get the IoU. It’s a number between 0 and 1, where 1 denotes a perfect match, and 0 denotes a complete miss. We can simply use a built-in Keras metric rather than building this ourselves:

We can now compile and fit our model:

## Next page (PDF 342) text, for continuity
Figure 11.6 A test image and its predicted segmentation mask

There are a couple of small artifacts in our predicted mask, caused by geometric shapes in the foreground and background. Nevertheless, our model appears to work nicely.

11.3 Using a pretrained segmentation model

In the image classification example from chapter 8, you saw how using a pretrained model could significantly boost your accuracy—especially when you only have a few samples to train on. Image segmentation is no different.

The Segment Anything Model ,1 or SAM for short, is a powerful pretrained segmentation model you can use for, well, almost anything. It was developed by Meta AI and released in April 2023. It was trained on 11 million images and their segmentation masks, covering over 1 billion object instances. This massive amount of training data

1 Kirillov et al., “Segment Anything,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, arXiv

(2023), https://arxiv.org/abs/2304.02643.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.285 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.275 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.274 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.144 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.279 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.284 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.180 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.136 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.181 (printed 157) — 5 Fundamentals of machine learning › 5.3 Improving model fit › 
The training curves now look exactly like they should: the model fits fast and starts overfitting after eight epochs (see figure 5.15).

Figure 5.15 Validation loss for a model with appropriate capacity

### PDF p.275 (printed 251) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
Figure 8.9 Training and validation metrics for a simple ConvNet

These plots are characteristic of overfitting. The training accuracy increases linearly over time, until it reaches nearly 100%, whereas the validation accuracy peaks around 80%. The validation loss reaches its minimum after only 10 epochs and then stalls, whereas the training loss keeps decreasing linearly as training proceeds.

Let’s check the test accuracy. We’ll reload the model from its saved file to evaluate it as it was before it started overfitting.

### PDF p.138 (printed 114) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › 
Figure 4.5 Training and validation accuracy

As you can see, the training loss decreases with every epoch, and the training accuracy increases with every epoch. That’s what you would expect when running gradient-descent optimization—the quantity you’re trying to minimize should be less with every iteration. But that isn’t the case for the validation loss and accuracy: they seem to peak at the fourth epoch. This is an example of what we warned against earlier: a model that performs better on the training data isn’t necessarily a model that will do better on data it has never seen before. In precise terms, what you’re seeing is overfitting: after the fourth epoch, you’re overoptimizing on the training data, and you end up learning representations that are specific to the training data and don’t generalize to data outside of the training set.

In this case, to prevent overfitting, you could stop training after four epochs. In general, you can use a range of techniques to mitigate overfitting, which we’ll cover in chapter 5. Let’s train a new model from scratch for four epochs and then evaluate it on the test data.

Listing 4.10 Training the model for four epochs

### PDF p.279 (printed 255) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
Let’s plot the results again; see figure 8.11. Thanks to data augmentation and dropout, we start overfitting much later, around epochs 60–70 (compared to epoch 10 for the original model). The validation accuracy ends up peaking above 85%—a big improvement over our first try.

Figure 8.11 Training and validation metrics with data augmentation

Let’s check the test accuracy.

Listing 8.22 Evaluating the model on the test set

### PDF p.157 (printed 133) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › 
According to this plot (see figure 4.11), validation MAE stops improving significantly after 120–140 epochs (this number includes the 10 epochs we omitted). Past that point, you start overfitting.

Once you’re finished tuning other parameters of the model (in addition to the number of epochs, you could also adjust the size of the intermediate layers), you can train a final production model on all of the training data, with the best parameters, and then look at its performance on the test data.

Figure 4.11 Validation MAE by epoch, excluding the first 10 data points

Listing 4.33 Training the final model

### PDF p.233 (printed 209) — 7 A deep dive on Keras › 7.3 Using built-in training and evaluation loops › The EarlyStopping and ModelCheckpoint callbacks
When you’re training a model, there are many things you can’t predict at the start. In particular, you can’t tell how many epochs will be needed to get to an optimal validation loss. Our examples so far have adopted the strategy of training for enough epochs that you begin overfitting, using the first run to figure out the optimal number of epochs, and then finally launching a new training run from scratch using this optimal number. Of course, this approach is wasteful. A much better way to handle this is to stop training when you measure that the validation loss is no longer improving. This can be achieved using the `EarlyStopping` callback.

The `EarlyStopping` callback interrupts training once a target metric being monitored has stopped improving for a fixed number of epochs. For instance, this callback allows you to interrupt training as soon as you start overfitting, thus avoiding having to retrain your model for a smaller number of epochs. This callback is typically used in combination with `ModelCheckpoint`, which lets you continually save the model during training (and, optionally, save only the current best model so far: the version of the model that achieved the best perf

### PDF p.188 (printed 164) — 5 Fundamentals of machine learning › 5.4 Improving generalization › 
Figure 5.19 shows how the bigger model fares compared to the reference model. The bigger model starts overfitting almost immediately, after just one epoch, and it overfits much more severely. Its validation loss is also noisier. It gets training loss near zero very quickly. The more capacity the model has, the more quickly it can model the training data (resulting in a low training loss), but the more susceptible it is to overfitting (resulting in a large difference between the training and validation loss).

Figure 5.19 Original model vs. much larger model on IMDB review classification

### PDF p.274 (printed 250) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › Fitting the model
Let’s plot the loss and accuracy of the model over the training and validation data during training (see figure 8.9).
