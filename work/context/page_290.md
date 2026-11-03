# Generation bundle for page_290  (PDF page 290, printed page 266)

Page kind: content
Chapter: 8 Image classification
Section: 8.3 Using a pretrained model
Subsection: 8.3.2 Fine-tuning a pretrained model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_290.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_290.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p290_e03] (code)
```
history = model.fit(
    augmented_train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks,
)
```
[p290_e04] (body) You can now finally evaluate this model on the test data (see figure 8.15):
[p290_e05] (code)
```
model = keras.models.load_model("fine_tuning.keras")
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")
```
[p290_e06] (figure_caption) Figure 8.15 Training and validation metrics for fine-tuning
[p290_e07] (body) Here, you get a test accuracy of 98.6% (again, your own results may be within half a percentage point). In the original Kaggle competition around this dataset, this would have been one of the top results. It’s not quite a fair comparison, however, since you used pretrained features that already contained prior knowledge about cats and dogs, which competitors couldn’t use at the time.
[p290_e08] (body) On the positive side, by using modern deep learning techniques, you managed to reach this result using only a small fraction of the training data that was available for the competition (about 10%). There is a huge difference between being able to train on 20,000 samples compared to 2,000 samples! Now you have a solid set of tools for dealing with image-classification problems—in particular, with small datasets.

## Figures
- Figure 8.15: Training and validation metrics for fine-tuning  | labels inside the figure: (none - image)

## Equation candidates (formulas written inline in the text)
- [p290_e03] history = model.fit(
    augmented_train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks,
)
- [p290_e05] model = keras.models.load_model("fine_tuning.keras")
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")

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

## Previous page (PDF 289) text, for continuity
Note that you should not unfreeze “batch normalization” layers (`BatchNormalization`). Batch normalization and its effect on fine-tuning is explained in the next chapter.

You already completed the first three steps when doing feature extraction. Let’s proceed with step 4: you’ll unfreeze your `conv_base`.

Partial fine-tuning

In this case, we chose to unfreeze and fine-tune all of the Xception convolutional base. However, when dealing with large pretrained models, you may sometimes only unfreeze some of the top layers of the convolutional base, and leave the lower layers frozen. You’re probably wondering, why only fine-tune some of the layers? Why the top ones specifically? Here’s why:

Earlier layers in the convolutional base encode more-generic, reusable features,

whereas layers higher up encode more-specialized features. It’s more useful to fine-tune the more specialized features because these are the ones that need to be repurposed on your new problem. There would be fast-decreasing returns in fine-tuning lower layers.

The more parameters you’re training, the more you’re at risk of overfitting. The

convolutional base has 15 million parameters, so it would be risky to attempt to train it on your small dataset. Thus, it can be a good strategy to fine-tune only the top three or four layers in the convolutional base. You’d do something like this:

Let’s start fine-tuning the model using a very low learning rate. The reason for using a low learning rate is that you want to limit the magnitude of the modifications you make to the representations of the layers you’re fine-tuning. Updates that are too large may harm these representations.

Listing 8.31 Fine-tuning the model

## Next page (PDF 291) text, for continuity
Summary

ConvNets excel at computer vision tasks. It’s possible to train one from scratch, even on a very small dataset, with decent results.

ConvNets work by learning a hierarchy of modular patterns and concepts to represent the visual world.

On a small dataset, overfitting will be the main issue. Data augmentation is a powerful way to fight overfitting when you’re working with image data.

It’s easy to reuse an existing ConvNet on a new dataset via feature extraction. This is a valuable technique for working with small image datasets.

As a complement to feature extraction, you can use fine-tuning, which adapts to a new problem some of the representations previously learned by an existing model. This pushes performance a bit further.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.287 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.276 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.279 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.285 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.258 — 8 Image classification › 8.1 Introduction to convnets
- PDF p.268 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.274 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.266 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.285 (printed 261) — 8 Image classification › 8.3 Using a pretrained model › 
Figure 8.13 Training and validation metrics for plain feature extraction

You reach a validation accuracy of slightly over 98%—much better than you achieved in the previous section with the small model trained from scratch. This is a bit of an unfair comparison, however, because ImageNet contains many dog and cat instances, which means that our pretrained model already has the exact knowledge required for the task at hand. This won’t always be the case when you use pretrained features.

However, the plots also indicate that you’re overfitting almost from the start—despite using dropout with a fairly large rate. That’s because this technique doesn’t use data augmentation, which is essential for preventing overfitting with small image datasets. Let’s check the test accuracy:

### PDF p.268 (printed 244) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
Unsurprisingly, the original dogs-versus-cats Kaggle competition, all the way back in 2013, was won by entrants who used ConvNets. The best entries achieved up to 95% accuracy. In this example, we will get fairly close to this accuracy (in the next section), even though we will train our models on less than 10% of the data that was available to the competitors.

This dataset contains 25,000 images of dogs and cats (12,500 from each class) and is 543 MB (compressed). After downloading and uncompressing the data, we’ll create a new dataset containing three subsets: a training set with 1,000 samples of each class, a validation set with 500 samples of each class, and a test set with 1,000 samples of each class. Why do this? Because many of the image datasets you’ll encounter in your career only contain a few thousand samples, not tens of thousands. Having more data available would make the problem easier—so it’s good practice to learn with a small dataset. The subsampled dataset we will work with will have the following directory structure:

### PDF p.265 (printed 241) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › Training a ConvNet from scratch on a small dataset
Having to train an image-classification model using very little data is a common situation, which you’ll likely encounter in practice if you ever do computer vision in a professional context. A “few” samples can mean anywhere from a few hundred to a few tens of thousands of images. As a practical example, we’ll focus on classifying images as dogs or cats. We’ll work with a dataset containing 5,000 pictures of cats and dogs (2,500 cats, 2,500 dogs), taken from the original Kaggle dataset. We’ll use 2,000 pictures for training, 1,000 for validation, and 2,000 for testing.

In this section, we’ll review one basic strategy to tackle this problem: training a new model from scratch using what little data we have. We’ll start by naively training a small ConvNet on the 2,000 training samples, without any regularization, to set a baseline for what can be achieved. This will get us to a classification accuracy of about 80%. At that

### PDF p.266 (printed 242) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
point, the main issue will be overfitting. Then we’ll introduce data augmentation, a powerful technique for mitigating overfitting in computer vision. By using data augmentation, we’ll improve the model to reach a test accuracy of about 84%.

In the next section, we’ll review two more essential techniques for applying deep learning to small datasets: feature extraction with a pretrained model and fine-tuning a pretrained model (which will get us to a final accuracy of 98.5%). Together, these three strategies—training a small model from scratch, doing feature extraction using a pretrained model, and fine-tuning a pretrained model—will constitute your future toolbox for tackling the problem of performing image classification with small datasets.

### PDF p.280 (printed 256) — 8 Image classification › 8.3 Using a pretrained model › Using a pretrained model
A common and highly effective approach to deep learning on small image datasets is to use a pretrained model. A pretrained model is a model that was previously trained on a large dataset, typically on a large-scale image-classification task. If this original dataset is large enough and general enough, then the spatial hierarchy of features learned by the pretrained model can effectively act as a generic model of the visual world, and hence its features can prove useful for many different computer vision problems, even though these new problems may involve completely different classes than those of the original task. For instance, you might train a model on ImageNet (where classes are mostly animals and everyday objects) and then repurpose this trained model for something as remote as identifying furniture items in images. Such portability of learned features across different problems is a key advantage of deep learning compared to many older, shallow learning approaches, and it makes deep learning very effective for small-data problems.

In this case, let’s consider a large ConvNet trained on the ImageNet dataset (1.4 million labeled images and 1,000 different classes). ImageNet co

### PDF p.288 (printed 264) — 8 Image classification › 8.3 Using a pretrained model › Fine-tuning a pretrained model
Another widely used technique for model reuse, complementary to feature extraction, is fine-tuning (see figure 8.15). Fine-tuning consists of unfreezing the frozen model base used for feature extraction and jointly training both the newly added part of the model (in this case, the fully connected classifier) and the base model. This is called fine-tuning because it slightly adjusts the more abstract representations of the model being reused to make them more relevant for the problem at hand.

We stated earlier that it’s necessary to freeze the pretrained convolution base first to be able to train a randomly initialized classifier on top. For the same reason, it’s only possible to fine-tune the convolutional base once the classifier on top has already been trained. If the classifier isn’t already trained, then the error signal propagating through the network during training will be too large, and the representations previously learned by the layers being fine-tuned will be destroyed. Thus, the steps for fine-tuning a network are as follows: 1 Add your custom network on top of an already trained base network. 2 Freeze the base network. 3 Train the part you added. 4 Unfreeze the base 

### PDF p.276 (printed 252) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
We get a test accuracy of 78.6% (due to the randomness of neural network initializations, you may get numbers within 1 percentage point of that).

Because you have relatively few training samples (2,000), overfitting will be your number-one concern. You already know about a number of techniques that can help mitigate overfitting, such as dropout and weight decay (L2 regularization). We’re now going to work with a new one, specific to computer vision and used almost universally when processing images with deep learning models: data augmentation.

### PDF p.135 (printed 111) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › Validating your approach
As you learned in chapter 3, a deep learning model should never be evaluated on its training data—it’s standard practice to use a “validation set” to monitor the accuracy of the model during training. Here, you’ll create a validation set by setting apart 10,000 samples from the original training data.

You might ask, why not simply use the test data to evaluate the model? That seems like it would be easier. The reason is that you’re going to want to use the results you get on the validation set to inform your next choices to improve training—for instance, your choice of what model size to use or how many epochs to train for. When you start doing this, your validation scores stop being an accurate reflection of the performance of the model on brand-new data, since the model has been deliberately modified to perform better on the validation data. It’s good to keep around a set of never-before-seen samples that you can use to perform the final evaluation round in a completely unbiased way, and that’s exactly what the test set is. We’ll talk more about this in the next chapter.

Listing 4.6 Setting aside a validation set
