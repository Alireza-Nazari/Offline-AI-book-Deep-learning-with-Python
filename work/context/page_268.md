# Generation bundle for page_268  (PDF page 268, printed page 244)

Page kind: content
Chapter: 8 Image classification
Section: 8.2 Training a convnet from scratch on a small dataset
Subsection: 8.2.2 Downloading the data
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_268.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_268.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p268_e03] (body) Unsurprisingly, the original dogs-versus-cats Kaggle competition, all the way back in 2013, was won by entrants who used ConvNets. The best entries achieved up to 95% accuracy. In this example, we will get fairly close to this accuracy (in the next section), even though we will train our models on less than 10% of the data that was available to the competitors.
[p268_e04] (body) This dataset contains 25,000 images of dogs and cats (12,500 from each class) and is 543 MB (compressed). After downloading and uncompressing the data, we’ll create a new dataset containing three subsets: a training set with 1,000 samples of each class, a validation set with 500 samples of each class, and a test set with 1,000 samples of each class. Why do this? Because many of the image datasets you’ll encounter in your career only contain a few thousand samples, not tens of thousands. Having more data available would make the problem easier—so it’s good practice to learn with a small dataset. The subsampled dataset we will work with will have the following directory structure:
[p268_e05] (code)
```
dogs_vs_cats_small/
...train/
......cat/
......dog/
...validation/
......cat/
......dog/
...test/
......cat/
......dog/
```
[p268_e06] (body) Let’s make it happen in a couple of calls to `shutil`, a Python library for running shell-like commands.
[p268_e07] (listing_caption) Listing 8.6 Copying images to training, validation, and test directories
[p268_e08] (code)
```
import os, shutil, pathlib
original_dir = pathlib.Path("train")
new_base_dir = pathlib.Path("dogs_vs_cats_small")
def make_subset(subset_name, start_index, end_index):
    for category in ("cat", "dog"):
        dir = new_base_dir / subset_name / category
        os.makedirs(dir)
```

## Code annotations printed next to listings (callouts, in order)
- Contains 1,000 cat images
- Contains 1,000 dog images
- Contains 500 cat images
- Contains 500 dog images
- Contains 1,000 cat images
- Contains 1,000 dog images
- Utility function to copy cat (respectively, dog) images from
- index start_index to index end_index to the subdirectory
- new_base_dir/{subset_name}/cat (respectively, dog).
- “subset_name” will be either “train,” “validation,” or “test.”
- Path to the directory where the
- original dataset was uncompressed
- Directory where
- we will store our
- smaller dataset

## Listings
- Listing 8.6: Copying images to training, validation, and test directories (code elements p268_e08)

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

## Previous page (PDF 267) text, for continuity
1 Go to https://www.kaggle.com/c/dogs-vs-cats/rules and click to Join the Competition. 2 Go to https://www.kaggle.com/settings and generate a Kaggle API key. With that we are ready to download the data in our notebook. First, log in with your Kaggle API key:

Then, download the competition data:

This downloads two new files, `train.zip` (the training data) and `test1.zip` (the test data). We’ll only use the training data here. Let’s unzip it:

All done!

The pictures in our dataset are medium-resolution color JPEGs. Figure 8.8 shows some examples.

Figure 8.8 Samples from the Dogs vs. Cats dataset. Sizes weren’t modified: the samples come in different sizes, colors, backgrounds, etc.

## Next page (PDF 269) text, for continuity
We now have 2,000 training images, 1,000 validation images, and 2,000 test images. Each split contains the same number of samples from each class: this is a balanced binary classification problem, which means classification accuracy will be an appropriate measure of success.

8.2.3

Building your model

We will reuse the same general model structure you saw in the first example: the ConvNet will be a stack of alternated `Conv2D` (with `relu` activation) and `MaxPooling2D` layers.

But because we’re dealing with bigger images and a more complex problem, we’ll make our model larger, accordingly: it will have two more `Conv2D MaxPooling2D` stages. This serves both to augment the capacity of the model and to further reduce the size of the feature maps so they aren’t overly large when we reach the pooling layer. Here, because we start from inputs of size 180 × 180 pixels (a somewhat arbitrary choice), we end up with feature maps of size 7 × 7 just before the `GlobalAveragePooling2D` layer. NOTE The depth of the feature maps progressively increases in the model (from 32 to 512), whereas the size of the feature maps decreases (from 180 × 180 to 7 × 7). This is a pattern you’ll see in almost all ConvNets.

Because we’re looking at a binary classification problem, we’ll end the model with a single unit (a `Dense` layer of size 1) and a `sigmoid` activation. This unit will encode the probability that the model is looking at one class or the other.

One last small difference: we will start the model with a `Rescaling` layer, which will rescale image inputs (whose values are originally in the [0, 255] range) to the [0, 1] range.

Listing 8.7 Instantiating a small ConvNet for dogs vs. cats classification

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.290 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.266 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.285 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.265 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.282 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.366 — 12 Object detection › 12.2 Training a YOLO model from scratch
- PDF p.287 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.421 — 14 Text classification › 14.3 Sets vs. sequences

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.265 (printed 241) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › Training a ConvNet from scratch on a small dataset
Having to train an image-classification model using very little data is a common situation, which you’ll likely encounter in practice if you ever do computer vision in a professional context. A “few” samples can mean anywhere from a few hundred to a few tens of thousands of images. As a practical example, we’ll focus on classifying images as dogs or cats. We’ll work with a dataset containing 5,000 pictures of cats and dogs (2,500 cats, 2,500 dogs), taken from the original Kaggle dataset. We’ll use 2,000 pictures for training, 1,000 for validation, and 2,000 for testing.

In this section, we’ll review one basic strategy to tackle this problem: training a new model from scratch using what little data we have. We’ll start by naively training a small ConvNet on the 2,000 training samples, without any regularization, to set a baseline for what can be achieved. This will get us to a classification accuracy of about 80%. At that

### PDF p.290 (printed 266) — 8 Image classification › 8.3 Using a pretrained model › 
Figure 8.15 Training and validation metrics for fine-tuning

Here, you get a test accuracy of 98.6% (again, your own results may be within half a percentage point). In the original Kaggle competition around this dataset, this would have been one of the top results. It’s not quite a fair comparison, however, since you used pretrained features that already contained prior knowledge about cats and dogs, which competitors couldn’t use at the time.

On the positive side, by using modern deep learning techniques, you managed to reach this result using only a small fraction of the training data that was available for the competition (about 10%). There is a huge difference between being able to train on 20,000 samples compared to 2,000 samples! Now you have a solid set of tools for dealing with image-classification problems—in particular, with small datasets.

### PDF p.266 (printed 242) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › Downloading the data
The Dogs vs. Cats dataset that we will use isn’t packaged with Keras. It was made available by Kaggle as part of a computer-vision competition in late 2013, back when ConvNets weren’t mainstream. You can download the original dataset from `www.kaggle
` `.com/c/dogs-vs-cats/data` (you’ll need to create a Kaggle account if you don’t already have one—don’t worry, the process is painless). You can also use the Kaggle API to download the dataset in Colab.

### PDF p.266 (printed 242) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › The relevance of deep learning for small-data problems
What qualifies as “enough samples” to train a model is relative—relative to the size and depth of the model you’re trying to train, for starters. It isn’t possible to train a ConvNet to solve a complex problem with just a few tens of samples, but a few hundred can potentially suffice if the model is small and well regularized and the task is simple. Because ConvNets learn local, translation-invariant features, they’re highly data efficient on perceptual problems. Training a ConvNet from scratch on a very small image dataset will still yield reasonable results despite a relative lack of data, without the need for any custom feature engineering. You’ll see this in action in this section.

What’s more, deep learning models are by nature highly repurposable: you can take, say, an image-classification or speech-to-text model trained on a large-scale dataset and reuse it on a significantly different problem with only minor changes. Specifically, in the case of computer vision, many pretrained classification models are publicly available for download and can be used to bootstrap powerful vision models out of very little data. This is one of the greatest strengths of deep learning: feature

### PDF p.282 (printed 258) — 8 Image classification › 8.3 Using a pretrained model › 
only the first few layers of the model to do feature extraction, rather than using the entire convolutional base.

In this case, because the ImageNet class set contains multiple dog and cat classes, it’s likely to be beneficial to reuse the information contained in the densely connected layers of the original model. But we’ll choose not to, so we can cover the more general case where the class set of the new problem doesn’t overlap the class set of the original model. Let’s put this in practice by using the convolutional base of our pretrained model to extract interesting features from cat and dog images and then train a dogs-versus-cats classifier on top of these features. We will use the KerasHub library to create all pretrained models used in this book. KerasHub contains Keras implementations of popular pretrained model architectures paired with pretrained weights that can be downloaded to your machine. It contains a number of ConvNets like Xception, ResNet, EfficientNet and MobileNet, as well as larger, generative models we will use in the later chapters of this book. Let’s try using it to instantiate the Xception model trained on the ImageNet dataset.

NOTE KerasHub comes as a

### PDF p.12 (printed xii) — contents ›  › 
8.2

Training a ConvNet from scratch on a small dataset 241

The relevance of deep learning for small-data problems 242 Downloading the data 242 ■ Building your model 245 Data preprocessing 247 ■ Using data augmentation 252

8.3

Using a pretrained model 256

Feature extraction with a pretrained model 256 ■ Fine-tuning a pretrained model 264

9

ConvNet architecture patterns 268 9.1

Modularity, hierarchy, and reuse 269 9.2 Residual connections 272 9.3 Batch normalization 276 9.4 Depthwise separable convolutions 278 9.5 Putting it together: A mini Xception-like model 280 9.6 Beyond convolution: Vision Transformers 282 10 Interpreting what ConvNets learn 284 10.1 Visualizing intermediate activations 285 10.2 Visualizing ConvNet filters 291

Gradient ascent in TensorFlow 294 ■ Gradient ascent in PyTorch 295 ■ Gradient ascent in JAX 295 ■ The filter visualization loop 296

10.3

Visualizing heatmaps of class activation 299

Getting the gradient of the top class: TensorFlow version 302 Getting the gradient of the top class: PyTorch version 302 Getting the gradient of the top class: JAX version 303 Displaying the class activation heatmap 304

10.4 Visualizing the latent space of a ConvN

### PDF p.285 (printed 261) — 8 Image classification › 8.3 Using a pretrained model › 
Figure 8.13 Training and validation metrics for plain feature extraction

You reach a validation accuracy of slightly over 98%—much better than you achieved in the previous section with the small model trained from scratch. This is a bit of an unfair comparison, however, because ImageNet contains many dog and cat instances, which means that our pretrained model already has the exact knowledge required for the task at hand. This won’t always be the case when you use pretrained features.

However, the plots also indicate that you’re overfitting almost from the start—despite using dropout with a fairly large rate. That’s because this technique doesn’t use data augmentation, which is essential for preventing overfitting with small image datasets. Let’s check the test accuracy:

### PDF p.335 (printed 311) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 11.2.1 Downloading a segmentation dataset
We’ll work with the Oxford-IIIT Pets dataset (https://www.robots.ox.ac.uk/~vgg/data/ pets/), which contains 7,390 pictures of various breeds of cats and dogs, together with foreground-background segmentation masks for each picture. A segmentation mask is the image segmentation equivalent of a label: it’s an image the same size as the input image, with a single color channel where each integer value corresponds to the class of the corresponding pixel in the input image. In our case, the pixels of our segmentation masks can take one of three integer values:

1 (foreground)

2 (background)

3 (contour)

Let’s start by downloading and uncompressing our dataset, using the `wget` and `tar` shell utilities:
