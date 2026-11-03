# Generation bundle for page_289  (PDF page 289, printed page 265)

Page kind: content
Chapter: 8 Image classification
Section: 8.3 Using a pretrained model
Subsection: 8.3.2 Fine-tuning a pretrained model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_289.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_289.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p289_e03] (body) Note that you should not unfreeze “batch normalization” layers (`BatchNormalization`). Batch normalization and its effect on fine-tuning is explained in the next chapter.
[p289_e04] (body) You already completed the first three steps when doing feature extraction. Let’s proceed with step 4: you’ll unfreeze your `conv_base`.
[p289_e05] (heading_subsection) Partial fine-tuning
[p289_e06] (sidebar) In this case, we chose to unfreeze and fine-tune all of the Xception convolutional base. However, when dealing with large pretrained models, you may sometimes only unfreeze some of the top layers of the convolutional base, and leave the lower layers frozen. You’re probably wondering, why only fine-tune some of the layers? Why the top ones specifically? Here’s why:
[p289_e07] (bullet) Earlier layers in the convolutional base encode more-generic, reusable features,
[p289_e08] (sidebar) whereas layers higher up encode more-specialized features. It’s more useful to fine-tune the more specialized features because these are the ones that need to be repurposed on your new problem. There would be fast-decreasing returns in fine-tuning lower layers.
[p289_e09] (bullet) The more parameters you’re training, the more you’re at risk of overfitting. The
[p289_e10] (sidebar) convolutional base has 15 million parameters, so it would be risky to attempt to train it on your small dataset. Thus, it can be a good strategy to fine-tune only the top three or four layers in the convolutional base. You’d do something like this:
[p289_e11] (code)
```
conv_base.trainable = True
for layer in conv_base.layers[:-4]:
    layer.trainable = False
```
[p289_e12] (body) Let’s start fine-tuning the model using a very low learning rate. The reason for using a low learning rate is that you want to limit the magnitude of the modifications you make to the representations of the layers you’re fine-tuning. Updates that are too large may harm these representations.
[p289_e13] (listing_caption) Listing 8.31 Fine-tuning the model
[p289_e14] (code)
```
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    metrics=["accuracy"],
)
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="fine_tuning.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]
```

## Listings
- Listing 8.31: Fine-tuning the model (code elements p289_e14)

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

## Previous page (PDF 288) text, for continuity
Figure 8.14 Training and validation metrics for feature extraction with data augmentation

We get a test accuracy of 98.4%. This is not an improvement over the previous model, which is a bit disappointing. This could be a sign that our data augmentation configuration does not exactly match the distribution of the test data. Let’s see if we can do better with our latest attempt.

8.3.2

Fine-tuning a pretrained model

Another widely used technique for model reuse, complementary to feature extraction, is fine-tuning (see figure 8.15). Fine-tuning consists of unfreezing the frozen model base used for feature extraction and jointly training both the newly added part of the model (in this case, the fully connected classifier) and the base model. This is called fine-tuning because it slightly adjusts the more abstract representations of the model being reused to make them more relevant for the problem at hand.

We stated earlier that it’s necessary to freeze the pretrained convolution base first to be able to train a randomly initialized classifier on top. For the same reason, it’s only possible to fine-tune the convolutional base once the classifier on top has already been trained. If the classifier isn’t already trained, then the error signal propagating through the network during training will be too large, and the representations previously learned by the layers being fine-tuned will be destroyed. Thus, the steps for fine-tuning a network are as follows: 1 Add your custom network on top of an already trained base network. 2 Freeze the base network. 3 Train the part you added. 4 Unfreeze the base network. 5 Jointly train both these layers and the part you added.

## Next page (PDF 290) text, for continuity
You can now finally evaluate this model on the test data (see figure 8.15):

Figure 8.15 Training and validation metrics for fine-tuning

Here, you get a test accuracy of 98.6% (again, your own results may be within half a percentage point). In the original Kaggle competition around this dataset, this would have been one of the top results. It’s not quite a fair comparison, however, since you used pretrained features that already contained prior knowledge about cats and dogs, which competitors couldn’t use at the time.

On the positive side, by using modern deep learning techniques, you managed to reach this result using only a small fraction of the training data that was available for the competition (about 10%). There is a huge difference between being able to train on 20,000 samples compared to 2,000 samples! Now you have a solid set of tools for dealing with image-classification problems—in particular, with small datasets.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.286 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.512 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.283 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.280 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.302 — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions
- PDF p.149 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.282 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.266 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.302 (printed 278) — 9 ConvNet architecture patterns › 9.4 Depthwise separable convolutions › 
The intuitive reason why is that batch normalization will center your inputs on zero, while your ReLU activation uses zero as a pivot for keeping or dropping activated channels: doing normalization before the activation maximizes the utilization of the ReLU. That said, this ordering best practice is not exactly critical, so if you do convolution-activation-batch normalization, your model will still train, and you won’t necessarily see worse results.

NOTE Batch normalization has many quirks. One of the main ones relates to fine-tuning: when fine-tuning a model that includes `BatchNormalization` layers, I recommend leaving these layers frozen (set their `trainable` attribute to `False`). Otherwise, they will keep updating their internal mean and variance, which can interfere with the very small updates applied to the surrounding `Conv2D` layers.

Now, let’s take a look at the last architecture pattern in our series: depthwise separable convolutions.

### PDF p.286 (printed 262) — 8 Image classification › 8.3 Using a pretrained model › Feature extraction together with data augmentation
Now, let’s review the second technique we mentioned for doing feature extraction, which is much slower and more expensive but allows you to use data augmentation during training: creating a model that chains the `conv_base` with a new dense classifier and training it end to end on the inputs.

To do this, we will first freeze the convolutional base. Freezing a layer or set of layers means preventing their weights from being updated during training. Here, if you don’t do this, then the representations that were previously learned by the convolutional base will be modified during training. Because the `Dense` layers on top are randomly initialized, very large weight updates would be propagated through the network, effectively destroying the representations previously learned. In Keras, you freeze a layer or model by setting its `trainable` attribute to `False`.

Listing 8.28 Creating the frozen convolutional base

### PDF p.266 (printed 242) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › 
point, the main issue will be overfitting. Then we’ll introduce data augmentation, a powerful technique for mitigating overfitting in computer vision. By using data augmentation, we’ll improve the model to reach a test accuracy of about 84%.

In the next section, we’ll review two more essential techniques for applying deep learning to small datasets: feature extraction with a pretrained model and fine-tuning a pretrained model (which will get us to a final accuracy of 98.5%). Together, these three strategies—training a small model from scratch, doing feature extraction using a pretrained model, and fine-tuning a pretrained model—will constitute your future toolbox for tackling the problem of performing image classification with small datasets.

### PDF p.512 (printed 488) — 16 Text generation › 16.3 Using a pretrained LLM › 16.3.2 Instruction fine-tuning
Instruction fine-tuning involves feeding the model input/output pairs—a user instruction followed by a model response. We combine these into a single sequence that becomes new training data for the model. To make it clear during training when an instruction or response ends, we can add special markers like `"[instruction]"` and `"[response]"` directly to the combined sequence. The precise markup will not matter much as long as it is consistent.

We can use the combined sequence as regular training data, with the same “guess the next word” loss we used to pretrain an LLM. By doing further training with examples containing desired responses, we are essentially bending the model’s output in the direction we want. We won’t be learning a latent space for language here; that’s already been done over trillions of tokens of pretraining. We are simply nudging the learned representation a bit to control the tone and content of the output. To begin, we will need a dataset of instruction-response pairs. Training chatbots is a hot topic, so there are many datasets made specifically for this purpose. We will use a dataset made public by the company Databricks. Employees contributed to a dataset 

### PDF p.531 (printed 507) — 16 Text generation › 16.5 Where are LLMs heading next? › 
LLM output is unreliable, and all LLMs will occasionally hallucinate factually incorrect information.

LLMs can be fine-tuned to follow instructions in a chat dialog. This type of

fine-tuning is called instruction fine-tuning:

– The simplest form of instruction fine-tuning involves directly training the

model on instruction and response pairs. – More advanced forms of instruction fine-tuning involve reinforcement learning.

The most common resource bottleneck when working with LLMs is accelerator memory.

LoRA is a technique to reduce memory usage by freezing most Transformer parameters and only updating a low-rank decomposition of attention projection weights.

LLMs can input or output data from different modalities if you can figure out how to frame these inputs or outputs as sequences in a sequence prediction problem.

A foundation model is a general term for models of any modality trained using self-supervision for a wide range of downstream tasks.

### PDF p.484 (printed 460) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 15.4.4 Fine-tuning a pretrained Transformer
Before we fine-tune our backbone to predict movie reviews, we need to update it so it outputs a binary classification label. The backbone outputs an entire sequence with shape `(batch_size, sequence_length, 768)`, where each 768-dimensional vector represents an input word in the context of its surrounding words. Before predicting a label, we must condense this sequence to a single vector per sample.

One option would be to do mean pooling or max pooling across the whole sequence, computing an average of all token vectors. What works slightly better is simply using the first token’s representation as the pooled value. This is due to the nature of the attention in our model—the first position in the final encoder layer will be able to attend to all other positions in the sequence and pull information from them. So rather than pooling information with something coarse, like taking an average, attention allows us to pool information contextually across the sequence. Let’s now add a classification head to our backbone. We will also add one final `Dense` projection with a nonlinearity before generating an output prediction.

### PDF p.282 (printed 258) — 8 Image classification › 8.3 Using a pretrained model › 
only the first few layers of the model to do feature extraction, rather than using the entire convolutional base.

In this case, because the ImageNet class set contains multiple dog and cat classes, it’s likely to be beneficial to reuse the information contained in the densely connected layers of the original model. But we’ll choose not to, so we can cover the more general case where the class set of the new problem doesn’t overlap the class set of the original model. Let’s put this in practice by using the convolutional base of our pretrained model to extract interesting features from cat and dog images and then train a dogs-versus-cats classifier on top of these features. We will use the KerasHub library to create all pretrained models used in this book. KerasHub contains Keras implementations of popular pretrained model architectures paired with pretrained weights that can be downloaded to your machine. It contains a number of ConvNets like Xception, ResNet, EfficientNet and MobileNet, as well as larger, generative models we will use in the later chapters of this book. Let’s try using it to instantiate the Xception model trained on the ImageNet dataset.

NOTE KerasHub comes as a

### PDF p.280 (printed 256) — 8 Image classification › 8.3 Using a pretrained model › Using a pretrained model
A common and highly effective approach to deep learning on small image datasets is to use a pretrained model. A pretrained model is a model that was previously trained on a large dataset, typically on a large-scale image-classification task. If this original dataset is large enough and general enough, then the spatial hierarchy of features learned by the pretrained model can effectively act as a generic model of the visual world, and hence its features can prove useful for many different computer vision problems, even though these new problems may involve completely different classes than those of the original task. For instance, you might train a model on ImageNet (where classes are mostly animals and everyday objects) and then repurpose this trained model for something as remote as identifying furniture items in images. Such portability of learned features across different problems is a key advantage of deep learning compared to many older, shallow learning approaches, and it makes deep learning very effective for small-data problems.

In this case, let’s consider a large ConvNet trained on the ImageNet dataset (1.4 million labeled images and 1,000 different classes). ImageNet co
