# Generation bundle for page_425  (PDF page 425, printed page 401)

Page kind: content
Chapter: 14 Text classification
Section: 14.4 Set models
Subsection: 14.4.1 Training a bag-of-words model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_425.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_425.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p425_e03] (code)
```
    bag_of_words_train_ds,
    validation_data=bag_of_words_val_ds,
    epochs=10,
    callbacks=[early_stopping],
)
```
[p425_e04] (body) Our model trains in much less than a minute, which is unsurprising given its size. The tokenization and encoding of our input is actually quite a bit more expensive than updating our model parameters. Let’s plot the model accuracy (figure 14.2):
[p425_e05] (code)
```
import matplotlib.pyplot as plt
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "r--", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.show()
```
[p425_e06] (figure_caption) Figure 14 2 Training and validation metrics for our bag of words model
[p425_e07] (body) We can see that validation performance levels off rather than significantly declining; our model is so simple it cannot really overfit. Let’s try evaluating it on our test set.

## Figures
- Figure None: Figure 14 2 Training and validation metrics for our bag of words model  | labels inside the figure: (none - image)

## Equation candidates (formulas written inline in the text)
- [p425_e03]     bag_of_words_train_ds,
    validation_data=bag_of_words_val_ds,
    epochs=10,
    callbacks=[early_stopping],
)

## Chapter outline
Chapter: 14 Text classification (PDF pages 405-444)
  - 14.1 A brief history of natural language processing (p.405)
  - 14.2 Preparing text data (p.408)
      - 14.2.1 Character and word tokenization (p.411)
      - 14.2.2 Subword tokenization (p.414)
  - 14.3 Sets vs. sequences (p.419)
      - 14.3.1 Loading the IMDb classification dataset (p.420)
  - 14.4 Set models (p.422)
      - 14.4.1 Training a bag-of-words model (p.423)
      - 14.4.2 Training a bigram model (p.427)
  - 14.5 Sequence models (p.429)
      - 14.5.1 Training a recurrent model (p.430)
      - 14.5.2 Understanding word embeddings (p.433)
      - 14.5.3 Using a word embedding (p.434)
      - 14.5.4 Pretraining a word embedding (p.438)
      - 14.5.5 Using the pretrained embedding for classification (p.442)

## Previous page (PDF 424) text, for continuity
You can see that after preprocessing, each sample in our batch is converted into a vector of 20,000 numbers, each tracking the presence or absence of a vocabulary term.

Next, we can train a very simple linear model. We will save our model-building code as a function so we can use it again later.

Listing 14.13 Building a bag-of-words regression model

Let’s take a look at our model’s summary:

This model is dead simple. We have only 20,001 parameters, one for each word in our vocabulary and one for a bias term. Let’s train it. We’ll add on the `EarlyStopping` callback first covered in chapter 7, which will automatically stop when training when the validation loss stops improving and restore weights from the best epoch.

Listing 14.14 Training the bag-of-words regression model

## Next page (PDF 426) text, for continuity
Listing 14.15 Evaluating the bag-of-words regression model

We can correctly predict the sentiment of a review 88% of the time with a training job light enough that it could run efficiently on a single CPU.

It is worth noting our choice of word tokenization in this example. The reason to avoid character-level tokenization here is pretty obvious—a “bag” of all characters in a movie review will tell you very little about its content. Subword tokenization with a large enough vocabulary would be a good choice, but there is little need for it here. Since the model we are training is so small, it’s convenient to use a vocabulary that is quick to train and have our weights correspond to actual English words.

Preprocessing text efficiently

In all applied machine learning, the speed and efficiency of preprocessing are important concerns. A faster program is always desirable, but this becomes more urgent when the cost of accelerators (GPUs and TPUs) is so high. You want to avoid letting expensive GPUs idle while you preprocess your input! Text preprocessing is unique because it must always run on a CPU. GPUs strictly handle numeric inputs, so all tokenization must happen before your GPU’s train step. One option is to precompute your tokenized input—tokenization does not depend on model weights, so you could tokenize all input text files and resave them as integer sequences before starting training. However, this is not always practical. Tokenizing text on the fly allows for more rapid experimentation. If you are running inference on an unseen example, there is no way to precompute the tokenized input; you need to tokenize and run a forward pass in rapid succession. The name of the game when preprocessing text input on the fly is to be “fast enough.” You want to ensure your exp …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.285 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.144 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.275 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.437 — 14 Text classification › 14.5 Sequence models
- PDF p.146 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.137 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.157 — 4 Classification and regression › 4.3 Predicting house prices: A regression example
- PDF p.156 — 4 Classification and regression › 4.3 Predicting house prices: A regression example

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.428 (printed 404) — 14 Text classification › 14.4 Set models › 
This model will be slightly larger than our bag-of words models (30,001 parameters instead of 20,001 parameters), but it still trains in about the same amount of time. How did it do?

Listing 14.18 Evaluating the bigram regression model

### PDF p.422 (printed 398) — 14 Text classification › 14.4 Set models › 14.4 Set models
The simplest approach we can take regarding the ordering of tokens in text is to discard it. We still tokenize our input reviews normally as a sequence of token IDs, but immediately after tokenization, we convert the entire training example to a set—a simple unordered “bag” of tokens that are either present or absent in a movie review.

The idea here is to use these sets to build a very simple model that assigns a weight to every individual word in a review. The presence of the word `"terrible"` would probably (though not always) indicate a bad review, and `"riveting"` might indicate a good review. We can build a small model that can learn these weights—called a bag-of-words model. For example, let’s say you had a simple input sentence and vocabulary:

### PDF p.437 (printed 413) — 14 Text classification › 14.5 Sequence models › 
With the embedding, we have reduced both our training time and model size by an order of magnitude. A learned embedding is clearly far more efficient than one-hot encoding our input.

However, the LSTM’s overall performance did not change. Accuracy was stubbornly around 84%, still a far cry from the bag-of-words and bigram models. Does this mean that a “structured embedding space” for input tokens is not that practically useful? Or is it not useful for text classification tasks? Quite the contrary, a well-trained token embedding space can dramatically improve the practical performance ceiling of a model like this. The issue in this particular case is with our training setup. We lack enough data in our 20,000 review examples to effectively train a good word embedding. By the end of our 10 training epochs, our train set accuracy has cracked 99%. Our model has begun to overfit and memorize our input, and it turns out it is doing so well before we have learned an optimal set of word embeddings for the task at hand. For cases like this, we can turn to pretraining. Rather than training our word embedding jointly with the classification task, we can train it separately, on more data, with

### PDF p.423 (printed 399) — 14 Text classification › 14.4 Set models › 14.4.1 Training a bag-of-words model
Let’s look a single batch of our preprocessed input data:

### PDF p.432 (printed 408) — 14 Text classification › 14.5 Sequence models › 
This model works, but it trains very slowly, especially compared to the lightweight model of the previous section. That’s because our inputs are quite large: each input sample is encoded as a matrix of size `(600, 30000)` (600 words per sample, 30,000 possible words). That is 18,000,000 floating-point numbers for a single movie review! Our bidirectional LSTM has a lot of work to do. In addition to being slow, the model only gets to 84% test accuracy—it doesn’t perform nearly as well as our very fast set-based models.

### PDF p.157 (printed 133) — 4 Classification and regression › 4.3 Predicting house prices: A regression example › 
According to this plot (see figure 4.11), validation MAE stops improving significantly after 120–140 epochs (this number includes the 10 epochs we omitted). Past that point, you start overfitting.

Once you’re finished tuning other parameters of the model (in addition to the number of epochs, you could also adjust the size of the intermediate layers), you can train a final production model on all of the training data, with the best parameters, and then look at its performance on the test data.

Figure 4.11 Validation MAE by epoch, excluding the first 10 data points

Listing 4.33 Training the final model

### PDF p.423 (printed 399) — 14 Text classification › 14.4 Set models › 14.4.1 Training a bag-of-words model
To do this text processing in code, it would be easy enough to extend our `WordTokenizer` from earlier in the chapter. An even easier solution is to use the `TextVectorization` layer built into Keras. The `TextVectorization` handles word and character tokenization and comes with several additional features, including multi-hot encoding of the layer output.

The `TextVectorization` layer, like many preprocessing layers in Keras, has an `adapt()` method to learn a layer state from input data. In the case of `TextVectorization`, `adapt()` will learn a vocabulary for a dataset on the fly by iterating over an input dataset. Let’s use it to tokenize and encode our input data. We will build a vocabulary of 20,000 words, a good starting place for text classification problems.

Listing 14.12 Applying a bag-of-words encoding to the IMDb reviews

### PDF p.13 (printed xiii) — contents ›  › 
14.3

Sets vs. sequences 395

Loading the IMDb classification dataset 396

14.4

Set models 398

Training a bag-of-words model 399 ■ Training a bigram model 403
