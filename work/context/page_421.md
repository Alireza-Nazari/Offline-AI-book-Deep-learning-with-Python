# Generation bundle for page_421  (PDF page 421, printed page 397)

Page kind: content
Chapter: 14 Text classification
Section: 14.3 Sets vs. sequences
Subsection: 14.3.1 Loading the IMDb classification dataset
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_421.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_421.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p421_e03] (listing_caption) Listing 14.9 Previewing a single IMDb review
[p421_e04] (code)
```
>>> print(open(imdb_extract_dir / "train" / "pos" / "4077_10.txt", "r").read())
I first saw this back in the early 90s on UK TV, i did like it then but i missed
the chance to tape it, many years passed but the film always stuck with me and i
lost hope of seeing it TV again, the main thing that stuck with me was the end,
the hole castle part really touched me, its easy to watch, has a great story,
great music, the list goes on and on, its OK me saying how good it is but
everyone will take there own best bits away with them once they have seen it,
yes the animation is top notch and beautiful to watch, it does show its age in a
very few parts but that has now become part of it beauty, i am so glad it has
came out on DVD as it is one of my top 10 films of all time. Buy it or rent it
just see it, best viewing is at night alone with drink and food in reach so you
don't have to stop the film.<br/>
<br/>Enjoy
```
[p421_e05] (body) Before we begin tokenizing our input text, we will make a copy of our training data with a few important modifications. We can ignore the unsupervised reviews for now and create a separate validation set to monitor our accuracy while training. We do this by splitting 20% of the training text files into a new directory.
[p421_e06] (listing_caption) Listing 14.10 Splitting validation from the IMDb dataset
[p421_e07] (code)
```
train_dir = pathlib.Path("imdb_train")
test_dir = pathlib.Path("imdb_test")
val_dir = pathlib.Path("imdb_val")
shutil.copytree(imdb_extract_dir / "test", test_dir)
val_percentage = 0.2
for category in ("neg", "pos"):
    src_dir = imdb_extract_dir / "train" / category
    src_files = os.listdir(src_dir)
    random.Random(1337).shuffle(src_files)
    num_val_samples = int(len(src_files) * val_percentage)
    os.makedirs(val_dir / category)
    for file in src_files[:num_val_samples]:
        shutil.copy(src_dir / file, val_dir / category / file)
    os.makedirs(train_dir / category)
    for file in src_files[num_val_samples:]:
        shutil.copy(src_dir / file, train_dir / category / file)
```
[p421_e08] (body) We are now ready to load the data. Remember how, in chapter 8, we used the `image_dataset_from_directory` utility to create a `Dataset` of images and their labels for a directory structure? You can do the exact same thing for text files using the `text_dataset_from_directory` utility. Let’s create three `Dataset` objects for training, validation, and testing.

## Code annotations printed next to listings (callouts, in order)
- Moves the test
- data unaltered
- Splits the training
- data into a train set
- and a validation set

## Listings
- Listing 14.9: Previewing a single IMDb review (code elements p421_e04)
- Listing 14.10: Splitting validation from the IMDb dataset (code elements p421_e07)

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

## Previous page (PDF 420) text, for continuity
Historically, most early applications of machine learning to NLP just involved bag-of-words models that discarded sequence data. Interest in sequence models only started rising in 2015, with the rebirth of RNNs. Today, both approaches remain relevant. Let’s see how they work and when to use which. We will demonstrate each approach on a well-known text classification benchmark: the IMDb movie review sentiment-classification dataset. In chapters 4 and 5, you worked with a prevectorized version of the IMDb dataset; now let’s process the raw IMDb text data, just like you would do when approaching a new text-classification problem in the real world.

14.3.1 Loading the IMDb classification dataset

To begin, let’s download and extract our dataset.

Listing 14.8 Downloading the IMDb movie review dataset

Let’s list out our directory structure:

We can see both a train and test set with positive and negative examples. Movie reviews with a low user rating on the IMDb site were sorted into the `neg/` directory and those with a high rating into the `pos/` directory. We can also see an `unsup/` directory, which is short for unsupervised. These are reviews deliberately left unlabeled by the dataset creator; they could be negative or positive reviews.

Let’s look at the content of a few of these text files. Whether you’re working with text or image data, remember to inspect what your data looks like before you dive into modeling. It will ground your intuition about what your model is actually doing.

## Next page (PDF 422) text, for continuity
Listing 14.11 Loading the IMDb dataset for use with Keras

Originally we had 25,000 training and testing examples each, and after our validation split, we have 20,000 reviews to train on and 5,000 for validation. Let’s try learning something from this data.

14.4 Set models

The simplest approach we can take regarding the ordering of tokens in text is to discard it. We still tokenize our input reviews normally as a sequence of token IDs, but immediately after tokenization, we convert the entire training example to a set—a simple unordered “bag” of tokens that are either present or absent in a movie review.

The idea here is to use these sets to build a very simple model that assigns a weight to every individual word in a review. The presence of the word `"terrible"` would probably (though not always) indicate a bad review, and `"riveting"` might indicate a good review. We can build a small model that can learn these weights—called a bag-of-words model. For example, let’s say you had a simple input sentence and vocabulary:

We would tokenize this tiny review as

Discarding order, we can turn this into a set of token IDs:

Finally, we could use a multi-hot encoding to transform the set to a fixed-sized vector with the same length as a vocabulary:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.482 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer
- PDF p.130 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.268 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.141 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.274 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.496 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.271 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.427 — 14 Text classification › 14.4 Set models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.130 (printed 106) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › The IMDb dataset
You’ll work with the IMDb dataset: a set of 50,000 highly polarized reviews from the Internet Movie Database. They’re split into 25,000 reviews for training and 25,000 reviews for testing, each set consisting of 50% negative and 50% positive reviews.

Just like the MNIST dataset, the IMDb dataset comes packaged with Keras. It has already been preprocessed: the reviews (sequences of words) have been turned into sequences of integers, where each integer stands for a specific word in a dictionary. This enables us to focus on model building, training, and evaluation. In chapter 14, you’ll learn how to process raw text input from scratch. The following code will load the dataset (when you run it the first time, about 80 MB of data will be downloaded to your machine).

Listing 4.1 Loading the IMDb dataset

### PDF p.482 (printed 458) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 15.4.3 Preprocessing IMDb movie reviews
We can reuse the IMDb loading code we used in chapter 14 unchanged. This will download the movie review data to a `train_dir` and `test_dir` and split a validation dataset into a `val_dir`:

### PDF p.13 (printed xiii) — contents ›  › 
14.3

Sets vs. sequences 395

Loading the IMDb classification dataset 396

14.4

Set models 398

Training a bag-of-words model 399 ■ Training a bigram model 403

### PDF p.130 (printed 106) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › The IMDb dataset
Listing 4.1 Loading the IMDb dataset
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000
)

### PDF p.482 (printed 458) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 15.4.3 Preprocessing IMDb movie reviews
After loading, we once again have a training set of 20,000 movie reviews and a validation set of 5,000 movie reviews.

Before fine-tuning our classification model, we must tokenize our movie reviews with the RoBERTa tokenizer we loaded. During pretraining, RoBERTa used a specific form of “packing” tokens into a sequence, similar to what we did for our translation model. Each sequence would begin with an `<s>` token, end with an `</s>` token, and be followed by any number of `<pad>` tokens like this:

### PDF p.366 (printed 342) — 12 Object detection › 12.2 Training a YOLO model from scratch › 
Lastly, let’s use `tf.data` to load our image data. We will load our images from disk, apply our preprocessing, and batch them. We should also split a validation set to monitor training.

Listing 12.9 Creating a dataset to train on

### PDF p.173 (printed 149) — 5 Fundamentals of machine learning › 5.2 Evaluating machine-learning models › Training, validation, and test sets
But if you repeat this many times— running one experiment, evaluating on the validation set, and modifying your model as a result—then you’ll leak an increasingly significant amount of information about the validation set into the model. At the end of the day, you’ll end up with a model that performs artificially well on the validation data because that’s what you optimized it for. You care about performance on completely new data, not the validation data, so you need to use a completely different, never-before-seen dataset to evaluate the model: the test dataset. Your model shouldn’t have had access to any information about the test set, even indirectly. If anything about the model has been tuned based on test set performance, then your measure of generalization will be flawed. Splitting your data into training, validation, and test sets may seem straightforward, but there are a few advanced ways to do it that can come in handy when little data is available. Let’s review three classic evaluation recipes: simple hold-out validation, K-fold validation, and iterated K-fold validation with shuffling. We’ll also

### PDF p.177 (printed 153) — 5 Fundamentals of machine learning › 5.3 Improving model fit › 
ordered by their class, taking the first 80% of the array as your training set and the remaining 20% as your test set will result in your training set containing only classes 0–7, whereas your test set contains only classes 8–9. This seems like a ridiculous mistake, but it’s surprisingly common. For this reason, you usually should randomly shuffle your data before splitting it into training and test sets.

The arrow of time—If you’re trying to predict the future given the past (for example, tomorrow’s weather, stock movements, and so on), you should not randomly shuffle your data before splitting it because doing so will create a temporal leak: your model will effectively be trained on data from the future. In such situations, you should always make sure all data in your test set is posterior to the data in the training set.

Redundancy in your data—If some data points in your data appear twice (fairly common with real-world data), then shuffling the data and splitting it into a training set and a validation set will result in redundancy between the training and validation sets. In effect, you’ll be testing on part of your training data, which is the worst thing you can do! Make su
