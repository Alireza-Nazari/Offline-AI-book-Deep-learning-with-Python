# Generation bundle for page_482  (PDF page 482, printed page 458)

Page kind: content
Chapter: 15 Language models and the Transformer
Section: 15.4 Classification with a pretrained Transformer
Subsection: 15.4.3 Preprocessing IMDb movie reviews
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_482.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_482.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p482_e03] (heading_subsection) 15.4.3 Preprocessing IMDb movie reviews
[p482_e04] (body) We can reuse the IMDb loading code we used in chapter 14 unchanged. This will download the movie review data to a `train_dir` and `test_dir` and split a validation dataset into a `val_dir`:
[p482_e05] (code)
```
zip_path = keras.utils.get_file(
    origin="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    fname="imdb",
    extract=True,
)
imdb_extract_dir = pathlib.Path(zip_path) / "aclImdb"
train_dir = pathlib.Path("imdb_train")
test_dir = pathlib.Path("imdb_test")
val_dir = pathlib.Path("imdb_val")
shutil.copytree(imdb_extract_dir / "test", test_dir, dirs_exist_ok=True)
val_percentage = 0.2
for category in ("neg", "pos"):
    src_dir = imdb_extract_dir / "train" / category
    src_files = os.listdir(src_dir)
    random.Random(1337).shuffle(src_files)
    num_val_samples = int(len(src_files) * val_percentage)
    os.makedirs(train_dir / category, exist_ok=True)
    os.makedirs(val_dir / category, exist_ok=True)
    for index, file in enumerate(src_files):
        if index < num_val_samples:
            shutil.copy(src_dir / file, val_dir / category / file)
        else:
            shutil.copy(src_dir / file, train_dir / category / file)
from keras.utils import text_dataset_from_directory
batch_size = 16
train_ds = text_dataset_from_directory(train_dir, batch_size=batch_size)
val_ds = text_dataset_from_directory(val_dir, batch_size=batch_size)
test_ds = text_dataset_from_directory(test_dir, batch_size=batch_size)
```
[p482_e06] (body) After loading, we once again have a training set of 20,000 movie reviews and a validation set of 5,000 movie reviews.
[p482_e07] (body) Before fine-tuning our classification model, we must tokenize our movie reviews with the RoBERTa tokenizer we loaded. During pretraining, RoBERTa used a specific form of “packing” tokens into a sequence, similar to what we did for our translation model. Each sequence would begin with an `<s>` token, end with an `</s>` token, and be followed by any number of `<pad>` tokens like this:

## Chapter outline
Chapter: 15 Language models and the Transformer (PDF pages 445-489)
  - 15.1 The language model (p.445)
      - 15.1.1 Training a Shakespeare language model (p.446)
      - 15.1.2 Generating Shakespeare (p.450)
  - 15.2 Sequence-to-sequence learning (p.452)
      - 15.2.1 English-to-Spanish translation (p.454)
      - 15.2.2 Sequence-to-sequence learning with RNNs (p.456)
  - 15.3 The Transformer architecture (p.461)
      - 15.3.1 Dot-product attention (p.463)
      - 15.3.2 Transformer encoder block (p.468)
      - 15.3.3 Transformer decoder block (p.470)
      - 15.3.4 Sequence-to-sequence learning with a Transformer (p.472)
      - 15.3.5 Embedding positional information (p.475)
  - 15.4 Classification with a pretrained Transformer (p.478)
      - 15.4.1 Pretraining a Transformer encoder (p.478)
      - 15.4.2 Loading a pretrained Transformer (p.479)
      - 15.4.3 Preprocessing IMDb movie reviews (p.482)
      - 15.4.4 Fine-tuning a pretrained Transformer (p.484)
  - 15.5 What makes the Transformer effective? (p.485)

## Previous page (PDF 481) text, for continuity
a backbone refers to any pretrained model that is not yet specialized for a task. The model we just loaded takes in an input sequence and embeds it to an output sequence with shape `(batch_size, sequence_length, 768)`, but it’s not set up for a particular loss function. You could use it for any number of downstream tasks—classifying sentences, identifying text spans with certain information, identifying parts of speech, etc.

Next, we will attach a classification head to this backbone that specializes it for our IMDb review classification fine-tuning. You can think of this as attaching different heads to a screwdriver: a Phillips head for one task, a flat head for another. Let’s take a look at our backbone. We loaded the smallest variant of RoBERTa here, but it still has 124 million parameters, which is the biggest model we have used in this book:

RoBERTa uses 12 Transformer encoder layers stacked on top of each other. That’s a big step up from our translation model!

## Next page (PDF 483) text, for continuity
It’s important to match the token ordering used for pretraining as closely as possible; the model will train more quickly and accurately if we do. KerasHub provides a layer for this type of token packing called the `StartEndPacker`. The layer appends start, end, and padding tokens, trimming long sequences to a given sequence length if necessary. Let’s use it along with our tokenizer.

Listing 15.21 Preprocessing IMDb movie reviews with RoBERTa’s tokenizer

Let’s take a look at a single preprocessed batch:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.421 — 14 Text classification › 14.3 Sets vs. sequences
- PDF p.420 — 14 Text classification › 14.3 Sets vs. sequences
- PDF p.422 — 14 Text classification › 14.4 Set models
- PDF p.130 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.454 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.427 — 14 Text classification › 14.4 Set models
- PDF p.429 — 14 Text classification › 14.5 Sequence models
- PDF p.430 — 14 Text classification › 14.5 Sequence models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.130 (printed 106) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › The IMDb dataset
You’ll work with the IMDb dataset: a set of 50,000 highly polarized reviews from the Internet Movie Database. They’re split into 25,000 reviews for training and 25,000 reviews for testing, each set consisting of 50% negative and 50% positive reviews.

Just like the MNIST dataset, the IMDb dataset comes packaged with Keras. It has already been preprocessed: the reviews (sequences of words) have been turned into sequences of integers, where each integer stands for a specific word in a dictionary. This enables us to focus on model building, training, and evaluation. In chapter 14, you’ll learn how to process raw text input from scratch. The following code will load the dataset (when you run it the first time, about 80 MB of data will be downloaded to your machine).

Listing 4.1 Loading the IMDb dataset

### PDF p.420 (printed 396) — 14 Text classification › 14.3 Sets vs. sequences › 14.3.1 Loading the IMDb classification dataset
Listing 14.8 Downloading the IMDb movie review dataset
import os, pathlib, shutil, random
zip_path = keras.utils.get_file(
    origin="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    fname="imdb",
    extract=True,
)
imdb_extract_dir = pathlib.Path(zip_path) / "aclImdb"

### PDF p.420 (printed 396) — 14 Text classification › 14.3 Sets vs. sequences › 14.3.1 Loading the IMDb classification dataset
To begin, let’s download and extract our dataset.

Listing 14.8 Downloading the IMDb movie review dataset

### PDF p.421 (printed 397) — 14 Text classification › 14.3 Sets vs. sequences › 
Before we begin tokenizing our input text, we will make a copy of our training data with a few important modifications. We can ignore the unsupervised reviews for now and create a separate validation set to monitor our accuracy while training. We do this by splitting 20% of the training text files into a new directory.

Listing 14.10 Splitting validation from the IMDb dataset

### PDF p.429 (printed 405) — 14 Text classification › 14.5 Sequence models › 14.5 Sequence models
This will allow our batch computation to proceed much faster, although we will need to be careful with our padding tokens to ensure they do not affect the quality of our model predictions.

To keep a manageable input size, we can truncate our IMDb reviews after the first 600 words. This is a reasonable choice, since the average review length is 233 words, and only 5% of reviews are longer than 600 words. Once again, we can use the `Vectorization` layer, which has an option for padding or truncating inputs and includes a `"[PAD]"` at index zero of the learned vocabulary.

### PDF p.430 (printed 406) — 14 Text classification › 14.5 Sequence models › 
Listing 14.19 Padding IMDb reviews to a fixed sequence length

### PDF p.440 (printed 416) — 14 Text classification › 14.5 Sequence models › 
After preprocessing, we can see that we have eight integer token IDs as context paired with a single token ID label.

The model we train with this data is exceedingly simple. We will use an `Embedding` layer to embed all context tokens and a `GlobalAveragePooling1D` to compute the average embedding of our “bag” of context tokens. Then, we use that average embedding to predict the value of our middle label token. That’s it! By repeatedly refining our embedding space so that we are good at predicting a word based on nearby word embeddings, we learn a rich embedding of tokens used in movie reviews.

Listing 14.28 Building a CBOW model

### PDF p.484 (printed 460) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 15.4.4 Fine-tuning a pretrained Transformer
Before we fine-tune our backbone to predict movie reviews, we need to update it so it outputs a binary classification label. The backbone outputs an entire sequence with shape `(batch_size, sequence_length, 768)`, where each 768-dimensional vector represents an input word in the context of its surrounding words. Before predicting a label, we must condense this sequence to a single vector per sample.

One option would be to do mean pooling or max pooling across the whole sequence, computing an average of all token vectors. What works slightly better is simply using the first token’s representation as the pooled value. This is due to the nature of the attention in our model—the first position in the final encoder layer will be able to attend to all other positions in the sequence and pull information from them. So rather than pooling information with something coarse, like taking an average, attention allows us to pool information contextually across the sequence. Let’s now add a classification head to our backbone. We will also add one final `Dense` projection with a nonlinearity before generating an output prediction.
