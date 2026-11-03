# Generation bundle for page_433  (PDF page 433, printed page 409)

Page kind: content
Chapter: 14 Text classification
Section: 14.5 Sequence models
Subsection: 14.5.2 Understanding word embeddings
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_433.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_433.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p433_e03] (body) Clearly, using one-hot encoding to turn words into vectors, which was the simplest thing we could do, wasn’t a great idea. There’s a better way—word embeddings.
[p433_e04] (heading_subsection) 14.5.2 Understanding word embeddings
[p433_e05] (body) When you encode something via one-hot encoding, you’re making a feature engineering decision. You’re injecting into your model a fundamental assumption about the structure of your feature space. That assumption is that the different tokens you’re encoding are all independent from each other: indeed, one-hot vectors are all orthogonal to one another. In the case of words, that assumption is clearly wrong. Words form a structured space: they share information with each other. The words “movie” and “film” are interchangeable in most sentences, so the vector that represents “movie” should not be orthogonal to the vector that represents “film”—they should be the same vector, or close enough.
[p433_e06] (body) To get more abstract, the geometric relationship between two-word vectors should reflect the semantic relationship between these words. For instance, in a reasonable word vector space, you would expect synonyms to be embedded into similar word vectors, and in general, you would expect the geometric distance (such as the cosine distance or L2 distance) between any two-word vectors to relate to the “semantic distance” between the associated words. Words that mean different things should lie far away from each other, whereas related words should be closer. Word embeddings are vector representations of words that achieve precisely this: they map human language into a structured geometric space. Whereas the vectors obtained through
[p433_e07] (body) one-hot
[p433_e08] (body) encoding
[p433_e09] (body) are binary, sparse (mostly made of zeros), and very high-dimensional (the same dimensionality as the number of words in the vocabulary), word embeddings are low-dimensional floating-point vectors (that is, dense vectors, as opposed to sparse vectors); see figure 14.3. It’s common to see word embeddings that are 256-dimensional, 512-dimensional, or 1,024-dimensional when dealing with very large vocabularies. On the other hand, one-hot encoding words generally leads to vectors that are 30,000-dimensional in the case
[p433_e10] (figure_caption) Figure 14.3 Word representations obtained from one-
[p433_e11] (body) of our current vocabulary. So word
[p433_e12] (figure_caption) hot encoding or hashing are sparse, high-dimensional,
[p433_e13] (body) embeddings pack more information
[p433_e14] (figure_caption) and hardcoded. Word embeddings are dense, relatively
[p433_e15] (body) into far fewer dimensions.
[p433_e16] (figure_caption) low-dimensional, and learned from data.

## Figures
- Figure 14.3: Word representations obtained from one-  | labels inside the figure: (none - image)
- Figure None: hot encoding or hashing are sparse, high-dimensional,  | labels inside the figure: (none - image)
- Figure None: and hardcoded. Word embeddings are dense, relatively  | labels inside the figure: (none - image)
- Figure None: low-dimensional, and learned from data.  | labels inside the figure: (none - image)

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

## Previous page (PDF 432) text, for continuity
This is quite the step up in size from the unigram and bigram models. At about 15 million parameters, this is one of the larger models we have trained in the book so far, with only a single LSTM layer. Let’s trying training the model.

Listing 14.22 Training the LSTM sequence model

How does it perform?

Listing 14.23 Evaluating the LSTM sequence model

This model works, but it trains very slowly, especially compared to the lightweight model of the previous section. That’s because our inputs are quite large: each input sample is encoded as a matrix of size `(600, 30000)` (600 words per sample, 30,000 possible words). That is 18,000,000 floating-point numbers for a single movie review! Our bidirectional LSTM has a lot of work to do. In addition to being slow, the model only gets to 84% test accuracy—it doesn’t perform nearly as well as our very fast set-based models.

## Next page (PDF 434) text, for continuity
Besides being dense representations, word embeddings are also structured representations, and their structure is learned from data. Similar words get embedded in close locations, and further, specific directions in the embedding space are meaningful. To make this clearer, let’s look at a concrete example. In figure 14.4, four words are embedded on a 2D plane: cat, dog, wolf, and tiger. With the vector representations we chose here, some semantic relationships

Figure 14.4 A toy

between these words can be encoded as geometric transfor

example of a word-

mations. For instance, the same vector allows us to go from

embedding space

cat to tiger and from dog to wolf: this vector could be interpreted as the “from pet to wild animal” vector. Similarly, another vector lets us go from dog to cat and from wolf to tiger, which could be interpreted as a “from canine to feline” vector.

In real-world word-embedding spaces, typical examples of meaningful geometric transformations are “gender” vectors and “plural” vectors. For instance, by adding a “female” vector to the vector “king,” we obtain the vector “queen.” By adding a “plural” vector, we obtain “kings.” Word-embedding spaces typically feature thousands of such interpretable and potentially useful vectors. Let’s look at how to use such an embedding space in practice.

14.5.3 Using a word embedding

Is there an ideal word-embedding space that perfectly maps human language and can be used for any NPL task? Possibly, but we have yet to compute anything of the sort. Also, there is no single human language we could attempt to map—there are many different languages, and they aren’t isomorphic to one another because a language is the reflection of a specific culture and a particular context. More pragmatically, what makes a good  …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.486 — 15 Language models and the Transformer › 15.5 What makes the Transformer effective?
- PDF p.419 — 14 Text classification › 14.3 Sets vs. sequences
- PDF p.437 — 14 Text classification › 14.5 Sequence models
- PDF p.423 — 14 Text classification › 14.4 Set models
- PDF p.142 — 4 Classification and regression › 4.2 Classifying newswires: A multiclass classification example
- PDF p.435 — 14 Text classification › 14.5 Sequence models
- PDF p.475 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.438 — 14 Text classification › 14.5 Sequence models

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.475 (printed 451) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 15.3.5 Embedding positional information
The idea behind a positional embedding is very simple: to give the model access to word order information, we will add the word’s position in the sentence to each word embedding. Our input word embeddings will have two components: the usual word vector, which represents the word independently of any specific context, and a position vector, which represents the position of the word in the current sentence. Hopefully, the model will then figure out how to best use this additional information.

The most straightforward scheme to add position information would be concatenating each word’s position to its embedding vector. You’d add a “position” axis to the vector and fill it with `0` for the first word in the sequence, `1` for the second, and so on. However, that may not be ideal because the positions can potentially be very large integers, which will disrupt the range of values in the embedding vector. As you know, neural networks don’t like very large input values or discrete input distributions. The “Attention is all you need” authors used an interesting trick to encode word positions: they added to the word embeddings a vector containing values in the range `[-1, 1]` that varied cy

### PDF p.486 (printed 462) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › 
The distance function used to compare tokens is cosine distance in both cases. Even the dimensionality of the embedding space is similar: a vector with somewhere between 1,000 and 10,000 dimensions to represent each word. At this point, you might interject: A Transformer is trained to predict missing words in a sequence, not to group tokens in an embedding space. How does the language model loss function relate to Word2Vec’s objective of maximizing the dot-product between co-occurring tokens? The answer is the attention mechanism. Attention is, by far, the most critical component in the Transformer architecture. It’s a mechanism for learning a new token embedding space by linearly recombining token embeddings from some prior space, in weighted combinations that give greater importance to tokens that are already “closer” to each other (i.e., that have a higher dot-product). It will tend to pull together the vectors of already close tokens, resulting over time in a space where token correlation relationships turn into embedding proximity relationships (in terms of cosine distance). Transformers work by learning a series of incrementally refined embedding spaces, each based on recombi

### PDF p.485 (printed 461) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › 15.5 What makes the Transformer effective?
In 2013, at Google, Tomas Mikolov and his colleagues noticed something remarkable. They were building a pretrained embedding called “Word2Vec,” similar to the Continuous Bag of Words (CBOW) embedding we built in the last chapter. Much like our CBOW model, their training objective sought to turn correlation relationships between words into distance relationships in the embedding space: a vector was associated with each word in a vocabulary, and the vectors were optimized so that the dot-product (cosine proximity) between vectors representing frequently co-occurring words would

### PDF p.437 (printed 413) — 14 Text classification › 14.5 Sequence models › 
With the embedding, we have reduced both our training time and model size by an order of magnitude. A learned embedding is clearly far more efficient than one-hot encoding our input.

However, the LSTM’s overall performance did not change. Accuracy was stubbornly around 84%, still a far cry from the bag-of-words and bigram models. Does this mean that a “structured embedding space” for input tokens is not that practically useful? Or is it not useful for text classification tasks? Quite the contrary, a well-trained token embedding space can dramatically improve the practical performance ceiling of a model like this. The issue in this particular case is with our training setup. We lack enough data in our 20,000 review examples to effectively train a good word embedding. By the end of our 10 training epochs, our train set accuracy has cracked 99%. Our model has begun to overfit and memorize our input, and it turns out it is doing so well before we have learned an optimal set of word embeddings for the task at hand. For cases like this, we can turn to pretraining. Rather than training our word embedding jointly with the classification task, we can train it separately, on more data, with

### PDF p.486 (printed 462) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › 
They found that the resulting embedding space did much more than capture semantic similarity. It featured some form of emergent learning—a sort of “word arithmetic.” A vector existed in the space that you could add to many male nouns to obtain a point that would land close to its female equivalent, as in `V(king) - V(man) + V(woman) = ` `V(queen)`, a gender vector. This was quite surprising; the model had not been trained for this in any explicit way. There seemed to be dozens of such magic vectors—a plural vector, a vector to go from wild animals’ names to their closest pet equivalent, etc. Fast-forward about 10 years—we are now in the age of large, pretrained Transformer models. On the surface, these models couldn’t seem any further from the primitive Word2Vec model. A Transformer can generate perfectly fluent language—a feat Word2Vec was entirely incapable of. As we will see in the next chapter, such models can seem knowledgeable about virtually any topic. And yet, they actually have a lot in common with good old Word2Vec. Both models seek to embed tokens (words or subwords) in a vector space. Both rely on the same fundamental principle to learn this space: tokens that appear to

### PDF p.419 (printed 395) — 14 Text classification › 14.3 Sets vs. sequences › 14.3 Sets vs. sequences
How a machine learning model should represent individual tokens is a relatively uncontroversial question: they’re categorical features (values from a predefined set), and we know how to handle those. They should be encoded as dimensions in a feature space or as category vectors (token vectors in this case). A much more problematic question, however, is how to encode the ordering of tokens in text.

The problem of order in natural language is an interesting one: unlike the steps of a timeseries, words in a sentence don’t have a natural, canonical order. Different languages order similar words in very different ways. For instance, the sentence structure of English is quite different from that of Japanese. Even within a given language, you can typically say the same thing in different ways by reshuffling the words a bit. If you were to fully randomize the words in a short sentence, you could still sometimes figure out what it was saying—though, in many cases, significant ambiguity would arise. Order is clearly important, but its relationship to meaning isn’t straightforward. How to represent word order is the pivotal question from which different kinds of NLP architectures spring. The

### PDF p.439 (printed 415) — 14 Text classification › 14.5 Sequence models › 
text in a dataset, where we continuously attempt to guess a missing word based on the words that appear to its direct right and left (figure 14.6). For example, if our “bag” of surrounding words contained the words “sail,” “wave,” and “mast,” we might guess that the middle word is “boat” or “ocean.”

Figure 14.6 The Continuous Bag of Words predicts a word based on its surrounding context with a shallow neural network.

In our particular IMDb classification problem, we are interested in “priming” the word embedding of the LSTM model we just trained. We can reuse the `TextVectorization` vocabulary we computed earlier. All we are trying to do here is to learn a good 64-dimensional vector for each word in this vocabulary.

We can create a new `TextVectorization` layer with the same vocabulary that does not truncate or pad input. We will preprocess the output tokens of this layer by sliding a context window across our text.

Listing 14.26 Removing padding from our `TextVectorization` preprocessing layer

### PDF p.427 (printed 403) — 14 Text classification › 14.4 Set models › 14.4.2 Training a bigram model
Of course, we can intuitively guess that discarding all word order is very reductive because even atomic concepts can be expressed via multiple words: the term “United States” conveys a concept that is quite distinct from the meaning of the words “states” and “united” taken separately. A movie that is “not bad” and a movie that is “bad” should probably get different sentiment scores.

Therefore, it is usually a good idea to inject some knowledge of local word ordering into a model, even for these simple set-based models we are currently building. One easy way to do that is to consider bigrams—a term for two tokens that appear consecutively in the input text. Given our example “this movie made me cry,” `{"this",` `"movie",` `"made",` `"me",` `"cry"}` is the set of all word unigrams in the input, and `{"this` `movie",` `"movie` `made",` `"made me",` `"me cry"}` is the set of all bigrams. The bag-of-words model we just trained could equivalently be called a unigram model, and the term n-gram refers to an ordered sequence of n tokens for any n. To add bigrams to our model, we want to consider the frequency of all bigrams while building our vocabulary. We could do this in two ways: by c
