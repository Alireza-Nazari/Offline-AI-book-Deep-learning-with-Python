# Generation bundle for page_486  (PDF page 486, printed page 462)

Page kind: content
Chapter: 15 Language models and the Transformer
Section: 15.5 What makes the Transformer effective?
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_486.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_486.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p486_e03] (body) be closer to 1, while the dot-product between vectors representing rarely co-occurring words would be closer to 0.
[p486_e04] (body) They found that the resulting embedding space did much more than capture semantic similarity. It featured some form of emergent learning—a sort of “word arithmetic.” A vector existed in the space that you could add to many male nouns to obtain a point that would land close to its female equivalent, as in `V(king) - V(man) + V(woman) = ` `V(queen)`, a gender vector. This was quite surprising; the model had not been trained for this in any explicit way. There seemed to be dozens of such magic vectors—a plural vector, a vector to go from wild animals’ names to their closest pet equivalent, etc. Fast-forward about 10 years—we are now in the age of large, pretrained Transformer models. On the surface, these models couldn’t seem any further from the primitive Word2Vec model. A Transformer can generate perfectly fluent language—a feat Word2Vec was entirely incapable of. As we will see in the next chapter, such models can seem knowledgeable about virtually any topic. And yet, they actually have a lot in common with good old Word2Vec. Both models seek to embed tokens (words or subwords) in a vector space. Both rely on the same fundamental principle to learn this space: tokens that appear together end up close in the embedding space. The distance function used to compare tokens is cosine distance in both cases. Even the dimensionality of the embedding space is similar: a vector with somewhere between 1,000 and 10,000 dimensions to represent each word. At this point, you might interject: A Transformer is trained to predict missing words in a sequence, not to group tokens in an embedding space. How does the language model loss function relate to Word2Vec’s objective of maximizing the dot-product between co-occurring tokens? The answer is the attention mechanism. Attention is, by far, the most critical component in the Transformer architecture. It’s a mechanism for learning a new token embedding space by linearly recombining token embeddings from some prior space, in weighted combinations that give greater importance to tokens that are already “closer” to each other (i.e., that have a higher dot-product). It will tend to pull together the vectors of already close tokens, resulting over time in a space where token correlation relationships turn into embedding proximity relationships (in terms of cosine distance). Transformers work by learning a series of incrementally refined embedding spaces, each based on recombining elements from the previous one. Attention provides Transformers with two crucial properties:
[p486_e05] (bullet) The embedding spaces they learn are semantically continuous—that is, moving a bit in an embedding space only changes the human-facing meaning of the corresponding tokens by a bit. The Word2Vec space also exhibited this property.
[p486_e06] (bullet) The embedding spaces they learn are semantically interpolative—that is, taking the intermediate point between two points in an embedding space produces a point representing the “intermediate meaning” between the corresponding

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

## Previous page (PDF 485) text, for continuity
Listing 15.22 Extending the base RoBERTa model for classification

With that, we are ready to fine-tune and evaluate the model on the IMDb dataset.

Listing 15.23 Training the RoBERTa classification model

Finally, let’s evaluate the trained model:

In just a single epoch of training, our model reached 93%, a noticeable improvement from the 90% ceiling we hit in our last chapter. Of course, this is a far more expensive model to use than the simple bigram classifier we built previously, but there are clear benefits to using such a large model. And this is all with the smaller size of the RoBERTa model. Using the larger 300 million parameter model, we could achieve an accuracy of over 95%.

15.5 What makes the Transformer effective?

In 2013, at Google, Tomas Mikolov and his colleagues noticed something remarkable. They were building a pretrained embedding called “Word2Vec,” similar to the Continuous Bag of Words (CBOW) embedding we built in the last chapter. Much like our CBOW model, their training objective sought to turn correlation relationships between words into distance relationships in the embedding space: a vector was associated with each word in a vocabulary, and the vectors were optimized so that the dot-product (cosine proximity) between vectors representing frequently co-occurring words would

## Next page (PDF 487) text, for continuity
tokens. This comes from the fact that each new embedding space is built by interpolating between vectors from the previous space. This is not entirely unlike the way the brain learns. The key learning principle in the brain is Hebbian learning—in short, “neurons that fire together, wire together.” Correlation relationships between neural firing events (which may represent actions or perceptual inputs) are turned into proximity relationships in the brain network, just like the Transformer and Word2Vec turn correlation relationships into vector proximity relationships. Both are maps of a space of information. Of course, there are significant differences between Word2Vec and the Transformer. Word2Vec was not designed for generative text sampling. A Transformer can get far bigger and can encode vastly more complex transformations. The thing is, Word2Vec is very much a toy model: it is to today’s language models as a logistic regression on MNIST pixels is to state-of-the-art computer vision models. The fundamental principles are mostly the same, but the toy model lacks any meaningful representation power. Word2Vec wasn’t even a deep neural network—it had a shallow, single-layer architecture. Meanwhile, today’s Transformer models have the highest representation power of any model anyone has ever trained—they feature dozens of stacked attention and feedforward layers, and their parameter count ranges in the billions. Like with Word2Vec, the Transformer learns useful semantic functions as a by-product of organizing tokens into a vector space. But thanks to this increased representation power and a much more refined autoregressive optimization objective, we’re no longer confined to linear transformations like a gender vector or a plural vector. Transformers can store arbitrarily …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.488 — 15 Language models and the Transformer › 15.5 What makes the Transformer effective?
- PDF p.434 — 14 Text classification › 14.5 Sequence models
- PDF p.433 — 14 Text classification › 14.5 Sequence models
- PDF p.474 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.438 — 14 Text classification › 14.5 Sequence models
- PDF p.437 — 14 Text classification › 14.5 Sequence models
- PDF p.435 — 14 Text classification › 14.5 Sequence models
- PDF p.475 — 15 Language models and the Transformer › 15.3 The Transformer architecture

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.488 (printed 464) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › 
internet, there is a lot of data: facts, places, people, dates, things, and relationships. But it’s also—perhaps primarily—a database of programs.

They’re different from the kind of programs you’re used to dealing with, mind you. These are not like Python programs—series of symbolic statements processing data step by step. Instead, these vector programs are highly nonlinear functions that map the latent embedding space unto itself, analogous to Word2Vec’s magic vectors, but far more complex. In the next chapter, we will push Transformer models to an entirely new scale. Models will use billions of parameters and train on trillions of words. Output from these models can often feel like magic—like an intelligent operator sitting inside our model and pulling the strings. But it’s important to remember that these models are fundamentally interpolative—thanks to attention, they learn an interpolative embedding space for a significant chunk of all text written in the English language. Wandering this embedding space can lead to interesting, unexpected generalizations, but it cannot synthesize something fundamentally new with anything close to genuine, human-level intelligence.

### PDF p.434 (printed 410) — 14 Text classification › 14.5 Sequence models › 
Besides being dense representations, word embeddings are also structured representations, and their structure is learned from data. Similar words get embedded in close locations, and further, specific directions in the embedding space are meaningful. To make this clearer, let’s look at a concrete example. In figure 14.4, four words are embedded on a 2D plane: cat, dog, wolf, and tiger. With the vector representations we chose here, some semantic relationships

Figure 14.4 A toy

between these words can be encoded as geometric transfor

example of a word-

mations. For instance, the same vector allows us to go from

embedding space

cat to tiger and from dog to wolf: this vector could be interpreted as the “from pet to wild animal” vector. Similarly, another vector lets us go from dog to cat and from wolf to tiger, which could be interpreted as a “from canine to feline” vector.

In real-world word-embedding spaces, typical examples of meaningful geometric transformations are “gender” vectors and “plural” vectors. For instance, by adding a “female” vector to the vector “king,” we obtain the vector “queen.” By adding a “plural” vector, we obtain “kings.” Word-embedding spaces typicall

### PDF p.433 (printed 409) — 14 Text classification › 14.5 Sequence models › 14.5.2 Understanding word embeddings
When you encode something via one-hot encoding, you’re making a feature engineering decision. You’re injecting into your model a fundamental assumption about the structure of your feature space. That assumption is that the different tokens you’re encoding are all independent from each other: indeed, one-hot vectors are all orthogonal to one another. In the case of words, that assumption is clearly wrong. Words form a structured space: they share information with each other. The words “movie” and “film” are interchangeable in most sentences, so the vector that represents “movie” should not be orthogonal to the vector that represents “film”—they should be the same vector, or close enough.

To get more abstract, the geometric relationship between two-word vectors should reflect the semantic relationship between these words. For instance, in a reasonable word vector space, you would expect synonyms to be embedded into similar word vectors, and in general, you would expect the geometric distance (such as the cosine distance or L2 distance) between any two-word vectors to relate to the “semantic distance” between the associated words. Words that mean different things should lie far away 

### PDF p.437 (printed 413) — 14 Text classification › 14.5 Sequence models › 
With the embedding, we have reduced both our training time and model size by an order of magnitude. A learned embedding is clearly far more efficient than one-hot encoding our input.

However, the LSTM’s overall performance did not change. Accuracy was stubbornly around 84%, still a far cry from the bag-of-words and bigram models. Does this mean that a “structured embedding space” for input tokens is not that practically useful? Or is it not useful for text classification tasks? Quite the contrary, a well-trained token embedding space can dramatically improve the practical performance ceiling of a model like this. The issue in this particular case is with our training setup. We lack enough data in our 20,000 review examples to effectively train a good word embedding. By the end of our 10 training epochs, our train set accuracy has cracked 99%. Our model has begun to overfit and memorize our input, and it turns out it is doing so well before we have learned an optimal set of word embeddings for the task at hand. For cases like this, we can turn to pretraining. Rather than training our word embedding jointly with the classification task, we can train it separately, on more data, with

### PDF p.499 (printed 475) — 16 Text generation › 16.2 Training a mini-GPT › 
prediction layer because they deal with our vocabulary space. The token embedding weight has shape `(vocab_size, hidden_dim)` to embed every possible token. Our output projection has shape `(hidden_dim, vocab_size)` to make a floating-point prediction for every possible token.

We can actually tie these two weight matrices together. To compute our model’s final predictions, we will multiply our hidden states by the transpose of our token embedding matrix. You can very much think of our final projection as a “reverse embedding.” It maps from hidden space to token space, whereas an embedding maps from token space to hidden space. It turns out that using the same weights for this input and output projection is a good idea. Adding this to our `PositionalEmbedding` is simple; we will just add a `reverse` argument to the `call` method, which computes the projection by the transpose of the token embedding.

Listing 16.5 A positional embedding layer that can reverse a text embedding

### PDF p.434 (printed 410) — 14 Text classification › 14.5 Sequence models › 14.5.3 Using a word embedding
Is there an ideal word-embedding space that perfectly maps human language and can be used for any NPL task? Possibly, but we have yet to compute anything of the sort. Also, there is no single human language we could attempt to map—there are many different languages, and they aren’t isomorphic to one another because a language is the reflection of a specific culture and a particular context. More pragmatically, what makes a good word-embedding space depends heavily on your task: the perfect word-embedding space for an English-language movie review sentiment-analysis model may look different from the ideal embedding space for an English-language legal document classification model because the importance of certain semantic relationships varies from task to task.

It’s thus reasonable to learn a new embedding space with every new task. Fortunately, backpropagation makes this easy, and Keras makes it even easier. It’s about learning the weights of the Keras `Embedding` layer. The `Embedding` layer is best understood as a dictionary that maps integer indices (which stand for specific words) to dense vectors. It takes integers as input, looks them up in an internal dictionary, and return

### PDF p.475 (printed 451) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 15.3.5 Embedding positional information
The idea behind a positional embedding is very simple: to give the model access to word order information, we will add the word’s position in the sentence to each word embedding. Our input word embeddings will have two components: the usual word vector, which represents the word independently of any specific context, and a position vector, which represents the position of the word in the current sentence. Hopefully, the model will then figure out how to best use this additional information.

The most straightforward scheme to add position information would be concatenating each word’s position to its embedding vector. You’d add a “position” axis to the vector and fill it with `0` for the first word in the sequence, `1` for the second, and so on. However, that may not be ideal because the positions can potentially be very large integers, which will disrupt the range of values in the embedding vector. As you know, neural networks don’t like very large input values or discrete input distributions. The “Attention is all you need” authors used an interesting trick to encode word positions: they added to the word embeddings a vector containing values in the range `[-1, 1]` that varied cy

### PDF p.440 (printed 416) — 14 Text classification › 14.5 Sequence models › 
After preprocessing, we can see that we have eight integer token IDs as context paired with a single token ID label.

The model we train with this data is exceedingly simple. We will use an `Embedding` layer to embed all context tokens and a `GlobalAveragePooling1D` to compute the average embedding of our “bag” of context tokens. Then, we use that average embedding to predict the value of our middle label token. That’s it! By repeatedly refining our embedding space so that we are good at predicting a word based on nearby word embeddings, we learn a rich embedding of tokens used in movie reviews.

Listing 14.28 Building a CBOW model
