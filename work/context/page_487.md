# Generation bundle for page_487  (PDF page 487, printed page 463)

Page kind: content
Chapter: 15 Language models and the Transformer
Section: 15.5 What makes the Transformer effective?
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_487.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_487.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p487_e03] (body) tokens. This comes from the fact that each new embedding space is built by interpolating between vectors from the previous space. This is not entirely unlike the way the brain learns. The key learning principle in the brain is Hebbian learning—in short, “neurons that fire together, wire together.” Correlation relationships between neural firing events (which may represent actions or perceptual inputs) are turned into proximity relationships in the brain network, just like the Transformer and Word2Vec turn correlation relationships into vector proximity relationships. Both are maps of a space of information. Of course, there are significant differences between Word2Vec and the Transformer. Word2Vec was not designed for generative text sampling. A Transformer can get far bigger and can encode vastly more complex transformations. The thing is, Word2Vec is very much a toy model: it is to today’s language models as a logistic regression on MNIST pixels is to state-of-the-art computer vision models. The fundamental principles are mostly the same, but the toy model lacks any meaningful representation power. Word2Vec wasn’t even a deep neural network—it had a shallow, single-layer architecture. Meanwhile, today’s Transformer models have the highest representation power of any model anyone has ever trained—they feature dozens of stacked attention and feedforward layers, and their parameter count ranges in the billions. Like with Word2Vec, the Transformer learns useful semantic functions as a by-product of organizing tokens into a vector space. But thanks to this increased representation power and a much more refined autoregressive optimization objective, we’re no longer confined to linear transformations like a gender vector or a plural vector. Transformers can store arbitrarily complex vector functions—so complex, in fact, that it would be more accurate to refer to them as vector programs rather than functions. Word2Vec enabled you to do basic things like `plural(cat)` `→` `cats` or `male_to_` `female(king)` `→` `queen`. Meanwhile, a large Transformer model can do pure magic— things like `write_this_in_style_of_shakespeare("...your` `poem...")` `→` `"...new ` `poem..."`. And a single model can contain millions of such programs. You can see a Transformer as analogous to a database: it stores information you can retrieve via the tokens you pass in. But there are two important differences between a Transformer and a database. The first difference is that a Transformer is a continuous, interpolative kind of database. Instead of being stored as a set of discrete entries, your data is stored as a vector space—a curve. You can move around on the curve (it’s semantically continuous, as we discussed) to explore nearby, related points. And you can interpolate on the curve between different data points to find their in-between. This means that you can retrieve a lot more from your database than you put into it—although not all of it will be accurate or meaningful. Interpolation can lead to generalization, but it can also lead to hallucinations—a significant problem facing the generative language models trained today. The second difference is that a Transformer doesn’t just contain data. For models like RoBERTa, trained on hundreds of thousands of documents scraped from the

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

## Previous page (PDF 486) text, for continuity
be closer to 1, while the dot-product between vectors representing rarely co-occurring words would be closer to 0.

They found that the resulting embedding space did much more than capture semantic similarity. It featured some form of emergent learning—a sort of “word arithmetic.” A vector existed in the space that you could add to many male nouns to obtain a point that would land close to its female equivalent, as in `V(king) - V(man) + V(woman) = ` `V(queen)`, a gender vector. This was quite surprising; the model had not been trained for this in any explicit way. There seemed to be dozens of such magic vectors—a plural vector, a vector to go from wild animals’ names to their closest pet equivalent, etc. Fast-forward about 10 years—we are now in the age of large, pretrained Transformer models. On the surface, these models couldn’t seem any further from the primitive Word2Vec model. A Transformer can generate perfectly fluent language—a feat Word2Vec was entirely incapable of. As we will see in the next chapter, such models can seem knowledgeable about virtually any topic. And yet, they actually have a lot in common with good old Word2Vec. Both models seek to embed tokens (words or subwords) in a vector space. Both rely on the same fundamental principle to learn this space: tokens that appear together end up close in the embedding space. The distance function used to compare tokens is cosine distance in both cases. Even the dimensionality of the embedding space is similar: a vector with somewhere between 1,000 and 10,000 dimensions to represent each word. At this point, you might interject: A Transformer is trained to predict missing words in a sequence, not to group tokens in an embedding space. How does the language model loss function relate to Word2Vec’s objective o …

## Next page (PDF 488) text, for continuity
internet, there is a lot of data: facts, places, people, dates, things, and relationships. But it’s also—perhaps primarily—a database of programs.

They’re different from the kind of programs you’re used to dealing with, mind you. These are not like Python programs—series of symbolic statements processing data step by step. Instead, these vector programs are highly nonlinear functions that map the latent embedding space unto itself, analogous to Word2Vec’s magic vectors, but far more complex. In the next chapter, we will push Transformer models to an entirely new scale. Models will use billions of parameters and train on trillions of words. Output from these models can often feel like magic—like an intelligent operator sitting inside our model and pulling the strings. But it’s important to remember that these models are fundamentally interpolative—thanks to attention, they learn an interpolative embedding space for a significant chunk of all text written in the English language. Wandering this embedding space can lead to interesting, unexpected generalizations, but it cannot synthesize something fundamentally new with anything close to genuine, human-level intelligence.

Summary

A language model is a model that learns a specific probability distribution— `p(token|past tokens)`: – Language models have broad applications, but the most important is that you can generate text by calling them in a loop, where the output token at one time step becomes the input token in the next. – A masked language model learns a related probability distribution `p(tokens|surrounding` `tokens)` and can be helpful for classifying text and individual tokens. – A sequence-to-sequence language model learns to predict the next token given both past tokens in a target sequence and an entirely sep …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.478 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer
- PDF p.485 — 15 Language models and the Transformer › 15.5 What makes the Transformer effective?
- PDF p.474 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.593 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.489 — 15 Language models and the Transformer › 15.5 What makes the Transformer effective?
- PDF p.469 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.522 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.476 — 15 Language models and the Transformer › 15.3 The Transformer architecture

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.485 (printed 461) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › 15.5 What makes the Transformer effective?
In 2013, at Google, Tomas Mikolov and his colleagues noticed something remarkable. They were building a pretrained embedding called “Word2Vec,” similar to the Continuous Bag of Words (CBOW) embedding we built in the last chapter. Much like our CBOW model, their training objective sought to turn correlation relationships between words into distance relationships in the embedding space: a vector was associated with each word in a vocabulary, and the vectors were optimized so that the dot-product (cosine proximity) between vectors representing frequently co-occurring words would

### PDF p.478 (printed 454) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
Subjectively, the Transformer performs significantly better than the GRU-based translation model. It’s still a toy model, but it’s a better toy model.

The Transformer is a powerful architecture that has laid the basis for an explosion of interest in text-processing models. It’s also fairly complex, as deep learning models go. After seeing all of these implementation details, one might reasonably protest that this all seems quite arbitrary. There are so many small details to take on faith. How could we possibly know this choice and configuration of layers is optimal? The answer is simple—it’s not. Over the years, a number of improvements have been proposed to the Transformer architecture by making changes to attention, normalization, and positional embeddings. Many new models in research today are replacing attention altogether with something less computationally complex as sequence lengths get very long. Eventually, perhaps by the time you are reading this book, something will have supplanted the Transformer as the dominant architecture used for language modeling. There’s a lot we can learn from the Transformer that will stand the test of time. At the end of this chapter, we will 

### PDF p.434 (printed 410) — 14 Text classification › 14.5 Sequence models › 
Besides being dense representations, word embeddings are also structured representations, and their structure is learned from data. Similar words get embedded in close locations, and further, specific directions in the embedding space are meaningful. To make this clearer, let’s look at a concrete example. In figure 14.4, four words are embedded on a 2D plane: cat, dog, wolf, and tiger. With the vector representations we chose here, some semantic relationships

Figure 14.4 A toy

between these words can be encoded as geometric transfor

example of a word-

mations. For instance, the same vector allows us to go from

embedding space

cat to tiger and from dog to wolf: this vector could be interpreted as the “from pet to wild animal” vector. Similarly, another vector lets us go from dog to cat and from wolf to tiger, which could be interpreted as a “from canine to feline” vector.

In real-world word-embedding spaces, typical examples of meaningful geometric transformations are “gender” vectors and “plural” vectors. For instance, by adding a “female” vector to the vector “king,” we obtain the vector “queen.” By adding a “plural” vector, we obtain “kings.” Word-embedding spaces typicall

### PDF p.627 (printed 603) — 20 Conclusions › 20.1 Key concepts in review › Transformers
A Transformer looks at a set of vectors (such as word vectors) and uses neural attention to transform each vector into a representation that is aware of the context provided by the other vectors in the set. When the set in question is an ordered sequence, you can also use positional encoding to create Transformers that can take into account both global context and word order, capable of processing long text paragraphs much more effectively than RNNs or 1D ConvNets.

Transformers can be used for any set-processing or sequence-processing task, including text classification, but they excel especially at sequence-to-sequence learning, such as translating paragraphs in a source language into a target language. A sequence-to-sequence Transformer is made of two parts:

A `TransformerEncoder` that turns an input vector sequence into a context-aware, order-aware output vector sequence

A `TransformerDecoder` that takes the output of the `TransformerEncoder`, as well as a target sequence, and predicts what should come next in the target sequence

If you’re only processing a single sequence (or set) of vectors, you’d only use the `TransformerEncoder`.

Following is a sequence-to-sequence Tran

### PDF p.489 (printed 465) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › 
The Transformer is a sequence modeling architecture that uses attention as the only mechanism to pass information across a sequence: – The Transformer works by stacking blocks of alternating attention and two-layer feedforward networks. – The Transformer can scale to many parameters and lots of training data while still improving accuracy in the language modeling problem. – Unlike RNNs, the Transformer involves no sequence-length loops at training time, making the model much easier to train in parallel across many machines. – A Transformer encoder uses bidirectional attention to build a rich representation of a sequence. – A Transformer decoder uses causal attention to predict the next word in a language model setup.

### PDF p.593 (printed 569) — 19 The future of AI › 19.1 The limitations of deep learning › 19.1.3 Deep learning models struggle to learn generalizable programs
Consider the problem of adding two numbers, represented as character sequences— like “4 3 5 7 + 8 9 3 6.” Try training a Transformer on hundreds of thousands of such digit pairs: you will reach a very high accuracy. Very high, but not 100%—you will keep regularly seeing incorrect answers, because the Transformer doesn’t manage to encode the exact addition algorithm (you know, the one you learned in primary school). It is instead guessing the output by interpolating between the data points it has seen at training time. This applies to state-of-the-art LLMs, too—at least those that weren’t explicitly hardcoded to execute snippets like “4357 + 8936” in Python to provide the right answer. They’ve seen enough examples of digit addition that they can add numbers, but they only have about 70% accuracy—quite underwhelming. Further, their accuracy is strongly dependent on which digits are being added, with more common digits leading to higher accuracy. The reason why a deep learning model does not end up learning an exact addition algorithm even after seeing millions of examples is that it is just a static chain of simple, continuous geometric transformations mapping one vector space into a

### PDF p.522 (printed 498) — 16 Text generation › 16.4 Going further with LLMs › 16.4.2 Multimodal LLMs
One obvious chatbot extension is the ability to handle new modalities of input. An assistant that can respond to audio input and process images would be far more useful than one that can only operate on text.

Extending a Transformer to different modalities can be done in a conceptually simple way. The Transformer is not a text-specific model; it’s a highly effective model for learning patterns in sequence data. If we can figure out how to coerce other data types into a sequence representation, we can feed this sequence into a Transformer and train with it. In fact, the Gemma model we just loaded does just that. The model comes with a separate 420-million-parameter image encoder that cuts an input image into 256 patches and encodes each patch as a vector with the same dimensionality as Gemma’s hidden transformer dimension. Each image will be embedded as a `(256,` `2560)` sequence. Because 2560 is the hidden dimensionality of the Gemma Transformer model, this image representation can simply be spliced into our text sequence after the token embedding layer. You can think of it like 256 special tokens representing the image, where each `(1,` `2560)` vector is sometimes called a “soft 

### PDF p.469 (printed 445) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
to a single matrix multiplication! That’s a real problem with the expressiveness of our model.

At some point, all recurrent cells pass the input vector for each token through a dense projection and apply an activation function; we need a plan for something similar. The authors of “Attention is all you need” decided to add this back in the simplest way possible—stacking a feedforward network of two dense layers with an activation in the middle. Attention passes information across the sequence, and the feedforward network updates the representation of individual sequence items. We are ready to start building a Transformer model. Let’s start by replacing the encoder of our translation model. We will use self-attention to pass information along the source sequence of English words. We will also add in two things we learned to be particularly important when building ConvNets back in chapter 9, normalization and residual connections.

Listing 15.14 A Transformer encoder block
