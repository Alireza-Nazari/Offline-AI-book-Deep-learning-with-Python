# Generation bundle for page_461  (PDF page 461, printed page 437)

Page kind: content
Chapter: 15 Language models and the Transformer
Section: 15.3 The Transformer architecture
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_461.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_461.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p461_e03] (body) Note that this inference setup, while very simple, is inefficient, since we reprocess the entire source sentence and the entire generated target sentence every time we sample a new word. In a practical application, you’d want to be careful not to recompute any state that has not changed. All we really need to predict a new token in the decoder is the current token and the previous RNN state, which we could cache before each loop iteration. There are many ways this toy model could be improved. We could use a deep stack of recurrent layers for both the encoder and the decoder, we could try other RNN layers like `LSTM`, and so on. Beyond such tweaks, however, the RNN approach to sequence-to-sequence learning has a few fundamental limitations:
[p461_e04] (bullet) The source sequence representation has to be held entirely in the encoder state vector, which significantly limits the size and complexity of the sentences you can translate.
[p461_e05] (bullet) RNNs have trouble dealing with very long sequences since they tend to progressively forget about the past—by the time you’ve reached the 100th token in either sequence, little information remains about the start of the sequence.
[p461_e06] (body) Recurrent neural networks dominated sequence-to-sequence learning in the mid-2010s. Google Translate circa 2017 was powered by a stack of seven large `LSTM` layers in a setup similar to what we just created. However, these limitations of RNNs eventually led to researchers developing a new style of sequence model, called the Transformer.
[p461_e07] (heading_section) 15.3 The Transformer architecture
[p461_e08] (body) In 2017, Vaswani et al. introduced the Transformer architecture in the seminal paper “Attention Is All You Need.”1 The authors were working on translation systems like the one we just built, and the critical discovery is right in the title. As it turned out, a simple mechanism called attention can be used to construct powerful sequence models that don’t feature recurrent layers at all. The idea of attention was not new and had been used in NLP systems for a couple of years when they published. But the idea that attention was so useful it could be the only mechanism used to pass information across a sequence was quite surprising at the time.
[p461_e09] (body) This finding unleashed nothing short of a revolution in natural language processing—and beyond. Attention has fast become one of the most influential ideas in deep learning. In this section, you’ll get an in-depth explanation of how it works and why it has proven so effective for sequence modeling. We’ll then use attention to rebuild our English-to-Spanish translation model. So, with all that as build-up, what exactly is attention? And how does it offer a replacement for the recurrent neural networks we have used so far? Attention was actually developed as a way to augment an RNN model like the one we just built. Researchers noticed that while RNNs excelled at modeling dependencies in
[p461_e10] (footnote) 1 Vaswani et al., “Attention Is All You Need,” Proceedings of the 31st International Conference on Neural Information Processing Systems (2017), https://arxiv.org/abs/1706.03762.

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

## Previous page (PDF 460) text, for continuity
Listing 15.13 Generating translations with a seq2seq RNN

The exact translations will vary from run to run, as the final model weights will depend on the random initializations of our weights and the random shuffling of our input data. Here’s what we got:

Our model works decently well for a toy model, although it still makes many basic mistakes.

## Next page (PDF 462) text, for continuity
a local neighborhood, they struggled with recall as sequences got longer. Say you were building a system to answer questions about a source document. If the document length got too long, RNN results would get plain bad, a far cry from human performance.

As a thought experiment, imagine using this book to build a weather prediction model. If you had enough time, you might read the entire book cover to cover, but when you actually implemented your model, you would pay special attention to just the timeseries chapter. Even within a chapter, you might find specific code samples and explanations you would refer to often. On the other hand, you would not be particularly worried about the details of image convolutions as you worked on your code. The overall word count of this book is well over 100,000, far beyond any sequence length we have tackled, but humans can be selective and contextual in how we pull information from text. RNNs, on the other hand, lack any mechanism to refer back to a previous section of a sequence directly. All information must, by design, be passed through an RNN cell’s internal state in a loop, through every position in a sequence. It’s a bit like finishing this book, closing it, and trying to implement that weather prediction model entirely from memory. The idea with attention is to build a mechanism by which a neural network can give more weight to some part of a sequence and less weight to others contextually, depending on the current input being processed (figure 15.3).

Figure 15.3 The general concept of attention in deep learning: input features get assigned attention scores, which can be used to inform the next representation of the input.

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.489 — 15 Language models and the Transformer › 15.5 What makes the Transformer effective?
- PDF p.468 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.457 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.488 — 15 Language models and the Transformer › 15.5 What makes the Transformer effective?
- PDF p.474 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.469 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.471 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.478 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.489 (printed 465) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › 
The Transformer is a sequence modeling architecture that uses attention as the only mechanism to pass information across a sequence: – The Transformer works by stacking blocks of alternating attention and two-layer feedforward networks. – The Transformer can scale to many parameters and lots of training data while still improving accuracy in the language modeling problem. – Unlike RNNs, the Transformer involves no sequence-length loops at training time, making the model much easier to train in parallel across many machines. – A Transformer encoder uses bidirectional attention to build a rich representation of a sequence. – A Transformer decoder uses causal attention to predict the next word in a language model setup.

### PDF p.468 (printed 444) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 15.3.2 Transformer encoder block
One way to use the `MultiHeadAttention` layer would be to add it to our existing RNN translation model. We could pass the sequence output from our encoder and decoder into an attention layer and use its output to update our target sequence before prediction. Attention would allow the model to handle long-range dependencies in text that the `GRU` layer will struggle with. This does, in fact, improve an RNN model’s capabilities and is how attention was first used in the mid-2010s.

However, the authors of “Attention is all you need” realized you could go further and use attention as a general mechanism for handling all sequence data in a model. Although so far we have only looked at attention as a way to handle information passing between two sequences, you could also use attention as a way to let a sequence attend to itself:

### PDF p.474 (printed 450) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
After training, we get to about 58% accuracy: on average, the model correctly predicts the next word in the Spanish sentence 58% of the time. Something is off here. Training is worse than the RNN model by 7 percentage points. Either this Transformer architecture is not what it was hyped up to be, or we missed something in our implementation. Can you spot what it is?

This section is ostensibly about sequence models. In the previous chapter, we saw how vital word order could be to meaning. And yet, the Transformer we just built wasn’t a sequence model at all. Did you notice? It’s composed of dense layers that process sequence tokens independently of each other and an attention layer that looks at the tokens as a set. You could change the order of the tokens in a sequence, and you’d get identical pairwise attention scores and the same context-aware representations. If you were to rearrange every word in every English source sentence completely, the model wouldn’t notice, and you’d still get the same accuracy. Attention is a set-processing mechanism, focused on the relationships between pairs of sequence elements—it’s blind to whether these elements occur at the beginning, at the end,

### PDF p.457 (printed 433) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 
However, there is a critical issue with this approach. Due to the step-by-step nature of RNNs, the model will only look at tokens `0...N` in the source sequence to predict token `N` in the target sequence. Consider translating the sentence, “I will bring the bag to you.” In Spanish, that would be “Te traeré la bolsa,” where “Te,” the first word of the translation, corresponds to “you” in the English source text. There’s simply no way to output the first word of the translation without seeing the last word of the source English text!

If you’re a human translator, you’d start by reading the entire source sentence before beginning to translate it. This is especially important if you’re dealing with languages with wildly different word ordering. And that’s precisely what standard sequence-to-sequence models do. In a proper sequence-to-sequence setup (see figure 15.2), you would first use an encoder RNN to turn the entire source sequence into a single representation of the source text. This could be the last output of the RNN or, alternatively, its final internal state vectors. We can use this representation as the initial state of a decoder RNN in the language model setup instead of a

### PDF p.468 (printed 444) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 15.3.2 Transformer encoder block
This is called self-attention, and it is quite powerful. With self-attention, each token can attend to every token in its own sequence, including itself, allowing the model to learn a representation of the word in context.

Consider an example sentence: “The train left the station on time.” Now, consider one word in the sentence: “station.” What kind of station are we talking about? Could it be a radio station? Maybe the International Space Station? With self-attention, the model could learn to give a high attention score to the pair of “station” and “train,” summing the vector used to represent “train” into the representation of the word “station.” Self-attention gives the model an effective way to go from representing a word in a vacuum to representing a word conditioned on all other tokens that appear in the sequence. This sounds a lot like what an RNN is supposed to do. Can we just go ahead and replace our RNN layers with `MultiHeadAttention`? Almost! But not quite; we still need an essential ingredient for any deep neural network—a nonlinear activation function. The `MultiHeadAttention` layer combines linear projections of every element in a source sequence, but that’s it. In 

### PDF p.469 (printed 445) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
to a single matrix multiplication! That’s a real problem with the expressiveness of our model.

At some point, all recurrent cells pass the input vector for each token through a dense projection and apply an activation function; we need a plan for something similar. The authors of “Attention is all you need” decided to add this back in the simplest way possible—stacking a feedforward network of two dense layers with an activation in the middle. Attention passes information across the sequence, and the feedforward network updates the representation of individual sequence items. We are ready to start building a Transformer model. Let’s start by replacing the encoder of our translation model. We will use self-attention to pass information along the source sequence of English words. We will also add in two things we learned to be particularly important when building ConvNets back in chapter 9, normalization and residual connections.

Listing 15.14 A Transformer encoder block

### PDF p.504 (printed 480) — 16 Text generation › 16.2 Training a mini-GPT › Cached generation
There’s still one more major inefficiency in the generation function we just built. Can you spot it? Each time we call our model, we call it for an entire sequence and then throw away everything but the predictions for a single position. This is wasteful—our sequence only changes by a single token between generation steps. When we did generation with an RNN in chapter 15, we could keep around our RNN state and only compute outputs for a single token at each step. This state vector contained all the information the model needed about the past sequence. Transformers that use a causal attention, like GPT, actually have a similar notion of state. If we think through the entire model we just built, you’ll note that attention is the only place where the model passes information from position to position. The feedforward blocks of a transformer only modify the hidden representation of each token position in isolation. Inside attention, we incorporate information about past tokens through the `key` and `value` vectors. For a given `query` at a position, we compute attention scores by dotting the `query` with all previous `key` vectors and combining all previous `value` vectors. These `key`

### PDF p.488 (printed 464) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › Summary
A language model is a model that learns a specific probability distribution— `p(token|past tokens)`: – Language models have broad applications, but the most important is that you can generate text by calling them in a loop, where the output token at one time step becomes the input token in the next. – A masked language model learns a related probability distribution `p(tokens|surrounding` `tokens)` and can be helpful for classifying text and individual tokens. – A sequence-to-sequence language model learns to predict the next token given both past tokens in a target sequence and an entirely separate, fixed source sequence. Sequence-to-sequence models are useful for problems like translation and question answering. – A sequence-to-sequence model usually has two separate components. An encoder computes a representation of the source sequence, and a decoder takes this representation as input and predicts the next token in a target sequence based on past tokens.

Attention is a mechanism that allows a model to pull information from anywhere in a sequence selectively based on the context of the token currently being processed: – Attention avoids the problems RNNs have with long-range de
