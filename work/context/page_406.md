# Generation bundle for page_406  (PDF page 406, printed page 382)

Page kind: content
Chapter: 14 Text classification
Section: 14.1 A brief history of natural language processing
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_406.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_406.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p406_e03] (body) it’s the reverse: usage comes first, and rules arise later. Natural language was shaped by an evolutionary process, much like biological organisms—that’s what makes it “natural.” Its “rules,” like the grammar of English, were formalized after the fact and are often ignored or broken by its users. As a result, while machine-readable language is highly structured and rigorous, natural language is messy—ambiguous, chaotic, sprawling, and constantly in flux.
[p406_e04] (body) Computer scientists have long fixated on the potential of systems that can ingest or produce natural language. Language, particularly written text, underpins most of our communications and cultural production. Centuries of human knowledge are stored via text; the internet is mostly text, and even our thoughts are based on language! The practice of using computers to interpret and manipulate language is called natural language processing, or NLP for short. It was first proposed as a field of study immediately following World War II, where some thought we could view understanding language as a form of “code cracking,” where natural language is the “code” used to transmit information. In the early days of the field, many people naively thought that you could write down the “rule set of English,” much like one can write down the rule set of LISP. In the early 1950s, researchers at IBM and Georgetown demonstrated a system that could translate Russian into English. The system used a grammar with six hardcoded rules and a lookup table with a couple of hundred elements (words and suffixes) to translate 60 handpicked Russian sentences accurately. The goal was to drum up excitement and funding for machine translation, and in that sense, it was a huge success. Despite the limited nature of the demo, the authors claimed that within five years, translation would be a solved problem. Funding poured in for the better part of a decade. However, generalizing such a system proved to be maddeningly difficult. Words change their meaning dramatically depending on context. Any grammar rules needed countless exceptions. Developing a program that could shine on a few handpicked examples was simple enough, but building a robust system that could compete with human translators was another matter. An influential US report a decade later picked apart the lack of progress, and funding dried up. Despite these setbacks and repeated swings from excitement to disillusionment, handcrafted rules held out as the dominant approach well into the 1990s. The problems were obvious, but there was simply no viable alternative to writing down symbolic rules to describe grammar. However, as faster computers and greater quantities of data became available in the late 1980s, research began to head in a new direction. When you find yourself building systems that are big piles of ad hoc rules, as a clever engineer, you’re likely to start asking, “Could I use a corpus of data to automate the process of finding these rules? Could I search for the rules within some rule space, instead of having to come up with them myself?” And just like that, you’ve graduated to doing machine learning. In the late 1980s, we started seeing machine learning approaches to natural language processing. The earliest ones were based on decision trees—the intent was literally to

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

## Previous page (PDF 405) text, for continuity
Text classification

This chapter covers

An introduction to the field of natural language

processing (NLP)

Preprocessing text input into numeric input

Building simple text classification models

This chapter will lay the foundation for working with text input that we will build on in the next two chapters of this book. By the end of this chapter, you will be able to build a simple text classifier in a number of different ways. This will set the stage for building more complicated models, like the Transformer, in the next chapter.

14.1 A brief history of natural language processing

In computer science, we refer to human languages, like English or Mandarin, as “natural” languages to distinguish them from languages that were designed for machines, like LISP, Assembly, and XML. Every machine language was designed: its starting point was an engineer writing down a set of formal rules to describe what statements you can make and what they mean. The rules came first, and people only started using the language once the rule set was complete. With human language,

## Next page (PDF 407) text, for continuity
automate the development of the kind of if/then/else rules of hardcoded language systems. Then, statistical approaches started gaining speed, starting with logistic regression. Over time, learned parametric models took over, and linguistics came to be seen by some as a hindrance when baked directly into a model. Frederick Jelinek, an early speech recognition researcher, joked in the 1990s, “Every time I fire a linguist, the performance of the speech recognizer goes up.”

Much as computer vision is pattern recognition applied to pixels, the modern field of NLP is all about pattern recognition applied to words in text. There’s no shortage of practical applications:

Given the text of an email, what is the probability that it is spam? (text classification)

Given an English sentence, what is the most likely Russian translation?

(translation)

Given an incomplete sentence, what word will likely come next? (language

modeling)

The text-processing models you will train in this book won’t possess a human-like understanding of language; rather, they simply look for statistical regularities in their input data, which turns out to be sufficient to perform well on a wide array of real-world tasks.

In the last decade, NLP researchers and practitioners have discovered just how shockingly effective it can be to learn the answer to narrow statistical questions about text. In the 2010s, researchers began applying LSTM models to text, dramatically increasing the number of parameters in NLP models and the compute resources required to train them. The results were encouraging—LSTMs could generalize to unseen examples with far greater accuracy than previous approaches, but they eventually hit limits. LSTMs struggled to track dependencies in long chains of text with many sentences and pa …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.445 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.27 — 1 What is deep learning? › 1.3 Machine learning
- PDF p.408 — 14 Text classification › 14.2 Preparing text data
- PDF p.28 — 1 What is deep learning? › 1.4 Learning rules and representations from data
- PDF p.30 — 1 What is deep learning? › 1.4 Learning rules and representations from data
- PDF p.419 — 14 Text classification › 14.3 Sets vs. sequences
- PDF p.443 — 14 Text classification › 14.5 Sequence models
- PDF p.461 — 15 Language models and the Transformer › 15.3 The Transformer architecture

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.27 (printed 3) — 1 What is deep learning? › 1.3 Machine learning › 
knowledge stored in explicit databases. This approach is known as symbolic AI. It was the dominant paradigm in AI from the 1950s to the late 1980s, and it reached its peak popularity during the expert systems boom of the 1980s.

Although symbolic AI proved suitable to solve well-defined, logical problems, such as playing chess, it turned out to be intractable to figure out explicit rules for solving more complex, fuzzy problems, such as image classification, speech recognition, or natural language translation. A new approach arose to take symbolic AI’s place: machine learning.

### PDF p.461 (printed 437) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 15.3 The Transformer architecture
In 2017, Vaswani et al. introduced the Transformer architecture in the seminal paper “Attention Is All You Need.”1 The authors were working on translation systems like the one we just built, and the critical discovery is right in the title. As it turned out, a simple mechanism called attention can be used to construct powerful sequence models that don’t feature recurrent layers at all. The idea of attention was not new and had been used in NLP systems for a couple of years when they published. But the idea that attention was so useful it could be the only mechanism used to pass information across a sequence was quite surprising at the time.

This finding unleashed nothing short of a revolution in natural language processing—and beyond. Attention has fast become one of the most influential ideas in deep learning. In this section, you’ll get an in-depth explanation of how it works and why it has proven so effective for sequence modeling. We’ll then use attention to rebuild our English-to-Spanish translation model. So, with all that as build-up, what exactly is attention? And how does it offer a replacement for the recurrent neural networks we have used so far? Attention was actually 

### PDF p.600 (printed 576) — 19 The future of AI › 19.3 How to build intelligence › 
The progress of AI over the past 70 years bears striking similarities to this evolution. Early AI systems were pure automatons, like the ELIZA chat program from the 1960s, or SHRDLU,2 a 1970 AI capable of manipulating simple objects from natural language commands. In the 1990s and 2000s, we saw the rise of machine learning systems capable of local generalization that could deal with some level of uncertainty and novelty. In the 2010s, deep learning further expanded the local generalization power of these systems by enabling engineers to use much larger datasets and much more expressive models. Today, we may be on the cusp of the next evolutionary step. We are moving toward systems that achieve broad generalization, which I define as the ability to deal with unknown unknowns within a single broad domain of tasks (including situations the system was not trained to handle and that its creators could not have anticipated). Examples are a self-driving car capable of safely dealing with any situation you throw at it or a domestic robot that could pass the “Woz test of intelligence”—entering a random kitchen and making a cup of coffee.3 By combining deep learning and painstakingly handcra

### PDF p.446 (printed 422) — 15 Language models and the Transformer › 15.1 The language model › 
What about other text-based tasks like question answering or translation? For many real-world problems, we are interested in a model that can generate a text output for a given input. Just like we needed tokenizers and embeddings to help us handle text on the way in to a model, we must build up some techniques before we can produce text on the way out. We don’t need to start from scratch here; we can continue to use the idea of an integer sequence as a natural numeric representation for text. In the previous chapter, we covered tokenizing a string, where we split inputs into tokens and map each token to an int. We can detokenize a sequence by proceeding in reverse—map ints back to string tokens and join them together. With this approach, our problem becomes building a model that can predict an integer sequence of tokens. The simplest option to consider might be to train a direct classifier over the space of all possible output integer sequences, but some back-of-the-envelope math will quickly show this is intractable. With a vocabulary of 20,000 words, there are 20,000 ^ 4, or 160 quadrillion possible 4-word sequences, and fewer atoms in the universe than possible 20-word sequences

### PDF p.30 (printed 6) — 1 What is deep learning? › 1.4 Learning rules and representations from data › 
What if we tried systematically searching for different sets of automatically generated representations of the data and rules based on them, identifying good ones using the percentage of digits being correctly classified in some development dataset as feedback? We would then be doing machine learning. Learning, in the context of machine learning, describes an automatic search process for data transformations that produce useful representations of some data, guided by some feedback signal—representations that are amenable to simpler rules solving the task at hand. These transformations can be coordinate changes (like in our 2D coordinates classification example) or a histogram of pixels and counting loops (like in our digits classification example), but they could also be linear projections, translations, and nonlinear operations (such as “Select all points such that x > 0”), and so on. Machine learning algorithms aren’t usually creative in finding these transformations; they’re merely searching through a predefined set of operations, called a hypothesis space. For instance, the space of all possible coordinate changes would be our hypothesis space in the 2D coordinates classificati

### PDF p.36 (printed 12) — 1 What is deep learning? › 1.10 Beware of the short-term hype › 
then making fast progress on natural language processing tasks from 2017 to 2022, and culminating with a wave of transformative generative AI applications from 2022 to now.

Deep learning has enabled major breakthroughs, all in extremely challenging problems that had long eluded machines:

Fluent and highly versatile chatbots such as ChatGPT and Gemini

Programming assistants like GitHub Copilot

Photorealistic image generation

Human-level image classification

Human-level speech transcription

Human-level handwriting transcription and printed text transcription

Dramatically improved machine translation

Dramatically improved text-to-speech conversion

Human-level autonomous driving, already deployed to the public in Phoenix, San Francisco, Los Angeles, and Austin as of 2025

Improved recommender systems, as used by YouTube, Netflix, or Spotify

Superhuman Go, Chess, and Poker playing

We’re still exploring the full extent of what deep learning can do. We’ve started applying it with great success to a wide variety of problems that were thought to be impossible to solve just a few years ago—automatically transcribing the tens of thousands of ancient manuscripts held in the Vatican

### PDF p.445 (printed 421) — 15 Language models and the Transformer › 15.1 The language model › This chapter covers
How to generate text with a deep learning model

Training a model to translate from English to Spanish

The Transformer, a powerful architecture for text

modeling problems

With the basics of text preprocessing and modeling covered in the previous chapter, this chapter will tackle some more involved language problems such as machine translation. We will build up a solid intuition for the Transformer model that powers products like ChatGPT and has helped trigger a wave of investment in natural language processing (NLP).

### PDF p.479 (printed 455) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
model were released a year after “Attention Is All You Need.” The model structure was exactly the same as the encoder part of the translation Transformer we just built. This encoder model is bidirectional in that every position in the sequence can attend to positions in front of and behind it. This means it’s a good model for computing a rich representation of input text, but not a model meant to run generation in a loop.

BERT was trained in sizes between 100 million and 300 million parameters, much bigger than the 14 million parameter Transformer we just trained. This meant the model needed a lot of training data to perform well. To achieve this, the authors used a riff on the classic language modeling setup called masked language modeling. To pretrain the model, we take a sequence of text and replace about 15% of the tokens with a special `[MASK]` token. The model will attempt to predict the original masked token values during training. While the classic language model, sometimes called a causal language model, attempts to predict `p(token|past` `tokens)`, the masked language model attempts to predict `p(token|surrounding` `tokens)`. This training setup is unsupervised. You don’
