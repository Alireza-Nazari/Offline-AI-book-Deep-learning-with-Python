# Generation bundle for page_455  (PDF page 455, printed page 431)

Page kind: content
Chapter: 15 Language models and the Transformer
Section: 15.2 Sequence-to-sequence learning
Subsection: 15.2.1 English-to-Spanish translation
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_455.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_455.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p455_e03] (bullet) We need to preserve the `"[start]"` and `"[end]"` tokens that we’ve inserted. By default, the characters `[` and `]` would be stripped, but we want to keep them around so we can distinguish the word `"start"` from the start token `"[start]"`.
[p455_e04] (bullet) Punctuation is different from language to language! In the Spanish `Text` `Vectorization` layer, if we’re going to strip punctuation characters, we need to also strip the character ¿.
[p455_e05] (body) Note that for a non-toy translation model, we would treat punctuation characters as separate tokens rather than stripping them since we would want to be able to generate correctly punctuated sentences. In our case, for simplicity, we’ll get rid of all punctuation.
[p455_e06] (listing_caption) Listing 15.9 Learning token vocabularies for English and Spanish text
[p455_e07] (code)
```
import string
import re
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", ""
    )
vocab_size = 15000
sequence_length = 20
english_tokenizer = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
spanish_tokenizer = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)
train_english_texts = [pair[0] for pair in train_pairs]
train_spanish_texts = [pair[1] for pair in train_pairs]
english_tokenizer.adapt(train_english_texts)
spanish_tokenizer.adapt(train_spanish_texts)
```
[p455_e08] (body) Finally, we can turn our data into a `tf.data` pipeline. We want it to return a tuple `(inputs, target, sample_weights)` where `inputs` is a dict with two keys, `"english"` (the tokenized English sentence) and `"spanish"` (the tokenized Spanish sentence), and `target` is the Spanish sentence offset by one step ahead. `sample_weights` here will

## Listings
- Listing 15.9: Learning token vocabularies for English and Spanish text (code elements p455_e07)

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

## Previous page (PDF 454) text, for continuity
15.2.1 English-to-Spanish translation

We’ll be working with an English-to-Spanish translation dataset. Let’s download it:

The text file contains one example per line: an English sentence, followed by a tab character, followed by the corresponding Spanish sentence. Let’s parse this file:

Our `text_pairs` look like this:

Let’s shuffle them and split them into the usual training, validation, and test sets:

Next, let’s prepare two separate `TextVectorization` layers: one for English and one for Spanish. We’re going to need to customize the way strings are preprocessed:

## Next page (PDF 456) text, for continuity
be used to tell Keras which labels to use when calculating our loss and metrics. Our output translations are not all equal in length, and some of our label sequences will be padded with zeros. We only care about predictions for non-zero labels that represent actual translated text.

This matches the “off by one” label set up in the generation model we just built, with the addition of the fixed encoder inputs, which will be handled separately in our model.

Listing 15.10 Tokenizing and preparing the translation data

Here’s what our dataset outputs look like:

The data is now ready—time to build some models.

15.2.2 Sequence-to-sequence learning with RNNs

Before we try the twin encoder/decoder setup we previously mentioned, let’s think through simpler options. The easiest, naive way to use RNNs to turn one sequence

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.448 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.459 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.460 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.457 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.477 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.447 — 15 Language models and the Transformer › 15.1 The language model
- PDF p.458 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.412 — 14 Text classification › 14.2 Preparing text data

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.457 (printed 433) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 
However, there is a critical issue with this approach. Due to the step-by-step nature of RNNs, the model will only look at tokens `0...N` in the source sequence to predict token `N` in the target sequence. Consider translating the sentence, “I will bring the bag to you.” In Spanish, that would be “Te traeré la bolsa,” where “Te,” the first word of the translation, corresponds to “you” in the English source text. There’s simply no way to output the first word of the translation without seeing the last word of the source English text!

If you’re a human translator, you’d start by reading the entire source sentence before beginning to translate it. This is especially important if you’re dealing with languages with wildly different word ordering. And that’s precisely what standard sequence-to-sequence models do. In a proper sequence-to-sequence setup (see figure 15.2), you would first use an encoder RNN to turn the entire source sequence into a single representation of the source text. This could be the last output of the RNN or, alternatively, its final internal state vectors. We can use this representation as the initial state of a decoder RNN in the language model setup instead of a

### PDF p.448 (printed 424) — 15 Language models and the Transformer › 15.1 The language model › 
To map this input to a sequence of integers, we can again use the `TextVectorization` layer we saw in the last chapter. To learn a character-level vocabulary instead of a word-level vocabulary, we can change our `split` argument. Rather than the default `"whitespace"` splitting, we instead split by `"character"`. We will do no standardization here—to keep things simple, we will preserve case and pass punctuation through unaltered.

Listing 15.3 Learning a character-level vocabulary with the `TextVectorization` layer

### PDF p.459 (printed 435) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 
We picked accuracy as a crude way to monitor validation set performance during training. We get to 65% accuracy: on average, the model correctly predicts the next word in the Spanish sentence 65% of the time. However, in practice, next-token accuracy isn’t a great metric for machine translation models, in particular because it makes the assumption that the correct target tokens from `0` to `N` are already known when predicting token `N + 1`. In reality, during inference, you’re generating the target sentence from scratch, and you can’t rely on previously generated tokens being 100% correct. When working on a real-world machine translation system, metrics must be more carefully designed. There are standard metrics, such as a BLEU score, that measure the similarity of the machine-translated text to a set of high-quality reference translations and can tolerate slightly misaligned sequences.

At last, let’s use our model for inference. We’ll pick a few sentences in the test set and check how our model translates them. We’ll start from the seed token `"[start]"` and feed it into the decoder model, together with the encoded English source sentence. We’ll retrieve a next-token prediction,

### PDF p.409 (printed 385) — 14 Text classification › 14.2 Preparing text data › 
Splitting takes us from a single string to a token sequence, but we still need to transform our string tokens into numeric inputs. By far the most common approach is to map each token to a unique integer index, often called indexing our input. This is a flexible and reversible representation of our tokenized input that can work with a wide range of modeling approaches. Later on, we can decide how to map from token indices into a latent space ingested by the model.

For character tokens, we could use ASCII lookups to index each token—for example, `ord('A') `→` 65` and `ord('z') `→` 122`. However, this can scale poorly when you start to consider other languages—there are over a million characters in the Unicode specification! A more robust technique is to build a mapping from specific tokens in our training data to indices that occur in the data we care about, which in NLP is called a vocabulary. This has the nice property of working for word-level tokens as easily as for character-level tokens. Let’s take a look at how we might use a vocabulary to transform text. We will build a simple Python dictionary that maps tokens to indices, split our input into tokens, and finally index our 

### PDF p.474 (printed 450) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
After training, we get to about 58% accuracy: on average, the model correctly predicts the next word in the Spanish sentence 58% of the time. Something is off here. Training is worse than the RNN model by 7 percentage points. Either this Transformer architecture is not what it was hyped up to be, or we missed something in our implementation. Can you spot what it is?

This section is ostensibly about sequence models. In the previous chapter, we saw how vital word order could be to meaning. And yet, the Transformer we just built wasn’t a sequence model at all. Did you notice? It’s composed of dense layers that process sequence tokens independently of each other and an attention layer that looks at the tokens as a set. You could change the order of the tokens in a sequence, and you’d get identical pairwise attention scores and the same context-aware representations. If you were to rearrange every word in every English source sentence completely, the model wouldn’t notice, and you’d still get the same accuracy. Attention is a set-processing mechanism, focused on the relationships between pairs of sequence elements—it’s blind to whether these elements occur at the beginning, at the end,

### PDF p.458 (printed 434) — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning › 
Next, let’s add the decoder—a simple `GRU` layer that takes as its initial state the encoded source sentence. On top of it, we add a `Dense` layer that produces a probability distribution over the Spanish vocabulary for each output step. Here, we want to predict the next tokens based only on what came before, so a `Bidirectional` RNN would break training by making the loss function trivially easy.

Listing 15.12 Building a sequence-to-sequence decoder

### PDF p.446 (printed 422) — 15 Language models and the Transformer › 15.1 The language model › 
What about other text-based tasks like question answering or translation? For many real-world problems, we are interested in a model that can generate a text output for a given input. Just like we needed tokenizers and embeddings to help us handle text on the way in to a model, we must build up some techniques before we can produce text on the way out. We don’t need to start from scratch here; we can continue to use the idea of an integer sequence as a natural numeric representation for text. In the previous chapter, we covered tokenizing a string, where we split inputs into tokens and map each token to an int. We can detokenize a sequence by proceeding in reverse—map ints back to string tokens and join them together. With this approach, our problem becomes building a model that can predict an integer sequence of tokens. The simplest option to consider might be to train a direct classifier over the space of all possible output integer sequences, but some back-of-the-envelope math will quickly show this is intractable. With a vocabulary of 20,000 words, there are 20,000 ^ 4, or 160 quadrillion possible 4-word sequences, and fewer atoms in the universe than possible 20-word sequences

### PDF p.513 (printed 489) — 16 Text generation › 16.3 Using a pretrained LLM › 
Our prompt template gives our examples a predictable structure. Although Gemma is not a sequence-to-sequence model like our English-to-Spanish translator, we can still use it in a sequence-to-sequence setting by training on prompts like this and only generating the output after the `"[response]"` marker.

Let’s make a `tf.data.Dataset` and split some validation data:
