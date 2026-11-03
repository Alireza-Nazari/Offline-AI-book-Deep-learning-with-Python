# Generation bundle for page_501  (PDF page 501, printed page 477)

Page kind: content
Chapter: 16 Text generation
Section: 16.2 Training a mini-GPT
Subsection: 16.2.2 Pretraining the model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_501.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_501.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p501_e03] (figure_caption) Figure 16.2 Warmup makes our updates to model parameters smaller at the beginning of training and can help with stability.
[p501_e04] (body) We will train our model using one pass over our 1 billion tokens, split across eight epochs so we can occasionally check our validation set loss and accuracy.
[p501_e05] (body) We are training a miniature version of GPT, using 3× fewer parameters than GPT-1 and 100× fewer overall training steps. But despite this being two orders of magnitude cheaper to train than the smallest GPT model, this call to `fit()` will be the most computationally expensive training run in the entire book. If you are running the code as you read, set things off and take a breather!
[p501_e06] (listing_caption) Listing 16.8 Training the mini-GPT model
[p501_e07] (code)
```
num_epochs = 8
steps_per_epoch = num_train_batches // num_epochs
validation_steps = num_val_batches
mini_gpt.compile(
    optimizer=keras.optimizers.Adam(schedule),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[„accuracy"],
)
mini_gpt.fit(
    train_ds,
    validation_data=val_ds,
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)
```

## Code annotations printed next to listings (callouts, in order)
- Set these to a lower value if you
- don’t want to wait for training.

## Listings
- Listing 16.8: Training the mini-GPT model (code elements p501_e07)

## Figures
- Figure 16.2: Warmup makes our updates to model parameters smaller at the beginning of training and can help with stability.  | labels inside the figure: (none - image)

## Chapter outline
Chapter: 16 Text generation (PDF pages 490-531)
  - 16.1 A brief history of sequence generation (p.492)
  - 16.2 Training a mini-GPT (p.494)
      - 16.2.1 Building the model (p.497)
      - 16.2.2 Pretraining the model (p.500)
      - 16.2.3 Generative decoding (p.502)
      - 16.2.4 Sampling strategies (p.504)
  - 16.3 Using a pretrained LLM (p.508)
      - 16.3.1 Text generation with the Gemma model (p.509)
      - 16.3.2 Instruction fine-tuning (p.512)
      - 16.3.3 Low-Rank Adaptation (LoRA) (p.514)
  - 16.4 Going further with LLMs (p.519)
      - 16.4.1 Reinforcement Learning with Human Feedback (RLHF) (p.519)
      - 16.4.2 Multimodal LLMs (p.522)
      - 16.4.3 Retrieval Augmented Generation (RAG) (p.525)
      - 16.4.4 “Reasoning” models (p.526)
  - 16.5 Where are LLMs heading next? (p.528)

## Previous page (PDF 500) text, for continuity
This model has 41 million parameters, which is large for models in this book but quite small compared to most LLMs today, which range from a couple of billion to trillions of parameters.

16.2.2 Pretraining the model

Training a large Transformer is famously finicky—the model is sensitive to initializations of parameters and choice of optimizer. When many Transformer layers are stacked, it is easy to suffer from exploding gradients, where parameters update too quickly and our loss function does not converge. A trick that works well is to linearly ease into a full learning rate over a number of warmup steps, so our initial updates to our model parameters are small. This is easy to implement in Keras with a `LearningRateSchedule`.

Listing 16.7 Defining a custom learning rate schedule

We can plot our learning rate over time to make sure it is what we expect (figure 16.2):

## Next page (PDF 502) text, for continuity
What is a logit?

As we compile our model, you will notice a new value for the loss `SparseCategorical`

`Crossentropy(from_logits=True)`. What is a logit?

The output projection at the end of our transformer model does not contain the usual softmax activation. You can think of this output as a bunch of “unnormalized log probabilities” for each token. If you exponentiate each output value and normalize all values to sum to 1 (this is all the `softmax` function does), you will get a probability value. A common term for an “unnormalized log probability” is a logit, and logits can be easier to work with when generating text, as we will see in the next section. Keras gives you a choice of where to apply the `softmax` function. For classification problems, you can either use a `softmax` as the last activation of the model and output probabilities or move the `softmax` into the loss function and output logits. To do the

latter, you should pass `SparseCategoricalCrossentropy(from_logits=True)` as a

classification loss.

After training, our model can predict the next token in a sequence about 36% of the time on our validation set, though such a metric is just a crude heuristic.

Note that our model is undertrained. Our validation loss will continue to tick down after each epoch, which is unsurprising given that we used a hundred times fewer training steps than GPT-1. Training for longer would be a great idea, but we would need both time and money to pay for compute. Let’s play around with our mini-GPT model.

16.2.3 Generative decoding

To sample some output from our model, we can follow the approach we used to generate Shakespeare or Spanish translations in chapter 15. We feed a prompt of fixed tokens into the model. For each position in the input sequence, the model outputs …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.517 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.499 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.494 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.493 — 16 Text generation › 16.1 A brief history of sequence generation
- PDF p.135 — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example
- PDF p.508 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.495 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.96 — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.499 (printed 475) — 16 Text generation › 16.2 Training a mini-GPT › 
Let’s build our model. We will stack eight decoder layers into a single “mini” GPT model.

We will also turn on a Keras setting called mixed precision to speed up training. This will allow Keras to run some of the model’s computations much faster by sacrificing some numerical fidelity. For now, this will remain a little mysterious, but a full explanation is waiting in chapter 18.

Listing 16.6 Creating a mini-GPT functional model

### PDF p.493 (printed 469) — 16 Text generation › 16.1 A brief history of sequence generation › 
In 2018, a year after the “Attention Is All You Need” paper we discussed in the last chapter, a group of researchers at an organization called OpenAI put out a new paper “Improving Language Understanding by Generative Pre-Training.”3 They combined a few ingredients:

Unsupervised pretraining of a language model—essentially training a model to “guess the next token” in a sequence, as we did with our Shakespeare generator in chapter 15

The Transformer architecture

Textual data on various topics via thousands of self-published books

The authors showed that such a pretrained model could be fine-tuned to achieve state-of-the-art performance on a wide array of text classification tasks—from gauging the similarity of two sentences to answering a multiple-choice question. They called the pretrained model GPT, short for Generative Pretrained Transformer.

GPT didn’t come with any modeling or training advancements. What was interesting about the results was that such a general training setup could beat out more involved techniques across a number of tasks. There was no complex text normalization, no need to customize the model architecture or training data per benchmark, just a lot of pre

### PDF p.494 (printed 470) — 16 Text generation › 16.2 Training a mini-GPT › 16.2 Training a mini-GPT
To begin pretraining our mini-GPT, we will need a lot of text data. GPT-1 used a dataset called BooksCorpus, which contained a number of free, self-published books added to the dataset without the explicit permission of the authors. The dataset has since been taken down by its publishers.

We will use a more recent pretraining dataset called the “Colossal Clean Crawled Corpus” (C4), released by Google in 2020. At 750 GB, it’s far bigger than we could reasonably train on for a book example, so we will use less than 1% of the overall corpus. Let’s start by downloading and extracting our data.

Listing 16.1 Downloading a portion of the C4 dataset

### PDF p.517 (printed 493) — 16 Text generation › 16.3 Using a pretrained LLM › 
After training, we get to 55% accuracy when guessing the next word in our model’s response. That’s a huge jump from the 35% accuracy of our mini-GPT model. This shows the power of a larger model and more extensive pretraining.

### PDF p.530 (printed 506) — 16 Text generation › 16.5 Where are LLMs heading next? › 
We can note that while the total number of tokens used for pretraining has climbed steadily and massively, model parameter counts have varied substantially since GPT-3. In part, this is because we now know GPT-3 was undertrained, but it is also for a more practical reason. When deploying a model, it’s often worth it to sacrifice performance for a smaller model that fits on cheaper hardware. A really good model won’t help very much if it’s prohibitively expensive to run. There’s another reason we might not be able to just scale up these models thoughtlessly: we are starting to run out of pretraining data! Tech companies are starting to have trouble finding more high-quality, public, human-written content to throw at pretraining. Models are even starting to “eat their own tail” by training on a significant portion of content created by other LLMs, which runs into a whole other host of concerns. This is one of the reasons reinforcement learning is getting a lot of attention recently. If you can create a difficult, self-contained environment that generates new problems for an LLM to attempt, you will have found a way to continue training using the model’s own output—no need to scrounge

### PDF p.508 (printed 484) — 16 Text generation › 16.3 Using a pretrained LLM › 16.3 Using a pretrained LLM
Now that we’ve trained a mini-language model from scratch, let’s try using a billion-parameter pretrained model and see what it can do. Given how prohibitively expensive pretraining a Transformer can be, most of the industry has centered around using pretrained models developed by a relatively short list of companies. This is not purely a cost concern but also an environmental one—generative model training is now making up a large percentage of the total data center power consumption of large tech companies.

Meta published some environmental data on Llama 2, an LLM it published in 2023. It’s a good bit smaller than GPT-3, but it needed an estimated 1.3 million kilowatt hours of electricity to train—the daily power usage of about 45,000 American households. If every organization using an LLM ran pretraining themselves, the scale of energy use would be a noticeable percentage of global energy consumption. Let’s play around with a pretrained generative model from Google called Gemma. We will use the third version of the Gemma model, which was released to the public in 2025. To keep the examples in this book accessible, we will use the smallest variation of

### PDF p.96 (printed 72) — 3 Introduction to TensorFlow, PyTorch, JAX, and Keras › 3.3 Introduction to TensorFlow › 
For simplicity, we’ll do batch training instead of mini-batch training: we’ll run each training step (gradient computation and weight update) on the entire data, rather than iterate over the data in small batches. On one hand, this means that each training step will take much longer to run, since we compute the forward pass and the gradients for 2,000 samples at once. On the other hand, each gradient update will be much more effective at reducing the loss on the training data, since it will encompass information from all training samples instead of, say, only 128 random samples. As a result, we will need many fewer steps of training, and we should use a larger learning rate than what we would typically use for mini-batch training (we’ll use `learning_rate = 0.1`, as previously defined).

Listing 3.21 The batch training loop

### PDF p.135 (printed 111) — 4 Classification and regression › 4.1 Classifying movie reviews: A binary classification example › Validating your approach
You’ll now train the model for 20 epochs (20 iterations over all samples in the training data), in mini-batches of 512 samples. At the same time, you’ll monitor loss and accuracy on the 10,000 samples that you set apart. You do so by passing the validation data as the `validation_data` argument to `model.fit()`.

Listing 4.7 Training your model
