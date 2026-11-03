# Generation bundle for page_531  (PDF page 531, printed page 507)

Page kind: content
Chapter: 16 Text generation
Section: 16.5 Where are LLMs heading next?
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_531.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_531.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p531_e03] (bullet) LLM output is unreliable, and all LLMs will occasionally hallucinate factually incorrect information.
[p531_e04] (bullet) LLMs can be fine-tuned to follow instructions in a chat dialog. This type of
[p531_e05] (quote) fine-tuning is called instruction fine-tuning:
[p531_e06] (body) – The simplest form of instruction fine-tuning involves directly training the
[p531_e07] (body) model on instruction and response pairs. – More advanced forms of instruction fine-tuning involve reinforcement learning.
[p531_e08] (bullet) The most common resource bottleneck when working with LLMs is accelerator memory.
[p531_e09] (bullet) LoRA is a technique to reduce memory usage by freezing most Transformer parameters and only updating a low-rank decomposition of attention projection weights.
[p531_e10] (bullet) LLMs can input or output data from different modalities if you can figure out how to frame these inputs or outputs as sequences in a sequence prediction problem.
[p531_e11] (bullet) A foundation model is a general term for models of any modality trained using self-supervision for a wide range of downstream tasks.

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

## Previous page (PDF 530) text, for continuity
We can note that while the total number of tokens used for pretraining has climbed steadily and massively, model parameter counts have varied substantially since GPT-3. In part, this is because we now know GPT-3 was undertrained, but it is also for a more practical reason. When deploying a model, it’s often worth it to sacrifice performance for a smaller model that fits on cheaper hardware. A really good model won’t help very much if it’s prohibitively expensive to run. There’s another reason we might not be able to just scale up these models thoughtlessly: we are starting to run out of pretraining data! Tech companies are starting to have trouble finding more high-quality, public, human-written content to throw at pretraining. Models are even starting to “eat their own tail” by training on a significant portion of content created by other LLMs, which runs into a whole other host of concerns. This is one of the reasons reinforcement learning is getting a lot of attention recently. If you can create a difficult, self-contained environment that generates new problems for an LLM to attempt, you will have found a way to continue training using the model’s own output—no need to scrounge the web for more morsels of quality text. None of the solutions we touched on will be a silver bullet for the issues facing LLMs. At the end of the day, the fundamental problem remains that LLMs are wildly inefficient at learning compared to humans. Model capabilities only come from training on many orders of magnitude more text than people will read in their lifetimes. As scaling LLMs continues, so too will more fundamental research in how to make models that can learn quickly with limited data. Still, LLMs represent the ability to build fluent natural language interfaces, and that alone wil …

## Next page (PDF 532) text, for continuity
Image generation

This chapter covers

Variational autoencoders

Diffusion models

Using a pretrained text-to-image model

Exploring the latent image spaces learned by

text-to-image models

The most popular and successful application of creative AI today is image generation: learning latent visual spaces and sampling from them to create entirely new pictures, interpolated from real ones—pictures of imaginary people, imaginary places, imaginary cats and dogs, and so on.

17.1 Deep learning for image generation

In this section and the next, we’ll review some high-level concepts pertaining to image generation, alongside implementation details relative to two of the main techniques in this domain: variational autoencoders (VAEs) and diffusion models. Do note that the techniques we present here aren’t specific to images—you could develop

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.512 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.519 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.517 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.525 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.515 — 16 Text generation › 16.3 Using a pretrained LLM
- PDF p.289 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.528 — 16 Text generation › 16.5 Where are LLMs heading next?
- PDF p.484 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.512 (printed 488) — 16 Text generation › 16.3 Using a pretrained LLM › 16.3.2 Instruction fine-tuning
Instruction fine-tuning involves feeding the model input/output pairs—a user instruction followed by a model response. We combine these into a single sequence that becomes new training data for the model. To make it clear during training when an instruction or response ends, we can add special markers like `"[instruction]"` and `"[response]"` directly to the combined sequence. The precise markup will not matter much as long as it is consistent.

We can use the combined sequence as regular training data, with the same “guess the next word” loss we used to pretrain an LLM. By doing further training with examples containing desired responses, we are essentially bending the model’s output in the direction we want. We won’t be learning a latent space for language here; that’s already been done over trillions of tokens of pretraining. We are simply nudging the learned representation a bit to control the tone and content of the output. To begin, we will need a dataset of instruction-response pairs. Training chatbots is a hot topic, so there are many datasets made specifically for this purpose. We will use a dataset made public by the company Databricks. Employees contributed to a dataset 

### PDF p.519 (printed 495) — 16 Text generation › 16.4 Going further with LLMs › 16.4.1 Reinforcement Learning with Human Feedback (RLHF)
The type of instruction fine-tuning we just did is often called supervised fine-tuning. It is supervised because we are curating, by hand, a list of example prompts and responses we want from the model.

### PDF p.512 (printed 488) — 16 Text generation › 16.3 Using a pretrained LLM › 
Of course, this is utter nonsense, but the model could not find a more likely way to complete this prompt.

Hallucinations and uncontrollable output are fundamental problems with language models. If there is a silver bullet, we have yet to find it. However, one approach that helps immensely is to further fine-tune a model with examples of the specific types of generative outputs you would like. In the specific case of wanting to build a chatbot that can follow instructions, this type of training is called instruction fine-tuning. Let’s try some instruction fine-tuning with Gemma to make it a lot more useful as a conversation partner.

### PDF p.525 (printed 501) — 16 Text generation › 16.4 Going further with LLMs › Foundation models
As LLMs venture into different modalities, the “large language model” moniker can become a bit misleading. They do model language, but also images, audio, maybe even structured data. In the next chapter, we will see a distinct architecture, called diffusion models, that works quite differently in terms of underlying structure but has a similar feel—they too are trained on massive amounts of data at “internet scale” with a self-supervised loss.

An umbrella term for models like this is foundation models. More specifically, a foundation model is any model that is trained on broad data (generally using self-supervision at scale) that can be fine-tuned to a wide range of downstream tasks. In general, you can think of a foundation model as learning to reconstruct data pulled from large swaths of the internet, given a partial representation of it. While LLMs are the first and best-known of these models, there are many others. The hallmarks of a foundation model are the self-supervised learning objective (a reconstruction loss) and the fact that these models are not specialized to a single task and can be used for a number of downstream purposes. This is an important and striking shift th

### PDF p.288 (printed 264) — 8 Image classification › 8.3 Using a pretrained model › Fine-tuning a pretrained model
Another widely used technique for model reuse, complementary to feature extraction, is fine-tuning (see figure 8.15). Fine-tuning consists of unfreezing the frozen model base used for feature extraction and jointly training both the newly added part of the model (in this case, the fully connected classifier) and the base model. This is called fine-tuning because it slightly adjusts the more abstract representations of the model being reused to make them more relevant for the problem at hand.

We stated earlier that it’s necessary to freeze the pretrained convolution base first to be able to train a randomly initialized classifier on top. For the same reason, it’s only possible to fine-tune the convolutional base once the classifier on top has already been trained. If the classifier isn’t already trained, then the error signal propagating through the network during training will be too large, and the representations previously learned by the layers being fine-tuned will be destroyed. Thus, the steps for fine-tuning a network are as follows: 1 Add your custom network on top of an already trained base network. 2 Freeze the base network. 3 Train the part you added. 4 Unfreeze the base 

### PDF p.514 (printed 490) — 16 Text generation › 16.3 Using a pretrained LLM › 16.3.3 Low-Rank Adaptation (LoRA)
If we ran `fit()` right now on a Colab GPU with 16 GB of device memory, we would quickly trigger an out of memory error. But we’ve already loaded the model and run generation, so why would we run out of memory now?

Our 1-billion-parameter model takes up about 3.7 GB of memory. You can see it in our previous model summary. The `Adam` optimizer we have been using will need to track three extra floating-point numbers for each parameter—the actual gradients, a velocity value, and a momentum value. All told, it comes out to 15 GB just for the weights and optimizer state. We also need a few gigabytes of memory to keep track of intermediate values in the forward pass of the model, but we have none left to spare. Running `fit()` would crash on the first train step. This is a common problem when training LLMs. Because these models have large parameter counts, the throughput of your GPUs and CPUs is a secondary concern to fitting the model on accelerator memory. We’ve seen earlier in this book how we can freeze certain parts of a model during fine-tuning. What we did not mention is that this will save a lot of memory! We do not need to track any optimizer variables for frozen parameters—the

### PDF p.289 (printed 265) — 8 Image classification › 8.3 Using a pretrained model › Partial fine-tuning
In this case, we chose to unfreeze and fine-tune all of the Xception convolutional base. However, when dealing with large pretrained models, you may sometimes only unfreeze some of the top layers of the convolutional base, and leave the lower layers frozen. You’re probably wondering, why only fine-tune some of the layers? Why the top ones specifically? Here’s why:

Earlier layers in the convolutional base encode more-generic, reusable features,

whereas layers higher up encode more-specialized features. It’s more useful to fine-tune the more specialized features because these are the ones that need to be repurposed on your new problem. There would be fast-decreasing returns in fine-tuning lower layers.

The more parameters you’re training, the more you’re at risk of overfitting. The

convolutional base has 15 million parameters, so it would be risky to attempt to train it on your small dataset. Thus, it can be a good strategy to fine-tune only the top three or four layers in the convolutional base. You’d do something like this:

### PDF p.528 (printed 504) — 16 Text generation › 16.5 Where are LLMs heading next? › 16.5 Where are LLMs heading next?
Given the trajectory of LLMs discussed at the beginning of this chapter, it may seem obvious where LLMs will be heading. More parameters! Even better performance! In a general sense, that’s probably correct, but our trajectory might not be quite so linear.

If you have a fixed budget for pretraining, say, a million dollars, you can roughly think of it as buying you a fixed amount of compute or floating-point operations (flops). You can either spend those flops on training with more data or training a bigger model. Recent research has pointed out that GPT-3, at 175 billion parameters, was way too big for its computing budget. Training a smaller model on more data would have led to
