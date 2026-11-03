# Generation bundle for page_015  (PDF page 15, printed page xv)

Page kind: front_matter
Chapter: contents
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_015.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_015.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p015_e04] (body) 17
[p015_e05] (body) Image generation 508 17.1
[p015_e06] (body) Deep learning for image generation 508
[p015_e07] (quote) Sampling from latent spaces of images 509 ■ Variational autoencoders 510 ■ Implementing a VAE with Keras 513
[p015_e08] (body) 17.2
[p015_e09] (body) Diffusion models 518
[p015_e10] (quote) The Oxford Flowers dataset 520 ■ A U-Net denoising autoencoder 521 ■ The concepts of diffusion time and diffusion schedule 523 ■ The training process 525 ■ The generation process 527 ■ Visualizing results with a custom callback 528 It’s go time! 529
[p015_e11] (body) 17.3
[p015_e12] (body) Text-to-image models 531
[p015_e13] (quote) Exploring the latent space of a text-to-image model 533
[p015_e14] (body) 18
[p015_e15] (body) Best practices for the real world 538 18.1
[p015_e16] (body) Getting the most out of your models 539
[p015_e17] (quote) Hyperparameter optimization 539 ■ Model ensembling 546
[p015_e18] (body) 18.2
[p015_e19] (body) Scaling up model training with multiple devices 548
[p015_e20] (quote) Multi-GPU training 548 ■ Distributed training in practice 550 TPU training 555
[p015_e21] (body) 18.3
[p015_e22] (body) Speeding up training and inference with lower-precision computation 556
[p015_e23] (quote) Understanding floating-point precision 556 ■ Float16 inference 558 ■ Mixed-precision training 559 ■ Using loss scaling with mixed precision 559 ■ Beyond mixed precision: float8 training 560 ■ Faster inference with quantization 561
[p015_e24] (body) 19
[p015_e25] (body) The future of AI 564 19.1
[p015_e26] (body) The limitations of deep learning 564
[p015_e27] (quote) Deep learning models struggle to adapt to novelty 565 Deep learning models are highly sensitive to phrasing and other distractors 567 ■ Deep learning models struggle to learn generalizable programs 569 ■ The risk of anthropomorphizing machine-learning models 569
[p015_e28] (body) 19.2
[p015_e29] (body) Scale isn’t all you need 570
[p015_e30] (quote) Automatons vs. intelligent agents 571 ■ Local generalization vs. extreme generalization 573 ■ The purpose of intelligence 575 Climbing the spectrum of generalization 575

## Chapter outline
Chapter: contents (PDF pages 8-16)

## Previous page (PDF 14) text, for continuity
14.5

Sequence models 405

Training a recurrent model 406 ■ Understanding word embeddings 409 ■ Using a word embedding 410 ■ Pretraining a word embedding 414 ■ Using the pretrained embedding for classification 418

15

Language models and the Transformer 421 15.1

The language model 421

Training a Shakespeare language model 422 ■ Generating Shakespeare 426

15.2

Sequence-to-sequence learning 428

English-to-Spanish translation 430 ■ Sequence-to-sequence learning with RNNs 432

15.3

The Transformer architecture 437

Dot-product attention 439 ■ Transformer encoder block 444 Transformer decoder block 446 ■ Sequence-to-sequence learning with a Transformer 448 ■ Embedding positional information 451

15.4

Classification with a pretrained Transformer 454

Pretraining a Transformer encoder 454 ■ Loading a pretrained Transformer 455 ■ Preprocessing IMDb movie reviews 458 Fine-tuning a pretrained Transformer 460

15.5

What makes the Transformer effective? 461 16 Text generation 466 16.1 A brief history of sequence generation 468 16.2 Training a mini-GPT 470

Building the model 473 ■ Pretraining the model 476 Generative decoding 478 ■ Sampling strategies 480

16.3

Using a pretrained LLM 484

Text generation with the Gemma model 485 ■ Instruction fine-tuning 488 ■ Low-Rank Adaptation (LoRA) 490

16.4

Going further with LLMs 495

Reinforcement Learning with Human Feedback (RLHF) 495 Multimodal LLMs 498 ■ Retrieval Augmented Generation (RAG) 501 ■ “Reasoning” models 502

16.5

Where are LLMs heading next? 504

## Next page (PDF 16) text, for continuity
19.3

How to build intelligence 576

The kaleidoscope hypothesis 577 ■ The essence of intelligence: Abstraction acquisition and recombination 578 The importance of setting the right target 578 ■ A new target: On-the-fly adaptation 580 ■ ARC Prize 581 ■ The test-time adaptation era 582 ■ ARC-AGI 2 583

19.4

The missing ingredients: Search and symbols 584

The two poles of abstraction 585 ■ Cognition as a combination of both kinds of abstraction 587 ■ Why deep learning isn’t a complete answer to abstraction generation 588 ■ An alternative approach to AI: Program synthesis 589 ■ Blending deep learning and program synthesis 590 ■ Modular component recombination and lifelong learning 592 ■ The long-term vision 593

20

Conclusions 595 20.1

Key concepts in review 595

Various approaches to artificial intelligence 596 ■ What makes deep learning special within the field of machine learning 596 How to think about deep learning 597 ■ Key enabling technologies 598 ■ The universal machine learning workflow 599 ■ Key network architectures 600

20.2

Limitations of deep learning 605 20.3 What might lie ahead 606 20.4 Staying up to date in a fast-moving field 607

Practice on real-world problems using Kaggle 607 ■ Read about the latest developments on arXiv 607 ■ Explore the Keras ecosystem 608

20.5

Final words 608

index 609

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.593 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.636 — index › 
- PDF p.629 — 20 Conclusions › 20.2 Limitations of deep learning
- PDF p.637 — index › 
- PDF p.7 — brief contents › 
- PDF p.641 — index › 
- PDF p.639 — index › 
- PDF p.11 — contents › 

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.561 (printed 537) — 17 Image generation › 17.3 Text-to-image models › Summary
Image generation with deep learning is done by learning latent spaces that capture statistical information about a dataset of images. By sampling and decoding points from the latent space, you can generate never-before-seen images. There are three major tools to do this: VAEs, diffusion models, and GANs.

VAEs result in highly structured, continuous latent representations. For this reason, they work well for doing all sorts of image editing in latent space: face swapping, turning a frowning face into a smiling face, and so on. They also work nicely for doing latent space–based animations, such as animating a walk along a cross section of the latent space, showing a starting image slowly morphing into different images in a continuous way.

Diffusion models result in very realistic outputs and are the dominant method of image generation today. They work by repeatedly denoising an image, starting from pure noise. They can easily be conditioned on text captions to create text-to-image models.

Stable Diffusion 3 is a state-of-the-art pretrained text-to-image model that you can use to create highly realistic images of your own.

The visual latent space learned by such text-to-image diff

### PDF p.7 (printed vii) — brief contents ›  › brief contents
1

■ What is deep learning? 1

2

■ The mathematical building blocks of neural networks 16

3

■ Introduction to TensorFlow, PyTorch, JAX, and Keras 60

4

■ Classification and regression 104

5

■ Fundamentals of machine learning 136

6

■ The universal workflow of machine learning 171

7

■ A deep dive on Keras 190

8

■ Image classification 231

9

■ ConvNet architecture patterns 268

10

■ Interpreting what ConvNets learn 284

11

■ Image segmentation 308

12

■ Object detection 329

13

■ Timeseries forecasting 351

14

■ Text classification 381

15

■ Language models and the Transformer 421

16

■ Text generation 466

17

■ Image generation 508

18

■ Best practices for the real world 538

19

■ The future of AI 564

20

■ Conclusions 595

v

### PDF p.641 (printed 617) — index ›  › P
randomized A/B testing 188 padding 238, 239, 322 RandomSearch tuner 542 panoptic segmentation 310 R-CNN (region-based convolutional neural parameters, defined 539

networks), two-stage detectors 331 pattern recognition 586 real-world best practices, lower-precision computation 556–562 perception, defined 586 faster inference with quantization 561 pip install tensorboard command 213 float8 training 560 plot_model() utility 199 float16 inference 558 PositionalEmbedding layer 452, 474 positional encoding 603 floating-point precision 557 position embedding 451 loss scaling with mixed precision 560 positive prompt 532 mixed-precision training 559 predict() function 191, 301, 479, 489 recombination 578 prediction error or loss value 105 reconstruction loss 512 prediction or output 105 recurrent dropout 372–375 predict() method 115, 124, 203, 259, 260 recurrent layers in Keras 368–372 premade search spaces 546 recurrent models, training 406 preparing data 179–180 region proposals 331 preprocessor layer 489 regression 104, 126–134, 602 pretrained embedding, using for classification 418 building model 128 pretrained models 256–266 California Housing Price dataset 126 feature extraction with

### PDF p.644 (printed 620) — index ›  › U
attribute 262

underfitting 137–142 parameters 39

ambiguous features 139 weights 215 noisy training data 138 training rare features and spurious correlations 140–142 diffusion models 525 U-Net denoising autoencoder 522 inference vs. 215 unigrams 403 lower-precision computation 556–562 universal workflow of machine learning training arguments 215, 218 deploying model 183–189 training data 147 developing model 179–183 training loops 10, 39, 206–214 update_state() method 207, 222 callbacks 208, 210 upsampling stage 522 monitoring and visualization with use_casual_mask 448 TensorBoard 212 using fit() with custom training loops 222–225

### PDF p.534 (printed 510) — 17 Image generation › 17.1 Deep learning for image generation › 
Of course, there are still challenges to overcome. Like with all deep learning models, the latent space doesn’t encode a consistent model of the physical world, so you might occasionally see hands with extra fingers, incoherent lighting, or garbled objects. The coherence of generated images is still an area of active research. In the case of figure 17.2, despite having seen tens of thousands of images of people riding bikes, the model doesn’t understand in a human sense what it means to ride a bike—concepts like pedaling, steering, or maintaining upright balance. That’s why your bike-riding horse is unlikely to get depicted pedaling with its hind legs in a believable manner, the way a human artist would draw it.

There’s a range of different strategies for learning such latent spaces of image representations, each with its own characteristics. The most common types of image generation models are

Diffusion models

Variational autoencoders (VAEs)

Generative adversarial networks (GANs)

While previous editions of this book covered GANs, they have gradually fallen out of fashion in recent years and have been all but replaced by diffusion models. In this edition, we’ll cover both VAEs

### PDF p.532 (printed 508) — 17 Image generation › 17.1 Deep learning for image generation › This chapter covers
Variational autoencoders

Diffusion models

Using a pretrained text-to-image model

Exploring the latent image spaces learned by

text-to-image models

The most popular and successful application of creative AI today is image generation: learning latent visual spaces and sampling from them to create entirely new pictures, interpolated from real ones—pictures of imaginary people, imaginary places, imaginary cats and dogs, and so on.

### PDF p.637 (printed 613) — index ›  › 
distributed training, accessing multiple

building model 245 GPUs 551 ConvNets 232–241 multi-GPU training 548–550 data augmentation 252 grad attribute 217 data preprocessing 247, 248, 250 .grad() function 302 deep learning for small-data problems 242 gradient ascent downloading data 242 in input space 291 pretrained models 256–266 in JAX 295 training ConvNet from scratch on small in PyTorch 295

dataset 241 in TensorFlow 294 ImageClassifier 301 gradient-based optimization 39–50 ImageConverter layer 258 backpropagation algorithm 46–50 image generation 508 derivatives 41–43 deep learning for 509–518 stochastic gradient descent 43–46 diffusion models 518–530 gradient descent 40, 291, 571, 596 text-to-image models 531–537 tuning key parameters 154 image normalization layer 525 GradientTape 294, 302 image segmentation 308, 309 scope 216 computer vision tasks 308–311 ground-truth or annotations 105 preparing test images 321 GRU (Gated Recurrent Unit) 375, 424, 426, 428, prompting model with target points 323–327 439 training segmentation model from scratch layers 434, 444, 449, 604 311–318 types of 310

### PDF p.532 (printed 508) — 17 Image generation › 17.1 Deep learning for image generation › 17.1 Deep learning for image generation
In this section and the next, we’ll review some high-level concepts pertaining to image generation, alongside implementation details relative to two of the main techniques in this domain: variational autoencoders (VAEs) and diffusion models. Do note that the techniques we present here aren’t specific to images—you could develop
