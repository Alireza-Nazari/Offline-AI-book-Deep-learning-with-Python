# Generation bundle for page_562  (PDF page 562, printed page 538)

Page kind: chapter_opener
Chapter: 18 Best practices for the real world
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_562.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_562.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p562_e01] (chapter_number) 18
[p562_e02] (chapter_title) Best practices for
[p562_e03] (chapter_title) the real world
[p562_e04] (chapter_covers) This chapter covers
[p562_e05] (bullet) Hyperparameter tuning
[p562_e06] (bullet) Model ensembling
[p562_e07] (bullet) Training Keras models on multiple GPUs or on TPU
[p562_e08] (bullet) Mixed-precision training
[p562_e09] (bullet) Quantization
[p562_e10] (body) You’ve come quite far since the beginning of this book. You can now train image classification models, image segmentation models, models for classification or regression on vector data, timeseries forecasting models, text classification models, sequence-to-sequence models, and even generative models for text and images. You’ve got all the bases covered.
[p562_e11] (body) However, your models so far have all been trained at a small scale—on small datasets, with a single GPU—and they generally haven’t reached the best achievable performance on each dataset we’ve looked at. This book is, after all, an introductory book. If you are to go out into the real world and achieve state-of-the-art results on brand new problems, there’s still a bit of a chasm that you’ll need to cross.

## Chapter outline
Chapter: 18 Best practices for the real world (PDF pages 562-587)
  - 18.1 Getting the most out of your models (p.563)
      - 18.1.1 Hyperparameter optimization (p.563)
      - 18.1.2 Model ensembling (p.570)
  - 18.2 Scaling up model training with multiple devices (p.572)
      - 18.2.1 Multi-GPU training (p.572)
      - 18.2.2 Distributed training in practice (p.574)
      - 18.2.3 TPU training (p.579)
  - 18.3 Speeding up training and inference with lower-precision computation (p.580)
      - 18.3.1 Understanding floating-point precision (p.580)
      - 18.3.2 Float16 inference (p.582)
      - 18.3.3 Mixed-precision training (p.583)
      - 18.3.4 Using loss scaling with mixed precision (p.583)
      - 18.3.5 Beyond mixed precision: float8 training (p.584)
      - 18.3.6 Faster inference with quantization (p.585)

## Previous page (PDF 561) text, for continuity
This might feel like magic the first time you try it, but there’s nothing magic about it— interpolation is fundamental to the way deep neural networks learn. This will be the last substantive model we work with in the book, and it’s a great visual metaphor to end with. Deep neural networks are interpolation machines; they map complex, real-world probability distributions to low-dimensional manifolds. We can exploit this fact even for input as complex as human language and output as complex as natural images.

Summary

Image generation with deep learning is done by learning latent spaces that capture statistical information about a dataset of images. By sampling and decoding points from the latent space, you can generate never-before-seen images. There are three major tools to do this: VAEs, diffusion models, and GANs.

VAEs result in highly structured, continuous latent representations. For this reason, they work well for doing all sorts of image editing in latent space: face swapping, turning a frowning face into a smiling face, and so on. They also work nicely for doing latent space–based animations, such as animating a walk along a cross section of the latent space, showing a starting image slowly morphing into different images in a continuous way.

Diffusion models result in very realistic outputs and are the dominant method of image generation today. They work by repeatedly denoising an image, starting from pure noise. They can easily be conditioned on text captions to create text-to-image models.

Stable Diffusion 3 is a state-of-the-art pretrained text-to-image model that you can use to create highly realistic images of your own.

The visual latent space learned by such text-to-image diffusion models is fundamentally interpolative. You can see this by interpolati …

## Next page (PDF 563) text, for continuity
This chapter is about bridging that gap and giving you the best practices you’ll need as you go from machine learning student to a fully fledged machine learning engineer. We’ll review essential techniques for systematically improving model performance: hyperparameter tuning and model ensembling. Then we’ll look at how you can speed up and scale up model training, with multi-GPU and TPU training, mixed precision, and quantization.

18.1 Getting the most out of your models

Blindly trying out different architecture configurations works well enough if you just need something that works okay. In this section, we’ll go beyond “works okay” to “works great and wins machine learning competitions” via a quick guide to a set of must-know techniques for building state-of-the-art deep learning models.

18.1.1 Hyperparameter optimization

When building a deep learning model, you have to make many seemingly arbitrary decisions: How many layers should you stack? How many units or filters should go in each layer? Should you use `relu` as an activation, or a different function? Should you use `BatchNormalization` after a given layer? How much dropout should you use? And so on. These architecture-level parameters are called hyperparameters to distinguish them from the parameters of a model, which are trained via backpropagation.

In practice, experienced machine learning engineers and researchers build intuition over time as to what works and what doesn’t when it comes to these choices—they develop hyperparameter-tuning skills. But there are no formal rules. If you want to get to the very limit of what can be achieved on a given task, you can’t be content with such arbitrary choices. Your initial decisions are almost always suboptimal, even if you have very good intuition. You can refin …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.587 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation
- PDF p.572 — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices
- PDF p.21 — about this book › How this book is organized: A road map
- PDF p.570 — 18 Best practices for the real world › 18.1 Getting the most out of your models
- PDF p.266 — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset
- PDF p.643 — index › 
- PDF p.290 — 8 Image classification › 8.3 Using a pretrained model
- PDF p.637 — index › 

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.587 (printed 563) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › Summary
You can use hyperparameter tuning and KerasTuner to automate the tedium out of finding the best model configuration. But be mindful of validation-set overfitting!

An ensemble of diverse models can often significantly improve the quality of your predictions.

To further scale your workflows, you can use data parallelism to train a model on multiple devices, as long as the model is small enough to fit on a single device.

For larger models, you can also use model parallelism to split your model’s variables and computation across several devices.

You can speed up model training on GPUs or TPUs by turning on mixed precision—you’ll generally get a nice speed boost at virtually no cost.

You can also speed up inference by using `float16` precision or even `int8` quantization.

### PDF p.15 (printed xv) — contents ›  › 
17

Image generation 508 17.1

Deep learning for image generation 508

Sampling from latent spaces of images 509 ■ Variational autoencoders 510 ■ Implementing a VAE with Keras 513

17.2

Diffusion models 518

The Oxford Flowers dataset 520 ■ A U-Net denoising autoencoder 521 ■ The concepts of diffusion time and diffusion schedule 523 ■ The training process 525 ■ The generation process 527 ■ Visualizing results with a custom callback 528 It’s go time! 529

17.3

Text-to-image models 531

Exploring the latent space of a text-to-image model 533

18

Best practices for the real world 538 18.1

Getting the most out of your models 539

Hyperparameter optimization 539 ■ Model ensembling 546

18.2

Scaling up model training with multiple devices 548

Multi-GPU training 548 ■ Distributed training in practice 550 TPU training 555

18.3

Speeding up training and inference with lower-precision computation 556

Understanding floating-point precision 556 ■ Float16 inference 558 ■ Mixed-precision training 559 ■ Using loss scaling with mixed precision 559 ■ Beyond mixed precision: float8 training 560 ■ Faster inference with quantization 561

19

The future of AI 564 19.1

The limitations of deep l

### PDF p.21 (printed xxi) — about this book › How this book is organized: A road map › How this book is organized: A road map
This book contains 20 chapters, organized as follows:

Chapters 1–7: Foundations of deep learning—These chapters give you a solid grounding in deep learning. We discuss the history of the field, provide an introduction to coding with the major deep learning frameworks currently available (JAX, PyTorch, and TensorFlow), and outline a practical guide to applying machine learning to real-world problems. We look closely at the Keras library for training machine learning models.

Chapters 8–12: Computer vision—These chapters cover machine learning on image data. We train models to classify entire images and models to recognize and separate out individual objects in images. We look closely at ConvNets, a model architecture well suited to images and the first big success of the deep learning era.

Chapter 13: Timeseries forecasting—This chapter covers working with timeseries data. We take data that is measured over time, like the weather or sales data, and use past patterns to predict future trends.

Chapters 14–15: Natural language processing—These chapters cover applying deep learning to text. We build sentiment classifiers to predict whether text is positive or negative. We build an En

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

### PDF p.637 (printed 613) — index ›  › 
distributed training, accessing multiple

building model 245 GPUs 551 ConvNets 232–241 multi-GPU training 548–550 data augmentation 252 grad attribute 217 data preprocessing 247, 248, 250 .grad() function 302 deep learning for small-data problems 242 gradient ascent downloading data 242 in input space 291 pretrained models 256–266 in JAX 295 training ConvNet from scratch on small in PyTorch 295

dataset 241 in TensorFlow 294 ImageClassifier 301 gradient-based optimization 39–50 ImageConverter layer 258 backpropagation algorithm 46–50 image generation 508 derivatives 41–43 deep learning for 509–518 stochastic gradient descent 43–46 diffusion models 518–530 gradient descent 40, 291, 571, 596 text-to-image models 531–537 tuning key parameters 154 image normalization layer 525 GradientTape 294, 302 image segmentation 308, 309 scope 216 computer vision tasks 308–311 ground-truth or annotations 105 preparing test images 321 GRU (Gated Recurrent Unit) 375, 424, 426, 428, prompting model with target points 323–327 439 training segmentation model from scratch layers 434, 444, 449, 604 311–318 types of 310

### PDF p.478 (printed 454) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 15.4 Classification with a pretrained Transformer
After “Attention is all you need,” people started to notice how far Transformer training could scale, especially compared to models that had come before. As we just mentioned, one big plus was that the model is faster to train than RNNs. No more loops during training, which is always good when working with a GPU or TPU.

It is also a very data hungry model architecture. We actually got a little taste of this in the last section. While our RNN translation model plateaued in validation performance after 5 or so epochs, the Transformer model was still improving its validation score after 30 epochs of training. These observations prompted many to try scaling up the Transformer with more data, layers, and parameters—with great results. This caused a distinctive shift in the field toward large pretrained models that can cost millions to train but perform noticeably better on a wide range of problems in the text domain. For our last code example in the text section, we will revisit our IMDb text-classification problem, this time with a pretrained Transformer model.

### PDF p.572 (printed 548) — 18 Best practices for the real world › 18.2 Scaling up model training with multiple devices › 18.2.1 Multi-GPU training
While GPUs are getting more powerful every year, deep learning models are also getting increasingly larger, requiring ever more computational resources. Training on a single GPU puts a hard bound on how fast you can move. The solution? You could simply add more GPUs and start doing multi-GPU distributed training.

There are two ways to distribute computation across multiple devices: data parallelism and model parallelism. With data parallelism, a single model gets replicated on multiple devices or multiple machines. Each of the model replicas processes different batches of data, and then they merge their results. With model parallelism, different parts of a single model run on different devices, processing a single batch of data together at the same time. This works best with models that have a naturally parallel architecture, such as models that feature multiple branches. In practice, model parallelism is only used in the case of models that are too large to fit on any single device: it isn’t used as a way to speed up training of regular models but as a way to train larger models. Then, of course, you can also mix both data parallelism and model parallelism: a single model can be 

### PDF p.555 (printed 531) — 17 Image generation › 17.3 Text-to-image models › 17.3 Text-to-image models
We can use the same basic diffusion process to create a model that maps text input to image output. To do this we need to take a pretrained text encoder (think a transformer encoder like RoBERTa from chapter 15) that can map text to vectors in a continuous embedding space. Then we can train a diffusion model on `(prompt,` `image)` pairs, where each prompt is a short, textual description of the input image.

We can handle the image input in the same way as we did previously, mapping noisy input to a denoised output that progressively approaches our input image. Critically, we can extend this setup by also passing the embedded text prompt to the denoising model. So rather than our denoising model simply taking in a `noisy_images` input, our model will take two inputs: `noisy_images` and `text_embeddings`. This gives a leg up on the flower denoiser we trained previously. Instead of learning to remove noise from an image without any additional information, the model gets to use a textual representation of the final image to help guide the denoising process. After training is when things get a bit more fun. Because we have trained a model that can map pure noise to images conditioned on
