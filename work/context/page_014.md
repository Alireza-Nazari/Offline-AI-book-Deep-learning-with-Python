# Generation bundle for page_014  (PDF page 14, printed page xiv)

Page kind: front_matter
Chapter: contents
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_014.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_014.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p014_e04] (body) 14.5
[p014_e05] (body) Sequence models 405
[p014_e06] (quote) Training a recurrent model 406 ■ Understanding word embeddings 409 ■ Using a word embedding 410 ■ Pretraining a word embedding 414 ■ Using the pretrained embedding for classification 418
[p014_e07] (body) 15
[p014_e08] (body) Language models and the Transformer 421 15.1
[p014_e09] (body) The language model 421
[p014_e10] (quote) Training a Shakespeare language model 422 ■ Generating Shakespeare 426
[p014_e11] (body) 15.2
[p014_e12] (body) Sequence-to-sequence learning 428
[p014_e13] (quote) English-to-Spanish translation 430 ■ Sequence-to-sequence learning with RNNs 432
[p014_e14] (body) 15.3
[p014_e15] (body) The Transformer architecture 437
[p014_e16] (quote) Dot-product attention 439 ■ Transformer encoder block 444 Transformer decoder block 446 ■ Sequence-to-sequence learning with a Transformer 448 ■ Embedding positional information 451
[p014_e17] (body) 15.4
[p014_e18] (body) Classification with a pretrained Transformer 454
[p014_e19] (quote) Pretraining a Transformer encoder 454 ■ Loading a pretrained Transformer 455 ■ Preprocessing IMDb movie reviews 458 Fine-tuning a pretrained Transformer 460
[p014_e20] (body) 15.5
[p014_e21] (body) What makes the Transformer effective? 461 16 Text generation 466 16.1 A brief history of sequence generation 468 16.2 Training a mini-GPT 470
[p014_e22] (quote) Building the model 473 ■ Pretraining the model 476 Generative decoding 478 ■ Sampling strategies 480
[p014_e23] (body) 16.3
[p014_e24] (body) Using a pretrained LLM 484
[p014_e25] (quote) Text generation with the Gemma model 485 ■ Instruction fine-tuning 488 ■ Low-Rank Adaptation (LoRA) 490
[p014_e26] (body) 16.4
[p014_e27] (body) Going further with LLMs 495
[p014_e28] (quote) Reinforcement Learning with Human Feedback (RLHF) 495 Multimodal LLMs 498 ■ Retrieval Augmented Generation (RAG) 501 ■ “Reasoning” models 502
[p014_e29] (body) 16.5
[p014_e30] (body) Where are LLMs heading next? 504

## Chapter outline
Chapter: contents (PDF pages 8-16)

## Previous page (PDF 13) text, for continuity
11.3

Using a pretrained segmentation model 318

Downloading the Segment Anything Model 319 ■ How Segment Anything works 319 ■ Preparing a test image 321 ■ Prompting the model with a target point 323 ■ Prompting the model with a target box 327

12

Object detection 329 12.1

Single-stage vs. two-stage object detectors 330

Two-stage R-CNN detectors 330 ■ Single-stage detectors 332

12.2

Training a YOLO model from scratch 332

Downloading the COCO dataset 332 ■ Creating a YOLO model 336 ■ Readying the COCO data for the YOLO model 339 Training the YOLO model 342

12.3

Using a pretrained RetinaNet detector 346 13 Timeseries forecasting 351 13.1 Different kinds of timeseries tasks 351 13.2 A temperature forecasting example 352

Preparing the data 356 ■ A commonsense, non-machine-learning baseline 359 ■ Let’s try a basic machine learning model 360 Let’s try a 1D convolutional model 362

13.3

Recurrent neural networks 364

Understanding recurrent neural networks 365 ■ A recurrent layer in Keras 368 ■ Getting the most out of recurrent neural networks 372 ■ Using recurrent dropout to fight overfitting 372 Stacking recurrent layers 375 ■ Using bidirectional RNNs 377

13.4

Going even further 379 14 Text classification 381 14.1 A brief history of natural language processing 381 14.2 Preparing text data 384

Character and word tokenization 387 ■ Subword tokenization 390

14.3

Sets vs. sequences 395

Loading the IMDb classification dataset 396

14.4

Set models 398

Training a bag-of-words model 399 ■ Training a bigram model 403

## Next page (PDF 15) text, for continuity
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

The limitations of deep learning 564

Deep learning models struggle to adapt to novelty 565 Deep learning models are highly sensitive to phrasing and other distractors 567 ■ Deep learning models struggle to learn generalizable programs 569 ■ The risk of anthropomorphizing machine-learning models 569

19.2

Scale isn’t all you need 570

Automatons vs. intelligent agents 571 ■ Local generalization vs. extreme generalization 573 ■ The purpose of intelligence 575 Climbing the spectrum of generalization 575

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.474 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.489 — 15 Language models and the Transformer › 15.5 What makes the Transformer effective?
- PDF p.639 — index › 
- PDF p.476 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.479 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer
- PDF p.453 — 15 Language models and the Transformer › 15.2 Sequence-to-sequence learning
- PDF p.478 — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer
- PDF p.643 — index › 

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.638 (printed 614) — index ›  › K
classification with pretrained Transformer

454–461 Kaggle 607 dot-product attention 439 kagglehub package 242, 485 embedding positional information 451 kaleidoscope hypothesis 577 generating Shakespeare 426–428 Keras 60, 190 loss function 462 building models in 192 seq2seq learning 428–437 ecosystem 608 sequence-to-sequence learning with evaluation loops 206–214 Transformer 449 fit() function 222– 225 training Shakespeare language model 422–426 Functional API 195, 197–200 Transformer 444, 447, 462–464 implementing VAEs with 513–518 Transformer architecture 437 low-level usage of metrics 221 last_conv_layer_output 302 mixing and matching different components 204 latent space 509, 568 relationship between frameworks 63 of images 509 Sequential model 192 subclassing Model class 202 of text-to-image models 533–537 summary 230 visualizing 306 training and evaluation loops 214 lateral connections 347 training loops 206–214 Layer API 191 training step functions 216–218 layered representations learning 7 using right tool for job 205 LayerNormalization layer 445 workflows 191 layer_output layer 167 Keras API, recurrent layers in 368–372 layers 596 keras.callbacks.Callback class 210 LayoutM

### PDF p.639 (printed 615) — index ›  › 
learning, seq2seq (sequence-to-sequence)

manifold hypothesis 144 428–437 .map(function, num_parallel_calls) method 249 English-to-Spanish translation 430–432 mask argument 412 with RNNs 433–437 masked language model 455, 464 LearningRateSchedule 476 mask_zero argument 412 lemmatization, defined 386–387 matmul operation 550 lifelong learning 607 Matplotlib axis object 322 linear transformations 110 MaxPooling1D layer 362 LLMs (large language models) 468, 495–504, 565 MaxPooling2D layer 232, 233, 234, 239, 245, 315 multimodal LLM 498–501 max-pooling layers 602 RAG (Retrieval Augmented Generation) 501 max-pooling operation 239–241 reasoning models 502–504 max tensor operation 240 RLHF 495–498 metrics using pretrained 484–494

handling in custom train_step() 226–228 local generalization 573, 574 low-level usage of 221 local minimum 45 writing 207 logit 478 MHR (Modularity-Hierarchy-Reuse) formula 269 logs argument 211 middle stage 522 LoRA (Low-Rank Adaptation) 490–494 mini-batch or just batch 105 loss.backward() method 217 mini-batch SGD (mini-batch stochastic gradient loss function 9, 181, 596

descent) 44 loss scaling with mixed precision 560 mini-GPT (Generative Pretrained Transfo

### PDF p.479 (printed 455) — 15 Language models and the Transformer › 15.4 Classification with a pretrained Transformer › 
model were released a year after “Attention Is All You Need.” The model structure was exactly the same as the encoder part of the translation Transformer we just built. This encoder model is bidirectional in that every position in the sequence can attend to positions in front of and behind it. This means it’s a good model for computing a rich representation of input text, but not a model meant to run generation in a loop.

BERT was trained in sizes between 100 million and 300 million parameters, much bigger than the 14 million parameter Transformer we just trained. This meant the model needed a lot of training data to perform well. To achieve this, the authors used a riff on the classic language modeling setup called masked language modeling. To pretrain the model, we take a sequence of text and replace about 15% of the tokens with a special `[MASK]` token. The model will attempt to predict the original masked token values during training. While the classic language model, sometimes called a causal language model, attempts to predict `p(token|past` `tokens)`, the masked language model attempts to predict `p(token|surrounding` `tokens)`. This training setup is unsupervised. You don’

### PDF p.643 (printed 619) — index ›  › 
stemming, defined 386–387

TensorFlow Serving 185 step fusing 556 tensor operations, derivatives of 42–43 stochastic gradient descent 43–46 tensors, defined 63 strided convolutions 239 test images, preparing 321 subclassing Model class 202 test-time training 582 rewriting previous example as subclassed text classification 381 model 202

augmenting text data 413 what subclassed models don't support 204 NLP (natural language processing) 382–384 subgraph isomorphism 586 preparing text data 384–394 subword tokenization 390–394 pretraining word embeddings 414 summary() method 193 sequence models 405 supervised sets vs. sequences 395–398 fine-tuning 495 setting models 398–404 learning 214 training recurrent model 406 symbolic AI 3 using pretrained embedding for symbolic tensor 196

classification 418 symbols 584–594 word embeddings 409, 410 blending deep learning and program text-conditioning 509 synthesis 590–592 text_dataset_from_directory utility 397 cognition as combination of both kinds of abstraction 587 text_embeddings input 531 deep learning and 588 text generation 466 long-term vision 594 history of sequence generation 468–470 modular component recombination and lifelong LLMs (l

### PDF p.489 (printed 465) — 15 Language models and the Transformer › 15.5 What makes the Transformer effective? › 
The Transformer is a sequence modeling architecture that uses attention as the only mechanism to pass information across a sequence: – The Transformer works by stacking blocks of alternating attention and two-layer feedforward networks. – The Transformer can scale to many parameters and lots of training data while still improving accuracy in the language modeling problem. – Unlike RNNs, the Transformer involves no sequence-length loops at training time, making the model much easier to train in parallel across many machines. – A Transformer encoder uses bidirectional attention to build a rich representation of a sequence. – A Transformer decoder uses causal attention to predict the next word in a language model setup.

### PDF p.642 (printed 618) — index ›  › 
RetinaNet detector 347–349

self-supervised learning 11, 214, 516 return_sequences argument 368 semantic segmentation 310 reusability 11 SeparableConv1D layer 362 RLHF (Reinforcement Learning with Human SeparableConv2D layer 292, 362, 602 Feedback) 495–498 separable convolution 269 using chatbots trained with 496–498 seq2seq (sequence-to-sequence) learning 428–437 RMSE (root mean squared error) 207

English-to-Spanish translation 430–432 RNNs (recurrent neural networks) 352, 364–378, with RNNs 433–437 591, 604 sequence generation, history of 468–470 advanced features of 372 sequence models 395, 405 bidirectional RNNs 377–378 sequences, sets vs. 395–398 overview of 365–367 sequence-to-sequence language model 464 recurrent dropout 373–375 sequence-to-sequence learning 603 recurrent layers in Keras 368–372 with Transformer 449 seq2seq learning with 433–437 sequential stacking recurrent layers 375 architecture 191 RoBERTa 464 class 230 ROC AUC (receiver operating characteristic model 191, 192 curve) 178 Shakespeare language model 422–428

### PDF p.438 (printed 414) — 14 Text classification › 14.5 Sequence models › 14.5.4 Pretraining a word embedding
The last decade of rapid advancement in NLP has coincided with the rise of pretraining as the dominant approach for text modeling problems. Once we move past simple set-based regression models to sequence models with millions or even billions of parameters, text models become incredibly data-hungry. We are usually limited by our ability to find labeled examples for a particular problem in the text domain.

The idea is to devise an unsupervised task to train model parameters that do not need labeled data. Pretraining data can be text in a similar domain to our final task, or even just arbitrary text in the languages we are interested in working with. Pretraining allows us to learn general patterns in language, effectively priming our model before we specialize it to the final task we are interested in. Word embeddings were one of the first big successes with text pretraining, and we will show how to pretrain a word embedding in this section. Remember the `unsup/` directory we ignored in our IMDb dataset preparation? It contains another 25,000 reviews— the same size as our training data. We will combine all our training data together and show how to pretrain the parameters of an `Emb

### PDF p.474 (printed 450) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
After training, we get to about 58% accuracy: on average, the model correctly predicts the next word in the Spanish sentence 58% of the time. Something is off here. Training is worse than the RNN model by 7 percentage points. Either this Transformer architecture is not what it was hyped up to be, or we missed something in our implementation. Can you spot what it is?

This section is ostensibly about sequence models. In the previous chapter, we saw how vital word order could be to meaning. And yet, the Transformer we just built wasn’t a sequence model at all. Did you notice? It’s composed of dense layers that process sequence tokens independently of each other and an attention layer that looks at the tokens as a set. You could change the order of the tokens in a sequence, and you’d get identical pairwise attention scores and the same context-aware representations. If you were to rearrange every word in every English source sentence completely, the model wouldn’t notice, and you’d still get the same accuracy. Attention is a set-processing mechanism, focused on the relationships between pairs of sequence elements—it’s blind to whether these elements occur at the beginning, at the end,
