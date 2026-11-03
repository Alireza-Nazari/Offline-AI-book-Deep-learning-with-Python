# Generation bundle for page_648  (PDF page 648, printed page 624)

Page kind: index
Chapter: index
Section: 
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_648.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_648.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p648_e01] (body) The Manning Early Access Program
[p648_e02] (body) Don’t wait to start learning! In MEAP, the Manning Early Access Program, you can read
[p648_e03] (body) books as they’re being created and long before they’re available in stores.
[p648_e04] (body) Here’s how MEAP works.
[p648_e05] (body) • Start now. Buy a MEAP and you’ll get all available chapters in PDF, ePub, Kindle,
[p648_e06] (body) and liveBook formats.
[p648_e07] (body) • Regular updates. New chapters are released as soon as they’re written. We’ll
[p648_e08] (body) let you know when fresh content is available.
[p648_e09] (body) • Finish faster. MEAP customers are the first to get final versions of all books!
[p648_e10] (body) Pre-order the print book, and it’ll ship as soon as it’s off the press.
[p648_e11] (body) • Contribute to the process. The feedback you share with authors makes the end
[p648_e12] (body) product better.
[p648_e13] (body) • No risk. You get a full refund or exchange if we ever have to cancel a MEAP.
[p648_e14] (body) Explore dozens of titles in MEAP at www.manning.com.

## Chapter outline
Chapter: index (PDF pages 633-650)

## Previous page (PDF 647) text, for continuity
A new online reading experience

liveBook, our online reading platform, adds a new dimension to your Manning books,

with features that make reading, learning, and sharing easier than ever. A liveBook

version of your book is included FREE with every Manning book.

This next generation book platform is more than an online reader. It’s packed with

unique features to upgrade and enhance your learning experience.

• Add your own notes and bookmarks

• One-click code copy

• Learn from other readers in the discussion forum

• Audio recordings and interactive exercises

• Read all your purchased Manning content in any browser, anytime, anywhere

As an added bonus, you can search every Manning book and video in liveBook—even

ones you don’t yet own. Open any liveBook, and you’ll be able to browse the content and

read anything you like.*

Find out more at www.manning.com/livebook-program.

*Open reading is limited to 10 minutes per book daily

## Next page (PDF 649) text, for continuity
Task Last-layer activation Loss function Metrics

Binary classification Sigmoid Binary crossentropy Binary accuracy, ROC AUC Multiclass, single-label Softmax Categorical crossentropy Categorical accuracy, classification Top-k categorical accuracy, ROC AUC Multiclass, multi-label Sigmoid Binary crossentropy Binary accuracy, ROC classification AUC Regression None Mean squared error Mean absolute error

Input Output Model

Vector data Class probability, regression value Densely connected network Timeseries Class probability, regression value RNN, Transformer Images Class probability, regression value ConvNet Text Class probability, regression value Transformer Text, images Text Transformer Text, images Images VAE, diffusion model

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.22 — about this book › liveBook discussion forum
- PDF p.6 — Praise for the Second Edition › 
- PDF p.580 — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation
- PDF p.646 — index › 
- PDF p.20 — about this book › Who should read this book
- PDF p.631 — 20 Conclusions › 20.4 Staying up to date in a fast-moving field
- PDF p.494 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.421 — 14 Text classification › 14.3 Sets vs. sequences

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.22 (printed xxii) — about this book › liveBook discussion forum › liveBook discussion forum
Purchase of Deep Learning with Python, Third Edition, includes free access to liveBook, Manning’s online reading platform. Using liveBook’s exclusive discussion features, you can attach comments to the book globally or to specific sections or paragraphs. It’s a snap to make notes for yourself, ask and answer technical questions, and receive help from the authors and other users. To access the forum, go to https://livebook .manning.com/book/deep-learning-with-python-third-edition/discussion.

Manning’s commitment to our readers is to provide a venue where a meaningful dialogue between individual readers and between readers and the authors can take place. It is not a commitment to any specific amount of participation on the part of the authors, whose contribution to the forum remains voluntary (and unpaid). We suggest you try asking the authors some challenging questions lest their interest stray! The forum and the archives of previous discussions will be accessible from the publisher’s website as long as the book is in print.

### PDF p.6 (printed vi) — Praise for the Second Edition ›  › 
For online information and ordering of this and other Manning books, please visit www.manning.com. The publisher offers discounts on this book when ordered in quantity. For more information, please contact

Special Sales Department Manning Publications Co. 20 Baldwin Road PO Box 761 Shelter Island, NY 11964 Email: orders@manning.com

© 2026 Manning Publications Co. All rights reserved.

No part of this publication may be reproduced, stored in a retrieval system, or transmitted, in any form or by means electronic, mechanical, photocopying, or otherwise, without prior written permission of the publisher.

Many of the designations used by manufacturers and sellers to distinguish their products are claimed as trademarks. Where those designations appear in the book, and Manning Publications was aware of a trademark claim, the designations have been printed in initial caps or all caps.

∞

Recognizing the importance of preserving what has been written, it is Manning’s policy to have the books we publish printed on acid-free paper, and we exert our best efforts to that end. Recognizing also our responsibility to conserve the resources of our planet, Manning books are printed on paper that

### PDF p.22 (printed xxii) — about this book › liveBook discussion forum › About the code
This book is an example of literate programming—it is simultaneously a prose explanation of deep learning concepts and a runnable Python program. Starting with chapter 2, each chapter will be interspersed with code listings that can be run as a Jupyter notebook, an interactive notebook for running Python code and visualizing data. We strongly suggest running, modifying, and playing with the examples in this book as you read to build your own intuitions on the concepts presented.

All code is written in the Python language with the deep learning library Keras. Keras can be run on top of TensorFlow, PyTorch, and JAX, the most popular low-level deep learning frameworks as of 2025. All code can be run on a local machine or directly in the browser using Google Colab, a hosted environment for Jupyter notebooks. You can get executable snippets of code from the liveBook version of this book, included with your print or eBook purchase, at https://livebook.manning.com/book/ deep-learning-with-python-third-edition. The complete code for the examples in the book is available for download from the Manning website at https://www.manning .com/books/deep-learning-with-python-third-edition and on G

### PDF p.494 (printed 470) — 16 Text generation › 16.2 Training a mini-GPT › 16.2 Training a mini-GPT
To begin pretraining our mini-GPT, we will need a lot of text data. GPT-1 used a dataset called BooksCorpus, which contained a number of free, self-published books added to the dataset without the explicit permission of the authors. The dataset has since been taken down by its publishers.

We will use a more recent pretraining dataset called the “Colossal Clean Crawled Corpus” (C4), released by Google in 2020. At 750 GB, it’s far bigger than we could reasonably train on for a book example, so we will use less than 1% of the overall corpus. Let’s start by downloading and extracting our data.

Listing 16.1 Downloading a portion of the C4 dataset

### PDF p.646 (printed 622) — index ›  › 
RELATED MANNING TITLES

Grokking Artificial Intelligence Algorithms by Rishal Hurbans

ISBN 9781617296185 392 pages, $59.99 July 2020

Deep Learning and the Game of Go by Max Pumperla and Kevin Ferguson Foreword by Thore Graepel

ISBN: 9781617295324 384 pages, $54.99 January 2019

Transfer Learning for Natural Language Processing by Paul Azunre

ISBN: 9781617297267 272 pages, $49.99 July 2021

Deep Learning with JavaScript by Shanqing Cai, Stanley Bileschi, Eric D. Nielsen, and François Chollet Foreword by Nikhil Thorat and Daniel Smilkov

ISBN 9781617296178 560 pages, $49.99 January 2020

For ordering information, go to www.manning.com

### PDF p.20 (printed xx) — about this book › Who should read this book › Who should read this book
This book is written for people with some Python programming experience who want to get started with machine learning and deep learning. But this book can also be valuable to many different types of readers:

If you’re a data scientist familiar with machine learning, this book will provide you with a solid, practical introduction to deep learning, the fastest-growing and most significant subfield of machine learning.

xviii

### PDF p.580 (printed 556) — 18 Best practices for the real world › 18.3 Speeding up training and inference with lower-precision computation › 
TPU runtime in addition to the GPU runtime. For more serious training runs, Google Cloud also makes available TPU v3 through v5, which are even faster.

When running Keras code with the JAX backend on a TPU-enabled notebook, you don’t need anything more than calling `keras.distribution.set_distribution
` `(distribution)` with a `DataParallel` or `ModelParallel` distribution instance to start using your TPU cores. Make sure to call it before creating your model! NOTE Because TPUs can process batches of data extremely quickly, the speed at which you can read data from Google Cloud Storage (GCS) can easily become a bottleneck. If your dataset is small enough, you should keep it in the memory of the virtual machine. You can do so by calling `dataset.cache()` on your `tf.data.Dataset` instance. That way, the data will only be read from GCS once.

### PDF p.213 (printed 189) — 6 The universal workflow of machine learning › 6.3 Deploying your model › 
As soon as your model has launched, you should be getting ready to train the next generation that will replace it:

Watch out for changes in the production data. Are new features becoming available? Should you expand or otherwise edit the label set?

Keep collecting and annotating data, and keep improving your annotation pipeline over time. In particular, you should pay special attention to collecting samples that seem to be difficult to classify for your current model—such samples are the most likely to help improve performance.

This concludes the universal workflow of machine learning—that’s a lot of things to keep in mind. It takes time and experience to become an expert, but don’t worry, you’re already a lot wiser than you were a few chapters ago. You are now familiar with the big picture—the entire spectrum of what machine learning projects entail. While most of this book will focus on the model development part, you’re now aware that it’s only one part of the entire workflow. Always keep in mind the big picture!
