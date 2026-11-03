# Generation bundle for page_183  (PDF page 183, printed page 159)

Page kind: content
Chapter: 5 Fundamentals of machine learning
Section: 5.4 Improving generalization
Subsection: 5.4.2 Feature engineering
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_183.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_183.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p183_e03] (heading_subsection) 5.4.1
[p183_e04] (heading_subsection) Dataset curation
[p183_e05] (body) You’ve already learned that generalization in deep learning originates from the latent structure of your data. If your data makes it possible to smoothly interpolate between samples, then you will be able to train a deep learning model that generalizes. If your problem is overly noisy or fundamentally discrete, like, say, list sorting, deep learning will not help you. Deep learning is curve fitting, not magic.
[p183_e06] (body) As such, it is essential that you make sure that you’re working with an appropriate dataset. Spending more effort and money on data collection almost always yields a much greater return on investment than spending the same on developing a better model:
[p183_e07] (bullet) Make sure you have enough data. Remember that you need a dense sampling of the input-cross-output space. More data will yield a better model. Sometimes, problems that seem impossible at first become solvable with a larger dataset.
[p183_e08] (bullet) Minimize labeling errors—visualize your inputs to check for anomalies, and proofread your labels.
[p183_e09] (bullet) Clean your data and deal with missing values (we cover this in the next chapter).
[p183_e10] (bullet) If you have many features and you aren’t sure which ones are actually useful, do feature selection.
[p183_e11] (body) A particularly important way you can improve the generalization potential of your data is feature engineering. For most machine learning problems, feature engineering is a key ingredient for success. Let’s take a look.
[p183_e12] (heading_subsection) 5.4.2
[p183_e13] (heading_subsection) Feature engineering
[p183_e14] (body) Feature engineering is the process of using your own knowledge about the data and about the machine learning algorithm at hand (in this case, a neural network) to make the algorithm work better by applying hardcoded (non-learned) transformations to the data before it goes into the model. In many cases, it isn’t reasonable to expect a machine learning model to be able to learn from completely arbitrary data. The data needs to be presented to the model in a way that will make the model’s job easier.
[p183_e15] (body) Let’s look at an intuitive example. Suppose you’re trying to develop a model that can take as input an image of a clock and can output the time of day (see figure 5.17). If you choose to use the raw pixels of the image as input data, then you have a difficult machine learning problem on your hands. You’ll need a convolutional neural network to solve it, and you’ll have to expend quite a bit of computational resources to train the network. But if you already understand the problem at a high level (you understand how humans read time on a clock face), then you can come up with much better input features for a machine learning algorithm: for instance, it’s easy to write a five-line Python script to follow the black pixels of the clock hands and output the `(x, y)` coordinates of the tip of each hand. Then a simple machine learning algorithm can learn to associate these coordinates with the appropriate time of day.

## Chapter outline
Chapter: 5 Fundamentals of machine learning (PDF pages 160-194)
  - 5.1 Generalization: The goal of machine learning (p.160)
      - 5.1.1 Underfitting and overfitting (p.161)
      - 5.1.2 The nature of generalization in deep learning (p.167)
  - 5.2 Evaluating machine-learning models (p.173)
      - 5.2.1 Training, validation, and test sets (p.173)
      - 5.2.2 Beating a common-sense baseline (p.176)
      - 5.2.3 Things to keep in mind about model evaluation (p.176)
  - 5.3 Improving model fit (p.177)
      - 5.3.1 Tuning key gradient descent parameters (p.177)
      - 5.3.2 Using better architecture priors (p.179)
      - 5.3.3 Increasing model capacity (p.179)
  - 5.4 Improving generalization (p.182)
      - 5.4.1 Dataset curation (p.183)
      - 5.4.2 Feature engineering (p.183)
      - 5.4.3 Using early stopping (p.185)
      - 5.4.4 Regularizing your model (p.185)

## Previous page (PDF 182) text, for continuity
Note that while it is standard to work with models that are way overparameterized for the problem at hand, there can definitely be such a thing as too much memorization capacity. You’ll know your model is too large if it starts overfitting right away. Here’s what happens for an MNIST model with three intermediate layers with 2,048 units each (see figure 5.16):

Figure 5.16 Effect of excessive model capacity on validation loss

5.4

Improving generalization

Once your model has shown to have some generalization power and to be able to overfit, it’s time to switch your focus toward maximizing generalization.

## Next page (PDF 184) text, for continuity
Figure 5.17 Feature engineering for reading the time on a clock

You can go even further: do a coordinate change and express the `(x, y)` coordinates as polar coordinates with regard to the center of the image. Your input will become the angle `theta` of each clock hand. At this point, your features are making the problem so easy that no machine learning is required; a simple rounding operation and dictionary lookup are enough to recover the approximate time of day.

That’s the essence of feature engineering: making a problem easier by expressing it in a simpler way. Make the latent manifold smoother, simpler, and better organized. It usually requires understanding the problem in depth. Before deep learning, feature engineering used to be the most important part of the machine learning workflow because classical shallow algorithms didn’t have hypothesis spaces rich enough to learn useful features by themselves. The way you presented the data to the algorithm was absolutely critical to its success. For instance, before convolutional neural networks became successful on the MNIST digit-classification problem, solutions were typically based on hardcoded features such as the number of loops in a digit image, the height of each digit in an image, a histogram of pixel values, and so on. Fortunately, modern deep learning removes the need for most feature engineering because neural networks are capable of automatically extracting useful features from raw data. Does this mean you don’t have to worry about feature engineering as long as you’re using deep neural networks? No, for two reasons:

Good features still allow you to solve problems more elegantly while using fewer resources. For instance, it would be ridiculous to solve the problem of reading a clock face using a convolut …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.171 — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning
- PDF p.193 — 5 Fundamentals of machine learning › 5.4 Improving generalization
- PDF p.179 — 5 Fundamentals of machine learning › 5.3 Improving model fit
- PDF p.28 — 1 What is deep learning? › 1.4 Learning rules and representations from data
- PDF p.172 — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning
- PDF p.34 — 1 What is deep learning? › 1.7 What makes deep learning different
- PDF p.29 — 1 What is deep learning? › 1.4 Learning rules and representations from data
- PDF p.30 — 1 What is deep learning? › 1.4 Learning rules and representations from data

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.171 (printed 147) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › Training data is paramount
While deep learning is indeed well suited to manifold learning, the power to generalize is more a consequence of the natural structure of your data than a consequence of any property of your model. You’ll only be able to generalize if your data forms a manifold where points can be interpolated. The more informative and the less noisy your features are, the better you will be able to generalize, since your input space will be simpler and better structured. Data curation and feature engineering are essential to generalization.

Further, because deep learning is curve fitting, for a model to perform well, it needs to be trained on a dense sampling of its input space. A “dense sampling” in this context means that the training data should densely cover the entirety of the input data manifold (see figure 5.11). This is especially true near decision boundaries. With a sufficiently dense sampling, it becomes possible to make sense of new inputs by interpolating between

### PDF p.172 (printed 148) — 5 Fundamentals of machine learning › 5.1 Generalization: The goal of machine learning › 
Figure 5.11 A dense sampling of the input space is necessary to learn a model capable of accurate generalization.

past training inputs, without having to use common-sense, abstract reasoning, or external knowledge about the world—all things that machine learning models have no access to. As such, you should always keep in mind that the best way to improve a deep learning model is to train it on more data or better data (of course, adding overly noisy or inaccurate data will harm generalization). A denser coverage of the input data manifold will yield a model that generalizes better. You should never expect a deep learning model to perform anything more than crude interpolation between its training samples, and thus, you should do everything you can to make interpolation as easy as possible. The only thing you will find in a deep learning model is what you put into it: the priors encoded in its architecture and the data it was trained on.

When getting more data isn’t possible, the next-best solution is to modulate the quantity of information that your model is allowed to store, or to add constraints on the smoothness of the model curve. If a network can only afford to memorize a s

### PDF p.193 (printed 169) — 5 Fundamentals of machine learning › 5.4 Improving generalization › Summary
The purpose of a machine learning model is to generalize: to perform accurately on never-before-seen inputs. It’s harder than it seems.

A deep neural network achieves generalization by learning a parametric model that can successfully interpolate between training samples. Such a model can be said to have learned the latent manifold of the training data. This is why deep learning models can only make sense of inputs that are very close to what they’ve seen during training.

The fundamental problem in machine learning is the tension between optimization and generalization: to attain generalization, you must first achieve a good fit to the training data, but improving your model’s fit to the training data will inevitably start hurting generalization after a while. Every single deep learning best practice deals with managing this tension.

The ability of deep learning models to generalize comes from the fact that they manage to learn to approximate the latent manifold of their data and can thus make sense of new inputs via interpolation.

It’s essential to be able to accurately evaluate the generalization power of your model while you’re developing it. You have at your disposal an arr

### PDF p.179 (printed 155) — 5 Fundamentals of machine learning › 5.3 Improving model fit › Using better architecture priors
You have a model that fits, but for some reason your validation metrics aren’t improving at all. They remain no better than what a random classifier would achieve: your model trains, but doesn’t generalize. What’s going on?

This is perhaps the worst machine learning situation you can find yourself in. It indicates that something is fundamentally wrong with your approach, and it may not be easy to tell what. Here are some tips. First, it may be that the input data you’re using simply doesn’t contain sufficient information to predict your targets: the problem as formulated is not solvable. This is what happened earlier when we tried to fit an MNIST model where the labels were shuffled: the model would train just fine, but validation accuracy would stay stuck at 10%, because it was plainly impossible to generalize with such a dataset. It may also be that the kind of model you’re using is not suited for the problem at hand. For instance, in chapter 13, you’ll see an example of a timeseries prediction problem where a densely connected architecture isn’t able to beat a trivial baseline, whereas a more appropriate recurrent architecture does manage to generalize well. Using a model that 

### PDF p.34 (printed 10) — 1 What is deep learning? › 1.7 What makes deep learning different › What makes deep learning different
Is there anything special about deep neural networks that makes them the “right” approach for companies to invest in and for researchers to flock to? Will we still be using deep neural networks in 20 years?

Deep learning has several properties that justify its status as an AI revolution, and it’s here to stay. We may not be using neural networks many decades from now, but whatever we use will directly inherit from modern deep learning and its core concepts. These important properties can be broadly sorted into three categories:

Simplicity—Deep learning makes problem solving much easier, because it automates what used to be the most crucial step in a machine learning workflow: feature engineering. Previous machine learning techniques—shallow learning—only involved transforming the input data into one or two successive representation spaces, which wasn’t expressive enough for most problems. As such, humans had to go to great lengths to make the initial input data more amenable to processing by these methods: they had to manually engineer good representations for their data. This is called feature engineering. Deep learning, on the other hand, completely automates this step: with de

### PDF p.205 (printed 181) — 6 The universal workflow of machine learning › 6.2 Developing a model › Beating a baseline
As you start working on the model itself, your initial goal is to achieve statistical power, as you saw in chapter 5—that is, to develop a small model that is capable of beating a simple baseline.

At this stage, these are the three most important things you should focus on:

Feature engineering—Filter out uninformative features (feature selection) and use your knowledge of the problem to develop new features that are likely to be useful.

Selecting the correct architecture priors—What type of model architecture will you use? A densely connected network, a ConvNet, a recurrent neural network, a Transformer? Is deep learning even a good approach for the task, or should you use something else?

Selecting a good enough training configuration—What loss function should you use? What batch size and learning rate?

### PDF p.266 (printed 242) — 8 Image classification › 8.2 Training a convnet from scratch on a small dataset › The relevance of deep learning for small-data problems
What qualifies as “enough samples” to train a model is relative—relative to the size and depth of the model you’re trying to train, for starters. It isn’t possible to train a ConvNet to solve a complex problem with just a few tens of samples, but a few hundred can potentially suffice if the model is small and well regularized and the task is simple. Because ConvNets learn local, translation-invariant features, they’re highly data efficient on perceptual problems. Training a ConvNet from scratch on a very small image dataset will still yield reasonable results despite a relative lack of data, without the need for any custom feature engineering. You’ll see this in action in this section.

What’s more, deep learning models are by nature highly repurposable: you can take, say, an image-classification or speech-to-text model trained on a large-scale dataset and reuse it on a significantly different problem with only minor changes. Specifically, in the case of computer vision, many pretrained classification models are publicly available for download and can be used to bootstrap powerful vision models out of very little data. This is one of the greatest strengths of deep learning: feature

### PDF p.29 (printed 5) — 1 What is deep learning? › 1.4 Learning rules and representations from data › 
output. The measurement is used as a feedback signal to adjust the way the algorithm works. This adjustment step is what we call learning. A machine learning model transforms its input data into meaningful outputs, a process that is “learned” from exposure to known examples of inputs and outputs. Therefore, the central problem in machine learning and deep learning is to meaningfully transform data: in other words, to learn useful representations of the input data at hand—representations that get us closer to the expected output. Before we go any further, what’s a representation? At its core, it’s a different way to look at data to represent or encode data. For instance, a color image can be encoded in the RGB format (red-green-blue) or in the HSV format (hue-saturation-value): these are two different representations of the same data. Some tasks that may be difficult with one representation can become easy with another. For example, the task “Select all red pixels in the image” is simpler in the RGB format, whereas “Make the image less saturated” is simpler in the HSV format. Machine learning models are all about finding appropriate representations for their input data—transformatio
