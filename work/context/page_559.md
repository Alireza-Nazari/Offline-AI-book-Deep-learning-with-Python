# Generation bundle for page_559  (PDF page 559, printed page 535)

Page kind: content
Chapter: 17 Image generation
Section: 17.3 Text-to-image models
Subsection: 17.3.1 Exploring the latent space of a text-to-image model
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_559.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_559.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p559_e03] (body) representation of the entire token sequence learned by the text encoder. This effectively gives our denoising model more information to work with. The authors do this for both the positive and negative prompts, so we have a total of four tensors here:
[p559_e04] (bullet) The positive prompt’s encoder sequence
[p559_e05] (bullet) The negative prompt’s encoder sequence
[p559_e06] (bullet) The positive prompt’s encoder vector
[p559_e07] (bullet) The negative prompt’s encoder vector
[p559_e08] (body) With our `generate()` function decomposed, we can now try walking the latent space between two text prompts. To do so, let’s build a function to interpolate between the text embeddings outputted by the model.
[p559_e09] (listing_caption) Listing 17.10 A function to interpolate text embeddings
[p559_e10] (code)
```
from keras import ops
def slerp(t, v1, v2):
    v1, v2 = ops.cast(v1, "float32"), ops.cast(v2, "float32")
    v1_norm = ops.linalg.norm(ops.ravel(v1))
    v2_norm = ops.linalg.norm(ops.ravel(v2))
    dot = ops.sum(v1 * v2 / (v1_norm * v2_norm))
    theta_0 = ops.arccos(dot)
    sin_theta_0 = ops.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = ops.sin(theta_t)
    s0 = ops.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    return s0 * v1 + s1 * v2
def interpolate_text_embeddings(e1, e2, start=0, stop=1, num=10):
    embeddings = []
    for t in np.linspace(start, stop, num):
        embeddings.append(
            (
                slerp(t, e1[0], e2[0]),
                e1[1],
                slerp(t, e1[2], e2[2]),
                e1[3],
            )
        )
    return embeddings
```
[p559_e11] (body) You’ll notice we use a special interpolation function called `slerp` to walk between our text embeddings. This is short for spherical linear interpolation—it’s a function that has been used in computer graphics for decades to interpolate points on a sphere.
[p559_e12] (body) Don’t worry too much about the math; it’s not important for our example, but it is important to understand the motivation. If we imagine our text manifold as a sphere and our two prompts as random points on that sphere, directly linearly interpolating

## Code annotations printed next to listings (callouts, in order)
- The second and fourth text
- embeddings are for the negative
- prompt, which we do not use.

## Listings
- Listing 17.10: A function to interpolate text embeddings (code elements p559_e10)

## Chapter outline
Chapter: 17 Image generation (PDF pages 532-561)
  - 17.1 Deep learning for image generation (p.532)
      - 17.1.1 Sampling from latent spaces of images (p.533)
      - 17.1.2 Variational autoencoders (p.534)
      - 17.1.3 Implementing a VAE with Keras (p.537)
  - 17.2 Diffusion models (p.542)
      - 17.2.1 The Oxford Flowers dataset (p.544)
      - 17.2.2 A U-Net denoising autoencoder (p.545)
      - 17.2.3 The concepts of diffusion time and diffusion schedule (p.547)
      - 17.2.4 The training process (p.549)
      - 17.2.5 The generation process (p.551)
      - 17.2.6 Visualizing results with a custom callback (p.552)
      - 17.2.7 It’s go time! (p.553)
  - 17.3 Text-to-image models (p.555)
      - 17.3.1 Exploring the latent space of a text-to-image model (p.557)

## Previous page (PDF 558) text, for continuity
Before we can do this, we will need to break up our high-level `generate()` function into its constituent parts. Let’s try that out.

Listing 17.9 Breaking down the `generate()` function

Our generation process has three distinct steps:

1 First, we take our prompts, tokenize them, and embed them with our text encoder. 2 Second, we take our text embeddings and pure noise and progressively “denoise”

the noise into an image. This is the same as the flower model we just built. 3 Lastly, we map our model outputs, which are from `[-1, 1]` back to `[0, 255]`, so we can render the image. One thing to note here is that our text embeddings actually contain four separate tensors:

Rather than only passing the final, embedded text vector to the denoising model, the Stable Diffusion authors chose to pass both the final output vector and the last

## Next page (PDF 560) text, for continuity
between these two points would land us inside the sphere. We would no longer be on its surface. We would like to stay on the surface of the smooth manifold learned by our text embedding— that’s where embedding points have meaning for our denoising model. See figure 17.16.

Of course, the manifold learned by our text embedding model is not actually spherical. But it’s a smooth surface of numbers all with the same rough magnitude—it is sphere-like, and interpolating as if we were on a sphere is a better approximation than interpolat

Figure 17.16 Spherical interpolation keeps us

ing as if we were on a line.

close to the surface of our manifold.

With our interpolation defined, let’s try walking between the text embeddings for two prompts and generating an image at each interpolated output. We will run our `slerp` function from 0.5 to 0.6 (out of 0 to 1) to zoom in on the middle of the interpolation right when the “morph” becomes visually obvious (figure 17.17):

Figure 17.17 Interpolating between two prompts and generating outputs

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.557 — 17 Image generation › 17.3 Text-to-image models
- PDF p.555 — 17 Image generation › 17.3 Text-to-image models
- PDF p.498 — 16 Text generation › 16.2 Training a mini-GPT
- PDF p.556 — 17 Image generation › 17.3 Text-to-image models
- PDF p.469 — 15 Language models and the Transformer › 15.3 The Transformer architecture
- PDF p.561 — 17 Image generation › 17.3 Text-to-image models
- PDF p.627 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.502 — 16 Text generation › 16.2 Training a mini-GPT

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.557 (printed 533) — 17 Image generation › 17.3 Text-to-image models › 17.3.1 Exploring the latent space of a text-to-image model
There is probably no better way to see the interpolative nature of deep neural networks than text diffusion models. The text encoder used by our model will learn a smooth, low-dimensional manifold to represent our input prompts. It’s continuous, meaning we have learned a space where we can walk from the text representation of one prompt to another, and each intermediate point will have semantic meaning. We can couple that with our diffusion process to morph between two images by simply describing each end state with a text prompt.

### PDF p.555 (printed 531) — 17 Image generation › 17.3 Text-to-image models › 17.3 Text-to-image models
We can use the same basic diffusion process to create a model that maps text input to image output. To do this we need to take a pretrained text encoder (think a transformer encoder like RoBERTa from chapter 15) that can map text to vectors in a continuous embedding space. Then we can train a diffusion model on `(prompt,` `image)` pairs, where each prompt is a short, textual description of the input image.

We can handle the image input in the same way as we did previously, mapping noisy input to a denoised output that progressively approaches our input image. Critically, we can extend this setup by also passing the embedded text prompt to the denoising model. So rather than our denoising model simply taking in a `noisy_images` input, our model will take two inputs: `noisy_images` and `text_embeddings`. This gives a leg up on the flower denoiser we trained previously. Instead of learning to remove noise from an image without any additional information, the model gets to use a textual representation of the final image to help guide the denoising process. After training is when things get a bit more fun. Because we have trained a model that can map pure noise to images conditioned on

### PDF p.556 (printed 532) — 17 Image generation › 17.3 Text-to-image models › 
Like the `CausalLM` task we covered last chapter, the `TextToImage` task is a high-level class for performing image generation conditioned on text input. It wraps tokenization and the diffusion process into a high-level generate call.

The Stable Diffusion model actually adds a second “negative prompt” to its model, which can be used to steer the diffusion process away from certain text inputs. There’s nothing magic here. To add a negative prompt, you could simply train a model on triplets: `(image,` `positive_
` `prompt,` `negative_prompt)`, where the positive prompt is a description of the image, and the negative prompt is a series of words that do not describe the image. By feeding the positive and negative text embedding to the denoiser, the denoiser will learn to steer the noise toward images that match the positive prompt and away from images that match the negative prompt (figure 17.14). Let’s try removing

Figure 17.14 Using a negative prompt to steer

the color blue from our input:

the model away from the color blue

### PDF p.533 (printed 509) — 17 Image generation › 17.1 Deep learning for image generation › 17.1.1 Sampling from latent spaces of images
The key idea of image generation is to develop a low-dimensional latent space of representations (which, like everything else in deep learning, is a vector space) where any point can be mapped to a “valid” image: an image that looks like the real thing. The module capable of realizing this mapping, taking as input a latent point and outputting an image (a grid of pixels), is usually called a generator, or sometimes a decoder. Once such a latent space has been learned, you can sample points from it, and, by mapping them back to image space, generate images that have never been seen before (see figure 17.1)—the in-betweens of the training images.

Figure 17.1 Using a latent vector space to sample new images

Further, text-conditioning makes it possible to map a space of prompts in natural language to the latent space (see figure 17.2), making it possible to do language-guided image generation—generating pictures that correspond to a text description. This category of models is called text-to-image models.

Interpolating between many training images in the latent space enables such models to generate infinite combinations of visual concepts, including many that no one had explicitly c

### PDF p.561 (printed 537) — 17 Image generation › 17.3 Text-to-image models › Summary
Image generation with deep learning is done by learning latent spaces that capture statistical information about a dataset of images. By sampling and decoding points from the latent space, you can generate never-before-seen images. There are three major tools to do this: VAEs, diffusion models, and GANs.

VAEs result in highly structured, continuous latent representations. For this reason, they work well for doing all sorts of image editing in latent space: face swapping, turning a frowning face into a smiling face, and so on. They also work nicely for doing latent space–based animations, such as animating a walk along a cross section of the latent space, showing a starting image slowly morphing into different images in a continuous way.

Diffusion models result in very realistic outputs and are the dominant method of image generation today. They work by repeatedly denoising an image, starting from pure noise. They can easily be conditioned on text captions to create text-to-image models.

Stable Diffusion 3 is a state-of-the-art pretrained text-to-image model that you can use to create highly realistic images of your own.

The visual latent space learned by such text-to-image diff

### PDF p.643 (printed 619) — index ›  › 
stemming, defined 386–387

TensorFlow Serving 185 step fusing 556 tensor operations, derivatives of 42–43 stochastic gradient descent 43–46 tensors, defined 63 strided convolutions 239 test images, preparing 321 subclassing Model class 202 test-time training 582 rewriting previous example as subclassed text classification 381 model 202

augmenting text data 413 what subclassed models don't support 204 NLP (natural language processing) 382–384 subgraph isomorphism 586 preparing text data 384–394 subword tokenization 390–394 pretraining word embeddings 414 summary() method 193 sequence models 405 supervised sets vs. sequences 395–398 fine-tuning 495 setting models 398–404 learning 214 training recurrent model 406 symbolic AI 3 using pretrained embedding for symbolic tensor 196

classification 418 symbols 584–594 word embeddings 409, 410 blending deep learning and program text-conditioning 509 synthesis 590–592 text_dataset_from_directory utility 397 cognition as combination of both kinds of abstraction 587 text_embeddings input 531 deep learning and 588 text generation 466 long-term vision 594 history of sequence generation 468–470 modular component recombination and lifelong LLMs (l

### PDF p.446 (printed 422) — 15 Language models and the Transformer › 15.1 The language model › 
What about other text-based tasks like question answering or translation? For many real-world problems, we are interested in a model that can generate a text output for a given input. Just like we needed tokenizers and embeddings to help us handle text on the way in to a model, we must build up some techniques before we can produce text on the way out. We don’t need to start from scratch here; we can continue to use the idea of an integer sequence as a natural numeric representation for text. In the previous chapter, we covered tokenizing a string, where we split inputs into tokens and map each token to an int. We can detokenize a sequence by proceeding in reverse—map ints back to string tokens and join them together. With this approach, our problem becomes building a model that can predict an integer sequence of tokens. The simplest option to consider might be to train a direct classifier over the space of all possible output integer sequences, but some back-of-the-envelope math will quickly show this is intractable. With a vocabulary of 20,000 words, there are 20,000 ^ 4, or 160 quadrillion possible 4-word sequences, and fewer atoms in the universe than possible 20-word sequences

### PDF p.524 (printed 500) — 16 Text generation › 16.4 Going further with LLMs › 
Each of our input prompts contains the special token `<start_of_image>`. This is turned into 256 placeholder values in our input sequence, which, in turn, is replaced with the soft tokens representing our image.

Training for a multimodal model like this is quite similar to regular LLM pretraining and fine-tuning. Usually, you would want to first pretrain your image encoder separately, like we first did in Chapter 8 of this book. Then you can simply do the same basic “guess the next word” pretraining and also feed in mixed image and text content combined into a single sequence. Our transformer would not be trained to output image soft tokens; we would simply zero the loss at these image token locations. It might seem almost magical that we can simply add image data to an LLM, but when we consider the power of the sequence model we’re working with, it’s really quite an expected result. We’ve taken a Transformer, recast our image input as sequence data, and done a lot of extra training. The model can preserve the original language model’s ability to ingest and produce text while learning to also embed images in the Transformer’s latent space.
