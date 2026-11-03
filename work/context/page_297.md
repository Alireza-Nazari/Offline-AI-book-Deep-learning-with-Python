# Generation bundle for page_297  (PDF page 297, printed page 273)

Page kind: content
Chapter: 9 ConvNet architecture patterns
Section: 9.2 Residual connections
Subsection: 
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_297.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_297.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p297_e03] (body) The fix is simple: just force each function in the chain to be nondestructive—to retain a noiseless version of the information contained in the previous input. The easiest way to implement this is called a residual connection. It’s dead easy: just add the input of a layer or block of layers back to its output (see figure 9.3). The residual connection acts as an information shortcut around destructive or noisy blocks (such as blocks that contain ReLU activations or dropout layers), enabling error gradient information from early
[p297_e04] (figure_caption) Figure 9.3 A residual connection
[p297_e05] (body) layers to propagate noiselessly through a deep net
[p297_e06] (figure_caption) around a processing block
[p297_e07] (body) work. This technique was introduced in 2015 with the ResNet family of models (developed by He et al. at Microsoft).1
[p297_e08] (body) In practice, you’d implement a residual connection like the following listing.
[p297_e09] (listing_caption) Listing 9.1 A residual connection in pseudocode
[p297_e10] (code)
```
x = ...
residual = x
x = block(x)
x = add([x, residual])
```
[p297_e11] (body) Note that adding the input back to the output of a block implies that the output should have the same shape as the input. This is not the case if your block includes convolutional layers with an increased number of filters or a max pooling layer. In such cases, use a 1 × 1 `Conv2D` layer with no activation to linearly project the residual to the desired output shape. You’d typically use `padding="same"` in the convolution layers in your target block to avoid spatial downsampling due to padding, and you’d use strides in the residual projection to match any downsampling caused by a max pooling layer.
[p297_e12] (listing_caption) Listing 9.2 The target block changing the number of output filters
[p297_e13] (code)
```
import keras
from keras import layers
inputs = keras.Input(shape=(32, 32, 3))
```
[p297_e14] (footnote) 1 Kaiming He et al., “Deep Residual Learning for Image Recognition,” Conference on Computer Vision and Pattern Recognition (2015), https://arxiv.org/abs/1512.03385.

## Code annotations printed next to listings (callouts, in order)
- Some input tensor
- Saves a reference to the original
- input. This is called the residual.
- This computation block can potentially
- Adds the original input to the layer’s
- be destructive or noisy, and that’s fine.
- output. The final output will thus
- always preserve full information
- about the original input.

## Listings
- Listing 9.1: A residual connection in pseudocode (code elements p297_e10)
- Listing 9.2: The target block changing the number of output filters (code elements p297_e13)

## Figures
- Figure 9.3: A residual connection  | labels inside the figure: Output, Residual, connection, +, Block, Input
- Figure None: around a processing block  | labels inside the figure: (none - image)

## Equation candidates (formulas written inline in the text)
- [p297_e11] You’d typically use `padding="same"` in the convolution layers in your target block to avoid spatial downsampling due to padding, and you’d use strides in the residual projection to match any downsampling caused by a max pooling layer.

## Chapter outline
Chapter: 9 ConvNet architecture patterns (PDF pages 292-307)
  - 9.1 Modularity, hierarchy, and reuse (p.293)
  - 9.2 Residual connections (p.296)
  - 9.3 Batch normalization (p.300)
  - 9.4 Depthwise separable convolutions (p.302)
  - 9.5 Putting it together: A mini Xception-like model (p.304)
  - 9.6 Beyond convolution: Vision Transformers (p.306)

## Previous page (PDF 296) text, for continuity
(continued)

chances are you can remove a few modules (or replace some trained features with random ones) with no loss of performance. This is made worse by the incentives that deep learning researchers face: by making a system more complex than necessary, they can make it appear more interesting or more novel and thus increase their chances of getting a paper through the peer review process. If you read lots of deep learning papers, you will notice that they’re often optimized for peer review in both style and content in ways that actively hurt clarity of explanation and reliability of results. For instance, mathematics in deep learning papers is rarely used for clearly formalizing concepts or deriving unobvious results—rather, it gets used as a signal of seriousness, like an expensive suit on a salesperson. The goal of research shouldn’t be merely to publish but to generate reliable knowledge. Crucially, understanding causality in your system is the most straightforward way to generate reliable knowledge. And there’s a very low-effort way to look into causality: ablation studies. Ablation studies consist of systematically trying to remove parts of a system—-that is, make it simpler—to identify where its performance actually comes from. If you find that X + Y + Z gives you good results, also try X, Y, Z, X + Y, X + Z, Y + Z and see what happens. If you become a deep learning researcher, cut through the noise in the research process: do ablation studies for your models. Always ask, could there be a simpler explanation? Is this added complexity really necessary? Why?

9.2

Residual connections

You probably know about the game of telephone, also called Chinese whispers in the UK and téléphone arabe in France, where an initial message is whispered in the ear of a player,  …

## Next page (PDF 298) text, for continuity
Listing 9.3 The target block including a max pooling layer

To make these ideas more concrete, here’s an example of a simple ConvNet structured into a series of blocks, each made of two convolution layers and one optional max pooling layer, with a residual connection around each block:

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.305 — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model
- PDF p.299 — 9 ConvNet architecture patterns › 9.2 Residual connections
- PDF p.546 — 17 Image generation › 17.2 Diffusion models
- PDF p.304 — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model
- PDF p.627 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.295 — 9 ConvNet architecture patterns › 9.1 Modularity, hierarchy, and reuse
- PDF p.626 — 20 Conclusions › 20.1 Key concepts in review
- PDF p.339 — 11 Image segmentation › 11.2 Training a segmentation model from scratch

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.546 (printed 522) — 17 Image generation › 17.2 Diffusion models › 
extract for a given input image—that’s the `noise_rates` input. Rather than outputting a denoised image, we make our model output a predicted noise mask, which we can subtract from the input to denoise it.

For our denoising model, we’re going to use a U-Net—a kind of ConvNet originally developed for image segmentation. It looks like figure 17.9.

Figure 17.9 Our U-Net-style denoising autoencoder architecture

This architecture features three stages:

1 A downsampling stage, made of several blocks of convolution layers, where the

inputs get downsampled from their original 128 × 128 size down to a much smaller size (in our case, 16 × 16). 2 A middle stage, where the feature map has a constant size. 3 An upsampling stage, where the feature map get upsampled back to 128 × 128. There is a 1:1 mapping between the blocks of the downsampling and upsampling stages: each upsampling block is the inverse of a downsampling block. Importantly, the model features concatenative residual connections going from each downsampling block to the corresponding upsampling block. These connections help avoid loss of image detail information across the successive downsampling and upsampling operations. Le

### PDF p.299 (printed 275) — 9 ConvNet architecture patterns › 9.2 Residual connections › 
    elif filters != residual.shape[-1]:
        residual = layers.Conv2D(filters, 1)(residual)
    x = layers.add([x, residual])
    return x
x = residual_block(x, filters=32, pooling=True)
x = residual_block(x, filters=64, pooling=True)
x = residual_block(x, filters=128, pooling=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

### PDF p.394 (printed 370) — 13 Timeseries forecasting › 13.3 Recurrent neural networks › 
you of residual connections, which you learned about in chapter 9: it’s pretty much the same idea.

To understand this process in detail, let’s start from the `SimpleRNN` cell (see figure 13.8). Because you’ll have a lot of weight matrices, index the `W` and `U` matrices in the cell with the letter `o` (`Wo` and `Uo`) for output.

Figure 13.8 The starting point of an `LSTM` layer: a `SimpleRNN`

Let’s add to this picture an additional data flow that carries information across timesteps. Call its values at different timesteps `Ct`, where C stands for carry. This information will have the following effect on the cell: it will be combined with the input connection and the recurrent connection (via a dense transformation: a dot product with a weight matrix followed by a bias add and the application of an activation function), and it will affect the state being sent to the next timestep (via an activation function and a multiplication operation). Conceptually, the carry dataflow is a way to modulate the next output and the next state (see figure 13.9). Simple so far.

Figure 13.9 Going from a SimpleRNN to an LSTM: adding a carry track

### PDF p.627 (printed 603) — 20 Conclusions › 20.1 Key concepts in review › 
When building a very deep ConvNet, it’s common to add batch normalization layers as well as residual connections—two architecture patterns that help gradient information flow smoothly through the network.

### PDF p.305 (printed 281) — 9 ConvNet architecture patterns › 9.5 Putting it together: A mini Xception-like model › 
Deep and narrow is better than broad and shallow.

Introducing residual connections around blocks of layers helps you train deeper networks.

It can be beneficial to introduce batch normalization layers after your convolution layers.

It can be beneficial to replace `Conv2D` layers with `SeparableConv2D` layers, which are more parameter efficient.

Let’s bring all of these ideas together into a single model. Its architecture resembles a smaller version of Xception. We’ll apply it to the dogs-versus-cats task from last chapter. For data loading and model training, simply reuse the exact same setup as what we used in chapter 8, section 8.2—but replace the model definition with the following ConvNet:

### PDF p.339 (printed 315) — 11 Image segmentation › 11.2 Training a segmentation model from scratch › 
The second half of the model is a stack of `Conv2DTranspose` layers. What are those? Well, the output of the first half of the model is a feature map of shape `(25,` `25,` `256)`, but we want our final output to predict a class for each pixel, matching the original spatial dimensions. The final model output will have shape `(200, 200, num_classes)`, which is `(200,` `200,` `3)` here. Therefore, we need to apply a kind of inverse of the transformations we’ve applied so far, something that will upsample the feature maps instead of downsampling them. That’s the purpose of the `Conv2DTranspose` layer: you can think of it as a kind of convolution layer that learns to upsample. If you have an input of shape `(100,` `100,` `64)` and you run it through the layer `Conv2D(128,` `3,` `strides=2, ` `padding="same")`, you get an output of shape `(50,` `50,` `128)`. If you run this output through the layer `Conv2DTranspose(64,` `3,` `strides=2,` `padding="same")`, you get back an output of shape `(100,` `100,` `64)`, the same as the original. So after compressing our inputs into feature maps of shape `(25,` `25,` `256)` via a stack of `Conv2D` layers, we can

### PDF p.498 (printed 474) — 16 Text generation › 16.2 Training a mini-GPT › 
Listing 16.4 A Transformer decoder block without cross-attention
from keras import layers
class TransformerDecoder(keras.Layer):
    def __init__(self, hidden_dim, intermediate_dim, num_heads):
        super().__init__()
        key_dim = hidden_dim // num_heads
        self.self_attention = layers.MultiHeadAttention(
            num_heads, key_dim, dropout=0.1
        )
        self.self_attention_layernorm = layers.LayerNormalization()
        self.feed_forward_1 = layers.Dense(intermediate_dim, activation="relu")
        self.feed_forward_2 = layers.Dense(hidden_dim)
        self.feed_forward_layernorm = layers.LayerNormalization()
        self.dropout = layers.Dropout(0.1)
    def call(self, inputs):
        residual = x = inputs
        x = self.self_attention(query=x, key=x, value=x, use_causal_mask=True)
        x = self.dropout(x)
        x = x + residual
        x = self.self_attention_layernorm(x)
        residual = x
        x = self.feed_forward_1(x)
        x = self.feed_forward_2(x)
        x = self.dropout(x)
        x = x + residual
        x = self.feed_forward_layernorm(x)
        return x

### PDF p.471 (printed 447) — 15 Language models and the Transformer › 15.3 The Transformer architecture › 
Listing 15.15 A Transformer decoder block
class TransformerDecoder(keras.Layer):
    def __init__(self, hidden_dim, intermediate_dim, num_heads):
        super().__init__()
        key_dim = hidden_dim // num_heads
        self.self_attention = layers.MultiHeadAttention(num_heads, key_dim)
        self.self_attention_layernorm = layers.LayerNormalization()
        self.cross_attention = layers.MultiHeadAttention(num_heads, key_dim)
        self.cross_attention_layernorm = layers.LayerNormalization()
        self.feed_forward_1 = layers.Dense(intermediate_dim, activation="relu")
        self.feed_forward_2 = layers.Dense(hidden_dim)
        self.feed_forward_layernorm = layers.LayerNormalization()
    def call(self, target, source, source_mask):
        residual = x = target
        x = self.self_attention(query=x, key=x, value=x, use_causal_mask=True)
        x = x + residual
        x = self.self_attention_layernorm(x)
        residual = x
        mask = source_mask[:, None, :]
        x = self.cross_attention(
            query=x, key=source, value=source, attention_mask=mask
        )
        x = x + residual
        x = self.cross_attention_layernorm(x)
        residual = x
   
