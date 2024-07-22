# Neural Network Architectures
Basic feed-forward networks were [previously discussed](DeepLearning_NOTES.md). Now different architectures of Neural Networks will be discussed.

Ways to specialize a pre-trained base model trained on a generic database:
- **Transfer Learning**: Freeze existing layer weights, add a few hidden layers at the end of base model's hidden layers and only train these. Existing output layer is usually replaced with a new one.
- **Fine Tuning**: All layers' weights are modified - no frozen layers. *Generally used when specialized task is significantly different from general task*. If we don't want to retrain, we can train a second *downstream model* that converts base model's output into specialized output.

Transfer Learning vs Downstream Models : In transfer learning, base model's output layer is discarded. Whereas while training downstream models, output of base model is used as input (which includes base model's output layer).

### Image Processing
Image Tensors' dimensions depend on the no. of channels:
- Grayscale - 1D (each pixel has only one colour value)
- Coloured  - 3D (each pixel has RGB colour values)

Each image is a sample, and each image pixel is a feature.

**Sliding vs Rolling Windows**: Sliding (overlapping window), Rolling (non-overlapping window)

### Natural Language Processing (NLP)
NLP Tasks:
- *Sentiment Analysis*: classify text as positive / negative
- *Translation*
- *Answer Questions* based on text, like who is the hero, what actions occurred, etc.
- *Summarize / Paraphrase*
- **Natural Language Generation (NLG)** - gen text that follows *prompt*. A popular technique is **Autoregression** in which one word is generated following prompt, then it's appended to prompt and then next word generated, and so on.
- *Logical Flow*: Sentence S1 asserts a premise, sentence S2 asserts condition based on premise. Verify if S2 logically follows from S1. **It's difficult, generally requires human help.**

A *Language Model* takes a sequence of words as input and tells us probability of it being a well-formed sentence. 

2 approaches:
- Character-based
- Word-based approach - used most of the time - each word is converted to a number, where the *vocabalary* has the list of all words and their corresponding indices / numbers.

## Convolutional Neural Networks (CNN)
Deep Learning Models in which Convolutional Layers play a significant role. *All-Convolution Networks* have all layers as only convolutional layers.

**Applications:** Very successful in images (classification, modification, generation), also used in other domains like NLP (understanding sentence structure, sentence classification).

We can make a neuron *filter* to detect yellow (one value to detect yellowness from 3 initial RGB values): $ReLU(R+G-B)$ and then *sweep* it over all image pixels. *Sweeping* corresponds to **convolution** matrix operation. To make this faster, we can make a huge grid of neurons, each with identical weights, and apply it at once to the image - this is called *weight sharing*.

Convolution of input matrix $A$ with kernel matrix $B$ involves flipping kernel and sliding it over input, calculating sum of element-wise product (where both matrices are of size $M \times N$):

$$C(i, j) = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} A(i+m, j+n) \cdot B(m, n)$$

We can also sweep over the image with a *spatial filter*, where several pixels centered over an *anchor* points are processed at once by a single neuron. Its *footprint / local receptive field* is the shape of the input pixels passed to a single neuron - eg. we can process a pixel and its 8 neighbours at once (3x3 square shape) - so the neuron has 9 input features (assuming grayscale image). Usually small square $side \in [1,9]$ with odd size is used.

*Feature* (NOT the same as before, where it simply meant columns) also refers to a particular type of structure that a neuron is looking for (eg. zebra stripe). So the filters are also called *feature detectors*, and their output *feature map* tells us (for each pixel) how well image around that pixel matched feature we're looking for. Applying one filter, then another filter on outputs of first filter, and so on, allows us to detect very complex features.

Around the border of the image, we add *Padding* - extra elements, usually 0, so that spatial filter can get values at corners also. Thickness of border padding depends on size of filter.

For RGB images (3 channels), spatial filter and footprint both have 3xNxN size (where N is side of square). *Every filter must have same no. of channels as the image it's filtering*.

*Convolution Layer*: applies many filter neurons simultaneously & independantly on the image. Usually same padding is used in all the filters of a layer. Its output tensor will have as many channels as filters in the layer.

*Initial (Random) Weights* are usually chosen automatically by library. Best algos are by He, Glorot.

**1D Convolution**: special case where filter moves only in one direction, width or height. Eg. NLP - grid where rows are words (each cell is a letter), and spatial filter of size same as width is swept only in vertical direction.

**1x1 Convolution** (feature/channel reduction): a layer can have many similar filters that can be combined. Eg. a layer has 300 filters, then next layer has all its filters 1x1 (i.e., sweeping through exactly one cell at a time), and reduce no. of layers to 175 - result is that 300 channels in first layer get compressed to 175 channels in next layer. Note that both layers have same width, height. Particularly effective if channels of input layer are correlated.

**Pooling / Downsampling** (reduce width, height): blur image (or output tensor of previous layer) by aggregating each rolling window into single value. Pooling allows our convolutions to be *shift invariant* (i.e., tolerate some error due to some parts of feature being slightly out of alignment)
- Average Pooling: *box / low-pass filter* averages cells in window using a kernel matrix having all 1s.
- **Preferred** Max Pooling: learns a lot quicker than average pooling
- *Striding*: combines pooling with layer output to make it faster. During layer inference, convolution filter of NxN size and step size > 1 is used. Most common is filter size = step size, which means non-overlapping windows.

**Upsampling** (increase width, height, keeping channels constant)
- *Transposed Convolution / Fractional Striding*: combines convolution & upsampling: add extra empty row & column (filled with zeros) in between each original element, then surround with ring of zeros as before and then do convolution.

NOTE: Striding gives different (but similar) output than convolution, followed by pooling. A network trained with convolution + pooling layer can't just be replaced with a striding layer (and vice versa). Similarly for Upsampling and Transposed Convolution.

**CNN Best Practices**:
- Choose small filter sizes - eg. 2x2, 3x3, 5x5
- Start with 2-3 Convolutional Layers - if model underfits (not learning enough), then increase layers.
- Use Max Pooling after 1-2 Convolutional Layers to reduce dimensionality - 2x2 pooling with stride 2 is a popular choice.
- Dense (fully connected) layers are optionally placed at the end of the network.

**CNN Applications besides Images**:
CNNs can also be used for non-image applications. CNNs are useful wherever there's a spatial relation between inputs. Here, "spatial" simply means continous - time series could also be used as input. 1D convolution with temporal data can be used - note that 1D convolution should be used instead of 2D is because inputs only have inherent ordering in one dimension (time) - see details [here](https://stats.stackexchange.com/a/550774/406211).

### Real-Life CNN Examples
- Convolution Network for classifying handwritten gray-scale digits (MNIST dataset of 60,000 images):

![CNN Digit Classifier](cnn_digit_classifier.png)

**NOTE:** In the convolution layers, 32 x (3x3) means 32 filter neurons, each having (3x3) footprint. Stride is not mentioned, that means default stride 1 is used.

Though only 2 convolution layers (represented as boxes with double borders) are used at start for feature detection, this is still a CNN since they do heavylifting. 2 fully-connected/dense (i.e., traditional non-convolutional layers) are used at the end (represented as boxes with crosses), with support layers (pooling, dropout, etc.). Since no padding is applied, convolutional output width, height are smaller than input (border pixels don't show in output) - this is ok in this case since most images in the dataset have 4 extra border pixels.

Here a *flatten* layer converts N-dimensional convolution output tensor into a 1D vector that can be input to dense/fully-connected layers.

We know we haven't overfit because training & validation accuracy curves don't diverge, despite 99% accuracy.

- VGG16: CNN with 16 layers for classifying photos from ImageNet dataset:

![VGG 16 CNN Layers](vgg_cnn_layers.png)

### Visualizing CNN Filters (neurons)
We can visualize what patterns an individual filter (CNN layer neuron) is detecting by finding images that maximizes filter's outputs.
Only considering the Convolutional layers:
- feed in random numbers initially for the image - in feature map for the filter, use filter's output's sum as replacement for error.
- Like backprop, compute gradients using this error, going back to first Convolutional layer. But DON'T adjust weights, instead adjust input values in the direction of *maximum error* using gradients of first convolutional layer. (Max error to stimulate the neuron as much as possible).

Another (easier for me!) method is to simply take a sample input image, and look at the output image for a single filter.

### Adverserial Inputs
VGG is quite accurate, but we can still make small changes input image's pixel values that aren't noticeable to human eye, but completely mess up CNN output. Such images are called *adversaries*, and change made to input image is called *adverserial preturbation*. Sometimes we find preturbations that mess up model output for any image, these are called **universal preturbation**.


## Autoencoder Architecture
Learns how best to encode a type of data. Mainly used for cleaning noise (eg. denoising grainy images), and for reducing dimensionality. Decoder (2nd half on autoencoder) can be used as a **Generator** that creates new data that looks like the training data. Note that it doesn't always compress input data into fewer latent variables - in Denoising, latent variables can be actually more than input variables!

**Blending**:
- Content Blending: add/superimpose x% of image1 and y% of image2
- Parametric / Representation Blending / *Interpolation*: For 2 images, find their latent vars, calc intermediate latent point and then decode it - resultant blended image conceptually represents half of both images.

Autoencoders allow us to blend compressed versions together, and (to a degree) recover the blended uncompressed data.

![Autoencoder Bottleneck](autoencoder_bottleneck.png)

Here the initial dense layers encode N input variables into 20 *latent variables*. Then the dense layers after that decode 20 compressed variables back into N output vars. The narrow area in between encoder and decoder is called *bottleneck*, and last encoding layer (just before bottleneck) is called *bottleneck layer*. The space of all latent variables is called *Latent Space*. Like in this example, we often build encoding layers that decrease no. of vars by 2 times (512 -> 256), and corresponding increase in vars in decoding layers. 

Autoencoder is **semi-supervised learning** because we give an explicit goal (output should equal input) but there aren't any manually determined classes / target variables associated with each input.

The structure of Latent Space isn't immediately apparent, however we can see that there is structure by decoding nearby points in latent space - the system tends to cluster together similar images.

### Convolutional Autoencoders

![Convolutional Autoencoder](convolutional_autoencoder.png)

Here pooling is used between convolutional layers to reduce image width, height in encoder. Upsampling is used in decoder to increase image width, height to original. This performs quite better (i.e., decodes latent variables better).

*Denoising Autoencoder*: Removing noise from image. Note that in below example, input of shape 28x28x1 (784 nums) is encoded into latent variables of shape 7x7x32 (1568 nums) which is about double of nums in original input! But this is ok in this case since we aren't concerened with compressing. Also note that pooling and upsampling layers have been combined with convolution here, which reduced one-third of training time.

![Denoising Autoencoder](denoising_autoencoder.png)

### Variational Autoencoders (VAE)
They have some degrees of randomness and are non-deterministic - for same input, latent vars output each time are slightly different.

1. All Latent variables should be gathered into one region of latent space, so we know range for random values.
2. Latent vars of similar inputs should be close together.
3. Minimize empty regions in latent space.

For point 1 (all latent vars in bounded space), add a constraint to error function that each latent var should approximately form unit gaussian (most points close to 0). 

*Reparametrization Trick*: From each latent variable from encoder, calculate 2 nums: center (mean), and standard deviation (spread). Then pick a random number from a Gaussian distribution with this mean and std-dev. The resultant number is usually very close to center. The new numbers are now used as Latent Variables by decoder - it gives a big error if new image is far from original image. System learns what should be center, spread for each latent variable's gaussian distribution.

The final loss/error function will also check the similarity between weights of encoder and decoder layers. It's done using **Kullback-Leibler (KL) Divergence**, which measures error from encoding information using encoder different from optimal encoder. Here we're asserting that optimal decoder is opposite of optimial encoder.

Eg. The below is based on MLP autoencoder from *Caffe* library (MLP = Multi-Layer Perceptron means fully connected neural network). It has 50 latent variables:

![Variational Autoencoder](variational_autoencoder.png)


## Recurrent Neural Networks (RNN)
RNN remembers something about each sample before processing next sample. It's also used for generating new sequences. *Recurrent* means that it uses the hidden state for each input.

- *Sequence* has multiple test samples whose order matters (eg. words in sentence, image frames in video). Each sample in the sequence is called a *token*.

NLP issues (using normal fully-connected networks):
- *Semantics* (structure of language) is fundamentally different from mathematical curves.
- Even a tiny error in prediction gives incomprehensible text (since one word to the next can be completely different).
- Word order is not preserved.

In NLP, it represents words as simple numbers - first word encountered in text is given number 0, next 1, etc. If a word that's already known is encountered, it's skipped. The *vocabalary* contains all words and their corresponding num

RNN keeps a *hidden state* - output is a function of both input and this hidden state. Each successive input (called a *time step*) updates state - order of inputs matters.

This is the *rolled-up* diagram of a *recurrent cell*, where the *delay* (little black box) represents its hidden state. Its *unrolled* diagram is like a state machine diagram that shows all states and inputs causing state transitions explicitly. The exported state of the layer is usually shown with a dotted line to show that it's available if needed, but can be ignored if not required. 

**A recurrent cell contains one or more neural networks within it.** A recurrent cell is often placed in its own layer, called a *Recurrent Layer*. A network dominated by recurrent layers is called a *Recurrent Neural Network (RNN)*.

Internal state of a recurrent cell is represented by a tensor with a *width* and *height*. If cells in RNN network have same width, it's called *network width*.

![Recurrent Neural Cell (rolled)](rolled_recurrent_neural_cell.png)

The unrolled diagram of a recurrent cell is shown like this:

![Unrolled Recurrent Neural Cell](unrolled_recurrent_cell.png)

During training, gradient error needs to be propogated backwards sequentially from last cell to first. But that's not possible because they are all same cell! This is solved using technique **Backpropogation Through Time (BPTT)**. 

During backprop, suppose we find that gradient of a weight is smaller in previous layer. Then this process will keep on repeating for each previous layer and make the gradient for that weight exponentially smaller. But that means learning will slow down or even stop (due to gradient close to 0)! This is called **Vanishing Gradient** problem. Its inverse is called **Exploding Gradient** problem where the gradient increases exponentially during backprop.

### Long Short-Term Memory (LSTM)
These gradient issues can be solved with solved with a better recurrent cell, where the internal state changes frequently (short-term) but some portion of information is kept in state for a long time.

![LSTM](long_short_term_memory.png)

**LSTM contains 3 internal neural networks**:
- one to "forget" (discard) state that's no longer needed (means move number representing that state to 0)
- one to "remember" (add new info) to state (means adding new number to the state tensor at appropriate location)
- one to "select" a version of the state as output

LSTM doesn't require repeated copies of itself, so it solves the gradient problems of the basic recurrent cell. Neural networks inside it are trained normally using backprop and optimization.

When we say RNN, LSTM cell is implied. *Gated Recurrent Unit (GRU)* (a popular variant of LSTM) can also be used.

**Tradeoff in training RNN** is that we can use larger networks (with either large no. of cells, or cells with larger state memory) for better predictions, but that will take more time and memory.

### RNN Architectures
- **CNN-LSTM**: Useful for things like classifying video frames - convolutional layers find & identify objects, while recurrent layers track how the objects move from one layer to next. Note that LSTM is required for tracking objects to deal with cases that normal code can't handle easily - object disappearing & reappearing later, object following non-linear motion (eg. spinning), objects changing appearance, multiple object interactions, accurately work with noisy data, etc.

- **Deep RNN**: Multiple RNN cells stacked on top of each other.

### Translation with RNN
- **bi-Directional RNN (bi-RNN)**: Used for translating text, so that we can use context from both previous and later words. Here we have 2 RNNs - one that tracks state from start of sentence, one from end. Inputs are given to both RNNs and then their outputs for each word are joined/concat together.

![bi-Directional RNN](bidirectional_rnn.png)

*Deep bi-RNN* has multiple layers of bi-Directional RNNs.

- **Seq2Seq**: Word-wise translation of text is problematic because sentences in different languages can have different lengths (no. of words) and word order (eg. noun before or after verb). Seq2Seq architecture translates input sequence to output sequence (possibly of different length) as a whole.

![Seq2Seq](rnn_seq2seq.png)

2 RNNs are used - encoder and decoder. Encoder RNN's word outputs are ignored - instead its final state is passed to decoder RNN. Decoder RNN uses this as its initial state and produces words using autoregression. *Context Vector* is the final state of the encoder RNN which is passed to decoder.

Decoder RNN gets a special start token as first input: `[START]`. It stops generating with autoregression when it produces special end token `[END]`.

*Long-Term Dependency Problem*: Seq2Seq performs well with smaller sentences but breaks down for larger ones. This is because Context Vector (only thing passed from encoder to decoder) has to decode all information in the sentence, but it's of fixed size - no matter how large we make the Context Vector, we can always make a larger sentence that it can't represent.

### RNN Limitations
- Processes one word at a time only.
- Fixed-size context vector has to encode all sequence info - breaks down for larger sequences.


## Attention Networks and Transformers

### Word / Token Embeddings
In RNN words are represented as words, but we can instead encode them as vectors. *Embedder* algorithms create these *Word/Token Embeddings*. It's also possible to encode sentences instead of words, but word embeddings are more common.

Word Embedding vectors are encoded such that words with similar meanings have vectors near each other. So now we can tolerate some imprecision in predicted word embedding. Embedding vector similarity can be done with cosine similarity.

Pretrained Word Embedder algos: GLoVE, word2vec, fastText

**ELMo (Embedding from Language Models)**: Contextualized Word Embeddings
- ELMo has 2 layers of forward and backward bi-RNNs, grouped by direction. Layer 1 of forward bi-RNN feeds Layer 2 of forward bi-RNN, and similarly for backward bi-RNN layers, before finally concatenating outputs of both forward & backwards. OTOH in Deep bi-RNNs, layer 1 outputs of forward and backward are concatenated, then fed to next layer.
- Each input word is turned into 2 word tensors - from forward and backward networks, that are then concatenated to form final word embeddings which have context of both previous and next words. 

![ELMo Contextualized Word Embeddings](elmo_contextual_word_embeddings.png)

ELMo, and similar algos like *Universal Language Model Fine-Tuning (ULMFit)*, are typically trained on general-purpose databases. When we need them for specific downstream tasks (eg. legal, medical, etc.), they are fine-tuned with additional examples of that domain.

### Self-Attention - Query Kev Value (QKV)
Calculating word embeddings of a single word (called *Query*):
- Find $Q$ tensor for Query word.
- For a given word that we're comparing Query against:
    - Find 2 tensors: Key $K$, Value $V$
    - Using dot product, calc tensor $RawAttentionScore = \frac{Q \cdot K^T}{\sqrt{d_k}}$ where $d_k$ is the no. of dimensions in Key Tensor $K$.
    - Calc tensor AttentionWeight to compare how well Query matches Key: $softmax(rawAttentionScore)$. 
      **We can view each element in resultant tensor after softmax as relevance $\in [0,1]$ of the corresponding element in Value tensor to our Query word.**
    - Now scale Value tensor with this weight: $AttentionWeight \circ V$ where $\circ$ means multiplying the 2 tensors element-wise.
- Finally add results for each Key word: $\sum AttentionWeight \circ V$

So final formula is (final embedding after QKV for a single Query word):

$$\sum softmax(\frac{Q \cdot K^T}{\sqrt{d_k}}) \circ V$$

![Query Key Value (QKV)](attention_query_key_value.png)

In above diagram, dashed line indicates division by $\sqrt{d_k}$ followed by softmax in attention weight.

**3 Neural Networks are required, one each for calculating Query, Key, Value tensors for each word.** Usually all 3 only have one fully-connected layer (no hidden layers). Each of them takes word embedding tensor as input (created by embedder models like word2vec) and outputs a tensor. Word embeddings are already in a space where similar words are close to one another - Query and Key networks fine-tune these so that they can be meaningfully compared for relevance. *Value network has to make values that can be usefully mixed (scaled and added).*

**Seperate networks for Query and Key allows asymmetric relationship - eg. for "it" word, "cat" can be relevant, but the reverse need not be true.**

*Softmax Saturation*: Suppose only $Q \cdot K^T$ was used in raw attention score. Then due to exponential nature of softmax, any large value would completely dominate (be close to 1) and everything else would be close to 0. If that happens, system can't learn. To prevent this, division by $\sqrt{d_k}$ is done before softmax.

A Query word is scored with every word in sentence, including itself. **Usually the word will score highest with itself - but sometimes it can score higher with a different word.** For example, the pronoun "it" can score highest with "cat" word if that is most relevant.

All words in sentence are processed parallelly (using them as Query word, and all words of sentence as Key words). Since none of the steps depend on how long the sentence is (i.e., they can all be done parallelly), **a long sentence can be processed in same time as short sentence** with only constraint being how much memory and computing power are available.

