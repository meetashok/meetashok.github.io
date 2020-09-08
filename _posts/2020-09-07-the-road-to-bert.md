---
layout: post
title: 'The Road to BERT & Beyond'
date: 2020-09-07
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

The field of deep learning has seen incredible growth in the last
decade. It was fueled by massive datasets, higher computational
throughput via GPUs & neural architectures that could perform
zero-to-few shot learning. Computer vision was the first field in which
AI broke the barrier of human performance. It was driven by CNN
architectures that lend themselves to transfer learnings. Similar
approach eluded NLP for most part of this decade but BERT changed that.
With the use of Transformer based encoder architecture & unsupervised
tasks based learning, NLP models are now seeing rapid evolution. This
paper has a high level recap of ideas that led to BERT. Towards the end
of the paper, few ideas for Flipkart's chatbot strategy are explored.

Introduction
============

The march of deep learning
--------------------------

In our pursuit of designing algorithms that learn from examples, the
classical machine learning approaches have a fundamental limitation.
When dealing with non-tabular data like speech, text & images, the
target outcome has little correlation with individual units of feature
information (amplitude, pixels, characters). So, the shallow
architecture of classical machine learning models is insufficient & has
to be augmented with hand-engineered domain specific features. Deep
learning models have two key competitive advantages:

1.  By design, they can learn hierarchically more complex features. They
    do this directly from raw data enabling zero-to-few shot learning in
    many real-world applications.

2.  Deep learning models can be easily configured to be more flexible by
    adding more layers. In classical machine learning, building a more
    flexible model usually means choosing a completely different
    paradigm (e.g. regression vs. gradient boosting).

As a result, while classical models hit a performance plateau, deep
learning models are likely to deliver higher performance as the number
of layers & input data volume increases (fig
[\[fig: bayes-error\]](#fig: bayes-error){reference-type="ref"
reference="fig: bayes-error"}).

\centering
![Deep learning
performance[]{label="fig: bayes-error"}](bayes-error.png){#fig: bayes-error
width="50%"}

Peculiarities of language
-------------------------

Convolutional neural networks (CNN) brought about a revolution in our
ability to encode information from images. Images are spatially
hierarchical. The problem of detecting a person in image can be broken
down into the following steps: locate face, body, limbs. To identify a
face, locate hair, eyes, mouth. To identify eyes, locate curves, edges,
etc. Convolutional filters learn simple features (edges) in initial
layers. Chaining them leads to more complex features (face, limbs).
Further, while relative position of pixels is relevant but it matters
only locally. To identify a dog in an image, only a portion of the image
may be relevant. This insight of locality leads to the idea of parameter
sharing.

After the success of CNNs, they were applied to text data too.
Empirically, they work well enough for some language tasks but text is
inherently different from images. The first challenge with text is
representation. For text data to be fed to a model, we need to represent
it with numbers. Unlike images & tabular-data, text is of variable
length. Also, the idea of locality isn't necessarily a good choice. Text
is sequential. Learning from text should take into account the previous
occurrences, perhaps extending up to the beginning of the sequence. So,
we need a different modeling architecture that can learn temporal
dependencies.

Representation
==============

One-hot encoding
----------------

Until recently the representation and the modeling architecture problems
have been solved independently. The simplest way to represent text with
numbers is to tokenize text, build a vocabulary (size \|v\|) and then
represent each token with one-hot vector, $o_j$. This approach has the
following issues:

1.  Curse of dimensionality: The feature space will have as many
    dimensions as the cardinality of the vocabulary. Also, the feature
    space is very sparse.

2.  One-hot vectors don't account for semantic similarity between words.
    For example, the representations of the words \"doctor\" and
    \"nurse\" will be as dissimilar from each other as they would be
    from the representation of the word, \"aircraft\".

3.  No information on ordering of tokens is captured.

Word embeddings
---------------

"*You shall know a word by the company it keeps*"

Word embeddings is another way to represent text tokens. Each token is
represented by a dense vector of fixed size that is independent of
vocabulary size. Word embeddings are able to encode semantic meaning of
words.

Google introduced word2vec in 2013 [@mikolov2013efficient]. They
proposed two methods for training the embeddings: Skip-gram model and
continuous bag-of-words (CBoW). In CBoW, provided a context, we try to
predict the word. In skip-gram model, the word is provided and we
predict the context. Skip-gram exhibits better performance on rare words
but CBoW is faster to train. The training process can be accelerated
using techniques like hierarchical softmax classification. Another
approach is to use negative sampling. This reduces the problem from
multi-class classification to binary classification.

Another approach for building word embeddings is to start by building a
co-occurrence matrix, $\mathbf{X} \in \mathbb{R}^{|\text{v}|\times|\text{v}|}$. The value $X_{ij}$ represents the number of times words, $w_i$ & $w_j$ co-occur within a distance, $d$ of each other (known as context length) in the
training corpus. Using SVD, the matrix $\mathbf{C}$ can be factorized as $\mathbf{C} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\top}$. Retaining the first $k$ columns in matrix $\mathbf{U}$ will give us the word embeddings. GloVe [@pennington2014glove] solves this matrix factorization problem but builds a weighted cost function (eq.[\[eq: glove\]](#eq: glove){reference-type="ref" reference="eq: glove"}) to give higher weight to rare words. The performance from GloVe based recurrent models improved upon word2vec models. One of the reasons for better performance is that GloVe vectors are able to take into account the global information about word co occurrences.

$$\begin{split}
    \min_{\theta, e} J(\theta, e) &= \sum_{i=1}^{|\text{v}|}\sum_{j=1}^{|\text{v}|} f(X_{ij}) \left( \theta_i^{\top}e_j + b_i + b^{\prime}_j - \log X_{ij} \right) \\
    e_j^{final} &= \frac{\theta_j + e_j}{2} 
\end{split}
\label{eq: glove}$$

One of the challenges with both word2vec and GloVe embeddings is that if
a word isn't seen in the training vocabulary, then it will be assigned
an `<UNK>` token and will be initialized with random embeddings (same
for all unseen words). fastText [@bojanowski2017enriching] from Facebook
solved for this issue by learning subword or character level embeddings.

Another challenge is that the word embeddings are same irrespective of
the context. A word can take on different meanings depending on other
words around it. The meaning of the word *bank* is different in the two
sentences ([\[eq: contextual\]](#eq: contextual){reference-type="ref"
reference="eq: contextual"}) below. ELMo [@peters2018deep] uses a
bi-directional LSTM to train context specific word embeddings.

$$\begin{split}
    s1&: \textit{I went to the bank to withdraw money.} \\
    s2&: \textit{The river bank is brimming with people.}
\end{split}
\label{eq: contextual}$$

Recurrent Neural Networks
=========================

Let's say we have a sequence of data (words, tokens, etc.),
$\{x_1, x_2, \dots, x_{t-1}\}$. To predict the next word in the
sequence, we could use an autoregressive approach,
$$x_t \sim P(x_t | x_{t-1}, x_{t-2}, \dots, x_1)$$

Another approach called latent autoregressive models is to maintain a
state of the past data. RNN is an example of this approach. Let's say we
want to encode the sentence, "I like cats\", the unrolled RNN
architecture will look like figure
[\[fig: rnn\]](#fig: rnn){reference-type="ref" reference="fig: rnn"}.

\centering
![Encoding a sequence using RNN[]{label="fig: rnn"}](rnn){#fig: rnn
width="45%"}

Each RNN cell uses the hidden state from previous cell $h_{t-1}$ &
updates it based on the input $x_t$ (eq.
[\[eq: rnn\]](#eq: rnn){reference-type="ref" reference="eq: rnn"}). The
weight matrices $W_h$ and $W_y$ are shared across the different time
stages.

$$\begin{split}
    h_t &= \tanh{\left(W_h[x_t, h_{t-1}] + b_h\right)} \\
    \hat{y}_{t} &= g{\left(W_yh_t + b_y\right)}
\end{split} 
\label{eq: rnn}$$

One of the challenges with RNN is the vanishing gradient problem. When
the sequence length is long, as partial gradients are multiplied in
back-propagation through time, the product becomes too small. This
happens if $dW < 1$. If $dW > 1$, then the gradient could explode but it
can be tackled by clipping.

Take this sentence as an example, \"I visited France in summer. I
explored $\dots$ \<long-passage\>$\dots$. But I didn't go to it's
largest city, Paris.\". A language model will have to learn long-range
dependencies to make a link between France & Paris.

GRUs [@chung2014empirical] are a class of recurrent networks that make
use of gates to either update or forget previous hidden states (eq.
[\[eq: gru\]](#eq: gru){reference-type="ref" reference="eq: gru"}). If
$\Gamma_u$ is close to 1, then the hidden state will get updated based
on latest input, $x_t$. Whereas, if $\Gamma_u$ is close to 0, then the
previous hidden state $h_{t-1}$ will be preserved. GRUs are a special
case of LSTMs but work equally well in practice. LSTMs
[@10.1162/neco.1997.9.8.1735] use an explicit forget gate, $\Gamma_f$
instead of $(1 - \Gamma_u)$.

$$\begin{split}
    \tilde h_t &= \tanh{\left(W_c[x_t, h_{t-1}] + b_c\right)} \\
    \Gamma_u &= \sigma\left(W_u[x_t, h_{t-1}] + b_u\right) \\
    h_t &= \Gamma_u \odot \tilde h_t + (1 - \Gamma_u) \odot h_{t-1}\\
    \hat{y}_{t} &= g{\left(W_yh_t + b_y\right)}
\end{split} 
\label{eq: gru}$$

Like with deep neural networks, recurrent networks can also be stacked.
Each layer will maintain its own hidden state and weight matrices. In
practice, 2-3 layers are sufficient for recurrent networks, unlike CNNs
where 100s of layers are usually stacked.

One flaw of using a unidirectional RNN is that it may be important to
understand future context. For example, consider the sentences in
([\[eq: bidirectional\]](#eq: bidirectional){reference-type="ref"
reference="eq: bidirectional"}). The word *glass* has different meanings
in the two sentences but it can't be discerned by knowing just the words
prior to it. Bi-directional RNNs solve this problem, by using two
networks side by side. The input sequence is fed to the first network
and the reversed sequence in fed to the second network. One of the
disadvantages of Bi-RNNs is that the input sequence has to be fully
processed before any output can be generated.

$$\begin{split}
    s1&: \textit{From inside the glass building}\dots \\
    s2&: \textit{From inside the glass tumbler}\dots
\end{split}
\label{eq: bidirectional}$$

Encoder-Decoder Architecture
============================

Translating text from one language to another requires an
encoder-decoder network [@sutskever2014sequence]. Until recently, both
these used to be recurrent networks. Input sequences
$\{x_1, x_2, \dots, x_{T_x}\}$ are fed to the encoder recurrent network.
The hidden state from the last time-step (context = $c$) of the encoder
network is passed to the decoder network which generates the output
sequence $\{y_1, y_2, \dots, y_{T_y}\}$. This architecture is shown in
fig.
[\[fig: encoder-decoder\]](#fig: encoder-decoder){reference-type="ref"
reference="fig: encoder-decoder"}.

\centering
![Encoder-decoder
architecture[]{label="fig: encoder-decoder"}](encoder-decoder){#fig: encoder-decoder
width="1\linewidth"}

Goal of decoder is to take the context vector, $c$, and then generate
tokens, $\{y_1, y_2, \dots, y_{T_y}\}$ sequentially, such that the
conditional probability of output tokens is maximized (eq.
[\[eq: encoder-decoder\]](#eq: encoder-decoder){reference-type="ref"
reference="eq: encoder-decoder"}).
$$P(y_1, \dots, y_{T_y} | c) = \prod_{i=1}^{T_y} P(y_t | c, y_{t-1}, \dots, y_1)
\label{eq: encoder-decoder}$$

If the maximum output sequence length is $T_y$, and the vocabulary size
is $|\text{v}_{y}|$, then the number of possible output sequences is of
the order of $\mathcal{O}(|\text{v}_y|^{T_y})$. Exhaustive search of all
possible sequences is virtually impossible. So, a greedy approach of
choosing the output token with highest probability at each output time
step could be adopted. But this approach is suboptimal. An improvement
is to use Beam search, where $k$ most likely tokens are retained at each
output time step.

A challenge in passing only the last encoder hidden state to the decoder
is that the entire input sequence has to be encoded in a single vector
of fixed size. This can be problematic for input sequences of large
length. Further, when a sequenced is being decoded, the output tokens
may be closely dependent on only a few of the input tokens. Decoder has
to implicitly figure out these dependencies from just one context
vector. An improvement is to pass all hidden states to the decoder. This
is known as the attention mechanism.

The input to an attention layer is called query. It returns an output
based on a set of learnable key, value pairs. Let's say we have $n$ key
& value pairs $(\mathbf{k}_i, \mathbf{v}_i)$, and the query vector is
$\mathbf{q}$. Then the output vector, $\mathbf{o}$ is calculated by eq.
[\[eq: attention\]](#eq: attention){reference-type="ref"
reference="eq: attention"}.

$$\begin{split}
        a_i &= \alpha(\mathbf{q}, \mathbf{k}_i) \\
        b_i &= \frac{\exp(a_i)}{\sum_{j=1}^n\exp(a_j)} \\
        \mathbf{o} &= \sum_{i=1}^n b_i \mathbf{v}_i \\
    \end{split}
    \label{eq: attention}$$

The scoring function, $\alpha$ calculates similarity between the query
and key vectors. Dot-product is a common choice for the scoring
function. This method derives heavily from Nadaraya--Watson kernel
regression [@kernal]. The advantage of choosing this method is that it
is parameter free (information is contained in the data) & given enough
data it converges to the optimal solution. Attention based neural
machine translation was proposed by Google in 2016 [@wu2016googles].

Self-attention & Transformer
============================

While the attention mechanism had been discovered by 2016, it was used
in conjunction with recurrent networks. In Dec 2017, Google researches
[@vaswani2017attention] proposed a simpler encoder-decoder network based
solely on attention in their paper titled \"Attention Is All You Need\".
They called it the Transformer. This model needed less time to train as
computations could be parallelized. It delivered state-of-the-art
performance on several machine translation tasks (+2 BLUE on
English-to-German translation).

The core idea of the transformer is that when the encoder is encoding a
particular token, it is able to take into account the weighted impact of
every input token (including itself) to build a representation for that
token (self-attention). Also, instead of using just one attention head,
they used 8 randomly initialized heads (multi-headed attention). They
also used positional encodings that allows for the architecture to forgo
recurrent mechanism completely. The encoder layer has two sub-layers.
The first is the multi-headed self-attention layer followed by a
fully-connected layer.

Other than the parallelized nature of training, the key win for the
Transformer was that through self attention, it made is possible for
encoder to easily learn long-range dependencies amongst token
irrespective of the distance between them.

BERT
====

BERT, introduced in 2018, built upon several previous ideas and with new
innovations, achieved state-of-the-art results on 11 GLUE benchmark
tasks. The Transformer architecture was built specifically for machine
translation, so it had an encoder-decoder network. BERT retained only
the encoder network. Using just the encoder layer, allowed BERT to be
pre-trained for task agonistic embeddings. Transfer learning was one of
the key breakthroughs in computer vision. This approach by BERT allowed
it to be fine-tuned for specific tasks.

Through the use of transformers, BERT could learn bi-directional context
for each input token. ULMFit [@howard2018universal] had also used the
pre-training approach but it depended on BiLSTM which isn't as effective
as transformers for learning long-range dependencies.

BERT uses word-piece embeddings that allows it to encode tokens that
aren't seen during training time. Through the use of byte-piece
encoding, it can also be extended to all Unicode characters.

For pre-training, BERT used two tasks. The first was masked-language
modeling that uses a cloze task. 15% of tokens were masked at training
time and BERT had to predict the right word based on the context. The
second was next-sentence prediction (NSP).

BERT used 12 transformer layers, whereas the original Transformer used
only 6. It uses an embedding size of 768 which is 1.5x the size used by
the original transformer paper. BERT was trained on 2.5B words from
wikipedia and another 800M words from book corpuses.

BERT provides a `[CLS]` token that can be used for classification tasks.
For other types of NLP tasks, a custom head is required on top of the
output from the $12^{th}$ layer. The BERT embeddings could be fine-tuned
during training for specific tasks but just using the pre-trained
embeddings from last 4 layers works relatively well.

Beyond BERT
===========

A number of advances have been made since the original BERT paper came
out. Most of these advances have been related to using improved
pre-training tasks, using a bigger corpus for pre-training and making
the model inference faster.

BERT base has 108M parameters. Having such large number of parameters
makes the inference task slow. DistilBERT [@sanh2019distilbert] used a
technique called distillation to train a smaller network to mimic the
performance from BERT. It was able to reduce the number of parameters by
40% while retaining 97% of the model performance.

ALBERT [@lan2019albert] uses two techniques to reduce the number of
parameter. First, instead of projecting the embeddings of more than 30k
tokens in the vocabulary to a 768 dimensional space, it first projects
to a 128 dimensional space and then to 768 dimensions. This helps in
reducing the number of parameters by  15M. Further, it uses weight
sharing amongst the 12 layers. These approaches reduce the number of
parameters for ALBERT to just 12M. This allows ALBERT to train larger
models. For example, ALBERT-xlarge has only 60M parameters but it has 24
transformer layers with a hidden size of 2048.

RoBERTa [@liu2019roberta] used 1000% more data for pre-training. It also
removed the NSP task from BERT & introduced dynamic masking which
changes the masked tokens between epochs.

XLNet [@yang2019xlnet] used more data (130 GB) and more computational
power. It also used permutation language modeling where all tokens are
predicted in random order. This helped the model in learning better
bidirectional representations.

T5 [@raffel2019exploring] (\"everything is text\") proposed a unified
text-to-text-format where the input and output are always text strings
([\[eq: t5\]](#eq: t5){reference-type="ref" reference="eq: t5"}). This
allowed the model to use the same model, loss function, and
hyperparameters for any NLP task. The authors also developed Colossal
Clean Crawled Corpus (C4) which is 2 orders of magnitude bigger than
Wikipedia.

$$\begin{split}
    input 1&: \textit{"translate english to german: I like cats"} \\
    input 2&: \textit{"cola sentence: I go to market"} \\
\end{split}
\label{eq: t5}$$

More recently, GPT-3 [@brown2020language] has created significant buzz
and led to many fascinating example like React code generation, Regex
from english, etc. Gwern Branwen wrote an essay about why AI will never
achieve human intelligence. When many experts concurred with his
thoughts, he revealed that the essay was actually written by GPT-3 [^1].
It is generally agreed that GPT-3's success is attributable to its
massive scale. It used over 300B tokens for training which cost millions
of dollars and it has over 175 billion parameters.

Many experts believe that this approach of throwing massive data sets &
huge computational power on existing architectures could lead to short
term efficiencies but this approach may not be sufficient to build truly
intelligent machines that could eventually pass the Turing test.

\bibliographystyle{ACM-Reference-Format}
\appendix
\onecolumn
Appendix
========

\centering
![American Express chatbot includes widgets for accepting specific types
of responses[]{label="fig: dialogue"}](amex.jpg){#fig: dialogue
width="0.6\linewidth"}

------------------------------------------------------------------------

\bigskip
Glassdoor has employee driven reviews on each company page. This is much
like customer reviews on a product page on an e-commerce website. They
summarize the reviews to bring out the most talked about aspects of a
company. A similar approach could be implemented to make it easier for
customers to comprehend a plethora of reviews.

\centering
![Glassdoor summarizes the reviews to highlight key positive & negative
aspects[]{label="fig: dialogue"}](glassdoor){#fig: dialogue
width="0.6\linewidth"}

[^1]: <https://twitter.com/MIT_CSAIL/status/1294282794722643968>
