---
layout: post
title: Intent Classification for Dialogue Systems 🤖
date:  2020-09-14
categories: nlp
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


<img style="float: right; margin: 10px; height: 150px" src=  
"{{site.baseurl}}/assets/intent/dialogue.png"
                    alt="CS50 Twitter Profile">

<p style="text-align: justify">
AI based chatbots are increasingly becoming ubiquitous as an alternative way for companies to provide customer service, drive engagement and enabling product discovery. Chatbots can help companies save cost and have the distinct advantage of being available 24x7. While open-domain chatbots are still at a research stage (In Apr 2020, Facebook released BlenderBot which they claimed improves on the previous state-of-the-art work called Meena from Google in terms of human-like engagement)[^blenderbot], many production implementation of closed-domain chatbots exist. Most of these are retrieval based models rather than generative models. In retrieval based models, the scope of answerable questions is limited, but this model provides significant control over accuracy of responses. Several out-of-the-box chatbot platforms are available like Rasa, Google’s DialogFlow, Amazon’s Lex, etc. But it is often possible to achieve superior performance by building a model from scratch. Building a closed-domain retrieval based chatbot from scratch involves three main steps:
</p>

1. Detecting intent from on customer query
2. Extracting key entities needed to build a response
3. Building and delivering a response

### Scope of this article
In this note, I present a short analysis for detecting intent from customer query. If labeled dataset is available, then this can thought of as a supervised classification problem. If a labeled dataset isn’t readily available, it can be curated using crowd-sourcing mechanisms at a fairly small cost. For this analysis, I used the dataset provided by clinc2. It contains 15,000 queries of training data split evenly into 150 distinct intents. Each query is associated with a single intent. The accompanying paper3 uses a variety of machine learning methods to train a model on this dataset. Authors achieve a maximum accuracy of 96.9% by fine-tuning a BERT model.


### References
[^blenderbot]: [https://ai.facebook.com/blog/state-of-the-art-open-source-chatbot/](https://ai.facebook.com/blog/state-of-the-art-open-source-chatbot/)