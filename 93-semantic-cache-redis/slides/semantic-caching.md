# Semantic Caching with Redis

---


## The idea
* Make your AI agent faster and more cost effective
* Inference cost and latency affect your performance
* Semantic caching can help you reduce inference cost and latency
* Example
  * Can I get a refund?
  * I want my money back
---

## Semantic caching
* Looks ate meaning
* Uses embeddings to measure how similar two questions are

![](../images/01.png)

## Plan
* Build a cache
* Use embeddings and measure distances
* User Redis to implement semantic cache
  * TTL, cache sizes for different users
  * Open embedding models for cache accuracy
---

## Cache performance
* Hit rate, precision, recall
* Cache size, memory usage
* Confusion matrix
* Latency
* Methods for enhancing cache performance
* LLM check and fuzzy matches
---

## AI Agent
* Breaks the question into smaller parts
* Cache warms up
* Model calls drop
---
