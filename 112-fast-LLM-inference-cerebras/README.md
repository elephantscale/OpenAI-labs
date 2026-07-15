# Fast LLM Inference with Cerebras
* https://learn.deeplearning.ai/courses/fast-llm-inference-with-cerebras/

## The bottleneck

When an LLM generates text, much of the time goes to moving the model's weights from memory to
the compute units, not to the math itself. On a GPU those weights sit off-chip, and a large
model may be split across several chips, so data travels back and forth before computation can
even start.

That delay compounds. An agentic workflow might generate hundreds of thousands of tokens before
it returns anything to the user.

## The course

In this short course, **Fast LLM Inference with Cerebras**, built in partnership with Cerebras
and taught by Zhenwei Gao, Sebastian Duerr, and Sarah Chieng, you'll build applications on
hardware designed to remove that bottleneck. Cerebras' **Wafer-Scale Engine (WSE-3)** is a
single chip about the size of a large dining plate, big enough to hold a model's weights
**on-chip**, right next to the compute units.

Tokens come out several times faster than on a typical GPU setup, and that changes both what you
can build and how you build it:

- **What you can build:** real-time, latency-sensitive applications that were impractical when
  every response carried a delay — like live translation and voice agents.
- **How you build it:** once a response lands before the user notices the wait, you can drop the
  loading spinners, async queues, and precomputed results, and just call the model directly.

## What you'll learn

- Compare how **GPUs, TPUs, and the Wafer-Scale Engine** handle the memory-to-compute
  bottleneck, and why keeping weights on-chip minimizes data movement.
- Build a **live personalization** use case that adapts a webpage to users as they interact
  with it.
- Assemble a **real-time, multi-tool workflow** that runs live analysis of market signals in
  one fast response.
- Adopt concrete habits for **multi-agent coding with Codex**, validating between tasks to catch
  issues early and ship cleaner code.
