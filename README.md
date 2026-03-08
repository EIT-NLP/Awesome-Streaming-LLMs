<!-- # Awesome-Streaming-LLMs/MLLMs -->
<div align="center">
<h1><img src="Assets/wave.png" height="34" style="vertical-align: middle;" /> From Static Inference to Dynamic Interaction: 

A survey of Streaming Large Language Models</h1></div>



<p align="center">
<a href="https://arxiv.org/abs/2603.04592" target="_blank"><img alt="Demo" src="https://img.shields.io/badge/arxiv-2603.04592-DA644E?logo=arxiv" /></a>
<a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome"></a>
<img src="https://img.shields.io/github/last-commit/EIT-NLP
/Awesome-Streaming-LLMs/main?logo=github&color=orange" alt="GitHub last commit (branch)">
</p>

📢 **News:** We released the first comprehensive survey on Streaming LLMs/MLLMs! See https://arxiv.org/abs/2603.04592.

## 1. Overview
This repository provides a comprehensive landscape of current **streaming LLMs/MLLMs**, covering multi-modal streaming applications across text, audio, and video.

We cut through the confusing terminology of "streaming generation", "streaming input" and "interactive streaming" by introducing a unified, formal definition for Streaming LLMs. Based on **Data Flow and Interaction Concurrency**, we categorize Streaming LLMs into three progressive paradigms.


<p>
  <img src="Assets/Top.png" width="45%" align="left" />
  <br><br>
  <b>👉 Category I: Output-Streaming LLMs</b><br>
  <i>(Left)</i> Performs streaming generation <i>after</i> static reading.<br>
  <b>👉 Category II: Sequential-Streaming LLMs</b><br>
  <i>(Middle)</i> Performs streaming generation <i>after</i> streaming reading.<br>
  <b>👉 Category III: Concurrent-Streaming LLMs</b><br>
  <i>(Right)</i> Performs streaming generation <i>while</i> streaming reading.
</p>
<br clear="left" />


### Formal Definition
We formulate the modeling process as a conditional probability distribution $P(Y|X)$, where $X = (x_1, \dots, x_M)$ denotes the bounded input stream and $Y = (y_1, \dots, y_N)$ denotes the output stream. This distribution can be factorized as:
$$P(Y|X) = \prod_{t=1}^{N} P\big(y_t | y_{<t}, h_{1:\phi(t)}(X);\theta\big),$$

where $\theta$ denotes the LLM parameters, $h_{\phi(t)}(X)=llm(x_{\phi(t)})$ represents the hidden states corresponding to the input $x_{\phi(t)}$, and $\phi(t)$ is a **interaction decision function** to determine the input stream visible at generation step $t$.

<!-- where:
- $\theta$ denotes the LLM parameters.
- $h_{\phi(t)}(X)=llm(x_{\phi(t)})$ represents the hidden states corresponding to the input $x_{\phi(t)}$.
- $\phi(t)$ is an **interaction decision function** to determine the input stream visible at generation step $t$. -->

Then:
- Output-Streaming LLMs: $\phi(t)=M$ for all $t\in \{1,2,...,N \}$, $h_{1:\phi(t)}(X) = h_{1:M}(X) = llm(X_{1:M}).$
- Sequential-Streaming LLMs: $\phi(t)=M$ for all $t\in \{1,2,...,N \}$, $h_{1:M}(X) = \{ llm(x_1), \dots, llm(x_M) \}.$
- Concurrent-Streaming LLMs: $1\le \dots \le \phi(t)\le \phi(t+1)\le \dots \le M$, $h_{\phi(t)}(X) = llm(X_{\phi(t)},y_{<t}).$


Concurrent-Streaming is built upon the foundation of the previous two paradigms, representing the evolution **from static inference, to continuous streaming perception, to full-duplex dynamic interaction.**

---

### Key Challenges
- Output-Streaming LLMs: Streaming Generation & Generation Efficiency.
- Sequential-Streaming LLMs: Continuous Perception & Streaming Context Management.
- Concurrent-Streaming LLMs: Architecture Adaptation & Interaction Decision Policy.

### Citation
If you find this repository useful, please cite this paper:
```tex
@article{Tong2026Streaming,
      title={From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models}, 
      author={Junlong Tong and Zilong Wang and YuJie Ren and Peiran Yin and Hao Wu and Wei Zhang and Xiaoyu Shen},
      year={2026},
      eprint={2603.04592},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.04592}, 
}
```


## 2. Content
- [1. Overview](#1-overview)
- [2. Content](#2-content)
- [3. Streaming Taxonomy](#3-streaming-taxonomy)
- [4. Streaming Applications and Tasks](#4-streaming-applications-and-tasks)
- [5. Streaming Benchmark](#5-streaming-benchmark)

## 3. Streaming Taxonomy
[To Do]
<!-- ### Output-streaming
### Sequential-streaming
### Concurrent-streaming -->



## 4. Streaming Applications and Tasks
<!-- ### Text Streaming Tasks
### Speech Streaming Tasks
### Video Streaming Tasks
### Other Streaming Tasks -->
[To Do]

## 5. Streaming Benchmark
[To Do]
