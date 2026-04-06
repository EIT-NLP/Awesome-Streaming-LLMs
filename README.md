<!-- # Awesome-Streaming-LLMs/MLLMs -->
<div align="center">
<h1><img src="Assets/wave.png" height="34" style="vertical-align: middle;" /> Awesome Streaming LLMs</h1></div>



<p align="center">
<a href="https://arxiv.org/abs/2603.04592" target="_blank"><img alt="Demo" src="https://img.shields.io/badge/arxiv-2603.04592-DA644E?logo=arxiv" /></a>
<a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome"></a>
<img src="https://img.shields.io/github/last-commit/EIT-NLP/Awesome-Streaming-LLMs/main?logo=github&color=orange" alt="GitHub last commit (branch)">
</p>

> <strong> From Static Inference to Dynamic Interaction: A survey of Streaming Large Language Models </strong>
>
> <a href="https://scholar.google.com/citations?user=Amv2QE8AAAAJ" rel="nofollow">Junlong Tong</a><sup>1,2</sup>, 
<a href="https://scholar.google.com/citations?user=CEiW_HQAAAAJ" rel="nofollow">Zilong Wang</a><sup>2</sup>, 
Yujie Ren</a><sup>2</sup>, 
Peiran Yin</a><sup>2</sup>, 
<a href="https://harrisonwu42.github.io/" rel="nofollow">Hao Wu</a><sup>2</sup>, 
<a href="https://scholar.google.com/citations?user=Z7u9yEoAAAAJ" rel="nofollow">Wei Zhang</a><sup>2</sup>, 
<a href="https://chin-gyou.github.io/" rel="nofollow">Xiaoyu Shen</a><sup>†,2</sup> 
>
> <sup>1</sup>Shanghai Jiao Tong University
>
> <sup>2</sup>Institute of Digital Twin, Eastern Institute of Technology, Ningbo
>
> <sup>†</sup> Corresponding Author.
>
> Contact: jl-tong@sjtu.edu.cn, xyshen@eitech.edu.cn

If you find our paper of this resource helpful, please consider cite:
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

> [!IMPORTANT]
> We actively maintain this repository and welcome community contributions.
> If you would like to:
> 
> - Add newly released Streaming LLMs/MLLMs papers  
> - Propose refinements to our taxonomy  
> - Correct or update existing entries  
> - Discuss classification or methodology  
> 
> Please submit a [pull request](https://github.com/EIT-NLP/Awesome-MLLM-Compression/pulls) or contact the authors.

## 📢 News <a id="news"></a>
* **[2026.04] Our survey has been accepted by ACL 2026 Findings.**
* **[2026.03]** We released the **first comprehensive [survey](https://arxiv.org/abs/2603.04592)** on Streaming LLMs/MLLMs!

## 💡 Overview <a id="overview"></a>
This repository provides a comprehensive landscape of current **streaming LLMs/MLLMs**, covering multi-modal streaming applications across text, audio, and video.

We cut through the confusing terminology of "streaming generation", "streaming input" and "interactive streaming" by introducing a unified, formal definition for Streaming LLMs. Based on **Data Flow and Interaction Concurrency**, we categorize Streaming LLMs into three progressive paradigms.


<table>
  <tr>
    <td width="45%">
      <img src="Assets/Top.png" width="100%">
    </td>
    <td>
      <b>👉 Category I: Output-Streaming LLMs</b> <br>
      <i>(Left)</i> Performs streaming generation <i>after</i> static reading.<br>
      <b>👉 Category II: Sequential-Streaming LLMs</b> <br>
      <i>(Middle)</i> Performs streaming generation <i>after</i> streaming reading.<br>
      <b>👉 Category III: Concurrent-Streaming LLMs</b> <br>
      <i>(Right)</i> Performs streaming generation <i>while</i> streaming reading.<br>
    </td>
  </tr>
</table>


### Formal Definition
We formulate the modeling process as a conditional probability distribution $P(Y|X)$, where $X = (x_1, \dots, x_M)$ denotes the bounded input stream and $Y = (y_1, \dots, y_N)$ denotes the output stream. This distribution can be factorized as:

$$P(Y|X) = \prod_{t=1}^{N} P\big(y_t | y_{\lt t}, h_{1:\phi(t)}(X);\theta\big),$$

where $\theta$ denotes the LLM parameters, $h_{\phi(t)}(X)=llm(x_{\phi(t)})$ is the hidden states corresponding to the input $x_{\phi(t)}$, and $\phi(t)$ is a **interaction decision function** to determine the input stream visible at generation step $t$.

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


## 📚 Contents <a id="contents"></a>
- [1. Overview](#overview)
- [2. Content](#content)
- [3. Streaming Taxonomy](#tax)
- [4. Streaming Applications and Tasks](#app)
- [5. Streaming Benchmark](#benchmark)

## ✨ Streaming Taxonomy <a id="tax"></a>
[To Do]
<!-- ### Output-streaming
### Sequential-streaming
### Concurrent-streaming -->



## 🔧 Streaming Applications and Tasks <a id="app"></a>
<!-- ### Text Streaming Tasks
### Speech Streaming Tasks
### Video Streaming Tasks
### Other Streaming Tasks -->
[To Do]

## 📋 Streaming Benchmark <a id="benchmark"></a>
[To Do]
