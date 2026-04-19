<!-- # Awesome-Streaming-LLMs/MLLMs -->
<div align="center">
<h1><img src="Assets/wave.png" height="34" style="vertical-align: middle;" /> From Static Inference to Dynamic Interaction: 

A survey of Streaming Large Language Models</h1></div>


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
<a href="https://chin-gyou.github.io/" rel="nofollow">Xiaoyu Shen</a><sup>2</sup> 
>
> <sup>1</sup>Shanghai Jiao Tong University, <sup>2</sup>Institute of Digital Twin, Eastern Institute of Technology, Ningbo
>
> Contact: jl-tong@sjtu.edu.cn, xyshen@eitech.edu.cn






## 📢 News <a id="news"></a>
* **[2026.04] Our survey has been accepted by ACL 2026 Findings.**
* **[2026.03]** We released the **first comprehensive [survey](https://arxiv.org/abs/2603.04592)** on Streaming LLMs/MLLMs!





## 💡 1. Overview <a id="overview"></a>
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

### 1.1 Formal Definition
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

### 1.2 Key Challenges & Core Goal
**Output-Streaming LLMs**
* Streaming Generation & Generation Efficiency
<!-- 
| Core challenge | What it focuses on | Representative directions |
| --- | --- | --- |
| Streaming Generation | How to reveal outputs progressively after static reading, while preserving controllability and output quality. | Token-wise, Block-wise, Refinement-based |
| Generation Efficiency | How to reduce latency and memory cost during long streaming generation. | Decode Acceleration, Memory Efficiency | -->

**Sequential-Streaming LLMs**
* Continuous Perception & Streaming Context Management
<!-- | Core challenge | What it focuses on | Representative directions |
| --- | --- | --- |
| Continuous Perception | How to encode dynamic inputs incrementally without repeatedly recomputing the full history. | Atomic Encoding, Fragmented Encoding |
| Streaming Context Management | How to retain, compress, and access long streaming histories under limited resource budgets. | Memory Retention, KV Cache Management, Attention Optimization | -->

**Concurrent-Streaming LLMs**
* Architecture Adaptation & Proactive Interaction Decision

<!-- | Core challenge | What it focuses on | Representative directions |
| --- | --- | --- |
| Architecture Adaptation | How to resolve attention and position conflicts when reading and writing overlap in the same interaction loop. | Re-encoded Streaming, Concatenated Streaming, Interleaved Streaming, Grouped Streaming |
| Proactive Interaction Decision | How to decide when to read, wait, interrupt, or respond under the latency-quality trade-off. | Rule-based Policy, SFT-based Policy, RL-based Policy | -->

> [!NOTE]
> Since Concurrent-Streaming LLMs build on top of the previous two paradigms, we emphasize the the entirely new challenges uniquely introduced by concurrency, while shared issues such as streaming context management are not repeated.

![Main-png](Assets/Main.png)

## 📚 2. Contents <a id="contents"></a>
- [1. Overview](#overview)
- [2. Contents](#contents)
- [3. Streaming Taxonomy](#tax)
- [4. Output-Streaming LLMs](#output-streaming)
- [5. Sequential-Streaming LLMs](#sequential-streaming)
- [6. Concurrent-Streaming LLMs](#concurrent-streaming)
- [7. Streaming Applications and Tasks](#app)
- [8. Streaming Benchmark](#benchmark)

## ✨ 3. Streaming Taxonomy <a id="tax"></a>
| Main Category | Second-Level Category | Third-Level Category | Explanation |
| --- | --- | --- | --- |
| **Output-Streaming LLMs** | Streaming Generation | Token-wise | The model reads the full input first and then streams outputs one unit at a time; this is the standard autoregressive streaming setting. |
|  |  | Block-wise | The model still finishes reading before writing, but generates blocks, chunks, or sentences to reduce serial latency. |
|  |  | Refinement-based | The model reveals outputs progressively by iterative refinement or denoising, rather than only extending left to right. |
|  | Streaming Efficiency | Decode Acceleration | Methods that keep output-streaming generation but improve speed through speculative decoding, multi-token prediction, or shorter execution paths. |
|  |  | Memory Efficiency | Methods that reduce KV-cache or long-context cost during progressive output generation. |
| **Sequential-Streaming LLMs** | Incremental Encoding | Atomic Encoding | The input arrives as stable units, such as tokens or fixed discrete chunks, and the model processes them incrementally before generation starts. |
|  |  | Fragmented Encoding | Continuous signals are partitioned into streaming units by fixed or semantic boundaries, then processed incrementally before generation. |
|  | Streaming Context Management | Memory Retention | Methods that decide what historical streaming content should be kept, merged, or evicted over time. |
|  |  | KV Cache Management | Methods that compress, retrieve, or reorganize internal KV states under long streaming inputs. |
|  |  | Attention Optimization | Methods that redesign attention access patterns so the model can process long input streams efficiently. |
| **Concurrent-Streaming LLMs** | Architecture Adaptation | Re-encoded Streaming | New inputs trigger re-encoding of the history so the model can preserve batch-like dependencies while reading and writing concurrently. |
|  |  | Concatenated Streaming | New inputs and generated outputs are concatenated into a single sequence so their order is unified during concurrent interaction. |
|  |  | Interleaved Streaming | Input and output events are interleaved on a shared timeline to support continuous read-write overlap. |
|  |  | Grouped Streaming | Input and output are placed in separate groups, and cross-group interaction is designed explicitly to avoid structural conflicts. |
|  | Interaction Policy | Rule-based Policy | Read-write timing is controlled by fixed schedules or threshold-based triggers. |
|  |  | SFT-based Policy | Read-write timing is learned from supervised fine-tuning signals. |
|  |  | RL-based Policy | Read-write timing is learned as a sequential decision process that optimizes latency-quality trade-offs. |

## 📋 4. Output-Streaming LLMs <a id="output-streaming"></a>
### 4.1 Streaming Generation
#### 4.1.1 Token-wise
| Paper | Source | Modality |
| --- | --- | --- |
| GPT-4 Technical Report. [\[paper\]](https://arxiv.org/abs/2303.08774) | arXiv 2023 | Text-out |
| SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities. [\[paper\]](https://aclanthology.org/2023.findings-emnlp.1055.pdf) [\[code\]](https://github.com/0nutation/SpeechGPT) | EMNLP 2023 Findings | Speech-out |
|Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation. [\[paper\]](https://arxiv.org/pdf/2406.06525)[\[code\]](https://github.com/foundationvision/llamagen)| arXiv 2024| Image-out|
| Videopoet: A large language model for zero-shot video generation. [\[paper\]](https://arxiv.org/abs/2312.14125) [\[project\]](https://sites.research.google/videopoet/) | ICML 2024 | Video-out |

#### 4.1.2 Block-wise
<!-- | Paper | Source | Modality |
| --- | --- | --- |
| Let’s Predict Sentence by Sentence. [\[paper\]](https://arxiv.org/abs/2505.22202) [code TBD] | arXiv 2025 | Text-out |
| From Next-Token to Next-Block: A Principled Adaptation Path for Diffusion LLMs. [\[paper\]](https://arxiv.org/abs/2512.06776) [code TBD] | arXiv 2025 | Text-out |
| Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models. [\[paper\]](https://arxiv.org/abs/2503.09573) [code TBD] | arXiv 2025 | Text-out | -->

#### 4.1.3 Refinement-based
<!-- | Paper | Source | Modality |
| --- | --- | --- |
| SoundStorm: Efficient Parallel Audio Generation. [\[paper\]](https://arxiv.org/abs/2305.09636) [code TBD] | arXiv 2023 | Speech-out |
| MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer. [\[paper\]](https://arxiv.org/abs/2409.00750) [code TBD] | arXiv 2024 | Speech-out |
| DetailFlow: 1D Coarse-to-Fine Autoregressive Image Generation via Next-Detail Prediction. [\[paper\]](https://arxiv.org/abs/2505.21473) [code TBD] | arXiv 2025 | Image-out | -->

### 4.2 Streaming Efficiency
#### 4.2.1 Decode Acceleration
<!-- | Paper | Source | Modality |
| --- | --- | --- |
| Accelerating Large Language Model Decoding with Speculative Sampling. [\[paper\]](https://arxiv.org/abs/2302.01318) [code TBD] | arXiv 2023 | Text-out |
| Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. [\[paper\]](https://arxiv.org/abs/2401.10774) [code TBD] | arXiv 2024 | Text-out |
| EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees. [\[paper\]](https://arxiv.org/abs/2406.16858) [code TBD] | arXiv 2024 | Text-out |
| LiveSpeech: Low-Latency Zero-Shot Text-to-Speech via Autoregressive Modeling of Audio Discrete Codes. [\[paper\]](https://arxiv.org/abs/2406.02897) [code TBD] | arXiv 2024 | Speech-out |
| Accelerating Autoregressive Speech Synthesis Inference With Speech Speculative Decoding. [\[paper\]](https://arxiv.org/abs/2505.15380) [code TBD] | arXiv 2025 | Speech-out | -->

#### 4.2.2 Memory Efficiency
<!-- | Paper | Source | Modality |
| --- | --- | --- |
| Efficient Streaming Language Models with Attention Sinks. [\[paper\]](https://arxiv.org/abs/2309.17453) [code TBD] | arXiv 2023 | Text-out |
| ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference. [\[paper\]](https://arxiv.org/abs/2502.00299) [code TBD] | arXiv 2025 | Text-out |
| Head-Aware KV Cache Compression for Efficient Visual Autoregressive Modeling. [\[paper\]](https://arxiv.org/abs/2504.09261) [code TBD] | arXiv 2025 | Image-out |
| Memory-Efficient Visual Autoregressive Modeling with Scale-Aware KV Cache Compression. [\[paper\]](https://arxiv.org/abs/2505.19602) [code TBD] | arXiv 2025 | Image-out | -->

## 📋 5. Sequential-Streaming LLMs <a id="sequential-streaming"></a>
### 5.1 Incremental Encoding
#### 5.1.1 Atomic Encoding
<!-- | Paper | Source | Modality |
| --- | --- | --- |
| Turning Whisper into Real-Time Transcription System. [\[paper\]](https://arxiv.org/abs/2307.14743) [code TBD] | arXiv 2023 | Speech-in |
| Simul-Whisper: Attention-Guided Streaming Whisper with Truncation Detection. [\[paper\]](https://arxiv.org/abs/2406.10052) [code TBD] | arXiv 2024 | Speech-in |
| Moshi: A Speech-Text Foundation Model for Real-Time Dialogue. [\[paper\]](https://arxiv.org/abs/2410.00037) [code TBD] | arXiv 2024 | Speech / Text-in | -->

#### 5.1.2 Fragmented Encoding
<!-- | Paper | Source | Modality |
| --- | --- | --- |
| End-to-End Simultaneous Speech Translation with Differentiable Segmentation. [\[paper\]](https://arxiv.org/abs/2305.16093) [code TBD] | arXiv 2023 | Speech-in |
| Semantic VAD: Low-Latency Voice Activity Detection for Speech Interaction. [\[paper\]](https://arxiv.org/abs/2305.12450) [code TBD] | arXiv 2023 | Speech-in |
| dmel: Speech Tokenization Made Simple. [\[paper\]](https://arxiv.org/abs/2407.15835) [code TBD] | arXiv 2024 | Speech-in | -->

### 5.2 Streaming Context Management
#### 5.2.1 Memory Retention
<!-- | Paper | Source | Modality |
| --- | --- | --- |
| Flash-VStream: Memory-Based Real-Time Understanding for Long Video Streams. [\[paper\]](https://arxiv.org/abs/2406.08085) [code TBD] | arXiv 2024 | Video-in |
| StreamingTOM: Streaming Token Compression for Efficient Video Understanding. [\[paper\]](https://arxiv.org/abs/2510.18269) [code TBD] | arXiv 2025 | Video-in |
| StreamForest: Efficient Online Video Understanding with Persistent Event Memory. [\[paper\]](https://arxiv.org/abs/2509.24871) [code TBD] | arXiv 2025 | Video-in | -->

#### 5.2.2 KV Cache Management
<!-- | Paper | Source | Modality |
| --- | --- | --- |
| StreamKV: Streaming Video Question-Answering with Segment-Based KV Cache Retrieval and Compression. [\[paper\]](https://arxiv.org/abs/2511.07278) [code TBD] | arXiv 2025 | Video-in |
| StreamMem: Query-Agnostic KV Cache Memory for Streaming Video Understanding. [\[paper\]](https://arxiv.org/abs/2508.15717) [code TBD] | arXiv 2025 | Video-in |
| InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding. [\[paper\]](https://arxiv.org/abs/2506.15745) [code TBD] | arXiv 2025 | Video-in | -->

#### 5.2.3 Attention Optimization
<!-- | Paper | Source | Modality |
| --- | --- | --- |
| Efficient Streaming Language Models with Attention Sinks. [\[paper\]](https://arxiv.org/abs/2309.17453) [code TBD] | arXiv 2023 | Text-in |
| SirLLM: Streaming Infinite Retentive LLM. [\[paper\]](https://arxiv.org/abs/2405.12528) [code TBD] | arXiv 2024 | Text-in |
| LServe: Efficient Long-Sequence LLM Serving with Unified Sparse Attention. [\[paper\]](https://arxiv.org/abs/2502.14866) [code TBD] | arXiv 2025 | Text-in | -->

## 📋 6. Concurrent-Streaming LLMs <a id="concurrent-streaming"></a>
### 6.1 Architecture Adaptation
#### 6.1.1 Re-encoded Streaming
<!-- | Paper | Source | Modality |
| --- | --- | --- |
| SiLLM: Large Language Models for Simultaneous Machine Translation. [\[paper\]](https://arxiv.org/abs/2402.13036) [code TBD] | arXiv 2024 | T→T |
| TransLLaMA: LLM-Based Simultaneous Translation System. [\[paper\]](https://arxiv.org/abs/2402.04636) [code TBD] | arXiv 2024 | T→T |
| SimulS2S-LLM: Unlocking Simultaneous Inference of Speech LLMs for Speech-to-Speech Translation. [\[paper\]](https://arxiv.org/abs/2504.15509) [code TBD] | ACL 2025 | S→S | -->

#### 6.1.2 Concatenated Streaming
<!-- | Paper | Source | Modality |
| --- | --- | --- |
| Qwen2.5-Omni Technical Report. [\[paper\]](https://arxiv.org/abs/2503.20215) [code TBD] | arXiv 2025 | Text / Speech / Video |
| Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming. [\[paper\]](https://arxiv.org/abs/2408.16725) [code TBD] | arXiv 2024 | Speech / Text |
| LLMVoX: Autoregressive Streaming Text-to-Speech Model for Any LLM. [\[paper\]](https://arxiv.org/abs/2503.04724) [code](https://github.com/mbzuai-oryx/LLMVoX) | ACL Findings 2025 | T→S | -->

#### 6.1.3 Interleaved Streaming
<!-- | Paper | Source | Modality |
| --- | --- | --- |
| Interleaved Speech-Text Language Models Are Simple Streaming Text-to-Speech Synthesizers. [\[paper\]](https://arxiv.org/abs/2412.16102) [code TBD] | arXiv 2024 | T→S |
| CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models. [\[paper\]](https://arxiv.org/abs/2412.10117) [code TBD] | arXiv 2024 | T→S |
| StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Assistant. [\[paper\]](https://arxiv.org/abs/2505.05467) [code TBD] | arXiv 2025 | V→T | -->

#### 6.1.4 Grouped Streaming
<!-- | Paper | Source | Modality |
| --- | --- | --- |
| StreamChat: Chatting with Streaming Video. [\[paper\]](https://arxiv.org/abs/2412.08646) [code TBD] | arXiv 2024 | V→T |
| LLM as Effective Streaming Processor: Bridging Streaming-Batch Mismatches with Group Position Encoding. [\[paper\]](https://arxiv.org/abs/2505.16983) [code TBD] | arXiv 2025 | T→T / S→T |
| StreamingThinker: Large Language Models Can Think While Reading. [\[paper\]](https://arxiv.org/abs/2510.17238) [code TBD] | arXiv 2025 | T→T |
| Speak While Watching: Unleashing TRUE Real-Time Video Understanding Capability of Multimodal Large Language Models. [\[paper\]](https://arxiv.org/abs/2601.06843) [code TBD] | arXiv 2026 | V→T | -->

### 6.2 Interaction Policy
#### 6.2.1 Rule-based Policy
<!-- | Paper | Source | Modality |
| --- | --- | --- |
| SimulS2S-LLM: Unlocking Simultaneous Inference of Speech LLMs for Speech-to-Speech Translation. [\[paper\]](https://arxiv.org/abs/2504.15509) [code TBD] | ACL 2025 | S→S |
| SASST: Leveraging Syntax-Aware Chunking and LLMs for Simultaneous Speech Translation. [\[paper\]](https://arxiv.org/abs/2508.07781) [code TBD] | arXiv 2025 | S→T |
| Asynchronous Reasoning: Training-Free Interactive Thinking LLMs. [\[paper\]](https://arxiv.org/abs/2512.10931) [code TBD] | arXiv 2025 | T→T | -->

#### 6.2.2 SFT-based Policy
<!-- | Paper | Source | Modality |
| --- | --- | --- |
| SiLLM: Large Language Models for Simultaneous Machine Translation. [\[paper\]](https://arxiv.org/abs/2402.13036) [code TBD] | arXiv 2024 | T→T |
| Speech ReaLLM: Real-Time Streaming Speech Recognition with Multimodal LLMs by Teaching the Flow of Time. [\[paper\]](https://arxiv.org/abs/2406.09569) [code TBD] | arXiv 2024 | S→T |
| LLaMA-Omni: Seamless Speech Interaction with Large Language Models. [\[paper\]](https://arxiv.org/abs/2409.06666) [code TBD] | arXiv 2024 | S→T | -->

#### 6.2.3 RL-based Policy
<!-- | Paper | Source | Modality |
| --- | --- | --- |
| SeqPO-SiMT: Sequential Policy Optimization for Simultaneous Machine Translation. [\[paper\]](https://arxiv.org/abs/2505.20622) [code TBD] | arXiv 2025 | T→T |
| Seed LiveInterpret 2.0: End-to-End Simultaneous Speech-to-Speech Translation with Your Voice. [\[paper\]](https://arxiv.org/abs/2507.17527) [code TBD] | arXiv 2025 | S→S |
| MMDuet2: Enhancing Proactive Interaction of Video MLLMs with Multi-Turn Reinforcement Learning. [\[paper\]](https://arxiv.org/abs/2512.06810) [code TBD] | arXiv 2025 | V→T |
| Interleaved Reasoning for Large Language Models via Reinforcement Learning. [\[paper\]](https://arxiv.org/abs/2505.19640) [code TBD] | arXiv 2025 | T→T | -->

## 🔧 7. Streaming Applications and Tasks <a id="app"></a>
[To Do]
<!-- This section can be used as a task-oriented reverse index after the taxonomy stabilizes. A practical next step is to reorganize the papers above by task, while keeping each paper in only one primary taxonomy family. -->

## 📋 8. Streaming Benchmark <a id="benchmark"></a>
[To Do]
<!-- This section can summarize benchmark suites by modality, latency metric, interaction policy setting, and evaluation protocol. -->


## Welcome Contributions
> We actively maintain this repository and welcome community contributions.
> If you would like to:
> 
> - Add newly released Streaming LLMs/MLLMs papers  
> - Propose refinements to our taxonomy  
> - Correct or update existing entries  
> - Discuss classification or methodology  
> 
> Please submit a [pull request](https://github.com/EIT-NLP/Awesome-Streaming-LLMs/pulls) or contact the authors.


## Citation
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
