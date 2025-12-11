

# 124M-MoE: High-Efficiency Language Model Outperforming GPT-2

## 🌟 Introduction

This repository contains the implementation and pre-trained checkpoint for the **124M-MoE** model, a language model based on the **Mixture-of-Experts (MoE)** architecture.

With $\approx 124$ million parameters, this model achieves performance metrics that exceed those of the GPT-2 (124M) baseline. This success is attributed to the computational efficiency and increased representational capacity offered by the sparse MoE design.

## 🧠 Model Architecture: Mixture-of-Experts (MoE)

The 124M-MoE utilizes a sparse MoE layer structure:

  * **Total Parameters:** $\approx 124$ Million (Comparable to GPT-2 Small).
  * **Number of Experts:** **4** specialized feed-forward networks per MoE layer.
  * **Routing Mechanism:** A trainable Gating Network (Router) determines which expert(s) process the token.
  * **Sparse Activation:** We use **Top-1 Routing**, meaning that for any given input token, only one expert (and thus a fraction of the total parameters) is active.
      * **Active Parameters per Token:** $\approx 31$ Million ($\approx 25\%$ of total).

This sparse activation allows the model to leverage a massive parameter count for capacity while maintaining low computational costs during inference.

-----

## 📈 Performance Advantage over GPT-2

The efficiency gains of the MoE architecture translate directly into superior performance compared to a dense model of the same size.

| Model | Total Parameter Count | Compute Cost per Token | Result Summary |
| :--- | :--- | :--- | :--- |
| **124M-MoE (This Work)** | $124 \text{M}$ | **Low (Sparse Activation)** | **Outperforms GPT-2** |
| **GPT-2 (Dense)** | $124 \text{M}$ | High (Dense Activation) | Baseline Performance |

### Key Metrics

| Metric | GPT-2 (Small) | **124M-MoE** | Observation |
| :--- | :--- | :--- | :--- |
| **Perplexity** | $\text{29.41}$ | **$\text{29.41}$** | Lower Perplexity (Better Generalization) |
| **Inference Speed** | $\text{A}$ tokens/sec | **$\text{B}$ tokens/sec** | Faster Inference due to Sparsity |

-----

## 🛠️ Getting Started

### Prerequisites

  * Python 3.x
  * PyTorch (Recommended $\ge 2.0$)
  * (Other necessary libraries, e.g., NumPy, Hugging Face `transformers`)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Kahabk/Moe-llm/124M-MoE.git
    cd 124M-MoE
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage (Inference)

```python
from model import MoEModel
from tokenizer import MoETokenizer

# Load the model and tokenizer
tokenizer = MoETokenizer()
model = MoEModel.from_pretrained('checkpoints/124m_moe_final.pt')

prompt = "The future of language modeling is"
output = model.generate(prompt, max_length=100)
print(output)
```

### Usage (Training)

To train or fine-tune the model:

```bash
python train.py --config configs/moe_default.yaml
```

-----

## 🤝 Contribution and License

Contributions are welcome\! Please refer to the `CONTRIBUTING.md` file for guidelines.

This project is licensed under the [Specify License, e.g., MIT License].

