# Italian Sentence Boundary Detection

This repository studies **sentence boundary detection for Italian literary text**, comparing **encoder-based supervised models** with **decoder-based prompting approaches**.

The task is formulated as **token-level binary classification**, where each token is labeled according to whether it ends a sentence. The problem is challenging due to punctuation ambiguity, long-range context, and severe class imbalance.

---

## Approaches

The repository contains two complementary modeling paradigms:

### Encoder-Based Models

The `encoder/` directory contains supervised approaches based on transformer encoders and feature-based classifiers. These models are fine-tuned for token-level classification and serve as strong in-domain and out-of-domain baselines.

Refer to `encoder/README.md` for details on models, training, and evaluation.

---

### Decoder-Based Models

The `decoder/` directory explores sentence boundary detection using decoder-only large language models through prompt-based strategies. These methods evaluate whether sentence splitting can be performed without fine-tuning, relying solely on prompting and structured output parsing.

Refer to `decoder/README.md` for details on prompting strategies and experiments.

---

## Repository Structure

```
.
├── encoder/
├── decoder/
└── README.md
```

---

## Notes

* Both approaches use the same underlying task definition and dataset.
* The repository is designed to enable direct comparison between supervised encoder models and prompt-based decoder models for sentence segmentation.

---

## License

Released under the Apache 2.0 License.

---
