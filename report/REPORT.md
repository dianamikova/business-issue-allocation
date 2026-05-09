# Business Issue Allocation
### Classifying Business Data Problems Using Sentence Embeddings

**Diana Mikova**
Uppsala University - Information Retrieval 5LN712, 2026

---

## Links

| Resource | URL |
|----------|-----|
| GitHub | https://github.com/dianamikova/business-issue-allocation |
| Dataset | https://huggingface.co/datasets/dianamikova/business-issue-allocation |
| Model | https://huggingface.co/dianamikova/business-issue-allocation-classifier |
| Demo | https://huggingface.co/spaces/dianamikova/business-issue-allocation-demo |

---

## Problem and Motivation

Organisations repeatedly face the same data challenges — inconsistent reports, slow dashboards, disconnected systems — yet struggle to route these problems to the right solution because business users lack the technical vocabulary to describe what they need. A business analyst who says "we keep getting different numbers from different teams" is describing a data governance problem, not a reporting problem. This mismatch causes unnecessary meetings and delays.

The aim of this project is to classify a plain-language business problem description into the most appropriate data engineering solution category. The nine categories are: `stream_processing`, `etl_pipeline`, `data_warehouse`, `data_lake`, `api_integration`, `ml_feature_store`, `data_caching`, `data_governance`, and `data_quality`.

---

## Dataset

The dataset was created specifically for this project and contains 465 labeled examples across the nine categories. It was generated using Claude Sonnet 4.6 (Anthropic) and Codex, drawing on enterprise contexts including banking, healthcare, government, and logistics. All examples were reviewed and corrected by the author based on professional experience in data engineering.

An important quality iteration occurred during the project: initial examples used language that was too technical. These were rewritten in business-friendly language (e.g. "we need alerts the moment suspicious transactions happen" instead of "we need a stream processing pipeline for fraud detection"). This improved classifier confidence on real-world business inputs.

Each example has two labels: `label_it` (technical, used for training) and `label_bus` (business-friendly equivalent), retained for future use in presenting predictions in plain language to non-technical users. The dataset is split 80/20 into train and test sets using stratified sampling with a fixed random seed for reproducibility.

---

## Method and Results

Text is encoded using a pretrained sentence transformer model as a frozen feature extractor (Reimers & Gurevych, 2019). Two embedding models were compared: `all-mpnet-base-v2`, based on the MPNet architecture (Song et al., 2020), and `all-MiniLM-L6-v2` as a lighter baseline. Three classifiers were trained and evaluated on each. The approach follows the structure of the example project provided for this course (Moëll, 2025).

**Embedding model: `sentence-transformers/all-MiniLM-L6-v2`**

| Classifier | Accuracy | Macro F1 |
|-----------|----------|----------|
| **SVM** | **87.1%** | **86.4%** |
| Logistic Regression | 83.9% | 83.2% |
| Random Forest | 80.6% | 80.0% |

**Embedding model: `sentence-transformers/all-mpnet-base-v2`**

| Classifier | Accuracy | Macro F1 |
|-----------|----------|----------|
| **SVM** | **88.2%** | **88.4%** |
| Logistic Regression | 83.9% | 84.1% |
| Random Forest | 80.6% | 79.9% |


SVM with all-mpnet-base-v2 was selected as the final model. A notable finding is that Random Forest shows almost no improvement from the higher-quality embedding model, while SVM gains +2.0% macro F1 — suggesting SVM makes better use of richer vector representations. The demo uses SVM for prediction and a separately trained Logistic Regression to display a more reliable confidence score.

---

## Reflection on Working with AI

Working with AI tools on this assignment was genuinely useful but required constant critical thinking. Claude and Codex were helpful for generating the dataset, scaffolding the code, and thinking through options — though the results always need to be reviewed, questioned, and sometimes corrected. 

The most important lesson was about data quality. The first version of the dataset was generated too quickly and the examples sounded too technical. A business user would never say "we need a stream processing pipeline" — they would say "we need to know about problems the moment they happen." I had to go back, identify the problematic examples, and rewrite them iteratively with the help of Codex. It notably improved the demo.

Early confidence scores from the SVM appeared low in the demo because SVM is not naturally probabilistic. Claude initially presented this as acceptable, but after testing the demo I pushed for a better solution - using a separate Logistic Regression for the confidence score, which produced more meaningful results.

The experience confirmed that AI tools are powerful for generating first drafts and scaffolding, but human judgement — especially domain expertise — remains essential for quality. Everything AI generated in this project was verified, tested, and in many cases corrected before it was used.

---

## References

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.
- Song, K., Tan, X., Qin, T., Lu, J., & Liu, T. (2020). MPNet: Masked and Permuted Pre-training for Language Understanding. NeurIPS 2020. https://arxiv.org/abs/2004.09297
- Moëll, B. (2025). Swedish Health Source Triage. Hugging Face Spaces. https://huggingface.co/spaces/birgermoell/swedish-health-source-triage-demo
