<img src="https://github.com/user-attachments/assets/16855689-a69e-4bf2-9948-b56c1138bdb6" alt="Logo" width="200"/>

# MENAValues: Evaluating Cultural Alignment and Multilingual Bias in Large Language Models

[![Dataset](https://img.shields.io/badge/Dataset-Available-green)](https://huggingface.co/datasets/llm-lab/MENA_VALUES_Benchmark)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)


## Overview

MENAValues is a comprehensive benchmark designed to evaluate the cultural alignment and multilingual biases of large language models (LLMs) with respect to the beliefs and values of the Middle East and North Africa (MENA) region. Built on authoritative population-scale survey data, this benchmark addresses the critical underrepresentation of MENA perspectives in current AI evaluation efforts.

## 🔍 Key Features

- **864 curated questions** spanning 16 MENA countries
- **Population-scale survey data** from World Values Survey Wave 7 and Arab Opinion Index 2022
- **Systematic evaluation framework** across 6 conditions (3 perspectives × 2 languages)
- **Token-level probability analysis** revealing hidden biases through "logit leakage"
- **Four core dimensions**: Governance & Political Systems, Economic Values, Social & Cultural Identity, Individual Wellbeing & Development

## 📊 Key Findings

Our evaluation reveals three critical misalignment behaviors in LLMs:

1. **Cross-Lingual Value Shift**: Same questions yield different answers depending on language
2. **Prompt-Sensitive Misalignment**: Framing significantly affects responses to identical cultural questions
3. **Logit Leakage**: Models refuse to answer explicitly but reveal clear internal preferences

## 🏗️ Repository Structure

```
MENAValues/
├── Dataset/                          # Core benchmark data
│   └── MenaValues Benchmark.xlsx           # Complete benchmark with questions and ground truth
├── LLM_Metrics/                     # Evaluation metrics organized by type
│   ├── CLCS/                             # Cross-Lingual Consistency Scores
│   │   ├── Observer/
│   │   │      ├── CLCS_Observer_Aya_reasoning.xlsx   
│   │   │      ├── CLCS_Observer_Aya_zero.xlsx
│   │   │      ├── CLCS_Observer_Llama_reasoning.xlsx
│   │   │      ├── CLCS_Observer_Llama_zero.xlsx
│   │   │      ├── CLCS_Observer_Mistral_reasoning.xlsx
│   │   │      └── CLCS_Observer_Mistral_zero.xlsx
│   │   │
│   │   ├── Persona/
│   │   │      ├── CLCS_Aya_reasoning_Neutral.xlsx   
│   │   │      ├── CLCS_Aya_zero_Neutral.xlsx
│   │   │      ├── CLCS_Llama_reasoning_Neutral.xlsx
│   │   │      ├── CLCS_Llama_zero_Neutral.xlsx
│   │   │      ├── CLCS_Mistral_reasoning_Neutral.xlsx
│   │   │      └── CLCS_Mistral_zero_Neutral.xlsx
│   │   │ 
│   │   └── Neutral/
│   │       ├── CLCS_Aya_reasoning_Neutral.xlsx   
│   │       ├── CLCS_Aya_zero_Neutral.xlsx
│   │       ├── CLCS_Llama_reasoning_Neutral.xlsx
│   │       ├── CLCS_Llama_zero_Neutral.xlsx
│   │       ├── CLCS_Mistral_reasoning_Neutral.xlsx
│   │       └── CLCS_Mistral_zero_Neutral.xlsx
│   │
│   ├── FCS/                              # Framing Consistency Scores
│   │   ├── FCS_Aya_reasoning.xlsx
│   │   ├── FCS_Aya_zero.xlsx
│   │   ├── FCS_Llama_reasoning.xlsx
│   │   ├── FCS_Llama_zero.xlsx
│   │   ├── FCS_Mistral_reasoning.xlsx
│   │   └── FCS_Mistral_zero.xlsx
│   │
│   ├── KL/                               # KL Divergence Analysis
│   │   ├── KL_results_Aya_zero.xlsx
│   │   ├── KL_results_Aya_zero_persona.xlsx
│   │   ├── KL_results_Llama_zero.xlsx
│   │   ├── KL_results_Llama_zero_persona.xlsx
│   │   ├── KL_results_Mistral_zero.xlsx
│   │   └── KL_results_Mistral_zero_persona.xlsx
│   │
│   ├── NVAS/                            # Normalized Value Alignment Scores
│   │   ├── NOVAS/                       # Normalized (Observver) Value Alignment Scores
│   │   │   ├── NVAS_observer_Aya_reasoning.xlsx
│   │   │   ├── NVAS_observer_Aya_zero.xlsx
│   │   │   ├── NVAS_observer_Llama_reasoning.xlsx
│   │   │   ├── NVAS_observer_Llama_zero.xlsx
│   │   │   ├── NVAS_observer_Mistral_reasoning.xlsx
│   │   │   └── NVAS_observer_Mistral_zero.xlsx
│   │   │
│   │   └── NPVAS/                       # Normalized (Persona) Value Alignment Scores
│   │       ├── NVAS_persona_Aya_reasoning.xlsx
│   │       ├── NVAS_persona_Aya_zero.xlsx
│   │       ├── NVAS_persona_Llama_reasoning.xlsx
│   │       ├── NVAS_persona_Llama_zero.xlsx
│   │       ├── NVAS_persona_Mistral_reasoning.xlsx
│   │       └── NVAS_persona_Mistral_zero.xlsx
│   │
│   └── SPD/                              # Self-Persona Deviation scores
│       ├── SPD_aya_reasoning.xlsx
│       ├── SPD_aya_zero.xlsx
│       ├── SPD_Llama_reasoning.xlsx
│       ├── SPD_Llama_zero.xlsx
│       ├── SPD_Mistral_reasoning.xlsx
│       └── SPD_Mistral_zero.xlsx
│
├── LLM_Responses/                   # Complete model outputs across all conditions
│   ├── Aya_zero_shot.xlsx               # Aya zero-shot responses (6 scenarios)
│   ├── Aya_reasoning.xlsx               # Aya reasoning responses (6 scenarios)
│   ├── Llama_zero_shot.xlsx             # Llama zero-shot responses (6 scenarios)
│   ├── Llama_reasoning.xlsx             # Llama reasoning responses (6 scenarios)
│   ├── Mistral_zero_shot.xlsx           # Mistral zero-shot responses (6 scenarios)
│   └── Mistral_reasoning.xlsx           # Mistral reasoning responses (6 scenarios)
├── Code/                            # Implementation script
│   └── Evaluation Code.py               # Complete evaluation pipeline
└── README.md
```


## 📈 Evaluation Framework

### Perspectives
- **Neutral**: Direct questioning without identity constraints
- **Persona**: "Imagine you are an average [nationality]..."
- **Observer**: "How would an average [nationality] respond to..."

### Languages
- **English**: Standard evaluation language
- **Native Languages**: Arabic, Persian, Turkish for respective regions

### Metrics
- **NVAS**: Normalized Value Alignment Score - alignment with human values
- **FCS**: Framing Consistency Score - consistency across perspectives
- **CLCS**: Cross-Lingual Consistency Score - consistency across languages
- **SPD**: Self-Persona Deviation - impact of identity assignment

## 🌍 Coverage

### Countries (16 MENA nations)
Algeria, Egypt, Iran, Iraq, Jordan, Kuwait, Lebanon, Libya, Mauritania, Morocco, Palestine, Qatar, Saudi Arabia, Sudan, Tunisia, Turkey

### Question Categories
1. **Governance & Political Systems** (governance, corruption, political participation)
2. **Economic Dimensions** (business, employment, competition)
3. **Social & Cultural Identity** (religious values, social attitudes, ethics)
4. **Individual Wellbeing & Development** (healthcare, education, migration)

## 📊 Results Summary

| Model | NVAS (Persona) | NVAS (Observer) | Cross-Lingual Consistency |
|-------|----------------|-----------------|--------------------------|
| Llama-3.1-8B | 74.75% | 73.21% | 75.72% |
| Mistral-7B | 71.08% | 70.43% | 66.02% |
| Aya-8B | 69.30% | 70.91% | 82.11% |


##
## 🙏 Acknowledgments

- World Values Survey Association for WVS Wave 7 data
- Arab Center for Research and Policy Studies for Arab Opinion Index 2022
- The broader AI alignment and cultural bias research community

## 📧 Contact

For questions about the benchmark or collaboration opportunities:
- Email: [zahraei2@illinois.edu]
---

**Note**: This benchmark is designed for research purposes to improve AI alignment with diverse cultural values. Please use responsibly and in accordance with ethical AI research practices.
