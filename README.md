<div align="center">
  <img src="https://github.com/user-attachments/assets/16855689-a69e-4bf2-9948-b56c1138bdb6" alt="Logo" width="250"/>
</div>

# MENAValues: A Benchmark for Cultural Alignment and Multilingual Bias in LLMs

[![Dataset](https://img.shields.io/badge/Dataset-Available-green)](https://huggingface.co/datasets/llm-lab/MENA_VALUES_Benchmark)
[![Paper](https://img.shields.io/badge/Paper-Read_Here-b31b1b.svg)](#how-to-cite)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

### I Am Aligned, But With Whom? MENA Values Benchmark for Evaluating Cultural Alignment and Multilingual Bias in LLMs

**Authors:** Pardis Sadat Zahraei, Ehsaneddin Asgari

---

## 📜 Overview

**MENAValues** is a novel benchmark designed to evaluate the cultural alignment and multilingual biases of large language models (LLMs) with the beliefs and values of the **Middle East and North Africa (MENA) region**. As AI models are deployed globally, their tendency to reflect predominantly Western, English-speaking perspectives creates significant alignment gaps. This benchmark addresses the critical underrepresentation of MENA perspectives in AI evaluation by providing a scalable framework for diagnosing cultural misalignment and fostering more inclusive AI.

![Main Figure illustrating the benchmark's evaluation methodology](https://github.com/user-attachments/assets/48cb4573-9b1e-493d-90fd-ed5531e4d377)

## 🔍 Key Features

-   **Empirically Grounded Data**: Features **864 questions** derived from large-scale, authoritative human surveys: the **World Values Survey (WVS-7)** and the **Arab Opinion Index 2022**.
-   **Broad Regional Coverage**: Captures the sociocultural landscape of **16 MENA countries** with population-level response distributions.
-   **Multi-Dimensional Evaluation**: Probes LLMs across a matrix of conditions, crossing three perspective framings with two language modes (English and native languages).
-   **Deep Analytical Framework**: Moves beyond surface-level responses to analyze token-level probabilities, revealing hidden biases and novel phenomena.
-   **Four Core Dimensions**: Questions are organized into four key pillars: Social & Cultural Identity, Economic Dimensions, Governance & Political Systems, and Individual Wellbeing & Development.

![Chart summarizing evaluation results across different models and metrics](https://github.com/user-attachments/assets/a52e4061-3613-493f-a09c-11dc582508d3)

---

## 📊 Key Findings

Our analysis reveals three critical and pervasive phenomena in state-of-the-art LLMs:

1.  **Cross-Lingual Value Shift**: Models provide drastically different answers to the same question when asked in English versus a native language (Arabic, Persian, or Turkish), indicating that cultural values are unstably encoded across languages.
2.  **Reasoning-Induced Degradation**: Prompting models to "think through" their answers by providing reasoning often *degrades* their cultural alignment, activating stereotypes or Western-centric logic instead of nuanced cultural knowledge.
3.  **Logit Leakage**: Models frequently issue surface-level refusals to sensitive questions (e.g., "I cannot answer that") while their internal logit probabilities reveal strong, high-confidence hidden preferences, suggesting safety training may only be masking underlying biases.


---

---

## 📈 Evaluation Framework

Our methodology systematically tests models under varying conditions to expose inconsistencies.

### Perspectives (Framing Styles)
-   **Neutral**: The LLM is queried directly without any identity constraints.
-   **Persona**: The model is instructed to adopt a national identity (e.g., "Imagine you are an average Saudi...").
-   **Observer**: The model is asked to act as a cultural analyst (e.g., "How would an average Saudi respond...").

### Languages
-   **English**: The dominant language of AI development.
-   **Native Languages**: Arabic, Persian, and Turkish, validated by native speakers.

### Reasoning Conditions
-   **Zero-Shot**: The model provides a direct, immediate answer without additional reasoning prompts.
-   **With-Reasoning**: The model is prompted to provide a brief explanation before its answer.

### Core Metrics
-   **$NVAS$**: Normalized Value Alignment Score — Measures alignment with ground-truth human values.
-   **$CLCS$**: Cross-Lingual Consistency Score — Measures response consistency between English and native languages.
-   **$FCS$**: Framing Consistency Score — Measures response consistency across Persona and Observer framings.
-   **$SPD$**: Self-Persona Deviation — Measures how much a model's response changes when assigned a persona.

---



