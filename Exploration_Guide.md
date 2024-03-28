# Hands-On Exploration Guide

This guide provides some suggestions for areas of the repository that participants might explore based on the topics covered in the morning lectures. Please note that these are simply suggestions and meant to help orient participants within the repository a bit better. As a participant, you need not follow this guide and should feel free to engage with any material in this repository that interests you. If it makes sense to dedicate your time completely to a single area of this repository, please do so to maximize your personal learning.

## Hands-On Day 1

In the lectures preceding Hands-On Day 1, we will have covered a number of topics including:

* Language Modeling and Evaluation Methods.
* Pre-training, fine-tuning, zero-shot and few-shot methods.
* Generation configuration and manipulation.
* Prompt Design, manual optimization, and ensembling.
* Challenges and capabilities associated with truly large language models.

As such, some areas of this repository that may be of interest are:

1. Hugging Face (HF) Basics: Fine-tuning NLP models using the HF API, using HF-hosted fine-tuned models for inference, and HF evaluation metrics.

    `src/reference_implementations/hugging_face_basics/`

2. Examples of Prompt design, manual optimization, zero- and few-shot prompting for various tasks. Tasks include classification, question answering, translation, summarization, and aspect-based sentiment analysis.

    `src/reference_implementations/prompting_vector_llms/llm_prompting_examples/`

    The README provides additional details for each of the examples in this folder.

    `src/reference_implementations/prompting_vector_llms/README.MD`

3. Some examples of ensembling multiple prompts to potentially improve performance.

    `src/reference_implementations/prompting_vector_llms/prompt_ensembling/`

## Hands-On Day 2

In the lectures preceding Hands-On Day 2, we will have covered several new topics including:

* Discrete and Continuous Prompt Optimization Techniques.
* Parameter Efficient Fine-Tuning.
* Introduction to fairness/bias analysis for NLP models, including through prompting.

As such, some new areas of this repository that could be of interest are:

1. An example of activation fine-tuning, with and without prompting.

    `src/reference_implementations/prompting_vector_llms/activation_fine_tuning/`

2. Examples of methods for fairness and bias analysis of LMs and LLMs

    `src/reference_implementations/fairness_measurement/`

    Notebooks for the BBQ, StereoSet, and Crow-S pairs tasks are well document while the README in the `opt_czarnowska_analysis/` folder provides details about the code therein.

3. Due to their complexity and computational intensity, implementations or discrete and continuous prompt optimization have been omitted from this iteration of the lab. However, a notebook considering the transferrability of prompts optimized with AutoPrompt is in `transferring_gradient_optimized_prompts/`. These prompts are difficult to interpret by a human but surprisingly produce significant task-specific improvements for the T5 model.

## Hands-On Day 3

In the lectures preceding Hands-On Day 3, we will have covered several new topics including:

* Chain-of-thought (CoT) prompting and reasoning generation in LLMs.
* Retrieval Augmented Generation (RAG) and advanced RAG techniques.

1. There are several implementations of CoT prompting techniques in the `prompting_vector_llms/llm_reasoning_prompting` folder. Exploring these notebooks should give an overview to some of the popular reasoning based prompting methods and why they work in practice.
2. There is a basic example of RAG in the folder `prompting_vector_llms/llm_basic_rag_example`. The example presumes an effective retrieval strategy, which is not guaranteed, but demonstrates why RAG is an effective strategy for open-ended question-answering.

On this third day, we encourage participants to continue exploring implementations and examples that interest them most, based on their previous days of investigation, within the repository.
