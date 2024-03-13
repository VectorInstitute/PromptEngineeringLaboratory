# Prompting Vector's Large Language Models

This folder contains a number of notebooks and reference implementations aimed at working with Vector's toolkit for interacting with LLMs hosted on Vector's cluster. The current models hosted on our cluster are:

1) [Falcon](https://falconllm.tii.ae/index.html) (7B and 40B parameters): Falcon is a recent LLM from the Technology Innovation Institute (TII). At the time of its release, it was a state-of-the-art LLM, beating the original LLaMA model on various tasks. It is open source and has a permissive commercial license. The largest model in the family is 180B. However, the 40B parameter model is capable of handling fairly complex tasks, such as incorporating chain-of-thought reasoning.
2) [LLaMA-2](https://arxiv.org/abs/2307.09288) (7B, 13B, 70B): LLaMA-2 is the most recent open-source LLM released by Meta. It is one of the highest performing open-source models. It has a limited commercial-use license. However, it is fully open for academic use. The version of the model used for the lab is __NOT__ the chat version. While the chat model may be better at following instructions due to IFT, this also introduces additional behavior modifications that we'd like to circumvent.

__At least two copies of each model will be hosted on the Vector cluster. Each copy will be accessible through ports 3001 or 4001 using the Kaleidoscope library. A third port will potentially be open at 5001 at the time of the lab.__

__NOTE__: Due to their size, and depending on the size of the sequence input, the largest models may take some time to respond. In addition, the model can be sensitive to Out-of-memory errors if a large batch of inputs and activations are requested. As such, the models are currently configured to only accept requests with batch sizes of 10 samples or less. Make sure to batch your requests to be under this size.

__NOTE__: LLaMA-2, of all sizes, has been configured to have a limited context of 512 (original context length of 4096). This is purely for memory management and inference speed. The API will throw an error if you try to send a generation request for a prompt with more than 512 tokens.

## LLM Configuration

Below, we briefly describe the contents of the folders and notebooks in this section of the repository. However, in a separate markdown file, we also provide some details associated with the generation configs used by the Vector hosted LLMs for those who are less familiar with some of the parameters used to control how they generate text. See

`src/reference_implementations/prompting_vector_llms/CONFIG_README.md`

Additional documentation of the Kaleidoscope toolkit may also be found [here](https://kaleidoscope-docs.readthedocs.io/en/latest/index.html) and in the git repository [here](https://github.com/vectorInstitute/kaleidoscope/).

## LLM Prompting Examples

`src/reference_implementations/prompting_vector_llms/llm_prompting_examples/`

This folder gathers a number of example demonstrations of LLM prompting. There are five notebooks considering different tasks. The notebooks explore the effects of manual prompt optimization, varying instruction prompts, zero- and few-shot performance. The tasks considered are:

1) Summarization
2) Classification: AG News task
3) Question-Answering: BoolQ
4) Basic Translation (French -> English)
5) Aspect-Based Sentiment Analysis
6) Activation Fine-tuning

The notebooks are loosely ordered, in the sense that they build somewhat on each other. The rough ordering follows the enumeration above. That is, `Summarization`, `Classification: AG News`, `Question-Answering: BoolQ`, `Translation`, and `ABSA`. There are additional details about each task below and in the respective notebooks.

There is also a notebook considering loading and prompting an instruction fine-tuned model in the form of Koala. Koala can be better at following direct instructions for tasks as Koala is instruction fine-tuned.

`src/reference_implementations/prompting_vector_llms/llm_prompting_examples/llm_prompt_ift_koala_local.ipynb`

The notebook doesn't implement a specific task, but is freely adaptable for experimentation. __Note__: Initializing this model can take some time and it is recommended that at least 32GB of CPU memory is reserved for the task.

### Summarization

This notebook explores the summarization capabilities of Falcon on a small sample of news articles. This includes the effects of variations on instructions, how postprocessing might be used to prepare the summaries, and the various behaviors depending on how the prompt is structured.

At the end of the notebook, we consider the performance of two prompts on a small sample of the CNN Daily News dataset, as measured by the ROUGE-1 Score.

### Classification: AGs News

The notebook considers the task of categorizing news into one of four categories. The task is based on the AG News dataset, which has news stories belonging to the categories of World, Sports, Business, and Science/Technology. The notebook measures performance on a small sample from the AG news test set. Manual prompt optimization, zero-shot, and few-shot prompts are considered. The notebook also explores the concept of "label spaces." That is, how do we map answers generated by the LLM to the labels we care about. Specifically, we consider pure response and logits extraction associated with a chosen label space.

### Question-Answering: BoolQ

This notebook implements an example of performing question-answering based on a provided context. The BoolQ task consists of a subject, passage, and question based on the passage with a boolean answer. The question always has a correct response based on the context. We consider zero-shot and few-shot performance on a small sample of the full BoolQ dataset. The notebook also compares performance using two different label spaces. The primary focus is on various ways to setup zero-shot prompting to improve accuracy.

### Translation

The notebook is constructed to provide a proof of concept in translation through prompting of Falcon. Falcon is trained on some multilingual datasets, and has been shown to be able to perform some basic translations. However, the model was intended to be a primarily English language model and was not specifically meant for translation. This notebook demonstrates the extent to which the model can provide translations in zero- and few-shot settings. However, the model is not particularly strong, as measured by the BLEU scores on a small sample from the NMT14 fr-en test set, compared with translation specific models.

### Aspect-Based Sentiment Analysis

The notebook considers the task of assigning sentiments (positive, negative, or neutral) towards certain aspects of the input. Here, the aspect term denotes the specific text that explicitly appears in the given text. The notebook shows results on a small subset of the dataset with customer reviews of laptops. Zero-shot and Few-shot tasks are also explored in this notebook.

## Activation Fine-tuning

The folder `activation_fine_tuning/` considers extracting intermediate layer activations for the LLaMA-2 models and training a minimal classifier (in this case a 2-layer MLP) to perform a downstream task with high accuracy. The experiments are described in much greater detail in the `activation_fine_tuning/README.md`. The results are quite interesting and motivate prompting as a means of substantially increasing sampling efficiency in this regime.

## Ensembling Example

In the folder `prompt_ensembling/` there are several examples of how one might use prompt ensembling to improve the accuracy of an LLM's responses on a downstream task. The idea is to combine prompts and/or generations in order to ask the model to perform a task in multiple ways, hopefully producing better performance by asking the same question in various forms.

The notebooks consider the Balanced Choice of Plausible Alternatives (CoPA) task, which is a harder version of the original CoPA task. We work with a small sample of the dataset. The task, in short, is, given a context and a premise of either cause or effect, the model must choose between two distinct sentences to determine which is the logical following sentence. An example is:

From the following choices,
1) The author faded into obscurity.
2) It was adapted into a movie.

and an __effect__ premise, which logically follows the sentence "The book became a huge failure." The answer is "The author faded into obscurity."

The first notebook (`copa_prompting_examples.ipynb`) considers several different ways to formulate prompts for this problem, including few-shot prompting, multiple choice question-answering, and likelihood estimation. Each performs the task with a different accuracy. In `bootstrap_ensembling.ipynb`, we consider "bootstrap" ensembles that combine the same prompt structure in different ways and uses vote ensembling to combine the responses. The final notebook, `prompt_ensembling.ipynb`, considers combining the responses of both LLaMA-2 and Falcon across different strategies for a combination of 5 different approaches. The responses are combined with a vote ensembling, achieving the highest accuracy on the task.

## Reasoning in Large Language Models

State-of-the-art LLMs, including Falcon and LLaMA-2, have made significant progress in performing tasks that require complex or multi-step reasoning, such as mathematics word problems. [Some studies](https://arxiv.org/abs/2305.06161) attribute such capabilities to the intentional inclusion of code in the LLM pre-training phase.

One of the most effective ways to help LLMs perform such tasks is through reasoning generation techniques. The most popular way to do this is through so-called Chain-of-Thought (CoT) prompting. The `cot_prompting.ipynb` notebook introduces both standard CoT prompting ([using few-shot examples](https://arxiv.org/pdf/2201.11903.pdf)) and [zero-shot CoT](https://arxiv.org/pdf/2205.11916.pdf). This notebook runs through these examples in details and demonstrates how these approaches work and why. The `measuring_zero_shot_cot.ipynb` notebook quantifies the benefits of zero-shot CoT prompting over standard few-shot prompting for the MultiArith task (a simple mathematics word problem task). Finally, the `self_consistent_prompting.ipynb` notebook considers an implementation of the [Self-Consistency Prompting Technique](https://arxiv.org/abs/2203.11171). The notebook simply considers one example of how self-consistency could be used to answer a math problem.

## Can we transfer gradient optimized prompts to LLaMA-2?

In the folder `transferring_gradient_optimized_prompts/`, we consider whether the gradient optimized prompts that we have found during discrete prompt optimization through gradient based search for the T5 model are transferrable to the LLaMA-2 7B model for sentiment analysis. This model is larger than T5, but has a different tokenizer and a different architecture. T5 has an encoder-decoder coupling, while LLaMA-2 is decoder-only. The prompts were optimized using AutoPrompt with a Vector implementation [here](https://github.com/VectorInstitute/PromptEngineering/blob/main/src/reference_implementations/prompt_zoo/gradient_search.py). Due to the compute intensive nature and complexity of the code, the implementation source code has not been included in the lab.
