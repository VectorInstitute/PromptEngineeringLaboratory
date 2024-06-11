import argparse
import os
import random
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import pipeline

TrueLabel = int
PredictedLabel = int
Category = str
Group = str
TestText = str
Model = str
RunID = str
Dataset = str
NumParams = float
TestEntry = Tuple[TrueLabel, Category, Group, TestText]
OutputEntry = Tuple[
    PredictedLabel, TrueLabel, Category, Group, TestText, Model, RunID, Dataset, NumParams,
]

PATH_STUB = "src/reference_implementations/fairness_measurement/resources"
TEST_FILE_PATH = f"{PATH_STUB}/czarnowska_templates/sentiment_fairness_tests.tsv"
MODEL = "OPT-6.7B"
NUM_PARAMS: float = 6.7  # billions
BATCH_SIZE = 4
N_SHOTS = 9

SEEDS = {
    "run_1": 2024,
    "run_2": 2025,
    "run_3": 2026,
    "run_4": 2027,
    "run_5": 2028,
}

MODEL_PATH = "/ssd005/projects/llm/opt-6.7b"
generator = pipeline("text-generation", model=MODEL_PATH, device="cuda")

# Czarnowska Labels
label_lookup = {
    "negative": 0,  # Negative
    "neutral": 1,  # Neutral
    "positive": 2,  # Positive
}  # Maps string labels to integers.

reverse_label_lookup = {label_int: label_str for label_str, label_int in label_lookup.items()}

number_of_demonstrations_per_label = N_SHOTS // 3
number_of_random_demonstrations = N_SHOTS - number_of_demonstrations_per_label * 3


def create_demonstrations(dataset: str) -> str:
    if dataset == "SST5":
        path = "src/reference_implementations/fairness_measurement/czarnowska_analysis/resources/processed_sst5.tsv"
    else:
        path = "src/reference_implementations/fairness_measurement/czarnowska_analysis/resources/processed_semeval.tsv"
    if dataset != "ZeroShot":
        df = pd.read_csv(path, sep="\t", header=0)
        # Trying to balance the number of labels represented in the demonstrations
        sample_df_negative = df.Valence[df.Valence.eq("Negative")].sample(number_of_demonstrations_per_label).index
        sample_df_neutral = df.Valence[df.Valence.eq("Neutral")].sample(number_of_demonstrations_per_label).index
        sample_df_positive = df.Valence[df.Valence.eq("Positive")].sample(number_of_demonstrations_per_label).index
        random_sampled_df = df.sample(number_of_random_demonstrations).index
        sampled_df = df.loc[
            sample_df_negative.union(sample_df_neutral).union(sample_df_positive).union(random_sampled_df)
        ]
        texts = sampled_df["Text"].tolist()
        valences = sampled_df["Valence"].tolist()

        demonstrations = ""
        for text, valence in zip(texts, valences):
            demonstrations = (
                f"{demonstrations}Text: {text}\nQuestion: What is the sentiment of the text?\nAnswer: {valence}.\n\n"
            )
        print("Example of demonstrations")
        print("---------------------------------------------------------------------")
        print(demonstrations)
        print("---------------------------------------------------------------------")
        return demonstrations
    else:
        return ""


def create_prompt_for_text(text: str, demonstrations: str, dataset: str) -> str:
    if dataset != "ZeroShot":
        return f"{demonstrations}Text: {text}\nQuestion: What is the sentiment of the text?\nAnswer:"
    else:
        return f"Text: {text}\nQuestion: Is the sentiment of the text negative, neutral, or positive?\nAnswer: The sentiment is"


def create_prompts_for_batch(input_texts: List[str], demonstrations: str, dataset: str) -> List[str]:
    prompts = []
    for input_text in input_texts:
        prompts.append(create_prompt_for_text(input_text, demonstrations, dataset))
    return prompts


def extract_predicted_label(sequence: str) -> str:
    match = re.search(r"positive|negative|neutral", sequence, flags=re.IGNORECASE)
    if match:
        return match.group().lower()
    else:
        # If no part of the generated response matches our label space, randomly choose one.
        print(f"Unable to match to a valid label in {sequence}")
        return random.choice(["positive", "negative", "neutral"])


def get_predictions_batched(input_texts: List[str], demonstrations: str, dataset: str) -> List[str]:
    predicted_labels = []
    prompts = create_prompts_for_batch(input_texts, demonstrations, dataset)
    batched_sequences = generator(prompts, do_sample=True, max_new_tokens=3, temperature=0.8, return_full_text=False)
    for prompt_sequence in batched_sequences:
        generated_text = prompt_sequence[0]["generated_text"]
        predicted_label = extract_predicted_label(generated_text)
        predicted_labels.append(predicted_label)
    assert len(predicted_labels) == len(input_texts)
    return predicted_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OPT6.7 Czarnowska Prompting")
    parser.add_argument("--run_id", action="store", type=str, help="Should be one of run_1...run_5", default="run_1")
    parser.add_argument(
        "--dataset",
        action="store",
        type=str,
        help="Labeled task-specific dataset to use for few-shots. Should be one of SST5, SemEval, or ZeroShot",
        default="SST5",
    )
    args = parser.parse_args()

    run_id = args.run_id
    dataset = args.dataset
    assert run_id in ["run_1", "run_2", "run_3", "run_4", "run_5"]
    assert dataset in ["SST5", "SemEval", "ZeroShot"]

    # Append results to this file.
    PREDICTION_FILE_PATH = (
        f"{PATH_STUB}/prompt_tuning_fairness_paper_preds/opt6_7b/opt_6_7b_prompt_predictions_{dataset}_{run_id}.tsv"
    )

    # Setting random seed according to run ID
    SEED = SEEDS[run_id]
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    tests: List[TestEntry] = []

    if not os.path.exists(PREDICTION_FILE_PATH):
        # If the prediction file doesn't exist, we create a new one and append the tsv header row.
        header_row = "\t".join(
            ["y_true", "y_pred", "category", "group", "text", "model", "run_id", "dataset", "num_params",]
        )
        header_row = header_row + "\n"
        with open(PREDICTION_FILE_PATH, "w") as prediction_file:
            prediction_file.write(header_row)

    # Figure out how many predictions we've made so far
    with open(PREDICTION_FILE_PATH, "r") as prediction_file:
        num_predictions_so_far = len(prediction_file.readlines()) - 1

    # Open the templates and only start reading from where we left off
    with open(TEST_FILE_PATH, "r") as template_file:
        template_files_lines = template_file.readlines()
        for line in template_files_lines[num_predictions_so_far:]:
            label_str, attribute, group, text = tuple(line.rstrip().split("\t"))
            # convert the label string to an int
            label = int(label_str)
            tests.append((label, attribute, group, text))

    test_batches = [tests[x : x + BATCH_SIZE] for x in range(0, len(tests), BATCH_SIZE)]
    demonstrations = create_demonstrations(dataset)
    example_prompt = create_prompt_for_text("I did not like that movie at all.", demonstrations, dataset)
    print(f"Example Prompt\n{example_prompt}")

    # Append to the output file instead of overwriting.
    with open(PREDICTION_FILE_PATH, "a") as prediction_file:
        for batch in tqdm(test_batches):
            output: List[OutputEntry] = []
            text_batch = [test_case[-1] for test_case in batch]  # Extract texts from the batch.
            predictions = get_predictions_batched(text_batch, demonstrations, dataset)

            for prediction, test_entry in zip(predictions, batch):
                label, attribute, group, text = test_entry
                output_entry = (
                    label,
                    label_lookup[prediction],
                    attribute,
                    group,
                    text,
                    MODEL,
                    run_id,
                    dataset,
                    NUM_PARAMS,
                )
                output.append(output_entry)

            output_lines = []
            for output_entry in output:
                output_lines.append(
                    "\t".join(map(str, output_entry)) + "\n"
                )  # Convert integers to string before concatenating.

            prediction_file.writelines(output_lines)
            prediction_file.flush()
