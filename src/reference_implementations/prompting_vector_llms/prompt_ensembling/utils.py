from typing import List, Optional, Tuple

import pandas as pd


def copa_preprocessor(path: str) -> List[Tuple[str, int, str, str, str]]:
    copa_df = pd.read_csv(path, sep="\t")
    premises = copa_df["premise"].to_list()
    # We process the labels to either 0 or 1 (they are 1 and 2 in the original dataset)
    labels = copa_df["label"].apply(lambda x: int(x) - 1).to_list()
    phrases = copa_df["phrase"].to_list()
    first_choices = copa_df["choice_1"].to_list()
    second_choices = copa_df["choice_2"].to_list()
    return list(zip(premises, labels, phrases, first_choices, second_choices))


def split_prompts_into_batches(prompts: List[str], batch_size: int = 10) -> List[List[str]]:
    return [prompts[x : x + batch_size] for x in range(0, len(prompts), batch_size)]


#########################################################
# Generation of first type of prompts
#########################################################


def create_first_prompt_label(first_choice: str, second_choice: str, label: int) -> str:
    return first_choice if label == 0 else second_choice


def structure_first_prompt_string(
    first_choice: str, second_choice: str, phrase: str, premise_phrase: str, label: Optional[str] = None
) -> str:
    prompt = f'"{first_choice}" or "{second_choice}"\n{phrase}{premise_phrase} '
    if label is None:
        return prompt
    return f"{prompt}{label}"


def create_first_prompt(
    demonstrations: List[Tuple[str, int, str, str, str]],
    premise: str,
    phrase: str,
    first_choice: str,
    second_choice: str,
) -> str:
    prompt = "Choose the sentence that best completes the phrase\n\n"
    for demo_premise, demo_label, demo_phrase, demo_first_choice, demo_second_choice in demonstrations:
        demo_first_choice = demo_first_choice.lower()
        demo_second_choice = demo_second_choice.lower()
        demo_label_str = create_first_prompt_label(demo_first_choice, demo_second_choice, demo_label)
        demo_phrase = demo_phrase.rstrip(".")
        premise_phrase = ", because" if "cause" in demo_premise else ", so"
        demonstration_str = structure_first_prompt_string(
            demo_first_choice, demo_second_choice, demo_phrase, premise_phrase, demo_label_str
        )
        prompt = f"{prompt}{demonstration_str}\n\n"
    premise_phrase = ", because" if "cause" in premise else ", so"
    phrase = phrase.rstrip(".")
    data_point_prompt = structure_first_prompt_string(first_choice, second_choice, phrase, premise_phrase)
    return f"{prompt}{data_point_prompt} "


#########################################################
# Generation of second type of prompts
#########################################################


def create_second_prompt_label(first_choice: str, second_choice: str, label: int) -> str:
    return first_choice if label == 0 else second_choice


def structure_second_prompt_string(phrase: str, premise_phrase: str, label: Optional[str] = None) -> str:
    prompt = f"{phrase}{premise_phrase}"
    if label is None:
        return prompt
    return f"{prompt} {label}"


def create_second_prompt(demonstrations: List[Tuple[str, int, str, str, str]], premise: str, phrase: str) -> str:
    prompt = "Complete the phrase with a logical phrase.\n\n"
    for demo_premise, demo_label, demo_phrase, demo_first_choice, demo_second_choice in demonstrations:
        demo_first_choice = demo_first_choice.lower()
        demo_second_choice = demo_second_choice.lower()
        demo_label_str = create_second_prompt_label(demo_first_choice, demo_second_choice, demo_label)
        demo_phrase = demo_phrase.rstrip(".")
        premise_phrase = ", because" if "cause" in demo_premise else ", so"
        demonstration_str = structure_second_prompt_string(demo_phrase, premise_phrase, demo_label_str)
        prompt = f"{prompt}{demonstration_str}\n\n"
    premise_phrase = ", because" if "cause" in premise else ", so"
    phrase = phrase.rstrip(".")
    data_point_prompt = structure_second_prompt_string(phrase, premise_phrase)
    return f"{prompt}{data_point_prompt} "


#########################################################
# Generation of multiple choice type of prompts
#########################################################


def create_mc_prompt_answer(label: int) -> str:
    return "A" if label == 0 else "B"


def structure_mc_prompt_string(
    phrase: str, first_choice: str, second_choice: str, premise_phrase: str, label: Optional[str] = None
) -> str:
    prefix = "From A or B which choice best completes the phrase?\nPhrase: "
    prompt = f"{prefix}{phrase}{premise_phrase}\nA: {first_choice}\nB: {second_choice}\nAnswer:"
    if label is None:
        return f"{prompt}"
    return f"{prompt} {label}"


def create_mc_prompt(
    demonstrations: List[Tuple[str, int, str, str, str]],
    premise: str,
    phrase: str,
    first_choice: str,
    second_choice: str,
) -> str:
    prompt = ""
    for demo_premise, demo_label, demo_phrase, demo_first_choice, demo_second_choice in demonstrations:
        demo_first_choice = demo_first_choice.lower()
        demo_second_choice = demo_second_choice.lower()
        demo_label_str = create_mc_prompt_answer(demo_label)
        demo_phrase = demo_phrase.rstrip(".")
        premise_phrase = ", because" if "cause" in demo_premise else ", so"
        demonstration_str = structure_mc_prompt_string(
            demo_phrase, demo_first_choice, demo_second_choice, premise_phrase, demo_label_str
        )
        prompt = f"{prompt}{demonstration_str}\n\n"
    premise_phrase = ", because" if "cause" in premise else ", so"
    phrase = phrase.rstrip(".")
    data_point_prompt = structure_mc_prompt_string(phrase, first_choice.lower(), second_choice.lower(), premise_phrase)
    return f"{prompt}{data_point_prompt}"
