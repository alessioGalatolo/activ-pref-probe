"""Dataset processing and loading utilities for APO."""

import random
import re
from typing import List, Dict, Any
from httpx import get
from collections import defaultdict
import pandas as pd
import json
from copy import deepcopy
from datasets import load_dataset, Dataset as HFDataset


def common_start(str1: str, str2: str) -> str:
    """Find common starting substring between two strings."""
    common_chars = []
    for c1, c2 in zip(str1, str2):
        if c1 == c2:
            common_chars.append(c1)
        else:
            break
    return "".join(common_chars)


def extract_hh(example: str) -> list[dict[str, str]]:
    """Extract prompt, chosen, and rejected from HH-RLHF dataset format."""
    prompt_text = common_start(example["chosen"], example["rejected"])

    if not prompt_text.endswith("\n\nAssistant: "):
        prompt_text = prompt_text[: prompt_text.rfind("\n\nAssistant: ")] + "\n\nAssistant: "

    chosen_line = example["chosen"][len(prompt_text):]
    rejected_line = example["rejected"][len(prompt_text):]

    prompt_text = prompt_text[: -len("\n\nAssistant: ")]

    prompt_lines = re.split(r"(\n\nAssistant: |\n\nHuman: )", prompt_text)
    prompt_lines = prompt_lines[1:]

    prompt = []
    for idx in range(0, len(prompt_lines), 2):
        role = "user" if prompt_lines[idx] == "\n\nHuman: " else "assistant"
        content = prompt_lines[idx + 1]
        prompt.append({"role": role, "content": content})

    chosen = [{"role": "assistant", "content": chosen_line}]
    rejected = [{"role": "assistant", "content": rejected_line}]

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def extract_afrisenti(example: dict, label_names: List[str] = None) -> dict:
    """
    Convert AfriSenti sentiment classification example to preference format.

    Args:
        example: Dict with 'tweet' and 'label' keys
        label_names: List of possible label names (e.g., ['positive', 'negative', 'neutral'])

    Returns:
        Dict with 'prompt', 'chosen', and 'rejected' keys in chat format
    """
    tweet = example["tweet"]
    correct_label = example["label"]

    if label_names is None:
        label_names = ["positive", "negative", "neutral"]

    prompt = [
        {
            "role": "user",
            "content": f"Classify the sentiment of the following tweet as positive, negative, or neutral:\n\n{tweet}"
        }
    ]

    chosen = [
        {
            "role": "assistant",
            "content": f"The sentiment of this tweet is {correct_label}."
        }
    ]

    other_labels = [label for label in label_names if label != correct_label]
    if other_labels:
        incorrect_label = random.choice(other_labels)
    else:
        incorrect_label = "unknown"

    rejected = [
        {
            "role": "assistant",
            "content": f"The sentiment of this tweet is {incorrect_label}."
        }
    ]

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def load_preference_dataset(dataset_name: str, split: str = "train", max_samples: int = None, dataset_args: str = None):
    """
    Load and format a preference dataset.

    Args:
        dataset_name: Name of the dataset to load
        split: Dataset split (train, validation, test)
        max_samples: Maximum number of samples to load
        dataset_args: Dataset-specific argument:
            - afrisenti: language code (e.g. 'amh', 'ha')
            - oasst2: ISO 639-1 language code (e.g. 'de', 'fr', 'es', 'zh'); defaults to 'de'
            - prism: group key

    Returns:
        List of dicts with 'prompt', 'chosen', and 'rejected' keys
    """
    print(f"Loading dataset: {dataset_name}")

    if "hh" in dataset_name:
        ds = load_dataset("Anthropic/hh-rlhf", split=split)
        # subsets (harmless-base, helpful-base, ...) are concatenated in a fixed
        # order, not interleaved, so shuffle before truncating to max_samples
        # ds = ds.shuffle(seed=42)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        return ds.map(extract_hh, remove_columns=ds.column_names, load_from_cache_file=False).to_list()

    elif "afrisenti" in dataset_name.lower():
        if dataset_args is None:
            raise ValueError("Language parameter required for AfriSenti dataset. Use --po-dataset-args to specify (e.g., 'amh', 'dz', 'ha')")

        print(f"Loading AfriSenti dataset for language: {dataset_args}")
        url = f"https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main/data/{dataset_args}/{split}.tsv"
        data = get(url).text.strip().split("\n")
        records = []
        for line in data[1:]:  # skip header
            parts = line.split("\t")
            if len(parts) >= 2:
                records.append({
                    "tweet": parts[0],
                    "label": parts[1],
                })
        ds = HFDataset.from_list(records)

        # this dataset is label-sorted, shuffle before selecting max samples to get a good label distribution
        ds = ds.shuffle(seed=42)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        unique_labels = list(set(ds["label"]))
        print(f"Found labels: {unique_labels}")

        return ds
        def extract_with_labels(example):
            return extract_afrisenti(example, label_names=unique_labels)

        return ds.map(extract_with_labels, remove_columns=ds.column_names, load_from_cache_file=False).to_list()

    elif "ultrafeedback" in dataset_name.lower():
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split + "_prefs")
        # we have to take shorter samples first to avoid OOM
        ds = ds.map(lambda x: {"prompt_length": len(x["prompt"])})
        ds = ds.sort("prompt_length")

        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        def format_ultrafeedback(example):
            prompt = example["prompt"]
            prompt = [{"role": "user", "content": prompt}]
            chosen = example["chosen"]
            rejected = example["rejected"]
            # chosen rejected also contain the prompt
            chosen.pop(0)
            rejected.pop(0)
            return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

        ds = ds.map(format_ultrafeedback, remove_columns=ds.column_names, load_from_cache_file=False)
        ds = ds.filter(lambda x: len(x["chosen"][0]["content"]) < 8000 and len(x["rejected"][0]["content"]) < 8000 and len(x["prompt"][0]["content"]) < 8000)
        return ds.to_list()

    elif "nectar" in dataset_name.lower():
        ds = load_dataset("berkeley-nest/Nectar", split=split)
        # we have to take shorter samples first to avoid OOM
        ds = ds.map(lambda x: {"prompt_length": len(x["prompt"])})
        ds = ds.sort("prompt_length")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        def format_nectar(example):
            prompt_text = example["prompt"]
            prompt_text = prompt_text[: -len("\n\nAssistant: ")]
            prompt_lines = re.split(r"(\n\nAssistant: |\n\nHuman: )", prompt_text)
            prompt_lines = prompt_lines[1:]

            prompt = []
            for idx in range(0, len(prompt_lines), 2):
                role = "user" if prompt_lines[idx] == "\n\nHuman: " else "assistant"
                content = prompt_lines[idx + 1]
                prompt.append({"role": role, "content": content})
            answers = example["answers"]
            best_answer = answers[0]["answer"]
            worst_answer = answers[-1]["answer"]
            chosen = [{"role": "assistant", "content": best_answer}]
            rejected = [{"role": "assistant", "content": worst_answer}]
            return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
        ds = ds.map(format_nectar, remove_columns=ds.column_names, load_from_cache_file=False)
        return ds.to_list()
    elif "cvalues" in dataset_name.lower():
        # BAAI/CValues-Comparison: Chinese preference dataset.
        # Columns: prompt, pos_resp (chosen), neg_resp (rejected), type, source.
        ds = load_dataset("Skepsun/cvalues_rlhf", split=split)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        def format_cvalues(example):
            prompt = [{"role": "user", "content": example["prompt"]}]
            chosen = [{"role": "assistant", "content": example["pos_resp"]}]
            rejected = [{"role": "assistant", "content": example["neg_resp"]}]
            return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

        return ds.map(format_cvalues, remove_columns=ds.column_names, load_from_cache_file=False).to_list()

    elif "oasst2" in dataset_name.lower():
        # OpenAssistant/oasst2: multilingual preference dataset.
        # Each row is a message node in a conversation tree with a 'rank' field.
        # dataset_args: ISO 639-1 language code to filter by (e.g. 'de', 'fr', 'es', 'zh').
        # Defaults to German ('de') when no language is specified.
        target_lang = dataset_args.lower() if dataset_args else "de"
        print(f"Loading oasst2 dataset for language: {target_lang}")

        ds = load_dataset("OpenAssistant/oasst2", split=split)

        # Group messages by tree so we can resolve prompt→response pairs.
        trees = defaultdict(list)
        for msg in ds:
            trees[msg["message_tree_id"]].append(msg)

        pairs = []
        for tree_msgs in trees.values():
            # Find the root prompter turn (no parent).
            root = next(
                (m for m in tree_msgs if m["parent_id"] is None and m["role"] == "prompter"),
                None,
            )
            if root is None or root["lang"] != target_lang:
                continue

            # Direct assistant replies to the root.
            responses = [
                m for m in tree_msgs
                if m["parent_id"] == root["message_id"]
                and m["role"] == "assistant"
                and not m.get("deleted", False)
                and m.get("rank") is not None
            ]
            if len(responses) < 2:
                continue

            # rank 0 = best response.
            responses.sort(key=lambda m: m["rank"])
            if responses[0]["rank"] == responses[-1]["rank"]:
                continue

            pairs.append({
                "prompt": [{"role": "user", "content": root["text"]}],
                "chosen": [{"role": "assistant", "content": responses[0]["text"]}],
                "rejected": [{"role": "assistant", "content": responses[-1]["text"]}],
            })

        if max_samples:
            pairs = pairs[:max_samples]
        return pairs

    elif "prism" in dataset_name.lower():
        # prism contains a lot of internal datasets, for now we just load all and return the first one
        # the first one should correspond to "Africa"
        prism = get_formatted_prism()
        if dataset_args:
            if dataset_args in prism:
                print(f"Loading PRISM group: {dataset_args} ({len(prism[dataset_args])} examples)")
                return prism[dataset_args][:max_samples] if max_samples else prism[dataset_args]
            else:
                print(f"Warning: specified group '{dataset_args}' not found in PRISM dataset. Available groups: {list(prism.keys())}")
        # if no args, return all values concatenated
        all_samples = [example for examples in prism.values() for example in examples]
        return all_samples[:max_samples] if max_samples else all_samples
    else:
        ds = load_dataset(dataset_name, split=split)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        return list(ds)


def prepare_sft_dataset(dataset_name: str, tokenizer, max_samples: int = 1000, dataset_args: str = None, offset: int = 0):
    """Prepare dataset for SFT.

    Args:
        dataset_name: Name of the dataset to load
        tokenizer: Tokenizer for the model
        max_samples: Maximum number of samples to load
        dataset_args: Arguments for datasets (e.g., AfriSenti)
        offset: Number of samples to skip at the beginning (for non-overlapping splits)

    Returns:
        Dataset formatted for SFT training
    """
    print(f"Loading SFT dataset: {dataset_name}")

    is_preference_dataset = any(
        kw in dataset_name.lower()
        for kw in ("hh-rlhf", "afrisenti", "cvalues", "oasst2")
    )

    if is_preference_dataset:
        print("Detected preference dataset, extracting 'chosen' responses for SFT")
        preference_data = load_preference_dataset(
            dataset_name,
            split="train",
            max_samples=max_samples + offset if max_samples else None,
            dataset_args=dataset_args
        )

        if offset > 0:
            preference_data = preference_data[offset:offset + max_samples]
        elif max_samples:
            preference_data = preference_data[:max_samples]

        sft_data = []
        for item in preference_data:
            sft_data.append({
                "prompt": item["prompt"],
                "completion": item["chosen"]
            })

        return HFDataset.from_list(sft_data)
    else:
        ds = load_dataset(dataset_name, split="train")

        if offset > 0 and max_samples:
            ds = ds.select(range(offset, min(offset + max_samples, len(ds))))
        elif max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        def format_alpaca(example):
            result = [
                {
                    "role": "system",
                    "content": example["instruction"]
                },
            ]
            if example.get("input"):
                result.append({
                    "role": "user",
                    "content": example["input"]
                })
            return {"prompt": result, "completion": [{"role": "assistant", "content": example["output"]}]}

        ds = ds.map(format_alpaca)
        return ds


def get_formatted_prism():
    # Define custom grouping criteria
    grouping_criteria = {
        'age_group': True,
        'country_region': True,
    }

    # Process dataset
    print("Starting PRISM dataset processing...")
    grouped_datasets = process_prism_dataset(
        grouping_criteria=grouping_criteria,
        create_pairs=True  # Set to False for single completion format
    )
    return grouped_datasets


def load_prism_data():
    """Load PRISM dataset from HuggingFace"""
    print("Loading PRISM dataset from HuggingFace...")
    survey_data = load_dataset("HannahRoseKirk/prism-alignment", "survey", split="train")
    conversations_data = load_dataset("HannahRoseKirk/prism-alignment", "conversations", split="train")

    return survey_data, conversations_data


def create_user_groups(survey_data, grouping_criteria: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Split users into groups based on survey responses.

    Args:
        survey_data: Survey dataset from PRISM
        grouping_criteria: Dict specifying how to group users
            Example: {
                'age': [(18, 30), (31, 50), (51, 100)],
                'country': ['US', 'UK', 'other'],
                'political_orientation': ['left', 'center', 'right']
            }

    Returns:
        Dict mapping group names to lists of user_ids
    """
    user_groups = defaultdict(list)

    # Convert to pandas for easier manipulation
    survey_df = pd.DataFrame(survey_data)

    # If no criteria specified, use default groupings
    if not grouping_criteria:
        grouping_criteria = {
            'age_group': True,  # Young (18-35), Middle (36-55), Senior (56+)
            'country_region': True,  # US, UK, EU, Asia, Other
        }

    for _, user_row in survey_df.iterrows():
        user_id = user_row['user_id']
        group_labels = []

        # Apply grouping criteria
        for criterion, values in grouping_criteria.items():
            if criterion == 'age_group' and values:
                age = user_row.get('age', 0)
                group_labels.append(age)

            elif criterion == 'country_region' and values:
                country = user_row.get('location')["special_region"]
                group_labels.append(country)

            elif criterion in user_row:
                # Generic criterion: check if value matches
                user_value = user_row[criterion]
                if isinstance(values, list) and user_value in values:
                    group_labels.append(f"{criterion}_{user_value}")
                elif isinstance(values, dict):
                    # Range-based grouping
                    for group_name, range_vals in values.items():
                        if range_vals[0] <= user_value <= range_vals[1]:
                            group_labels.append(group_name)

        # Create group key from labels
        if group_labels:
            group_key = "_".join(sorted(set(group_labels)))
            user_groups[group_key].append(user_id)
        else:
            user_groups['ungrouped'].append(user_id)

    return dict(user_groups)


def format_conversation(conv_data: Dict) -> List[Dict[str, Any]]:
    """
    Format a single conversation into the required format.

    Returns dict with either:
    - prompt, completion, preference (for single responses)
    - prompt, chosen, rejected (for preference pairs)
    """
    user_id = conv_data['user_id']
    turns = conv_data.get('conversation_history')

    data = []

    for i, turn in enumerate(turns):
        role = turn.get('role')
        content = turn.get('content', '')

        if role == 'user':
            if data:
                di = deepcopy(data[-1])
                di["prompt"].extend(di["chosen"])
            else:
                di = {"prompt": []}
            data.append(di)
            di["prompt"].append({"role": "user", "content": content})
        elif role == 'model':
            content = [{"role": "assistant", "content": content}]
            if turn["if_chosen"]:
                data[-1]["chosen"] = content
            else:
                data[-1]["rejected"] = content

    # filter out incomplete pairs
    data = [d for d in data if "chosen" in d and "rejected" in d]
    return data


def create_preference_pairs(conversations: List[Dict]) -> List[Dict]:
    """
    Create preference pairs (chosen/rejected) from conversations with same prompt.
    Groups conversations by prompt and creates pairs based on ratings.
    """
    # Group by prompt (using first user message as key)
    prompt_groups = defaultdict(list)
    
    for conv in conversations:
        if conv['prompt']:
            # Use first user message as grouping key
            prompt_key = json.dumps(conv['prompt'])
            prompt_groups[prompt_key].append(conv)
    
    preference_pairs = []
    
    for prompt_key, convs in prompt_groups.items():
        # If we have ratings, create pairs
        rated_convs = [c for c in convs if 'preference' in c]
        
        if len(rated_convs) >= 2:
            # Sort by preference score
            sorted_convs = sorted(rated_convs, key=lambda x: x['preference'], reverse=True)
            
            # Create pairs: higher rated vs lower rated
            for i in range(len(sorted_convs) - 1):
                if sorted_convs[i]['preference'] > sorted_convs[i + 1]['preference']:
                    pair = {
                        "prompt": sorted_convs[i]['prompt'],
                        "chosen": sorted_convs[i]['completion'],
                        "rejected": sorted_convs[i + 1]['completion'],
                        "chosen_model": sorted_convs[i]['model'],
                        "rejected_model": sorted_convs[i + 1]['model'],
                        "chosen_score": sorted_convs[i]['preference'],
                        "rejected_score": sorted_convs[i + 1]['preference']
                    }
                    preference_pairs.append(pair)

    return preference_pairs


def process_prism_dataset(grouping_criteria: Dict[str, Any] = None,
                          create_pairs: bool = True) -> Dict[str, List[Dict]]:
    """
    Main processing function.

    Args:
        grouping_criteria: How to group users (see create_user_groups)
        create_pairs: Whether to create chosen/rejected pairs or keep single completions

    Returns:
        Dict mapping preference group names to formatted datasets
    """
    # Load data
    survey_data, conversations_data = load_prism_data()

    # Create user groups
    print("Creating user groups...")
    user_groups = create_user_groups(survey_data, grouping_criteria or {})
    print(f"Created {len(user_groups)} user groups")
    
    # Process conversations by group
    grouped_datasets = {}
    
    for group_name, user_ids in user_groups.items():
        user_id_set = set(user_ids)
        
        # Filter conversations for this group
        group_conversations = []
        for conv in conversations_data:
            if conv['user_id'] in user_id_set:
                formatted = format_conversation(conv)
                group_conversations.extend(formatted)
        group_dataset = group_conversations
        
        grouped_datasets[group_name] = group_dataset
    
    return grouped_datasets


def save_grouped_datasets(grouped_datasets: Dict[str, List[Dict]], 
                         output_dir: str = "./prism_processed"):
    """Save processed datasets to JSON files"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for group_name, dataset in grouped_datasets.items():
        output_path = os.path.join(output_dir, f"{group_name}.json")
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved {group_name}: {len(dataset)} examples -> {output_path}")
