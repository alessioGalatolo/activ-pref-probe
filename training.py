"""Training functions for SFT and preference optimization."""

import os
from typing import List, Dict, Tuple
from datasets import Dataset as HFDataset
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from peft.helpers import check_if_peft_model
from trl import DPOTrainer, DPOConfig, KTOTrainer, KTOConfig, CPOTrainer, CPOConfig, SFTTrainer

from config import APOConfig
from dataset_utils import prepare_sft_dataset
from callbacks import FixedIntervalCheckpointCallback


def run_sft(config: APOConfig, model, tokenizer, sft_offset: int = 0):
    """Run supervised fine-tuning on the model.

    Args:
        config: APOConfig object
        model: The model to train
        tokenizer: Tokenizer for the model
        sft_offset: Number of samples to skip at the beginning (for non-overlapping splits)
    """
    # given our save strategy, this is true only if training was completed
    already_done = os.path.exists(f"{config.output_dir}/sft") and bool(os.listdir(f"{config.output_dir}/sft"))
    if already_done:
        # get folder that starts with 'checkpoint-'
        checkpoints = [d for d in os.listdir(f"{config.output_dir}/sft") if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoint = sorted(checkpoints)[-1]
            if check_if_peft_model(f"{config.output_dir}/sft/{checkpoint}/"):
                model = AutoPeftModelForCausalLM.from_pretrained(f"{config.output_dir}/sft/{checkpoint}/", device_map="auto")
            else:
                model = model.from_pretrained(f"{config.output_dir}/sft/{checkpoint}/", device_map="auto")
            return model
    print("\n" + "="*50)
    print("Running Supervised Fine-Tuning")
    print("="*50)
    
    dataset_args = None
    if config.sft_dataset == config.po_dataset and config.po_dataset_args:
        dataset_args = config.po_dataset_args

    dataset = prepare_sft_dataset(
        config.sft_dataset,
        tokenizer,
        config.sft_max_samples,
        dataset_args=dataset_args,
        offset=sft_offset
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=f"{config.output_dir}/sft",
        num_train_epochs=config.sft_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.virtual_batch_size // config.batch_size,
        learning_rate=config.learning_rate,
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
        use_cpu=config.debug,  # Force CPU if in debug mode to avoid OOM issues on GPU
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset,
    )
    resume = os.path.exists(f"{config.output_dir}/sft") and bool(os.listdir(f"{config.output_dir}/sft"))

    trainer.train(resume_from_checkpoint=resume)
    return model


def run_preference_optimization(
    config: APOConfig,
    model,
    tokenizer,
    dataset: List[Dict],
    use_probe_labels: bool = True,
    suffix: str = "",
) -> Tuple[any, List[str]]:
    """Run preference optimization with specified method.

    Returns:
        Tuple of (trained_model, list_of_checkpoint_paths)
    """
    print("\n" + "="*50)
    label_type = "Probe Labels" if use_probe_labels else "Original Labels"
    print(f"Running {config.po_method.upper()} ({label_type})")
    print("="*50)

    formatted_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }

    for item in dataset:
        formatted_data["prompt"].append(item["prompt"])
        formatted_data["chosen"].append(item["chosen"])
        formatted_data["rejected"].append(item["rejected"])

    train_dataset = HFDataset.from_dict(formatted_data)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if hasattr(model, 'merge_and_unload'):
        model = model.merge_and_unload()

    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config, adapter_name=f"po{suffix}")

    output_dir = f"{config.output_dir}/{config.po_method}{suffix}"

    report_to = "wandb" if config.use_wandb else "none"
    run_name = os.environ.get("RUN_NAME", f"{config.po_method}{suffix}")

    checkpoint_callback = None
    if config.enable_checkpoint_eval:
        checkpoint_callback = FixedIntervalCheckpointCallback(
            intervals=config.checkpoint_intervals,
            output_dir=output_dir,
            suffix=suffix
        )

    if config.po_method == "dpo":
        training_args = DPOConfig(
            use_cpu=config.debug,
            output_dir=output_dir,
            num_train_epochs=config.po_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.virtual_batch_size // config.batch_size,
            learning_rate=config.learning_rate,
            label_smoothing=config.dpo_label_smoothing,
            beta=config.beta,
            logging_steps=10,
            save_strategy="steps",
            save_steps=0.05,
            save_total_limit=1,
            remove_unused_columns=False,
            report_to=report_to,
            run_name=run_name,
            max_length=config.max_length,
            max_prompt_length=config.max_length // 2 if config.max_length is not None else None,
        )
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            callbacks=[checkpoint_callback] if checkpoint_callback else [],
        )

    elif config.po_method == "kto":
        kto_data = {"prompt": [], "completion": [], "label": []}
        for item in dataset:
            kto_data["prompt"].append(item["prompt"])
            kto_data["completion"].append(item["chosen"])
            kto_data["label"].append(True)
            kto_data["prompt"].append(item["prompt"])
            kto_data["completion"].append(item["rejected"])
            kto_data["label"].append(False)

        train_dataset = HFDataset.from_dict(kto_data)

        training_args = KTOConfig(
            use_cpu=config.debug,
            output_dir=output_dir,
            num_train_epochs=config.po_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.virtual_batch_size // config.batch_size,
            learning_rate=config.learning_rate,
            beta=config.beta,
            logging_steps=10,
            save_strategy="steps",
            save_steps=0.05,
            save_total_limit=1,
            report_to=report_to,
            run_name=run_name,
            # KTO overrides max_length if is None
            max_length=config.max_length if config.max_length is not None else 1e5,
            max_prompt_length=config.max_length // 2 if config.max_length is not None else 1e5 // 2,
        )

        trainer = KTOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            callbacks=[checkpoint_callback] if checkpoint_callback else [],
        )

    elif config.po_method == "cpo":
        training_args = CPOConfig(
            use_cpu=config.debug,
            output_dir=output_dir,
            num_train_epochs=config.po_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.virtual_batch_size // config.batch_size,
            learning_rate=config.learning_rate,
            beta=config.beta,
            logging_steps=10,
            save_strategy="steps",
            save_steps=0.05,
            save_total_limit=1,
            report_to=report_to,
            run_name=run_name,
            # CPO overrides max_length if is None
            max_length=config.max_length if config.max_length is not None else 1e5,
            max_prompt_length=config.max_length // 2 if config.max_length is not None else 1e5 // 2,
        )

        trainer = CPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            callbacks=[checkpoint_callback] if checkpoint_callback else [],
        )

    elif config.po_method == "ipo":
        training_args = DPOConfig(
            use_cpu=config.debug,
            output_dir=output_dir,
            num_train_epochs=config.po_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.virtual_batch_size // config.batch_size,
            learning_rate=config.learning_rate,
            beta=config.beta,
            loss_type="ipo",
            logging_steps=10,
            save_strategy="steps",
            save_steps=0.05,
            save_total_limit=1,
            remove_unused_columns=False,
            report_to=report_to,
            run_name=run_name,
            max_length=config.max_length,
            max_prompt_length=config.max_length // 2 if config.max_length is not None else None,
        )

        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            callbacks=[checkpoint_callback] if checkpoint_callback else [],
        )

    else:
        raise ValueError(f"Unknown PO method: {config.po_method}")

    resume = os.path.exists(output_dir) and bool(os.listdir(output_dir))
    print(f"Resuming from checkpoint: {resume}")
    trainer.train(resume_from_checkpoint=resume)

    checkpoint_paths = []
    if checkpoint_callback:
        for interval in config.checkpoint_intervals:
            checkpoint_dir = f"{output_dir}/checkpoint_{int(interval*100)}{suffix}"
            if os.path.exists(checkpoint_dir):
                checkpoint_paths.append(checkpoint_dir)

    return model, checkpoint_paths
