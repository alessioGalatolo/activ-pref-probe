"""Training callbacks for APO."""

import os
from typing import List
from transformers import TrainerCallback, TrainerControl, TrainerState


class FixedIntervalCheckpointCallback(TrainerCallback):
    """Callback to save checkpoints at fixed percentage intervals of training."""

    def __init__(self, intervals: List[float], output_dir: str, suffix: str = ""):
        """
        Args:
            intervals: List of percentages (e.g., [0.25, 0.5, 0.75, 1.0])
            output_dir: Base directory for saving checkpoints
            suffix: Suffix to add to checkpoint directory names (e.g., "_probe" or "_original")
        """
        self.intervals = sorted(intervals)
        self.output_dir = output_dir
        self.suffix = suffix
        self.saved_checkpoints = []
        self.checkpoint_steps = []
        self.max_steps = None

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Calculate checkpoint steps based on total training steps."""
        self.max_steps = state.max_steps
        self.checkpoint_steps = [int(self.max_steps * interval) for interval in self.intervals]
        print(f"Will save checkpoints at steps: {self.checkpoint_steps} (intervals: {self.intervals})")
        return control

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Check if we should save a checkpoint at this step."""
        current_step = state.global_step

        for i, checkpoint_step in enumerate(self.checkpoint_steps):
            interval = self.intervals[i]
            if current_step >= checkpoint_step and checkpoint_step not in self.saved_checkpoints:
                checkpoint_dir = f"{self.output_dir}/checkpoint_{int(interval*100)}{self.suffix}"
                print(f"Saving checkpoint at step {current_step} ({interval:.0%}) to {checkpoint_dir}")

                model = kwargs.get("model")
                if model is not None:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)

                    tokenizer = kwargs.get("tokenizer")
                    if tokenizer is not None:
                        tokenizer.save_pretrained(checkpoint_dir)

                self.saved_checkpoints.append(checkpoint_step)
                break

        return control
