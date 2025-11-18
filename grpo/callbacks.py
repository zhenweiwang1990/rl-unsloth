"""Training callbacks for GRPO."""

import json
import logging
from pathlib import Path
from typing import Dict

from transformers import TrainerCallback, TrainerState, TrainerControl

logger = logging.getLogger(__name__)


class AccuracyStopCallback(TrainerCallback):
    """Custom callback to stop training when accuracy reaches target and save best model."""
    
    def __init__(
        self,
        target_accuracy: float = 0.95,
        output_dir: str = "outputs/grpo",
        reward_tracker: Dict = None,
    ):
        self.target_accuracy = target_accuracy
        self.output_dir = output_dir
        self.best_accuracy = 0.0
        self.best_model_path = None
        self.reward_tracker = reward_tracker or {}
        
        logger.info(f"🎯 Target accuracy: {self.target_accuracy * 100:.1f}%")
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics is None:
            return control
        
        accuracy = self._calculate_accuracy_from_rewards()
        logger.info(f"📊 Step {state.global_step} - Current accuracy: {accuracy * 100:.2f}%")
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self._save_best_model(state, kwargs.get("model"), kwargs.get("tokenizer"))
            logger.info(f"✨ New best accuracy: {accuracy * 100:.2f}%")
        
        if accuracy >= self.target_accuracy:
            logger.info(f"🎉 Target accuracy {self.target_accuracy * 100:.1f}% reached!")
            logger.info(f"🏆 Final accuracy: {accuracy * 100:.2f}%")
            control.should_training_stop = True
        
        return control
    
    def _calculate_accuracy_from_rewards(self) -> float:
        """Calculate accuracy from tracked rewards."""
        if not self.reward_tracker.get("eval_rewards"):
            return 0.0
        
        rewards = self.reward_tracker["eval_rewards"]
        if len(rewards) == 0:
            return 0.0
        
        # Consider reward > 0.8 as correct answer
        correct_count = sum(1 for r in rewards if r > 0.8)
        accuracy = correct_count / len(rewards)
        
        return accuracy
    
    def _save_best_model(self, state: TrainerState, model, tokenizer):
        """Save the best model checkpoint."""
        if model is None:
            return
        
        best_model_dir = Path(self.output_dir) / "best_model"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(str(best_model_dir))
        if tokenizer:
            tokenizer.save_pretrained(str(best_model_dir))
        
        metadata = {
            "step": state.global_step,
            "accuracy": float(self.best_accuracy),
            "epoch": state.epoch,
        }
        
        with open(best_model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.best_model_path = str(best_model_dir)
        logger.info(f"💾 Best model saved to: {best_model_dir}")
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training."""
        logger.info("="*60)
        logger.info("Training Summary")
        logger.info("="*60)
        logger.info(f"Best accuracy achieved: {self.best_accuracy * 100:.2f}%")
        logger.info(f"Target accuracy: {self.target_accuracy * 100:.1f}%")
        
        if self.best_model_path:
            logger.info(f"Best model saved at: {self.best_model_path}")
        
        if self.best_accuracy >= self.target_accuracy:
            logger.info("✅ Target accuracy reached!")
        else:
            logger.info("⚠️  Target accuracy not reached")
        
        logger.info("="*60)
        
        return control

