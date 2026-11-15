import os
import shutil
import tempfile
import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import NirnayConfig
from src.model import NirnayDecisionModel
from src.trainer import NirnayTrainer

class TestCheckpointingAndResume(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = NirnayConfig(
            output_dir=self.temp_dir,
            save_total_limit=2,
            save_every_n_steps=2,
        )

        # Mock model and toy dataloader
        self.model = NirnayDecisionModel.from_pretrained(self.config, is_training=True)
        
        # Toy dummy data
        dummy_inputs = torch.randint(0, 100, (8, 16))
        dummy_masks = torch.ones((8, 16), dtype=torch.long)
        dummy_targets = torch.tensor([[0.5, 0.5, 0, 0, 0, 0]] * 8, dtype=torch.float32)
        dummy_active = torch.tensor([[True, True, False, False, False, False]] * 8, dtype=torch.bool)

        dataset = [
            {
                "input_ids": dummy_inputs[i],
                "attention_mask": dummy_masks[i],
                "targets": dummy_targets[i],
                "active_mask": dummy_active[i],
                "kinds": "noul",
            }
            for i in range(8)
        ]

        def collate_fn(batch):
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
                "targets": torch.stack([b["targets"] for b in batch]),
                "active_mask": torch.stack([b["active_mask"] for b in batch]),
                "kinds": [b["kinds"] for b in batch],
            }

        self.dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        self.trainer = NirnayTrainer(
            model=self.model,
            config=self.config,
            train_dataloader=self.dataloader,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_state(self):
        # Advance step and modify state
        self.trainer.global_step = 42
        self.trainer.current_epoch = 1

        step_dir = os.path.join(self.temp_dir, "checkpoint-step-42")
        self.trainer.save_checkpoint(step_dir)

        # Verify files exist in step checkpoint
        self.assertTrue(os.path.exists(os.path.join(step_dir, "trainer_state.pt")))
        self.assertTrue(os.path.exists(os.path.join(step_dir, "adapter_config.json")))

        # Verify checkpoint-latest exists and has identical state
        latest_dir = os.path.join(self.temp_dir, "checkpoint-latest")
        self.assertTrue(os.path.exists(latest_dir))
        self.assertTrue(os.path.exists(os.path.join(latest_dir, "trainer_state.pt")))

        # Create fresh trainer and restore
        new_trainer = NirnayTrainer(
            model=self.model,
            config=self.config,
            train_dataloader=self.dataloader,
        )
        new_trainer.load_checkpoint(latest_dir)

        self.assertEqual(new_trainer.global_step, 42)
        self.assertEqual(new_trainer.current_epoch, 1)

    def test_checkpoint_pruning(self):
        # Save checkpoints 1, 2, 3 with limit 2
        for s in [100, 200, 300]:
            self.trainer.global_step = s
            d = os.path.join(self.temp_dir, f"checkpoint-step-{s}")
            self.trainer.save_checkpoint(d)

        # Oldest step 100 should be pruned; 200 and 300 should remain
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir, "checkpoint-step-100")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "checkpoint-step-200")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "checkpoint-step-300")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "checkpoint-latest")))

if __name__ == "__main__":
    unittest.main()
