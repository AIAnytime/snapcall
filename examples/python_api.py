"""End-to-end with the Python API: Faker data -> train -> decide."""
import json
from pathlib import Path

from snapcall import DataConfig, DecisionModel, TrainConfig, Trainer
from snapcall.data import faker_gen

spec = json.loads((Path(__file__).parent / "support_routing_faker.json").read_text())
faker_gen.generate_dataset("support-routing", spec, n_per_option=80)

cfg = TrainConfig(base_model="answerdotai/ModernBERT-base", data=DataConfig(dataset="support-routing"),
                  epochs=2, batch_size=16, max_length=256, rlcd_epochs=0.5)
Trainer(cfg, run_dir="runs/support-routing").train()

model = DecisionModel.from_pretrained("runs/support-routing/model")
print(model.decide("My earbuds never showed up, order #A55120", {"team": model.meta["question"]}))
