# Chronos-2 + CoCo LoRA adapter (SWARM-Lab), same backtest loop as chronos_pred.py
from chronos.chronos2 import Chronos2Model, Chronos2Pipeline
from peft import PeftModel
from chronos_pred import run

base = Chronos2Model.from_pretrained("amazon/chronos-2")
model = PeftModel.from_pretrained(base, "SWARM-Lab/CoCo-Chronos2-LoRA").merge_and_unload().eval()
run(Chronos2Pipeline(model), "chronos_lora")
