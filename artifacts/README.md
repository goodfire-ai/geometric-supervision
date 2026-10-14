# Trained LoRAs

The four LoRAs from the main text, one seed each. The run scripts retrain them from scratch;
these are here so you can use them without training.

| file | what it is |
|---|---|
| `months/task_lora.safetensors` | rank 8 on the layer-3 MLP, trained with cross-entropy (seed 42) |
| `months/geometry_lora.safetensors` | rank 1 on the layer-4 MLP, trained with the geometric loss at layer 18 (seed 42) |
| `linebreak/task_lora.safetensors` | rank 1 on every projection of layers 0-5, trained with cross-entropy (seed 1) |
| `linebreak/geometry_lora.safetensors` | rank 1 on every projection of layers 0-5, trained with the geometric loss at layer 5 (seed 1) |

Each file stores the A and B matrices of every adapted layer and the LoRA scaling. To use one:

```python
from common.lora import attach_lora, load_adapter
from common.model import load_base_model, load_tokenizer

model, tokenizer = load_base_model("cuda"), load_tokenizer()
params, scaling = load_adapter("artifacts/months/geometry_lora.safetensors")
attach_lora(model, params, scaling)   # the model now runs with the LoRA
```
