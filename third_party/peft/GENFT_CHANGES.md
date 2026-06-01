# GenFT PEFT Notes

This vendored PEFT copy uses `NLU/peft` as the base.

Only GenFT-related files were checked against `alpaca-lora/peft`:

- `src/peft/tuners/genft/config.py`: identical.
- `src/peft/tuners/genft/__init__.py`: identical.
- `src/peft/tuners/genft/layer.py`: kept the NLU parameter shapes and added LLaMA/device-map-safe tensor placement.
- `src/peft/tuners/genft/model.py`: kept the NLU adapter update path and added LLaMA/device-map-safe tensor placement.

Unrelated PEFT code from `alpaca-lora/peft` was not merged.

