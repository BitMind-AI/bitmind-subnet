from pathlib import Path
import json


project_root = Path(__file__).parent.parent

with open(project_root / 'datasets.json', 'r') as f:
    DATASET_META = json.load(f)

with open(project_root / 'validator-models.json', 'r') as f:
    VALIDATOR_MODEL_META = json.load(f)

PROMPT_GENERATOR_ARGS = {
    m['model']: m for m in VALIDATOR_MODEL_META['prompt_generators']
}
print(PROMPT_GENERATOR_ARGS)
PROMPT_GENERATOR_NAMES = list(PROMPT_GENERATOR_ARGS.keys())

DIFFUSER_ARGS = {
    m['path']: {k: v for k, v in m.items() if k != 'path'}  
    for m in VALIDATOR_MODEL_META['diffusers']
}
print(DIFFUSER_ARGS)
DIFFUSER_NAMES = list(DIFFUSER_ARGS.keys())

