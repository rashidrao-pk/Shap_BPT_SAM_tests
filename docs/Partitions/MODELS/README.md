```bash
source /beegfs/home/mrashid/pt_312_sam/bin/activate

python - <<'PY'
import importlib.util

for name in ["transformers", "torch", "torchvision", "PIL", "huggingface_hub"]:
    print(name, bool(importlib.util.find_spec(name)))
PY
```
