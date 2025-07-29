

# ⚠️⚠️⚠️THIS AI'S LANGUAGE HAS NOT BEEN AUDITED SO IT MIGHT GAVE YOU A GOOFY RESPONSE OR NONSENSE


# 🧠 GenZ Bot Fine-Tuning Script

Welcome to the **GenZ Bot** fine-tuner — a glorious Python script that unleashes fine-tuning fury on **GPT-Neo 125M** using the Wikitext-2 dataset.

This repo contains:
- A script to fine-tune GPT-Neo 125M (`genz-bot.py`)
- All configs and setup required
- Instructions to run and customize your training

---

## 🔧 Requirements

- **Python**: 3.9 or 3.10
- **GPU**: Required — recommended at least **8GB VRAM**  
  > Training on CPU will make you age in dog years.

---

## 🧪 Tested Environment

| Component     | Version       |
|---------------|---------------|
| Python        | 3.10.x        |
| CUDA          | 12.8+         |
| PyTorch       | 2.3.0+ (with CUDA 12.8 support) |
| Transformers  | 4.41.1        |
| Datasets      | 2.19.0        |
| GPU Used      | RTX 3050 (4GB VRAM) ✅ *(but slow)* |

---

## 📦 Installation

Create a virtual environment and install the goods:

```bash
# Optional: Create venv
python -m venv .venv
source .venv/Scripts/activate  # On Windows
# or
source .venv/bin/activate      # On Linux/macOS

# Install dependencies (with CUDA 12.8 support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets pandas
```

---

## 🚨 Warnings

- This script will download ~500MB of Wikitext-2 and ~500MB of GPT-Neo.
- Trained model output will be **~4.93GB** (for GPT-Neo 125M after training).
- Don't upload the `neo-125m-finetuned/` directory to GitHub — it's **thicc**.
- On low-VRAM GPUs (like 4GB), training is **slow AF** (expect 3–4 hours).
- Set `per_device_train_batch_size=1` unless your GPU is a war machine (A100, 3090, etc.)

---

## 🚀 Training the Model

Run this script to start fine-tuning:

```bash
python genz-bot.py
```

By default, it uses:
- `EleutherAI/gpt-neo-125m`
- Dataset: `"wikitext"`, `"wikitext-2-raw-v1"`
- Tokenization: `max_length=128`, with `eos_token` padding
- Epochs: `3`
- Save: `./neo-125m-finetuned`

Modify inside `genz-bot.py` if you want custom model or dataset.

---

## 📁 Output

After training, you'll get:
```
models_learn/
├── neo-125m-finetuned/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── special_tokens_map.json
```

You can load it back later like:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./neo-125m-finetuned")
tokenizer = AutoTokenizer.from_pretrained("./neo-125m-finetuned")
```

Or use it in flask to receive and send HTTP data

```python
import requests
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./neo-125m-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")
    if not user_input:
        return jsonify({"error": "Message is required"}), 400
    
    inputs = tokenizer.encode(user_input, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=256, do_sample=True, temperature=0.9, top_k=50)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({"response": response})
```

---

## 📡 Future Ideas

- Hook this into a Flask API (`flask_server.py`)
- Upload model to [HuggingFace Hub](https://huggingface.co/)
- Fine-tune with your own chat data
- Upgrade to LoRA if you're using Colab or want lightweight finetuning

---

## 🧠 Author

Built by **The BatShitBananaDotNet**, who probably microwaved a GPU while writing this.

---
