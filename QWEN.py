import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import gc
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

MODELS_CONFIG = [
    {
        "name": "Qwen 1.5B",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "sae_release": "DGurgurov/DeepSeek-R1-Distill-Qwen-1.5B-sae",
        "sae_ids": [
            "blocks.1.hook_resid_pre"
        ]
    },
    {
        "name": "Qwen 7B",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "sae_release": "rufimelo/secure_code_qwen_coder_strd_16384",
        "sae_ids": [
            "blocks.0.hook_resid_post",
            "blocks.14.hook_resid_post"
        ]
    }
]

LANGUAGES = ["english", "hindi", "spanish"]
N_SAMPLES = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading dataset...")
data = []
for lang in LANGUAGES:
    ds = load_dataset("cardiffnlp/tweet_sentiment_multilingual", lang, trust_remote_code=True)
    samples = ds["test"].select(range(N_SAMPLES))

    for s in samples:
        data.append({
            "text": s["text"],
            "label": s["label"],
            "lang": lang
        })
print(f"Total samples: {len(data)}")

activation = None

def hook_fn(module, input, output):
    global activation
    if isinstance(output, tuple):
        activation = output[0]
    else:
        activation = output

def get_features(text, sae, current_model, current_tokenizer):
    global activation
    inputs = current_tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        _ = current_model(**inputs)

    dense = activation
    with torch.no_grad():
        sparse = sae.encode(dense)

    return sparse.mean(dim=1).squeeze().cpu()

for config in MODELS_CONFIG:
    print(f"\nStarting pipeline for: {config['name']}")

    print(f"Loading {config['model_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config['model_id'],
        torch_dtype=torch.float16,
        device_map="auto"
    )

    def predict(text):
        prompt = f"Classify sentiment as Positive, Neutral, or Negative.\n\nTweet: {text}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)

        resp = tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).lower()

        if "positive" in resp: return 2
        if "neutral" in resp: return 1
        if "negative" in resp: return 0
        return -1

    print("Running zero-shot evaluation...")
    lang_acc = defaultdict(lambda: [0, 0])

    for d in data:
        p = predict(d["text"])
        lang_acc[d["lang"]][1] += 1
        if p == d["label"]:
            lang_acc[d["lang"]][0] += 1

    print(f"Baseline accuracy ({config['name']}):")
    for lang, (c, t) in lang_acc.items():
        print(f"{lang}: {c/t:.2%}")

    for sae_id in config['sae_ids']:
        layer = int(sae_id.split('.')[1])

        print(f"\nAnalyzing layer {layer} ({sae_id})")

        try:
            sae = SAE.from_pretrained(
                release=config['sae_release'],
                sae_id=sae_id,
                device=DEVICE
            )

            # Simplified hook registration since both models are now Qwen architecture
            layer_module = model.model.layers[layer]

            hook = layer_module.register_forward_hook(hook_fn)

            feat_by_sent = defaultdict(list)
            feat_by_lang = defaultdict(list)

            for d in data:
                feat = get_features(d["text"], sae, model, tokenizer)
                feat_by_sent[d["label"]].append(feat)
                feat_by_lang[d["lang"]].append(feat)

            avg_sent = {k: torch.stack(v).mean(0) for k, v in feat_by_sent.items()}
            avg_lang = {k: torch.stack(v).mean(0) for k, v in feat_by_lang.items()}

            print("Sentiment features:")
            for k, v in avg_sent.items():
                print(f"Sentiment {k}: {(v > 0).sum().item()} active neurons")

            print("Language similarity (cosine):")
            langs = list(avg_lang.keys())
            for i in range(len(langs)):
                for j in range(i+1, len(langs)):
                    sim = torch.cosine_similarity(avg_lang[langs[i]], avg_lang[langs[j]], dim=0)
                    print(f"{langs[i]} vs {langs[j]}: {sim.item():.4f}")

            hook.remove()
            del sae

        except Exception as e:
            print(f"Could not process {sae_id}. Skipping...")
            print(f"Error detail: {e}")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"Cleaning up {config['name']} from memory...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

print("\nAll pipelines complete.")