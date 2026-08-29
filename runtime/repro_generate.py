"""Minimal reproduction of the verl HF-rollout generate call for Qwen3.5-9B."""
import os, sys, time, threading
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HF_HOME"] = "/data/lanyuqi/skillrise/runtime/cache/huggingface"
os.environ["HF_HUB_CACHE"] = os.environ["HF_HOME"] + "/hub"
os.environ["TMPDIR"] = "/data/lanyuqi/skillrise/runtime/cache/tmp"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL = "/data/lanyuqi/skillrise/runtime/models/Qwen3.5-9B"

def watchdog():
    time.sleep(90)
    print("WATCHDOG: 90s elapsed, still generating -> hang reproduced", flush=True)
    os._exit(2)

threading.Thread(target=watchdog, daemon=True).start()

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
print("tokenizer ok", flush=True)
print("pad:", tok.pad_token_id, "eos:", tok.eos_token_id, flush=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
).to("cuda")
model.eval()
print("model loaded", flush=True)

# two text prompts, left padded like verl
msgs1 = [{"role": "user", "content": "Look around the room and describe what you find."}]
msgs2 = [{"role": "user", "content": "Mix some chemicals and observe the reaction."}]
t1 = tok.apply_chat_template(msgs1, add_generation_prompt=True, tokenize=True, return_tensors="pt").input_ids
t2 = tok.apply_chat_template(msgs2, add_generation_prompt=True, tokenize=True, return_tensors="pt").input_ids
maxlen = max(t1.size(1), t2.size(1))
input_ids = torch.cat([
    torch.cat([torch.full((1, maxlen - t1.size(1)), tok.pad_token_id, dtype=torch.long), t1], dim=1),
    torch.cat([torch.full((1, maxlen - t2.size(1)), tok.pad_token_id, dtype=torch.long), t2], dim=1),
], dim=0).to("cuda")
attention_mask = (input_ids != tok.pad_token_id).long().to("cuda")
position_ids = attention_mask.cumsum(-1) - 1
position_ids[attention_mask == 0] = 0
print("input_ids", input_ids.shape, flush=True)

gen_cfg = GenerationConfig(do_sample=True, num_beams=1, top_p=1.0, top_k=0, temperature=0.7, num_return_sequences=1)

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        do_sample=True,
        max_new_tokens=64,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        generation_config=gen_cfg,
        output_scores=False,
        return_dict_in_generate=True,
        use_cache=True,
    )
print("generate DONE, seq:", out.sequences.shape, flush=True)
print("decoded:", tok.batch_decode(out.sequences, skip_special_tokens=True), flush=True)
