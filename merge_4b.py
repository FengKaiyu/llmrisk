#!/usr/bin/env python3
"""4B adapter 本地合并：adapter 传输格式 → 合并部署格式（与 v1.0 验证配置等价）。
用法：
  python3 merge_4b.py --base /models/Qwen3-4B --adapter /models/4b-v3-adapter \
      --out /models/4b-finetune-v3
合并后：ADAPTER_4B 不挂，MODEL_4B 指向 --out 目录即可。
"""
import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                             device_map="cpu", trust_remote_code=True)
model = PeftModel.from_pretrained(model, args.adapter)
model = model.merge_and_unload()
model.save_pretrained(args.out)
tok = AutoTokenizer.from_pretrained(args.base)
tok.save_pretrained(args.out)
print(f"[done] 合并完成 -> {args.out}")
