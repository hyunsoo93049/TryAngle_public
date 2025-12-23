#!/usr/bin/env python3
"""
Quick test to check if models load properly
"""

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import pipeline
import time

print("=" * 60)
print("Model Loading Test")
print("=" * 60)

# Test 1: Grounding DINO
print("\n1. Testing Grounding DINO...")
start = time.time()
try:
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
    print(f"   [OK] Grounding DINO loaded in {time.time() - start:.2f}s")
except Exception as e:
    print(f"   [FAIL] Grounding DINO failed: {e}")

# Test 2: Depth Anything V2
print("\n2. Testing Depth Anything V2...")
start = time.time()
try:
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"   [OK] Depth Anything loaded in {time.time() - start:.2f}s")
except Exception as e:
    print(f"   [FAIL] Depth Anything failed: {e}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)