# Fun-CosyVoice3-0.5B for zero-shot voice cloning TTS

## Summary

- Vendor: FunAudioLLM
- Model: `FunAudioLLM/Fun-CosyVoice3-0.5B`
- Task: Text-to-speech with zero-shot voice cloning from reference audio
- Mode: Offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to run CosyVoice3 zero-shot voice cloning on a
consumer GPU (RTX 5080 16 GB). The two-stage pipeline (talker → code2wav)
fits within 16 GB by allocating 40% to Stage 0 and 20% to Stage 1, leaving
headroom for the OS and speaker cache.

CosyVoice3 supports both Chinese and English synthesis. The reference audio
controls the cloned voice; the text to synthesise can be in either language
regardless of what language the reference audio is in.

## References

- Related example: [`examples/offline_inference/text_to_speech/cosyvoice3/end2end.py`](../../examples/offline_inference/text_to_speech/cosyvoice3/end2end.py)
- Deploy config: [`vllm_omni/deploy/cosyvoice3.yaml`](../../vllm_omni/deploy/cosyvoice3.yaml)
- Pipeline definition: [`vllm_omni/model_executor/models/cosyvoice3/pipeline.py`](../../vllm_omni/model_executor/models/cosyvoice3/pipeline.py)

## Hardware Support

## GPU

### 1x RTX 5080 16GB

#### Environment

- OS: Ubuntu 24.04.3 LTS (WSL2 on Windows)
- Python: 3.12.3
- CUDA: 13.0 (driver 581.95)
- vLLM version: 0.20.0
- vLLM-Omni commit: 77480215

#### Command

**Offline inference (no server needed):**

```bash
# Run from the repository root with the project venv activated
.venv/bin/python3 examples/offline_inference/text_to_speech/cosyvoice3/end2end.py \
    --model pretrained_models/Fun-CosyVoice3-0.5B \
    --tokenizer pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
    --text "你好，这是一个测试语音合成的句子。" \
    --prompt-text "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。" \
    --ref-audio tests/assets/cosyvoice3/zero_shot_prompt.wav
```

English synthesis works equally well:

```bash
.venv/bin/python3 examples/offline_inference/text_to_speech/cosyvoice3/end2end.py \
    --model pretrained_models/Fun-CosyVoice3-0.5B \
    --tokenizer pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
    --text "CosyVoice is undergoing a comprehensive upgrade, providing more accurate, stable, faster, and better voice generation capabilities." \
    --prompt-text "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。" \
    --ref-audio tests/assets/cosyvoice3/zero_shot_prompt.wav
```

To use your own reference audio, replace `--ref-audio` with any WAV file
sampled at ≥ 16 kHz and update `--prompt-text` with the exact transcript of
what is spoken in that file, keeping the
`"You are a helpful assistant.<|endofprompt|>"` prefix.

#### Verification

The script saves the generated audio to `output_0.wav` in the current
directory. A successful run ends with:

```
Generated Audio Shape: torch.Size([166080])
Saved audio to output_0.wav
```

166 080 samples at 22 050 Hz ≈ 7.5 seconds of audio.

#### Notes

- Memory usage: Stage 0 (talker) ~1.92 GiB, Stage 1 (code2wav) ~1.59 GiB.
  Both stages share GPU 0 via the deploy config (`gpu_memory_utilization: 0.4`
  for Stage 0, `0.2` for Stage 1).
- Key flags: `--model` and `--tokenizer` must point to separate paths.
  The tokenizer lives inside the model directory at `CosyVoice-BlankEN/`.
- Async chunking: Enabled by default in `cosyvoice3.yaml`. Stage 0 streams
  codec chunks to Stage 1 via `SharedMemoryConnector` for lower latency.
- ONNX Runtime: The speech tokenizer (ONNX) falls back to CPU on this setup
  because `CUDAExecutionProvider` is unavailable. This does not affect output
  quality but adds a small preprocessing overhead.
- WSL2 note: `pin_memory` is automatically disabled on WSL2 by vLLM, which
  may slightly reduce data-transfer throughput.
- Known limitations: `enforce_eager=true` is set for both stages in the deploy
  config (CUDA graphs are not verified for this checkpoint). Online serving via
  `vllm serve` is not tested on this hardware configuration.
