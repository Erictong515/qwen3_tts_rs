# TTS Inference Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `Qwen3TTSModel` in `model.rs` to the already-implemented `TTSInference` engine so `generate_custom_voice()` produces real audio instead of silence.

**Architecture:** `Qwen3TTSModel::from_pretrained()` creates a `TTSInference` internally and stores it. `generate_custom_voice()` delegates to `TTSInference::generate()`, wrapping the `(Vec<f32>, u32)` result into the existing `GenerationOutput` type. No changes needed in `voice-chat/`.

**Tech Stack:** Rust, qwen3_tts crate (MLX feature), existing `TTSInference` in `src/inference.rs`

---

## Chunk 1: Wire TTSInference into Qwen3TTSModel

### Task 1: Add TTSInference field to Qwen3TTSModel

**Files:**
- Modify: `/Users/I075399/qwen3_tts_rs/src/model.rs`

- [ ] **Step 1: Add inference import and field to Qwen3TTSModel**

In `model.rs`, add the import and update the struct. Find this section:

```rust
use crate::audio::{load_audio, AudioInput};
use crate::config::{GenerationConfig, Qwen3TTSConfig, TTSModelType, TokenizerType};
```

Add after existing imports:
```rust
use crate::inference::TTSInference;
```

Then add a field to `Qwen3TTSModel` struct (after `supported_languages` field):
```rust
    /// Inference engine (None if model files not fully loaded)
    inference: Option<TTSInference>,
```

- [ ] **Step 2: Initialize TTSInference in from_pretrained_with_device()**

In `from_pretrained_with_device()`, after the existing `let model_path = ...` block, add inference engine init just before the `Ok(Self { ... })` block:

```rust
        // Try to load full inference engine (requires model.safetensors + tokenizer)
        let inference = {
            let weights_path = Path::new(&model_path).join("model.safetensors");
            let tokenizer_path = Path::new(&model_path).join("tokenizer.json");
            if weights_path.exists() && tokenizer_path.exists() {
                match TTSInference::new(Path::new(&model_path), device) {
                    Ok(engine) => {
                        tracing::info!("TTSInference engine loaded");
                        Some(engine)
                    }
                    Err(e) => {
                        tracing::warn!("TTSInference load failed, using placeholder: {e}");
                        None
                    }
                }
            } else {
                tracing::warn!("model.safetensors or tokenizer.json missing, using placeholder TTS");
                None
            }
        };
```

Then add `inference,` to the `Ok(Self { ... })` initializer.

- [ ] **Step 3: Build and check for compile errors**

```bash
cd /Users/I075399/qwen3_tts_rs && cargo build --features mlx 2>&1 | grep -E "^error"
```

Expected: no `error` lines (warnings OK).

- [ ] **Step 4: Commit**

```bash
cd /Users/I075399/qwen3_tts_rs
git add src/model.rs
git commit -m "feat: add TTSInference field to Qwen3TTSModel"
```

---

### Task 2: Delegate generate_custom_voice() to TTSInference

**Files:**
- Modify: `/Users/I075399/qwen3_tts_rs/src/model.rs`

- [ ] **Step 1: Replace placeholder in generate_custom_voice()**

Find the placeholder block in `generate_custom_voice()` (around line 382):

```rust
        // Placeholder: Actual generation would happen here
        // For now, return a silent waveform
        let sample_rate = 24000;
        let duration_samples = sample_rate * 2; // 2 seconds of silence
        let waveform = vec![0.0f32; duration_samples as usize];

        tracing::info!(
            "Generated custom voice: text='{}', speaker='{}', language='{}'",
            texts[0],
            speakers[0].name(),
            languages[0].as_str()
        );

        Ok(GenerationOutput {
            waveforms: vec![waveform],
            sample_rate,
        })
```

Replace with:

```rust
        tracing::info!(
            "Generating custom voice: text='{}', speaker='{}', language='{}'",
            texts[0],
            speakers[0].name(),
            languages[0].as_str()
        );

        let (waveform, sample_rate) = if let Some(ref engine) = self.inference {
            engine.generate(&texts[0], speakers[0].name(), languages[0].as_str())?
        } else {
            tracing::warn!("TTSInference not loaded, returning silence");
            let sr = 24000u32;
            (vec![0.0f32; (sr * 2) as usize], sr)
        };

        Ok(GenerationOutput {
            waveforms: vec![waveform],
            sample_rate,
        })
```

- [ ] **Step 2: Build and check for compile errors**

```bash
cd /Users/I075399/qwen3_tts_rs && cargo build --features mlx 2>&1 | grep -E "^error"
```

Expected: no `error` lines.

- [ ] **Step 3: Run existing tests**

```bash
cd /Users/I075399/qwen3_tts_rs && cargo test --features mlx 2>&1 | tail -20
```

Expected: all tests pass (no model files needed — unit tests don't call generate).

- [ ] **Step 4: Commit**

```bash
cd /Users/I075399/qwen3_tts_rs
git add src/model.rs
git commit -m "feat: wire generate_custom_voice() to TTSInference engine"
```

---

### Task 3: Build voice-chat and verify end-to-end compile

**Files:**
- Read: `/Users/I075399/cc_projects/voice-chat/Cargo.toml` (verify dep points to local path)

- [ ] **Step 1: Build voice-chat with updated qwen3_tts**

```bash
cd /Users/I075399/cc_projects/voice-chat && cargo build 2>&1 | grep -E "^error"
```

Expected: no `error` lines (voice-chat `tts.rs` calls `model.generate_custom_voice()` — API unchanged).

- [ ] **Step 2: Confirm no API breakage**

The `voice-chat/src/pipeline/tts.rs` calls:
```rust
model.generate_custom_voice(&text, Speaker::new(&speaker), Language::from(language.as_str()), None, None)
```
This signature is unchanged. No edit needed.

- [ ] **Step 3: Commit voice-chat if any Cargo.lock changed**

```bash
cd /Users/I075399/cc_projects/voice-chat
git diff --stat
# Only commit if Cargo.lock changed
git add Cargo.lock
git commit -m "chore: update Cargo.lock after qwen3_tts wiring"
```

---

## Chunk 2: Smoke test with real model files (manual, optional)

> This chunk requires the model to be downloaded. Skip if model files are not available.

### Task 4: Download model and run smoke test

- [ ] **Step 1: Check if model exists**

```bash
ls /Users/I075399/cc_projects/voice-chat/models/
```

Expected: `ggml-base.bin` for Whisper (separate), and a TTS model directory.

- [ ] **Step 2: Run TTS binary smoke test**

```bash
cd /Users/I075399/qwen3_tts_rs
cargo run --features mlx --bin tts -- \
  --model-path ../qwen3_tts_rs/models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "Hello, this is a test." \
  --speaker Vivian \
  --language english \
  --output /tmp/test_output.wav 2>&1 | tail -20
```

Expected: `Generated N samples (X.XX seconds)`, output file created.

- [ ] **Step 3: Verify audio file**

```bash
file /tmp/test_output.wav
# Should show: RIFF (little-endian) data, WAVE audio, ...
```

---
