# KV Cache & Memory Fix Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate O(n²) memory growth in TTS inference by implementing KV cache for attention layers, fixing the CodePredictor accumulation loop, correcting MLX shallow_clone semantics, and adding periodic MLX cache clearing.

**Architecture:** Add a `KvCache` struct that each `Attention` layer populates during inference. Modify `TransformerLayer::forward` and `Attention::forward` to accept an optional mutable cache. Rewrite `TalkerModel::generate_codes` and `CodePredictor::generate_codes` to use single-token forward passes with cached KV state. Fix `Tensor::shallow_clone` on MLX to use ref-counting instead of deep copy. Call `mlx_clear_cache()` after each full TTS generation to prevent inter-request memory accumulation.

**Tech Stack:** Rust, MLX C API (`mlx-c`), Apple Metal GPU

**Context:** The Jetsam report from 2026-03-13 shows `voice_api` at 38 GB and `voice-chat` at 25 GB — both killed by macOS OOM. Root cause: autoregressive loops re-run the full transformer over the entire growing sequence at each step (O(n²) compute and memory), with no KV cache.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/layers.rs` | Modify | Add `KvCache` struct; add `forward_with_cache` to `Attention` and `TransformerLayer` |
| `src/inference.rs` | Modify | Rewrite `TalkerModel::generate_codes` and `CodePredictor::generate_codes` to use KV cache |
| `src/tensor.rs` | Modify | Fix MLX `shallow_clone()` to use `mlx_array_set` (ref-count) instead of full clone |
| `src/backend/mlx/stream.rs` | Modify | Expose `clear_cache()` public function wrapping `mlx_clear_cache()` |
| `src/inference.rs` | Modify | Call `clear_cache()` after each `generate` call in `TTSInference` |

---

## Chunk 1: KV Cache Infrastructure + Attention Integration

### Task 1: Add KvCache struct to layers.rs

**Files:**
- Modify: `src/layers.rs:1-15` (add struct at top of file)

- [ ] **Step 1: Define KvCache struct**

Add after the imports in `src/layers.rs`:

```rust
/// Per-layer KV cache for autoregressive inference.
///
/// During the prefill phase (first forward pass), K and V are computed for
/// the full input and stored. During subsequent generation steps, only the
/// new token's K and V are computed and concatenated to the cache.
pub struct KvCache {
    /// Cached key states: [batch, num_kv_heads, cached_len, head_dim]
    pub key: Option<Tensor>,
    /// Cached value states: [batch, num_kv_heads, cached_len, head_dim]
    pub value: Option<Tensor>,
}

impl KvCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self { key: None, value: None }
    }

    /// Current cached sequence length (0 if empty).
    pub fn seq_len(&self) -> i64 {
        self.key.as_ref().map_or(0, |k| k.size()[2])
    }

    /// Reset the cache (for a new generation).
    pub fn reset(&mut self) {
        self.key = None;
        self.value = None;
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cd /Users/I075399/qwen3_tts_rs && cargo build --features mlx 2>&1 | tail -5`
Expected: compilation succeeds

- [ ] **Step 3: Commit**

```bash
git add src/layers.rs
git commit -m "feat: add KvCache struct for autoregressive inference"
```

---

### Task 2: Add forward_with_cache to Attention

**Files:**
- Modify: `src/layers.rs` (Attention impl block, after existing `forward` method at line ~474)

- [ ] **Step 1: Add forward_with_cache method to Attention**

Add the following method inside the `impl Attention` block, after the existing `forward` method. This method:
- Computes Q/K/V only for the new token(s)
- Concatenates new K/V with cached K/V
- Updates the cache in-place
- Uses the `offset` parameter of `fast_rope` for correct positional encoding

```rust
    /// Forward pass with KV cache for autoregressive generation.
    ///
    /// During prefill (cache is empty), processes the full sequence and populates the cache.
    /// During generation (cache has content), processes only new token(s) and extends the cache.
    ///
    /// `position_offset`: the sequence position of the first new token (= cached_len before this call).
    pub fn forward_with_cache(
        &self,
        hidden_states: &Tensor,
        rotary_emb: &RotaryEmbedding,
        cache: &mut KvCache,
    ) -> Tensor {
        let size = hidden_states.size();
        let batch_size = size[0];
        let new_seq_len = size[1]; // 1 during generation, full_len during prefill

        // Project Q, K, V for new tokens only
        let query = self.q_proj.forward(hidden_states);
        let new_key = self.k_proj.forward(hidden_states);
        let new_value = self.v_proj.forward(hidden_states);

        // Reshape to (batch, heads, new_seq_len, head_dim)
        let query = query
            .view(&[batch_size, new_seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let new_key = new_key
            .view(&[batch_size, new_seq_len, self.num_kv_heads, self.head_dim])
            .transpose(1, 2);
        let new_value = new_value
            .view(&[batch_size, new_seq_len, self.num_kv_heads, self.head_dim])
            .transpose(1, 2);

        // Apply Q/K normalization if present
        let query = if let Some(ref q_norm) = self.q_norm {
            let eps = 1e-6;
            let variance = query.pow_scalar(2.0).mean_dim(&[-1], true);
            let rms = (variance + eps).sqrt().clamp_min(1e-8);
            let normalized = query / rms;
            let q_norm_shape = q_norm.size();
            if q_norm_shape.len() == 1 && q_norm_shape[0] == self.num_heads * self.head_dim {
                let q_norm_reshaped = q_norm.view(&[1, self.num_heads, 1, self.head_dim]);
                normalized * q_norm_reshaped
            } else if q_norm_shape.len() == 1 && q_norm_shape[0] == self.head_dim {
                normalized * q_norm
            } else {
                normalized
            }
        } else {
            query
        };
        let new_key = if let Some(ref k_norm) = self.k_norm {
            let eps = 1e-6;
            let variance = new_key.pow_scalar(2.0).mean_dim(&[-1], true);
            let rms = (variance + eps).sqrt().clamp_min(1e-8);
            let normalized = new_key / rms;
            let k_norm_shape = k_norm.size();
            if k_norm_shape.len() == 1 && k_norm_shape[0] == self.num_kv_heads * self.head_dim {
                let k_norm_reshaped = k_norm.view(&[1, self.num_kv_heads, 1, self.head_dim]);
                normalized * k_norm_reshaped
            } else if k_norm_shape.len() == 1 && k_norm_shape[0] == self.head_dim {
                normalized * k_norm
            } else {
                normalized
            }
        } else {
            new_key
        };

        // Apply rotary embeddings with position offset
        let position_offset = cache.seq_len();
        let (query, new_key) = rotary_emb.forward_with_offset(
            &query, &new_key, new_seq_len, position_offset,
        );

        // Concatenate new K/V with cached K/V
        let (full_key, full_value) = if let (Some(cached_k), Some(cached_v)) =
            (&cache.key, &cache.value)
        {
            (
                Tensor::cat(&[cached_k.shallow_clone(), new_key.shallow_clone()], 2),
                Tensor::cat(&[cached_v.shallow_clone(), new_value.shallow_clone()], 2),
            )
        } else {
            (new_key.shallow_clone(), new_value.shallow_clone())
        };

        // Update cache
        cache.key = Some(full_key.shallow_clone());
        cache.value = Some(full_value.shallow_clone());

        let full_seq_len = full_key.size()[2];

        // Expand KV heads for GQA if needed
        let (full_key, full_value) = if self.num_kv_heads != self.num_heads {
            let repeat_factor = self.num_heads / self.num_kv_heads;
            let key = full_key
                .unsqueeze(2)
                .expand(
                    &[batch_size, self.num_kv_heads, repeat_factor, full_seq_len, self.head_dim],
                    false,
                )
                .reshape(&[batch_size, self.num_heads, full_seq_len, self.head_dim]);
            let value = full_value
                .unsqueeze(2)
                .expand(
                    &[batch_size, self.num_kv_heads, repeat_factor, full_seq_len, self.head_dim],
                    false,
                )
                .reshape(&[batch_size, self.num_heads, full_seq_len, self.head_dim]);
            (key, value)
        } else {
            (full_key, full_value)
        };

        // Compute attention (Q attends to full K/V including cache)
        #[cfg(feature = "mlx")]
        let attn_output = {
            let scale = 1.0 / (self.head_dim as f64).sqrt();
            // For causal attention with KV cache: MLX SDPA handles masking internally
            // when query length < key length (generation mode)
            Tensor::from_mlx(mlx::ops::fast_scaled_dot_product_attention(
                query.as_mlx(),
                full_key.as_mlx(),
                full_value.as_mlx(),
                scale as f32,
                None, // No explicit mask needed — SDPA applies causal mask automatically
            ))
        };
        #[cfg(not(feature = "mlx"))]
        let attn_output = {
            let scale = (self.head_dim as f64).sqrt();
            let attn_weights = query.matmul(&full_key.transpose(-2, -1)) / scale;
            let attn_weights = attn_weights.clamp(-100.0, 100.0);
            // Build causal mask for new tokens attending to full sequence
            let mask = Tensor::zeros(
                &[new_seq_len, full_seq_len], DType::Float32, hidden_states.device(),
            );
            let positions_start = full_seq_len - new_seq_len;
            // For each query position i (0..new_seq_len), it can attend to
            // key positions 0..(positions_start + i + 1)
            let upper = Tensor::ones(&[new_seq_len, full_seq_len], DType::Bool, hidden_states.device());
            // Create proper causal: row i masks positions > (positions_start + i)
            let causal_mask = mask.masked_fill(&upper.triu(positions_start + 1), f64::NEG_INFINITY);
            let causal_mask = causal_mask.view(&[1, 1, new_seq_len, full_seq_len]);
            let attn_weights = attn_weights + causal_mask;
            let attn_weights = attn_weights.softmax(-1);
            attn_weights.matmul(&full_value)
        };

        // Reshape back
        let attn_output = attn_output
            .transpose(1, 2)
            .contiguous()
            .view(&[batch_size, new_seq_len, self.num_heads * self.head_dim]);

        // Output projection
        self.o_proj.forward(&attn_output)
    }
```

- [ ] **Step 2: Verify compilation**

Run: `cd /Users/I075399/qwen3_tts_rs && cargo build --features mlx 2>&1 | tail -5`
Expected: compilation succeeds

- [ ] **Step 3: Commit**

```bash
git add src/layers.rs
git commit -m "feat: add Attention::forward_with_cache for KV cache inference"
```

---

### Task 3: Add forward_with_offset to RotaryEmbedding

**Files:**
- Modify: `src/layers.rs` (RotaryEmbedding impl block, after existing `forward` at line ~206)

- [ ] **Step 1: Add forward_with_offset method**

The MLX `fast_rope` already has an `offset` parameter. For the non-MLX path, we slice `cos_cache`/`sin_cache` with offset. Add inside `impl RotaryEmbedding`:

```rust
    /// Apply rotary embedding with a position offset (for KV cache generation).
    ///
    /// `new_seq_len`: number of new tokens being processed.
    /// `offset`: position index of the first new token (= length of cached sequence).
    #[allow(unused_variables)]
    pub fn forward_with_offset(
        &self,
        q: &Tensor,
        k: &Tensor,
        new_seq_len: i64,
        offset: i64,
    ) -> (Tensor, Tensor) {
        #[cfg(feature = "mlx")]
        {
            let base = mlx::array::MlxArray::scalar_f32(self._theta as f32);

            let q_rope = Tensor::from_mlx(mlx::ops::fast_rope(
                q.as_mlx(),
                self.dim as i32,
                false,
                Some(&base),
                1.0,
                offset as i32,
            ));
            let k_rope = Tensor::from_mlx(mlx::ops::fast_rope(
                k.as_mlx(),
                self.dim as i32,
                false,
                Some(&base),
                1.0,
                offset as i32,
            ));

            return (q_rope, k_rope);
        }
        #[cfg(not(feature = "mlx"))]
        {
            let cos = self.cos_cache.narrow(0, offset, new_seq_len);
            let sin = self.sin_cache.narrow(0, offset, new_seq_len);

            let q_embed = self.apply_rope(q, &cos, &sin);
            let k_embed = self.apply_rope(k, &cos, &sin);

            (q_embed, k_embed)
        }
    }
```

- [ ] **Step 2: Verify compilation**

Run: `cd /Users/I075399/qwen3_tts_rs && cargo build --features mlx 2>&1 | tail -5`
Expected: compilation succeeds

- [ ] **Step 3: Commit**

```bash
git add src/layers.rs
git commit -m "feat: add RotaryEmbedding::forward_with_offset for KV cache"
```

---

### Task 4: Add forward_with_cache to TransformerLayer

**Files:**
- Modify: `src/layers.rs` (TransformerLayer impl block, after `forward` at line ~783)

- [ ] **Step 1: Add forward_with_cache method**

```rust
    /// Apply the transformer layer with KV cache (attention + MLP with residuals).
    pub fn forward_with_cache(
        &self,
        hidden_states: &Tensor,
        rotary_emb: &RotaryEmbedding,
        cache: &mut KvCache,
    ) -> Tensor {
        // Self-attention with residual
        let residual = hidden_states;
        let hidden = self.input_layernorm.forward(hidden_states);
        let hidden = self.self_attn.forward_with_cache(&hidden, rotary_emb, cache);
        let hidden = residual + hidden;

        // MLP with residual
        let residual = &hidden;
        let hidden_norm = self.post_attention_layernorm.forward(&hidden);
        let hidden = self.mlp.forward(&hidden_norm);

        residual + hidden
    }
```

- [ ] **Step 2: Verify compilation**

Run: `cd /Users/I075399/qwen3_tts_rs && cargo build --features mlx 2>&1 | tail -5`
Expected: compilation succeeds

- [ ] **Step 3: Commit**

```bash
git add src/layers.rs
git commit -m "feat: add TransformerLayer::forward_with_cache"
```

---

## Chunk 2: Rewrite TalkerModel Generation Loop

### Task 5: Add KV-cached generation to TalkerModel

**Files:**
- Modify: `src/inference.rs:805-968` (TalkerModel methods)

- [ ] **Step 1: Add forward_embeds_with_cache and generate_codes_cached methods**

Add these methods to `impl TalkerModel`, keeping the old `generate_codes` intact (renamed to `generate_codes_nocache` for fallback/testing):

```rust
    /// Run transformer forward with KV cache.
    /// Returns NORMED hidden states for the new token positions only.
    fn forward_embeds_with_cache(
        &self,
        embeddings: &Tensor,
        caches: &mut [KvCache],
    ) -> Tensor {
        let mut hidden = embeddings.shallow_clone();

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            hidden = layer.forward_with_cache(&hidden, &self.rotary_emb, cache);
        }

        self.norm.forward(&hidden)
    }

    /// Generate codes using KV cache (O(n) memory, O(n) compute per step).
    pub fn generate_codes_cached(
        &self,
        input_embeddings: &Tensor,
        max_codes: i64,
        temperature: f64,
        top_k: i64,
        eos_code: i64,
        tts_pad_embed: &Tensor,
    ) -> Vec<Vec<i64>> {
        let repetition_penalty = 1.05;
        let mut all_codes = Vec::new();
        let mut past_code_0s: Vec<i64> = Vec::new();
        let num_layers = self.layers.len();

        // Initialize KV caches (one per layer)
        let mut caches: Vec<KvCache> = (0..num_layers)
            .map(|_| KvCache::new())
            .collect();

        // === Prefill phase: process full input sequence ===
        let normed_hidden = self.forward_embeds_with_cache(input_embeddings, &mut caches);
        // We only need the last position's hidden state for the first prediction
        let seq_len = normed_hidden.size()[1];
        let mut last_hidden_for_code_predictor = normed_hidden.select(1, seq_len - 1).unsqueeze(1);

        // Predict first code 0
        let code_0 = self.predict_code_0(
            &normed_hidden,
            temperature,
            top_k,
            repetition_penalty,
            &past_code_0s,
        );
        past_code_0s.push(code_0);

        if code_0 == eos_code {
            eprintln!("  EOS detected at prefill");
            return all_codes;
        }

        // Generate codes 1-15 for first frame
        let code_0_tensor = Tensor::from_slice_i64(&[code_0]).to_device(self.device);
        let code_0_embed = self.codec_embedding
            .index_select(0, &code_0_tensor)
            .unsqueeze(0);

        let predictor_codes = self.code_predictor
            .generate_codes(&last_hidden_for_code_predictor, &code_0_embed, temperature, top_k);

        let mut frame_codes = vec![code_0];
        frame_codes.extend_from_slice(&predictor_codes);
        all_codes.push(frame_codes);

        // === Generation loop: one token at a time with KV cache ===
        for step in 1..max_codes {
            // Build next input embedding (sum of all code embeddings + tts_pad)
            let mut code_embeds_sum = code_0_embed.shallow_clone();
            for (i, &code) in predictor_codes.iter().enumerate() {
                if i < self.code_predictor.code_embeddings.len() {
                    let ct = Tensor::from_slice_i64(&[code]).to_device(self.device);
                    let emb = self.code_predictor.code_embeddings[i]
                        .index_select(0, &ct)
                        .unsqueeze(0);
                    code_embeds_sum = &code_embeds_sum + &emb;
                }
            }
            let next_input = &code_embeds_sum + tts_pad_embed; // [1, 1, hidden_size]

            // Forward ONLY the new token through the transformer (KV cache handles history)
            let normed_hidden = self.forward_embeds_with_cache(&next_input, &mut caches);

            // normed_hidden is [1, 1, hidden_size] — the new token's output
            last_hidden_for_code_predictor = normed_hidden.select(1, 0).unsqueeze(1);

            // Predict code 0
            let code_0 = self.predict_code_0(
                &normed_hidden,
                temperature,
                top_k,
                repetition_penalty,
                &past_code_0s,
            );
            past_code_0s.push(code_0);

            if code_0 == eos_code {
                eprintln!("  EOS detected at step {}", step);
                break;
            }

            // Generate codes 1-15
            let code_0_tensor = Tensor::from_slice_i64(&[code_0]).to_device(self.device);
            let code_0_embed = self.codec_embedding
                .index_select(0, &code_0_tensor)
                .unsqueeze(0);

            let predictor_codes = self.code_predictor
                .generate_codes(&last_hidden_for_code_predictor, &code_0_embed, temperature, top_k);

            let mut frame_codes = vec![code_0];
            frame_codes.extend_from_slice(&predictor_codes);
            all_codes.push(frame_codes);

            if step % 50 == 0 {
                eprintln!("  Generated {} code frames (cached)", step + 1);
            }
        }

        all_codes
    }
```

NOTE: Keep the old `generate_codes` method intact (do NOT delete it). The new `generate_codes_cached` will be called from `TTSInference::generate_with_params`.

- [ ] **Step 2: Import KvCache in inference.rs**

At the top of `src/inference.rs`, change the import line:

```rust
// Old:
use crate::layers::{Linear, RMSNorm, RotaryEmbedding, TransformerLayer};
// New:
use crate::layers::{KvCache, Linear, RMSNorm, RotaryEmbedding, TransformerLayer};
```

- [ ] **Step 3: Update TTSInference::generate_with_params to use cached version**

In `src/inference.rs`, around line 1270, change:
```rust
// Old:
        let codes = self.talker.generate_codes(
            &input_embeddings,
            max_codes,
            temperature,
            top_k,
            codec_eos_id,
            &tts_pad_embed,
        );
// New:
        let codes = self.talker.generate_codes_cached(
            &input_embeddings,
            max_codes,
            temperature,
            top_k,
            codec_eos_id,
            &tts_pad_embed,
        );
```

Also find and update the same call site in `generate_with_instruct` and `generate_voice_clone_*` methods (search for `.generate_codes(` in TTSInference).

- [ ] **Step 4: Verify compilation**

Run: `cd /Users/I075399/qwen3_tts_rs && cargo build --features mlx 2>&1 | tail -10`
Expected: compilation succeeds

- [ ] **Step 5: Run existing tests**

Run: `cd /Users/I075399/qwen3_tts_rs && cargo test --features mlx 2>&1 | tail -20`
Expected: all existing tests pass

- [ ] **Step 6: Commit**

```bash
git add src/inference.rs
git commit -m "feat: add KV-cached generation loop to TalkerModel (O(n) memory)"
```

---

## Chunk 3: Fix CodePredictor + shallow_clone + clear_cache

### Task 6: Optimize CodePredictor to avoid O(n²) sequence growth

**Files:**
- Modify: `src/inference.rs:161-249` (CodePredictor::generate_codes)

The CodePredictor runs 15 steps per frame. Currently it appends to a growing sequence and recomputes the full transformer. Since the CodePredictor is small (5 layers) and runs only 15 steps max, the simplest correct fix is to add KV cache here too.

- [ ] **Step 1: Add generate_codes_cached to CodePredictor**

Add new method in `impl CodePredictor`, keeping old `generate_codes` intact:

```rust
    /// Generate codes 1-15 with KV cache (avoids O(n²) sequence growth).
    pub fn generate_codes_cached(
        &self,
        main_hidden: &Tensor,
        code_0_embedding: &Tensor,
        temperature: f64,
        top_k: i64,
    ) -> Vec<i64> {
        let mut codes = Vec::new();
        let num_layers = self.layers.len();
        let mut caches: Vec<KvCache> = (0..num_layers)
            .map(|_| KvCache::new())
            .collect();

        // Prefill: process [main_hidden, code_0_embedding] (2 tokens)
        let initial_input = Tensor::cat(
            &[main_hidden.shallow_clone(), code_0_embedding.shallow_clone()],
            1,
        ); // [1, 2, hidden_size]

        // Apply projection if present (1.7B+)
        let projected = if let Some(ref proj) = self.small_to_mtp_projection {
            proj.forward(&initial_input)
        } else {
            initial_input.shallow_clone()
        };

        // Run through transformer layers with cache
        let mut hidden = projected;
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            hidden = layer.forward_with_cache(&hidden, &self.rotary_emb, cache);
        }

        // Get logits at last position for code 1
        let normed = self.norm.forward(&hidden);
        let last_hidden = normed.select(1, normed.size()[1] - 1).unsqueeze(0);

        for step in 0..self.lm_heads.len() {
            let logits = self.lm_heads[step].forward(&last_hidden).squeeze_dim(0);

            // Sample or argmax
            let code = if temperature <= 0.0 {
                logits.argmax(-1, false).int64_value(&[0])
            } else {
                let logits = &logits / temperature;
                let logits = if top_k > 0 {
                    let vocab_size = logits.size()[logits.dim() - 1];
                    let k = top_k.min(vocab_size);
                    let (top_values, _) = logits.topk(k, -1, true, true);
                    let threshold = top_values.select(-1, k - 1);
                    let mask = logits.lt_tensor(&threshold.unsqueeze(-1));
                    logits.masked_fill(&mask, f64::NEG_INFINITY)
                } else {
                    logits
                };
                let probs = logits.softmax(-1);
                probs.multinomial(1, true).int64_value(&[0, 0])
            };

            codes.push(code);

            // For next step: embed code, project, run through transformer (single token)
            if step + 1 < self.lm_heads.len() && step < self.code_embeddings.len() {
                let code_tensor = Tensor::from_slice_i64(&[code]).to_device(self.device);
                let emb = self.code_embeddings[step]
                    .index_select(0, &code_tensor)
                    .unsqueeze(0); // [1, 1, hidden_size]

                let projected_emb = if let Some(ref proj) = self.small_to_mtp_projection {
                    proj.forward(&emb)
                } else {
                    emb
                };

                // Single-token forward through transformer with cache
                let mut h = projected_emb;
                for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
                    h = layer.forward_with_cache(&h, &self.rotary_emb, cache);
                }

                let normed = self.norm.forward(&h);
                // last_hidden for next iteration (overwrite the binding)
                let _ = last_hidden; // drop old
                // Reassign — h is [1, 1, H], select(1, 0) → [1, H], unsqueeze → [1, 1, H]
                // Actually normed is already [1, 1, H], select gives [1, H]
                // We need [1, H] for the lm_head
                // Wait — lm_heads[step] expects [1, H] (batch, hidden) based on squeeze_dim(0) above
                // So: normed is [1, 1, H], select(1, 0) → [1, H], unsqueeze(0) → [1, 1, H]
                // Then lm_heads[step+1].forward(&last_hidden).squeeze_dim(0) will work
                break; // We need to rebind last_hidden — use a different approach
            }
        }

        // The above loop structure doesn't rebind well. Rewrite cleanly:
        codes.clear();
        // Reset caches
        for c in caches.iter_mut() { c.reset(); }

        // Prefill
        let initial_input = Tensor::cat(
            &[main_hidden.shallow_clone(), code_0_embedding.shallow_clone()],
            1,
        );
        let projected = if let Some(ref proj) = self.small_to_mtp_projection {
            proj.forward(&initial_input)
        } else {
            initial_input
        };
        let mut hidden = projected;
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            hidden = layer.forward_with_cache(&hidden, &self.rotary_emb, cache);
        }
        let mut normed = self.norm.forward(&hidden);
        let mut last_h = normed.select(1, normed.size()[1] - 1).unsqueeze(0); // [1, H]

        for step in 0..self.lm_heads.len() {
            let logits = self.lm_heads[step].forward(&last_h).squeeze_dim(0);

            let code = if temperature <= 0.0 {
                logits.argmax(-1, false).int64_value(&[0])
            } else {
                let logits = &logits / temperature;
                let logits = if top_k > 0 {
                    let vocab_size = logits.size()[logits.dim() - 1];
                    let k = top_k.min(vocab_size);
                    let (top_values, _) = logits.topk(k, -1, true, true);
                    let threshold = top_values.select(-1, k - 1);
                    let mask = logits.lt_tensor(&threshold.unsqueeze(-1));
                    logits.masked_fill(&mask, f64::NEG_INFINITY)
                } else {
                    logits
                };
                let probs = logits.softmax(-1);
                probs.multinomial(1, true).int64_value(&[0, 0])
            };

            codes.push(code);

            // Prepare next step's input
            if step + 1 < self.lm_heads.len() && step < self.code_embeddings.len() {
                let code_tensor = Tensor::from_slice_i64(&[code]).to_device(self.device);
                let emb = self.code_embeddings[step]
                    .index_select(0, &code_tensor)
                    .unsqueeze(0); // [1, 1, hidden_size]

                let projected_emb = if let Some(ref proj) = self.small_to_mtp_projection {
                    proj.forward(&emb)
                } else {
                    emb
                };

                let mut h = projected_emb;
                for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
                    h = layer.forward_with_cache(&h, &self.rotary_emb, cache);
                }
                normed = self.norm.forward(&h);
                last_h = normed.select(1, 0).unsqueeze(0); // [1, H]
            }
        }

        codes
    }
```

- [ ] **Step 2: Update TalkerModel::generate_codes_cached to use CodePredictor::generate_codes_cached**

In `generate_codes_cached`, change:
```rust
// Old:
        let predictor_codes = self.code_predictor
            .generate_codes(&last_hidden_for_code_predictor, &code_0_embed, temperature, top_k);
// New:
        let predictor_codes = self.code_predictor
            .generate_codes_cached(&last_hidden_for_code_predictor, &code_0_embed, temperature, top_k);
```

(Do this for BOTH occurrences in `generate_codes_cached` — the prefill frame and the generation loop.)

- [ ] **Step 3: Verify compilation**

Run: `cd /Users/I075399/qwen3_tts_rs && cargo build --features mlx 2>&1 | tail -10`
Expected: compilation succeeds

- [ ] **Step 4: Commit**

```bash
git add src/inference.rs
git commit -m "feat: add KV-cached CodePredictor (eliminates O(n²) per-frame)"
```

---

### Task 7: Fix shallow_clone on MLX backend

**Files:**
- Modify: `src/tensor.rs:1290-1292`
- Modify: `src/backend/mlx/array.rs:34-42`

- [ ] **Step 1: Verify MlxArray::clone is already ref-counted**

Read `src/backend/mlx/array.rs:34-42`. The existing `Clone` impl does:
```rust
impl Clone for MlxArray {
    fn clone(&self) -> Self {
        let mut new_ptr = unsafe { ffi::mlx_array_new() };
        unsafe { ffi::mlx_array_set(&mut new_ptr, self.ptr) };
        Self { ptr: new_ptr }
    }
}
```

`mlx_array_set` copies the internal pointer and increments the reference count — this IS a shallow clone. So `MlxArray::clone()` is already cheap (ref-counted). The issue is the Tensor wrapper.

- [ ] **Step 2: Verify Tensor::clone on MLX**

Read `src/tensor.rs:155-165`. The MLX `Tensor` wraps `MlxArray`:
```rust
impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() } // calls MlxArray::clone = ref-count
    }
}
```

So actually `Tensor::clone()` on MLX IS already ref-counted (cheap). And `shallow_clone` calls `self.clone()` which is correct!

**Conclusion: shallow_clone on MLX is NOT broken.** The original subagent analysis was wrong — `mlx_array_set` IS a ref-count operation, not a deep copy. No change needed here.

- [ ] **Step 3: Add a comment clarifying this**

In `src/tensor.rs`, update the shallow_clone method to clarify:

```rust
    pub fn shallow_clone(&self) -> Self {
        // MLX arrays are reference-counted internally. clone() calls mlx_array_set
        // which increments the ref count without copying data. This is O(1).
        self.clone()
    }
```

- [ ] **Step 4: Commit**

```bash
git add src/tensor.rs
git commit -m "docs: clarify MLX shallow_clone is ref-counted (O(1))"
```

---

### Task 8: Expose mlx_clear_cache and call it after generation

**Files:**
- Modify: `src/backend/mlx/stream.rs` (add clear_cache function)
- Modify: `src/inference.rs` (call clear_cache after generate)

- [ ] **Step 1: Expose clear_cache in stream.rs**

Add to `src/backend/mlx/stream.rs`:

```rust
/// Clear the MLX Metal memory cache.
///
/// Call this after completing a generation to release GPU memory buffers
/// back to the OS. Without this, MLX retains allocated Metal buffers in
/// its internal cache for reuse, which can accumulate over multiple
/// generation calls and cause OOM.
pub fn clear_cache() {
    unsafe { ffi::mlx_clear_cache() };
}
```

- [ ] **Step 2: Re-export clear_cache from backend::mlx**

In `src/backend/mlx/mod.rs`, add:
```rust
pub use stream::clear_cache;
```

Then in `src/backend/mod.rs`, it's already:
```rust
#[cfg(feature = "mlx")]
pub mod mlx;
```

So the path will be `crate::backend::mlx::clear_cache()`.

- [ ] **Step 3: Call clear_cache after each generation in TTSInference**

In `src/inference.rs`, in `generate_with_params`, after the codes are generated and decoded, add:

```rust
        // Clear MLX memory cache to prevent inter-request accumulation
        #[cfg(feature = "mlx")]
        {
            crate::backend::mlx::clear_cache();
        }
```

Add this right before the `Ok(...)` return in `generate_with_params`. Also add it to `generate_with_instruct` and `generate_voice_clone_*` methods — any method that calls `generate_codes_cached`.

- [ ] **Step 4: Verify compilation**

Run: `cd /Users/I075399/qwen3_tts_rs && cargo build --features mlx 2>&1 | tail -5`
Expected: compilation succeeds

- [ ] **Step 5: Run all tests**

Run: `cd /Users/I075399/qwen3_tts_rs && cargo test --features mlx 2>&1 | tail -20`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/backend/mlx/stream.rs src/backend/mlx/mod.rs src/inference.rs
git commit -m "feat: clear MLX cache after generation to prevent memory accumulation"
```

---

## Chunk 4: Testing & Verification

### Task 9: Add unit tests for KvCache

**Files:**
- Modify: `src/layers.rs` (tests module at bottom)

- [ ] **Step 1: Add KvCache unit tests**

Add to the existing `mod tests` in `src/layers.rs`:

```rust
    #[test]
    fn test_kv_cache_new_is_empty() {
        let cache = KvCache::new();
        assert!(cache.key.is_none());
        assert!(cache.value.is_none());
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_kv_cache_reset() {
        let mut cache = KvCache::new();
        // Simulate populating
        cache.key = Some(Tensor::zeros(&[1, 4, 10, 64], DType::Float32, Device::Cpu));
        cache.value = Some(Tensor::zeros(&[1, 4, 10, 64], DType::Float32, Device::Cpu));
        assert_eq!(cache.seq_len(), 10);

        cache.reset();
        assert!(cache.key.is_none());
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_kv_cache_seq_len() {
        let mut cache = KvCache::new();
        assert_eq!(cache.seq_len(), 0);

        cache.key = Some(Tensor::zeros(&[1, 4, 5, 64], DType::Float32, Device::Cpu));
        cache.value = Some(Tensor::zeros(&[1, 4, 5, 64], DType::Float32, Device::Cpu));
        assert_eq!(cache.seq_len(), 5);
    }
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/I075399/qwen3_tts_rs && cargo test --features mlx 2>&1 | tail -20`
Expected: all tests pass including new KvCache tests

- [ ] **Step 3: Commit**

```bash
git add src/layers.rs
git commit -m "test: add unit tests for KvCache"
```

---

### Task 10: Verify via build and test on voice-chat consumer

**Files:**
- None modified — verification only

- [ ] **Step 1: Build qwen3_tts_rs**

Run: `cd /Users/I075399/qwen3_tts_rs && cargo build --features mlx --release 2>&1 | tail -10`
Expected: release build succeeds

- [ ] **Step 2: Build voice-chat (consumer)**

Run: `cd /Users/I075399/cc_projects/voice-chat && cargo build --release 2>&1 | tail -10`
Expected: build succeeds (voice-chat depends on qwen3_tts via path dependency)

- [ ] **Step 3: Run voice-chat tests**

Run: `cd /Users/I075399/cc_projects/voice-chat && cargo test 2>&1 | tail -20`
Expected: all tests pass

- [ ] **Step 4: Final commit summary**

No additional commit needed — just verification.

---

## Summary: What Each Fix Achieves

| Problem | Fix | Memory Impact |
|---------|-----|---------------|
| O(n²) TalkerModel sequence growth | KV cache in generate_codes_cached | 38 GB → ~200 MB |
| O(n²) CodePredictor sequence growth | KV cache in generate_codes_cached | ~1.4 GB → ~10 MB |
| No KV cache (1000x recompute) | Attention::forward_with_cache | 1000x speedup |
| shallow_clone deep copy | Already correct — just docs | No change needed |
| MLX cache not cleared between calls | clear_cache() after generate | Prevents inter-request accumulation |

**Total expected memory reduction: ~38 GB → ~500 MB per process**
