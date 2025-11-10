<div style="height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">

# BLIVA: A Simple Multimodal LLM

## Better Handling of Text-Rich Visual Questions

<div style="margin-top: 40px; font-size: 0.85em; line-height: 1.6;">

**Authors**: Wenbo Hu, Yifan Xu, Yi Li, Weiyue Li, Zeyuan Chen, Zhuowen Tu

**University of California, San Diego | Coinbase Global, Inc.**

**AAAI 2024**

</div>

</div>

---

## 1. Introduction

### Vision Language Models (VLMs)

<div style="font-size: 0.85em; line-height: 1.6;">

**Large Language Models** have transformed natural language understanding by demonstrating impressive generalization across tasks in zero-shot and few-shot settings.

**Vision Language Models** extend LLMs by incorporating visual understanding:
- Demonstrated significant advancements in open-ended VQA tasks
- Handle various vision and language tasks through instruction tuning

</div>

### Key Challenge

<div style="background: #fff3cd; padding: 15px; border-radius: 8px; margin: 15px 0; font-size: 0.8em;">
<strong>⚠️ Problem:</strong> Images infused with text are common in real-world scenarios (documents, charts, memes, screenshots) but current models struggle with text-heavy content.
</div>

---

## 2. Problem Statement

### Current VLM Limitations

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; font-size: 0.75em;">

<div>
<strong style="color: #e74c3c;">❌ Existing Approaches</strong>

**Standard methods encode images into image embeddings:**
- Fixed token count limits representation
- Potentially lose recognition of text context
- Struggle with text-rich images

**Example: Q-Former in BLIP-2**
- Uses only 32 query tokens
- Cannot capture intricate text details
- Limited by token budget
</div>

<div>
<strong style="color: #27ae60;">✓ BLIVA Solution</strong>

**Leverages learned query embeddings:**
- Additional visual patch branches
- Utilizes encoder patch embeddings
- Better captures text information

**Key Insight:**
- Query embeddings help better understand image semantic meaning
- Direct encoder patch embeddings missed by query-dependent approach
</div>

</div>

### Research Questions

<div style="font-size: 0.8em; background: #e8f4f8; padding: 12px; border-radius: 8px; margin-top: 15px;">

1. **How does BLIVA enhance recognition of YouTube thumbnails with text?**
2. **How does our method compare to single image embedding approaches in text-rich VQA?**
3. **How do individual components influence success?**

</div>

---

## 3. Architecture Overview

### BLIVA System Design - Complete Pipeline

<div style="font-size: 0.7em;">

<div style="display: grid; grid-template-columns: 2fr 3fr; gap: 20px; align-items: start;">

<div>
<div style="background: #e3f2fd; padding: 8px; border-radius: 6px; margin-bottom: 8px; color: #1565c0;">
<strong>Type 1: Query Embeddings</strong><br>
BLIP-2, InstructBLIP → Q-Former
</div>

<div style="background: #fff3e0; padding: 8px; border-radius: 6px; margin-bottom: 8px; color: #e65100;">
<strong>Type 2: Patch Embeddings</strong><br>
LLaVA, Qwen-VL → Direct patches
</div>

<div style="background: #e8f5e9; padding: 8px; border-radius: 6px; margin-bottom: 12px; color: #2e7d32;">
<strong>✓ BLIVA = Both Combined</strong><br>
Dual pathway architecture
</div>

<div style="background: #f0f0f0; padding: 8px; border-radius: 6px; font-size: 0.95em; color: #424242;">
<strong>Key Flow:</strong><br>
1. Vision Encoder processes image<br>
2. Q-Former extracts queries<br>
3. Patches encoded separately<br>
4. Both feed Pre-trained LLM<br>
5. Instruction-aware generation
</div>
</div>

<div class="bliva-arch-diagram">
<!-- Top: Pre-trained LLM -->
<div style="grid-column: 2 / 5; text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 16px; border-radius: 10px; font-weight: bold; box-shadow: 0 4px 8px rgba(0,0,0,0.3); font-size: 1.3em;">
Pre-trained LLM
</div>

<!-- Middle Row: Projections and Embeddings -->
<div style="grid-column: 1 / 2; grid-row: 2;">
<div style="background: #fff3e0; padding: 12px; border: 3px solid #ff9800; border-radius: 8px; text-align: center; margin-bottom: 8px; color: #e65100; font-weight: bold; font-size: 1.1em;">
Projection
</div>
<div style="text-align: center; font-size: 2.5em; color: #667eea; font-weight: bold;">↓</div>
<div style="background: #e3f2fd; padding: 12px; border: 3px solid #2196F3; border-radius: 8px; text-align: center; color: #1565c0; font-weight: bold; font-size: 1.05em;">
Learned Query<br>Embeddings
</div>
</div>

<div style="grid-column: 3 / 4; grid-row: 2;">
<div style="background: #fff3e0; padding: 12px; border: 3px solid #ff9800; border-radius: 8px; text-align: center; margin-bottom: 8px; color: #e65100; font-weight: bold; font-size: 1.1em;">
Projection
</div>
<div style="text-align: center; font-size: 2.5em; color: #667eea; font-weight: bold;">↓</div>
<div style="background: #f3e5f5; padding: 12px; border: 3px solid #9c27b0; border-radius: 8px; text-align: center; color: #6a1b9a; font-weight: bold; font-size: 1.05em;">
Encoded Patch<br>Embeddings
</div>
</div>

<div style="grid-column: 4 / 5; grid-row: 2;">
<div style="background: #e8f5e9; padding: 12px; border: 3px solid #4caf50; border-radius: 8px; text-align: center; margin-bottom: 8px; color: #2e7d32; font-weight: bold; font-size: 1.1em;">
Projection<br>Embeddings
</div>
<div style="text-align: center; font-size: 2.5em; color: #667eea; font-weight: bold;">↓</div>
<div style="background: #e8f5e9; padding: 10px; border: 3px solid #4caf50; border-radius: 8px; text-align: center; color: #2e7d32; font-weight: bold; font-size: 1.05em;">
Encoded Patch
</div>
</div>

<!-- Q-Former Row -->
<div style="grid-column: 1 / 3; grid-row: 3; position: relative;">
<div style="background: #ffe0b2; padding: 15px; border: 3px solid #ff9800; border-radius: 10px; text-align: center; font-weight: bold; color: #e65100; font-size: 1.2em;">
Q-Former
<div style="font-size: 0.85em; font-weight: 600; margin-top: 6px; color: #d84315;">
Feed-Forward → Cross-Attention → Self-Attention
</div>
</div>
<div style="position: absolute; right: -35px; top: 50%; font-size: 2.5em; color: #ff9800; font-weight: bold;">→</div>
</div>

<div style="grid-column: 3 / 5; grid-row: 3; display: flex; align-items: center; gap: 10px;">
<div style="flex: 1; background: #e3f2fd; padding: 14px; border: 3px solid #2196F3; border-radius: 8px; text-align: center; color: #1565c0; font-weight: bold; font-size: 1.05em;">
Encoded<br>Patch<br>Embeddings
</div>
<div style="font-size: 2.5em; color: #2196F3; font-weight: bold;">→</div>
<div style="flex: 1; background: #fff9e6; padding: 14px; border: 3px solid #ffc107; border-radius: 8px; text-align: center; color: #f57f17; font-weight: bold; font-size: 1.05em;">
Queries
</div>
</div>

<!-- Bottom Row: Text and Vision Encoder -->
<div style="grid-column: 1 / 2; grid-row: 4; text-align: center;">
<div style="text-align: center; font-size: 2.5em; color: #667eea; font-weight: bold;">↑</div>
<div style="background: #e0e0e0; padding: 12px; border: 3px solid #757575; border-radius: 8px; margin-bottom: 10px; color: #424242; font-weight: bold; font-size: 1.05em;">
Text Embeddings
</div>
<div style="display: flex; align-items: center; justify-content: center; gap: 10px; color: #424242; font-weight: 600;">
<div style="background: #64b5f6; color: white; padding: 8px 12px; border-radius: 6px; font-size: 1em;">👤</div>
<div style="text-align: left; font-size: 0.95em;">What is this<br>image about?</div>
</div>
<div style="margin-top: 8px; font-weight: bold; font-size: 1em; color: #424242;">User Instruction</div>
</div>

<div style="grid-column: 3 / 5; grid-row: 4; text-align: center;">
<div style="text-align: center; font-size: 2.5em; color: #667eea; font-weight: bold;">↑</div>
<div style="background: #a5d6a7; padding: 14px; border: 3px solid #4caf50; border-radius: 10px; font-weight: bold; margin-bottom: 10px; color: #2e7d32; font-size: 1.2em;">
Vision Encoder
</div>
<div style="background: #fff; border: 3px solid #9e9e9e; border-radius: 8px; padding: 10px;">
<div style="background: #ff6b6b; color: white; padding: 6px; font-size: 1em; font-weight: bold; border-radius: 4px;">HOLLYWOOD</div>
<div style="font-size: 0.95em; margin-top: 6px; color: #424242; font-weight: 600;">Input Image</div>
</div>
</div>

</div>

</div>

</div>

---

## 3. Architecture - Detailed Pipeline

### Component Breakdown

<div style="font-size: 0.7em;">

| Component | Description | Output Dimension |
|-----------|-------------|------------------|
| **Vision Encoder** | EVA-CLIP (ViT-g/14) frozen | 257 × 1408 |
| **Q-Former** | Extract visual features using queries | 32 × hidden_dim |
| **Projection Layer** | Linear projection to LLM space | 32 × 4096 |
| **Encoded Patches** | Directly from vision encoder | 256 × 1408 |
| **Projection Layer** | Project patches to LLM space | 256 × 4096 |
| **LLM** | Vicuna-7B (frozen) | 4096-dim space |

</div>

### Information Flow

<div style="font-size: 0.75em; background: #e8f5e9; padding: 12px; border-radius: 8px; margin-top: 15px;">

**Image → Vision Encoder → Dual Path:**

1. **Query Path:** Q-Former extracts semantic features → Projection → LLM
2. **Patch Path:** Encoded patches preserve text details → Projection → LLM

**Result:** Combined embeddings capture both semantic understanding and text details

</div>

---

## 3. Architecture - Visual Representation

### Complete System Architecture

<div style="font-size: 0.65em;">

```
┌─────────────┐
│   Image     │
│  224×224×3  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│   Vision Encoder    │
│  EVA-CLIP (frozen)  │
│   Output: 257×1408  │
└──────┬──────────────┘
       │
       ├──────────────────────────┬────────────────────────┐
       │                          │                        │
       ▼                          ▼                        ▼
┌────────────┐         ┌──────────────────┐    ┌──────────────────┐
│  Q-Former  │         │  Projection_A    │    │  Text Queries    │
│  (32 query)│         │  (encoded patch) │    │                  │
└─────┬──────┘         └────────┬─────────┘    └────────┬─────────┘
      │                         │                       │
      ▼                         ▼                       ▼
┌────────────┐         ┌──────────────────┐    ┌──────────────────┐
│Projection_Q│         │  256 × 4096      │    │  Text Embeddings │
│ 32 × 4096  │         │  embeddings      │    │                  │
└─────┬──────┘         └────────┬─────────┘    └────────┬─────────┘
      │                         │                       │
      └────────────┬────────────┴───────────────────────┘
                   ▼
         ┌────────────────────┐
         │   Concatenation    │
         │   All Embeddings   │
         └─────────┬──────────┘
                   ▼
         ┌────────────────────┐
         │   Vicuna-7B LLM    │
         │     (frozen)       │
         └─────────┬──────────┘
                   ▼
         ┌────────────────────┐
         │   Text Response    │
         └────────────────────┘
```

</div>

---

## 4. Codebase Implementation

### Model Initialization

```python
class BLIVAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Vision Encoder (EVA-CLIP, frozen)
        self.vision_encoder = EVACLIPVisionModel.from_pretrained(
            "eva_clip_g", freeze=True
        )
        
        # Q-Former for learned query embeddings
        self.qformer = BertLMHeadModel(config.qformer_config)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, 32, config.qformer_config.hidden_size)
        )
        
        # Projection layers
        self.projection_q = nn.Linear(
            config.qformer_config.hidden_size, 
            config.llm_hidden_size
        )
        self.projection_a = nn.Linear(
            config.vision_hidden_size, 
            config.llm_hidden_size
        )
        
        # Language Model (Vicuna, frozen)
        self.llm = LlamaForCausalLM.from_pretrained(
            "lmsys/vicuna-7b-v1.5", freeze=True
        )
```

---

## 4. Codebase - Forward Pass

### Complete Forward Function

```python
def forward(self, images, input_ids, attention_mask):
    batch_size = images.size(0)
    
    # 1. Extract vision features (frozen)
    with torch.no_grad():
        vision_outputs = self.vision_encoder(images)
        # Output: [batch, 257, 1408]
        image_embeds = vision_outputs.last_hidden_state
    
    # 2. Q-Former path: learned query embeddings
    query_tokens = self.query_tokens.expand(batch_size, -1, -1)
    query_outputs = self.qformer(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        return_dict=True
    )
    # Output: [batch, 32, hidden_size]
    
    # 3. Project Q-Former outputs to LLM space
    query_embeds = self.projection_q(query_outputs.last_hidden_state)
    # Output: [batch, 32, 4096]
```

---

## 4. Codebase - Forward Pass (cont.)

```python
    # 4. Encoded patch path: preserve text details
    # Remove CLS token, use remaining 256 patches
    patch_embeds = image_embeds[:, 1:, :]  # [batch, 256, 1408]
    
    # 5. Project patches to LLM space
    patch_embeds = self.projection_a(patch_embeds)
    # Output: [batch, 256, 4096]
    
    # 6. Get text embeddings
    text_embeds = self.llm.model.embed_tokens(input_ids)
    # Output: [batch, seq_len, 4096]
    
    # 7. Concatenate all embeddings
    inputs_embeds = torch.cat([
        query_embeds,      # Learned semantic features
        patch_embeds,      # Direct patch details
        text_embeds        # Question text
    ], dim=1)
    
    # 8. Generate response with LLM (frozen)
    with torch.no_grad():
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
    
    return outputs
```

---

## 4. Codebase - Training Setup

### Two-Stage Training Paradigm

```python
# Stage 1: Pre-training (Align vision with LLM)
def pretrain_step(model, batch):
    images, captions = batch
    
    # Only train Q-Former and projection layers
    # Vision encoder and LLM remain frozen
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.llm.parameters():
        param.requires_grad = False
    
    # Forward pass
    outputs = model(images, captions)
    
    # Language modeling loss
    loss = F.cross_entropy(
        outputs.logits.view(-1, vocab_size),
        captions.view(-1)
    )
    
    return loss

# Dataset: 129M image-caption pairs from LAION
# Training: 6 hours on 8 A6000 GPUs (48 GPUs)
```

---

## 4. Codebase - Training Setup (cont.)

### Stage 2: Instruction Tuning

```python
def instruction_tuning_step(model, batch):
    images, instructions, answers = batch
    
    # Fine-tune Q-Former and both projection layers
    # Employ image caption pairs in InstructBLIP format
    
    # Trainable parameters:
    # - Q-Former weights
    # - projection_q (Q-Former → LLM)
    # - projection_a (patches → LLM)
    
    # Frozen parameters:
    # - vision_encoder (EVA-CLIP)
    # - llm (Vicuna-7B)
    
    # Forward with instruction format
    formatted_input = f"Question: {instructions} Answer:"
    outputs = model(images, formatted_input)
    
    # Cross-entropy loss on answer tokens only
    loss = compute_loss(outputs, answers, ignore_prompt=True)
    
    return loss

# Dataset: 558K image instruction pairs
# Training: Takes 6 hours on 8 A6000 GPUs
# Learning rate: 1e-4 with linear warmup (0.05)
# Batch size: 128 (16 per GPU × 8 GPUs)
```

---

## 5. Core Innovation

### Key Innovation: Dual Embedding Approach

<div style="font-size: 0.8em;">

**Problem with existing methods:**
- Q-Former/learned queries: Limited by fixed token count (typically 32)
- Direct patch embeddings: Miss semantic relationships

**BLIVA's Innovation:**

<div style="background: #e8f5e9; padding: 15px; border-radius: 8px; margin: 10px 0;">

**Combines BOTH approaches simultaneously:**

1. **Learned Query Embeddings (Q-Former)**: Extract instruction-aware semantic features
2. **Encoded Patch Embeddings**: Preserve fine-grained text details

**Result**: Better text understanding without increasing parameters significantly

</div>

### Technical Advantages

| Aspect | Benefit |
|--------|---------|
| **Information Preservation** | Captures both semantic meaning AND text details |
| **Efficiency** | Reuses frozen vision encoder, no extra encoding |
| **Scalability** | Only projection layers trained (lightweight) |
| **Text-Rich Performance** | +17.76% on OCR-VQA vs baseline InstructBLIP |

</div>

---

## 5. Core Innovation - Ablation Study

### Component Analysis

<div style="font-size: 0.75em;">

**Ablation experiments show impact of each component:**

```python
# Baseline: InstructBLIP (Q-Former only)
# Accuracy: 41.6% OCR-VQA

# + Add encoded patch embeddings
# Accuracy: 45.83% OCR-VQA (+4.23%)

# + Instruction tuning with patches
# Accuracy: 49.0% OCR-VQA (+7.4% total)
```

**Table: Adding Individual Techniques**

| Configuration | OCR-VQA | TextVQA | Visual7W | Average Δ |
|---------------|---------|---------|----------|-----------|
| Baseline (Q-former only) | 41.6 | 34.1 | 54.8 | - |
| + Patch embeddings | 45.8 | 43.1 | 56.6 | +4.0% |
| + Instruction tuning | 58.7 | 44.34 | 57.58 | +13.7% |
| + Fine-tune projection | 62.2 | 44.86 | 57.96 | +16.4% |
| **BLIVA (Full)** | **62.2** | **44.86** | **57.96** | **+16.4%** |

</div>

<div style="background: #fff3cd; padding: 10px; border-radius: 8px; margin-top: 10px; font-size: 0.75em;">
<strong>Key Finding:</strong> Adding encoded patch embeddings provides immediate improvement, demonstrating their value for text-rich understanding.
</div>

---

## 6. Comparison with Similar Models

### Text-Rich VQA Benchmark Results

<div style="font-size: 0.7em;">

**Zero-Shot OCR-Free Results on Text-Rich VQA:**

| Model | OCR-VQA | TextVQA | VizWiz | MSVRIT | Average |
|-------|---------|---------|--------|--------|---------|
| **Flamingo-9B** | 44.7 | 31.8 | - | - | 38.3 |
| **Flamingo-80B** | 50.6 | 35.0 | - | - | 42.8 |
| **BLIP-2 (FlanT5XXL)** | 45.9 | 42.5 | 19.6 | 28.8 | 34.2 |
| **InstructBLIP (Vicuna-7B)** | 50.1 | 45.2 | 34.5 | 39.4 | 42.3 |
| **InstructBLIP (Vicuna-13B)** | 50.7 | 44.8 | 33.4 | 40.5 | 42.4 |
| **MiniGPT-4** | 18.56 | - | - | - | - |
| **LLaVA (13B)** | 37.98 | 39.7 | - | - | 38.8 |
| **LLaVA-1.5 (13B)** | 42.1 | 58.2 | - | - | 50.2 |
| **Qwen-VL** | 58.6 | 63.8 | 35.2 | 47.3 | 51.2 |
| **BLIVA (Vicuna-7B)** | **57.96** | **45.83** | **42.9** | **23.81** | **42.6** |

</div>

<div style="background: #e8f4f8; padding: 12px; border-radius: 8px; margin-top: 10px; font-size: 0.75em;">
<strong>Note:</strong> BLIVA demonstrates robust performance with only 7B parameters, especially excelling on OCR-VQA (+17.76% vs InstructBLIP baseline).
</div>

---

## 6. Comparison - General VQA Benchmarks

### Performance on Standard VQA Tasks

<div style="font-size: 0.7em;">

**Zero-shot results on general (not text-rich) VQA benchmarks:**

| Model | VSR | IconQA | TextVQA | VizWiz | Flick30K | HM | Average |
|-------|-----|--------|---------|--------|----------|-----|---------|
| Flamingo-9B | - | - | 31.8 | - | 61.5 | - | - |
| BLIP-2 (FlanT5) | 53.7 | 44.7 | 42.5 | 19.6 | - | 40.4 | 40.2 |
| InstructBLIP (V-7B) | **54.3** | **43.1** | 45.2 | 34.5 | **82.4** | 58.7 | 53.2 |
| **BLIVA (V-7B)** | 54.3 | 43.1 | **45.83** | **42.9** | 78.9 | **62.2** | **54.5** |

</div>

### MME Benchmark Results

<div style="font-size: 0.65em;">

| Model | Perception | Cognition | Total |
|-------|------------|-----------|-------|
| InstructBLIP | 1212.8 | 291.8 | 1504.6 |
| MiniGPT-4 | 856.1 | 248.5 | 1104.6 |
| LLaVA (13B) | 1531.3 | 295.4 | 1826.7 |
| **BLIVA** | **1669.2** | **180.0** | **1849.2** |

</div>

<div style="background: #d4edda; padding: 10px; border-radius: 8px; margin-top: 10px; font-size: 0.75em;">
<strong>Key Achievement:</strong> BLIVA achieves competitive or superior performance across diverse benchmarks with simpler architecture.
</div>

---

## 6. Comparison - Architecture Comparison

### Model Architecture Comparison

<div style="font-size: 0.65em;">

| Model | Vision Encoder | Query Mechanism | Patch Embeddings | LLM | Parameters |
|-------|----------------|-----------------|------------------|-----|------------|
| **Flamingo** | NFNet | Perceiver + Cross-Attn | ✗ | Chinchilla 70B | 80B |
| **BLIP-2** | ViT-g/14 | Q-Former (32 queries) | ✗ | FlanT5-XXL | 12.1B |
| **InstructBLIP** | ViT-g/14 | Q-Former (32 queries) | ✗ | Vicuna-7B/13B | 8B/14B |
| **MiniGPT-4** | ViT-g/14 | Q-Former (32 queries) | ✗ | Vicuna-13B | 14B |
| **LLaVA** | CLIP ViT-L/14 | ✗ | ✓ Linear projection | Vicuna-13B | 13B |
| **LLaVA-1.5** | CLIP ViT-L/14 | ✗ | ✓ MLP projection | Vicuna-13B | 13B |
| **Qwen-VL** | ViT-bigG | ✗ | ✓ Cross-attention | Qwen-7B | 9.6B |
| **BLIVA** | EVA-CLIP g/14 | ✓ Q-Former (32) | ✓ Linear projection | Vicuna-7B | **7.9B** |

</div>

### Key Differentiators

<div style="font-size: 0.75em; background: #f3e5f5; padding: 12px; border-radius: 8px; margin-top: 10px;">

**BLIVA's Unique Approach:**
- **Only model** combining Q-Former learned queries + direct encoded patches
- Leverages strengths of both InstructBLIP (semantic) and LLaVA (detailed)
- Achieves strong results with **fewer parameters** (7.9B vs 13B+)
- Simple architecture: just two linear projection layers added

</div>

---

## 6. Comparison - YouTube Thumbnails Task

### Real-World Application: YouTube Thumbnail Understanding

<div style="font-size: 0.75em;">

**Task:** Answer questions about YouTube thumbnails paired with video descriptions

**Dataset:** YTBVQA collected by authors
- 11 categories × 150 videos = 1,650 samples
- Tests real-world text-rich image understanding

**Results:**

| Model | Accuracy |
|-------|----------|
| Random Baseline | ~9% |
| MiniGPT-4 (Zhu et al. 2023) | 47.75% |
| LLaVA (Liu et al. 2023) | 44.75% |
| InstructBLIP (Dai et al. 2023) | 82.2% |
| Qwen-VL (Vicuna-7B) | 83.5% |
| **BLIVA (Vicuna-7B)** | **84.0%** |

</div>

<div style="background: #e8f4f8; padding: 12px; border-radius: 8px; margin-top: 10px; font-size: 0.75em;">
<strong>Analysis:</strong> BLIVA demonstrates superior ability to extract visual information from images with embedded text, outperforming baseline InstructBLIP by 1.8 percentage points.
</div>

---

## 6. Comparison - Qualitative Analysis

### Visual Examples: Handling Complex Text

<div style="font-size: 0.75em;">

**BLIVA excels at:**

1. **Posters and Memes:** Accurately interprets feeling/hatefuln text with visual context
2. **Webpages:** Successfully localizes and reads specific UI elements  
3. **Charts/Diagrams:** Extracts numerical values and relationships

**Example Tasks:**

```
Q: "What is the image about?"
Image: [Hollywood sign on hillside]

BLIVA: "The image depicts the famous Hollywood sign 
located on a hillside, surrounded by mountains. The 
sign is prominently displayed in the center of the 
image, with its letters clearly visible."

✓ Correctly identifies text AND contextual scenery
```

**Comparison with baselines:**
- InstructBLIP: Sometimes misses text details
- LLaVA: Good at text but may miss semantic context
- **BLIVA**: Captures both text AND contextual understanding

</div>

---

## 7. Conclusion

### Summary of Contributions

<div style="font-size: 0.8em; line-height: 1.6;">

**BLIVA presents a simple yet effective approach for text-rich visual question answering:**

#### ✓ Core Innovation
- **Dual embedding approach**: Combines learned query embeddings (Q-Former) with encoded patch embeddings
- Addresses limitations of fixed token budget in existing VLMs
- Preserves fine-grained text information crucial for text-rich images

#### ✓ Technical Achievements
- **Significant improvements**: Up to 17.76% gain on OCR-VQA benchmark
- **Competitive performance**: Matches or exceeds larger models (13B) with only 7B parameters
- **Simple architecture**: Only adds two projection layers, maintains frozen vision encoder and LLM

#### ✓ Broad Applicability
- Strong performance on both text-rich (OCR-VQA, TextVQA) and general VQA benchmarks
- Real-world validation on YouTube thumbnails dataset
- Open-source release: https://github.com/mlpc-ucsd/BLIVA

</div>

---

## 7. Conclusion - Future Directions

### Limitations and Future Work

<div style="font-size: 0.8em;">

**Current Limitations:**

<div style="background: #fff3cd; padding: 12px; border-radius: 8px; margin: 10px 0;">

1. **Frozen Components**: Vision encoder and LLM remain frozen - unfreezing may improve catastrophic forgetting but increases training complexity
2. **Token Budget**: Still uses 32 query tokens + 256 patch tokens (288 total)
3. **Single-Image Focus**: Current work focuses on single images

</div>

**Future Research Directions:**

<div style="background: #e8f5e9; padding: 12px; border-radius: 8px;">

- Explore unfreezing vision encoder for better alignment
- Investigate optimal query token count and patch selection
- Extend to multi-image scenarios
- Apply technique to other multimodal tasks (video understanding, etc.)
- Develop more efficient projection mechanisms

</div>

</div>

---

## 7. Conclusion - Key Takeaways

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 12px; font-size: 0.85em;">

### 🎯 Key Takeaways

<div style="margin: 15px 0; line-height: 1.8;">

**1. Simple Yet Effective Design**
- Combines strengths of existing approaches (InstructBLIP + LLaVA)
- Minimal additional parameters (just projection layers)

**2. Superior Text-Rich Understanding**
- 17.76% improvement on OCR-VQA benchmark
- Robust performance across diverse text-heavy scenarios

**3. Efficient and Practical**
- Only 7.9B parameters vs 13B+ in competing models
- Two-stage training takes only 12 hours total on 8 GPUs
- Maintains frozen backbone (easy to deploy)

**4. Open and Accessible**
- Code and models freely available
- github.com/mlpc-ucsd/BLIVA

</div>

<div style="text-align: center; margin-top: 20px; padding-top: 15px; border-top: 2px solid rgba(255,255,255,0.3); font-size: 1.1em;">
<strong>BLIVA: Better text-rich VQA through intelligent embedding combination</strong>
</div>

</div>