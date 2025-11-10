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

<div style="font-size: 1em; color: #000;">

**Type 1: Query Embeddings** - BLIP-2, InstructBLIP → Q-Former

**Type 2: Patch Embeddings** - LLaVA, Qwen-VL → Direct patches

**✓ BLIVA = Both Combined** - Dual pathway architecture

**Key Flow:**
1. Vision Encoder processes image
2. Q-Former extracts queries  
3. Patches encoded separately
4. Both feed Pre-trained LLM
5. Instruction-aware generation

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

<div style="font-size: 0.58em; padding: 0;">

<div style="display: grid; grid-template-columns: 43% 57%; gap: 15px; align-items: start;">

<!-- LEFT: Visual Diagram -->
<div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 12px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);">

<!-- Input Image -->
<div style="text-align: center; margin-bottom: 8px;">
<div style="background: #fff; border: 2px solid #dee2e6; border-radius: 5px; padding: 6px; display: inline-block;">
<div style="background: #ff6b6b; color: white; padding: 3px 10px; border-radius: 3px; font-weight: bold; font-size: 0.9em;">🎬 HOLLYWOOD</div>
<div style="font-size: 0.8em; margin-top: 3px; color: #666;">Input Image</div>
</div>
</div>

<div style="text-align: center; font-size: 1.6em; color: #667eea; font-weight: bold; margin: 5px 0;">↓</div>

<!-- Vision Encoder -->
<div style="background: linear-gradient(135deg, #a5d6a7 0%, #81c784 100%); padding: 10px; border-radius: 6px; text-align: center; color: #1b5e20; font-weight: bold; border: 2px solid #4caf50; margin-bottom: 8px; font-size: 0.95em;">
Vision Encoder<br>
<span style="font-size: 0.8em; font-weight: normal;">EVA-CLIP (frozen)</span>
</div>

<!-- Split into 3 paths -->
<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; margin: 8px 0;">

<!-- Path 1: Q-Former -->
<div>
<div style="text-align: center; font-size: 1.4em; color: #667eea; font-weight: bold;">↓</div>
<div style="background: #ffe0b2; padding: 8px 4px; border-radius: 5px; text-align: center; border: 2px solid #ff9800; color: #e65100; font-weight: bold; font-size: 0.85em; min-height: 50px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
<div>Q-Former</div>
<div style="font-size: 0.8em; font-weight: normal;">(32 queries)</div>
</div>
<div style="text-align: center; font-size: 1.4em; color: #667eea; font-weight: bold; margin: 3px 0;">↓</div>
<div style="background: #e3f2fd; padding: 6px 4px; border-radius: 5px; text-align: center; border: 2px solid #2196F3; color: #1565c0; font-size: 0.8em; font-weight: 600;">
Proj_Q<br>32×4096
</div>
</div>

<!-- Path 2: Patches -->
<div>
<div style="text-align: center; font-size: 1.4em; color: #667eea; font-weight: bold;">↓</div>
<div style="background: #f3e5f5; padding: 8px 4px; border-radius: 5px; text-align: center; border: 2px solid #9c27b0; color: #6a1b9a; font-weight: bold; font-size: 0.85em; min-height: 50px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
<div>Encoded</div>
<div>Patches</div>
</div>
<div style="text-align: center; font-size: 1.4em; color: #667eea; font-weight: bold; margin: 3px 0;">↓</div>
<div style="background: #f3e5f5; padding: 6px 4px; border-radius: 5px; text-align: center; border: 2px solid #9c27b0; color: #6a1b9a; font-size: 0.8em; font-weight: 600;">
Proj_A<br>256×4096
</div>
</div>

<!-- Path 3: Text -->
<div>
<div style="text-align: center; font-size: 1.4em; color: #667eea; font-weight: bold;">↓</div>
<div style="background: #e0e0e0; padding: 8px 4px; border-radius: 5px; text-align: center; border: 2px solid #757575; color: #424242; font-weight: bold; font-size: 0.85em; min-height: 50px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
<div>Text</div>
<div>Instruction</div>
</div>
<div style="text-align: center; font-size: 1.4em; color: #667eea; font-weight: bold; margin: 3px 0;">↓</div>
<div style="background: #e0e0e0; padding: 6px 4px; border-radius: 5px; text-align: center; border: 2px solid #757575; color: #424242; font-size: 0.8em; font-weight: 600;">
Text<br>Embed
</div>
</div>

</div>

<!-- Convergence -->
<div style="text-align: center; font-size: 1.6em; color: #667eea; font-weight: bold; margin: 5px 0;">↓</div>

<!-- Concatenation -->
<div style="background: #fff9c4; padding: 8px; border-radius: 6px; text-align: center; border: 2px solid #fbc02d; color: #f57f17; font-weight: bold; margin-bottom: 6px; font-size: 0.9em;">
Concatenate All Embeddings
</div>

<div style="text-align: center; font-size: 1.6em; color: #667eea; font-weight: bold; margin: 5px 0;">↓</div>

<!-- LLM -->
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 10px; border-radius: 6px; text-align: center; color: white; font-weight: bold; border: 2px solid #5e35b1; margin-bottom: 6px; font-size: 0.95em;">
Vicuna-7B LLM<br>
<span style="font-size: 0.8em; font-weight: normal;">(frozen)</span>
</div>

<div style="text-align: center; font-size: 1.6em; color: #667eea; font-weight: bold; margin: 5px 0;">↓</div>

<!-- Output -->
<div style="background: #c8e6c9; padding: 8px; border-radius: 6px; text-align: center; border: 2px solid #4caf50; color: #2e7d32; font-weight: bold; font-size: 0.9em;">
💬 Text Response
</div>

</div>

<!-- RIGHT: Explanation -->
<div style="display: flex; flex-direction: column; gap: 8px;">

<div style="background: #e3f2fd; padding: 8px; border-radius: 5px; border-left: 3px solid #2196F3;">
<strong style="color: #1565c0; font-size: 1em;">1. Vision Encoding</strong>
<ul style="margin: 4px 0 0 0; padding-left: 16px; line-height: 1.25;">
<li>Input: Image 224×224×3</li>
<li>Encoder: EVA-CLIP ViT-g/14 (frozen)</li>
<li>Output: 257×1408 visual features</li>
</ul>
</div>

<div style="background: #fff3e0; padding: 8px; border-radius: 5px; border-left: 3px solid #ff9800;">
<strong style="color: #e65100; font-size: 1em;">2. Dual Pathway Processing</strong>
<div style="margin-top: 4px;">

<div style="margin-bottom: 5px;">
<strong style="color: #d84315; font-size: 0.92em;">Path A - Query Embeddings:</strong>
<ul style="margin: 2px 0 0 0; padding-left: 16px; line-height: 1.2; font-size: 0.92em;">
<li>Q-Former with 32 learned queries</li>
<li>FF → Cross-Attn → Self-Attn layers</li>
<li>Projects to 32×4096 embeddings</li>
</ul>
</div>

<div style="margin-bottom: 5px;">
<strong style="color: #6a1b9a; font-size: 0.92em;">Path B - Patch Embeddings:</strong>
<ul style="margin: 2px 0 0 0; padding-left: 16px; line-height: 1.2; font-size: 0.92em;">
<li>Direct encoded patches from encoder</li>
<li>Preserves fine-grained text info</li>
<li>Projects to 256×4096 embeddings</li>
</ul>
</div>

<div>
<strong style="color: #424242; font-size: 0.92em;">Path C - Text Input:</strong>
<ul style="margin: 2px 0 0 0; padding-left: 16px; line-height: 1.2; font-size: 0.92em;">
<li>User instruction tokenized</li>
<li>Text embeddings prepared</li>
</ul>
</div>

</div>
</div>

<div style="background: #e8f5e9; padding: 8px; border-radius: 5px; border-left: 3px solid #4caf50;">
<strong style="color: #2e7d32; font-size: 1em;">3. LLM Integration</strong>
<ul style="margin: 4px 0 0 0; padding-left: 16px; line-height: 1.25;">
<li>All embeddings concatenated together</li>
<li>Vicuna-7B LLM processes (frozen)</li>
<li>Generates contextual text response</li>
</ul>
</div>

<div style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); padding: 8px; border-radius: 5px; border: 2px solid #9c27b0;">
<strong style="color: #6a1b9a; font-size: 0.95em;">🎯 Key Innovation:</strong>
<div style="margin-top: 4px; line-height: 1.25; color: #424242; font-size: 0.92em;">
Combines semantic understanding from queries with text-aware visual details from patches for superior text-rich image understanding.
</div>
</div>

</div>

</div>

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

<div style="font-size: 0.68em;">

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 15px;">

<!-- Left: VLM Architecture Comparison -->
<div>
<strong style="font-size: 1.1em; color: #2c3e50;">VLM Architecture Comparison</strong>
<div style="margin-top: 10px; background: #f8f9fa; padding: 12px; border-radius: 8px; line-height: 1.4;">

**a) Flamingo:** Fixed query embeddings + XATTN Layer

**b) BLIP-2 / InstructBLIP:** Q-Former with learned queries only

**c) LLaVA:** Direct encoded patch embeddings only

**d) BLIVA (Ours):** Merges learned query embeddings + encoded patch embeddings

<div style="background: #e8f5e9; padding: 8px; border-radius: 5px; margin-top: 8px; border-left: 3px solid #4caf50;">
<strong style="color: #2e7d32;">✓ BLIVA combines both approaches</strong> for superior text-rich understanding
</div>

</div>
</div>

<!-- Right: Ablation Results -->
<div>
<strong style="font-size: 1.1em; color: #2c3e50;">Progressive Component Addition</strong>

<div style="background: #2d2d2d; color: #f8f8f2; padding: 12px; border-radius: 6px; margin-top: 10px; font-size: 0.95em; line-height: 1.5;">
<span style="color: #6272a4;"># Baseline: InstructBLIP (Q-Former only)</span><br>
<span style="color: #6272a4;"># Accuracy: 41.6% OCR-VQA</span><br><br>

<span style="color: #6272a4;"># + Add encoded patch embeddings</span><br>
<span style="color: #6272a4;"># Accuracy: 45.83% OCR-VQA (+4.23%)</span><br><br>

<span style="color: #6272a4;"># + Instruction tuning with patches</span><br>
<span style="color: #6272a4;"># Accuracy: 49.0% OCR-VQA (+7.4% total)</span>
</div>

</div>

</div>

<strong style="font-size: 1.05em; color: #2c3e50;">Table: Adding Individual Techniques</strong>

<div style="margin-top: 8px;">

| Configuration | OCR-VQA | TextVQA | Visual7W | Average Δ |
|---------------|---------|---------|----------|-----------|
| Baseline (Q-former only) | 41.6 | 34.1 | 54.8 | - |
| + Patch embeddings | **45.8** | 43.1 | 56.6 | **+4.0%** |
| + Instruction tuning | **58.7** | 44.34 | 57.58 | **+13.7%** |
| + Fine-tune projection | **62.2** | 44.86 | 57.96 | **+16.4%** |
| **BLIVA (Full)** | **62.2** | **44.86** | **57.96** | **+16.4%** |

</div>

</div>

<div style="background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%); padding: 12px; border-radius: 8px; margin-top: 12px; font-size: 0.72em; border-left: 4px solid #ffc107;">
<strong style="color: #856404;">🔍 Key Finding:</strong> Adding encoded patch embeddings provides <strong>immediate +4.23% improvement</strong>, demonstrating their value for text-rich understanding. Full system achieves <strong>+16.4% average improvement</strong> over baseline.
</div>

---

## 6. Comparison with Similar Models

### Text-Rich VQA Benchmark Results

<div style="font-size: 0.62em;">

<strong style="font-size: 1.1em; color: #2c3e50;">Zero-Shot OCR-Free Results on Text-Rich VQA</strong>

<div style="margin-top: 10px; overflow-x: auto;">

| Model | OCR-VQA | TextVQA | VizWiz | MSVRIT | Average |
|-------|---------|---------|--------|--------|---------|
| Flamingo-9B | 44.7 | 31.8 | - | - | 38.3 |
| Flamingo-80B | 50.6 | 35.0 | - | - | 42.8 |
| BLIP-2 (FlanT5XXL) | 45.9 | 42.5 | 19.6 | 28.8 | 34.2 |
| InstructBLIP (V-7B) | 50.1 | 45.2 | 34.5 | 39.4 | 42.3 |
| InstructBLIP (V-13B) | 50.7 | 44.8 | 33.4 | 40.5 | 42.4 |
| MiniGPT-4 | 18.56 | - | - | - | - |
| LLaVA (13B) | 37.98 | 39.7 | - | - | 38.8 |
| LLaVA-1.5 (13B) | 42.1 | 58.2 | - | - | 50.2 |
| Qwen-VL | 58.6 | 63.8 | 35.2 | 47.3 | 51.2 |
| **BLIVA (V-7B)** | **57.96** | **45.83** | **42.9** | **23.81** | **42.6** |

</div>

</div>

<div style="background: linear-gradient(135deg, #e8f4f8 0%, #d1e7f0 100%); padding: 10px; border-radius: 6px; margin-top: 12px; font-size: 0.65em; border-left: 4px solid #2196F3;">
<strong style="color: #1565c0;">📊 Key Insight:</strong> BLIVA achieves robust performance with only <strong>7B parameters</strong>, excelling on OCR-VQA with <strong>+17.76% improvement</strong> vs InstructBLIP baseline, demonstrating superior text-rich understanding.
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

<div style="font-size: 0.72em;">

<div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 15px; border-radius: 8px; border-left: 5px solid #2196F3; margin-bottom: 15px;">
<strong style="color: #1565c0; font-size: 1.3em;">1. Simple Yet Effective Design</strong>
<ul style="margin: 8px 0 0 0; padding-left: 20px; line-height: 1.4; color: #424242;">
<li>Combines strengths of existing approaches (InstructBLIP + LLaVA)</li>
<li>Minimal additional parameters (just projection layers)</li>
</ul>
</div>

<div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); padding: 15px; border-radius: 8px; border-left: 5px solid #ff9800; margin-bottom: 15px;">
<strong style="color: #e65100; font-size: 1.3em;">2. Superior Text-Rich Understanding</strong>
<ul style="margin: 8px 0 0 0; padding-left: 20px; line-height: 1.4; color: #424242;">
<li><strong>17.76% improvement</strong> on OCR-VQA benchmark</li>
<li>Robust performance across diverse text-heavy scenarios</li>
</ul>
</div>

<div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); padding: 15px; border-radius: 8px; border-left: 5px solid #4caf50; margin-bottom: 15px;">
<strong style="color: #2e7d32; font-size: 1.3em;">3. Efficient and Practical</strong>
<ul style="margin: 8px 0 0 0; padding-left: 20px; line-height: 1.4; color: #424242;">
<li>Only <strong>7.9B parameters</strong> vs 13B+ in competing models</li>
<li>Two-stage training: <strong>12 hours total</strong> on 8 GPUs</li>
<li>Maintains frozen backbone (easy to deploy)</li>
</ul>
</div>

<div style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); padding: 15px; border-radius: 8px; border-left: 5px solid #9c27b0;">
<strong style="color: #6a1b9a; font-size: 1.3em;">4. Open and Accessible</strong>
<ul style="margin: 8px 0 0 0; padding-left: 20px; line-height: 1.4; color: #424242;">
<li>Code and models freely available</li>
<li><strong>github.com/mlpc-ucsd/BLIVA</strong></li>
</ul>
</div>

</div>

<div style="text-align: center; margin-top: 20px; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; color: white; font-size: 0.85em;">
<strong style="font-size: 1.2em;">BLIVA: Better text-rich VQA through intelligent embedding combination</strong>
</div>
</div>

</div>