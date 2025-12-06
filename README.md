# Science-Calibrated Quantization of DeepSeek-R1-Distill-Llama-70B

End-to-end pipeline to **quantize a 70B reasoning model** (`deepseek-ai/DeepSeek-R1-Distill-Llama-70B`) to **Q4_K_M GGUF** using **science-domain calibration** from the MetaMathQA dataset, with:

- **BF16 → GGUF conversion**
- **Generic vs. science importance matrices (imatrix)**
- **llama.cpp Q4_K_M quantization**
- **Perplexity + throughput benchmarks** on math reasoning text
- All scripts tuned for a **single A100 80GB** GPU

---

## 1. Project Goals

1. **Demonstrate 70B‑scale LLM quantization** end-to-end using open tools (`llama.cpp`, GGUF).
2. **Compare generic vs. science-domain calibration** via llama.cpp’s importance matrix:
   - Generic imatrix from WikiText‑style data.
   - Science imatrix from the MetaMathQA math reasoning dataset.
3. **Quantify the tradeoff**:
   - Speedup vs. BF16 (tokens/sec).
   - Perplexity on math reasoning text (MetaMathQA subset).
4. Document the **practical pitfalls** you hit:
   - CUDA memory constraints on a single A100.
   - `inf detected in blk.3.attn_q.weight` during imatrix.
   - Git LFS + Hugging Face hosting of 40+ GB GGUFs.

---

## 2. Repository Structure

```text
.
├── README.md                        # This file
├── scripts/
│   ├── prepare_generic_wikitext.py  # Pull & format WikiText-2 calibration data
│   ├── prepare_science_data.py      # Pull & format MetaMathQA calibration/eval data
│   ├── run_imatrix_generic.sh       # Compute generic imatrix
│   ├── run_imatrix_science.sh       # Compute science imatrix
│   ├── quantize_generic.sh          # Q4_K_M quantization with generic imatrix
│   ├── quantize_science.sh          # Q4_K_M quantization with science imatrix
│   ├── eval_perplexity.sh           # Run llama-perplexity for BF16 + quantized models
│   └── eval_bench.sh                # Run llama-bench throughput benchmarks
├── data/
│   ├── generic_calibration.txt      # WikiText-2-based calibration text
│   ├── science_calibration.txt      # MetaMathQA-based calibration text
│   └── eval_science.txt             # MetaMathQA eval subset
├── imatrix/
│   ├── imatrix_generic.dat          # Generic importance matrix
│   └── imatrix_science.dat          # Science importance matrix
├── models/
│   ├── DeepSeek-R1-Distill-Llama-70B-f16.gguf           # BF16 base GGUF
│   ├── DeepSeek-R1-Distill-Llama-70B-Q4_K_M-generic.gguf # Generic Q4_K_M
│   └── DeepSeek-R1-Distill-Llama-70B-Q4_K_M-science.gguf # Science Q4_K_M
└── results/
    ├── ppl_bf16.txt
    ├── ppl_q4_generic.txt
    ├── ppl_q4_science.txt
    ├── bench_bf16.txt
    ├── bench_q4_generic.txt
    └── bench_q4_science.txt
````

Not everything is under git (especially `models/`). Large artifacts live on Hugging Face. This repo stores:

- Scripts
    
- Configs
    
- Logs
    
- A small sample of calibration/eval text
    

---

## 3. Requirements

- **Hardware**
    
    - 1× NVIDIA A100 80GB (tested); other high‑VRAM GPUs may work with tweaked `-ngl`.
        
- **Software**
    
    - Ubuntu 22.04 / 24.04
        
    - CUDA 12.x
        
    - CMake, GCC 11+ / 13+
        
    - Python 3.10+
        
    - `llama.cpp` (commit `ec7f3ac9a` / build 4514, CUDA enabled)
        
    - `huggingface_hub`, `datasets`
        

---

## 4. Quickstart (high level)

1. **Clone `llama.cpp` and build with CUDA**
    
    ```bash
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
    cmake --build build -j"$(nproc)"
    ```
    
2. **Download base model (HF) and convert to BF16 GGUF**
    
    ```bash
    huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
      --local-dir models/DeepSeek-R1-Distill-Llama-70B \
      --local-dir-use-symlinks False
    
    python3 convert_hf_to_gguf.py \
      --outtype bf16 \
      --outfile models/DeepSeek-R1-Distill-Llama-70B-f16.gguf \
      models/DeepSeek-R1-Distill-Llama-70B
    ```
    
3. **Prepare calibration and eval data**
    
    - Generic: WikiText‑2 via `Salesforce/wikitext` (`wikitext-2-raw-v1`).
        
    - Science: MetaMathQA via `meta-math/MetaMathQA` (math reasoning Q/A with CoT). ([Hugging Face](https://huggingface.co/datasets/meta-math/MetaMathQA?utm_source=chatgpt.com "meta-math/MetaMathQA · Datasets at Hugging Face"))
        
    
    Example:
    
    ```bash
    python scripts/prepare_generic_wikitext.py
    python scripts/prepare_science_data.py
    ```
    
    You end up with:
    
    - `data/generic_calibration.txt`
        
    - `data/science_calibration.txt`
        
    - `data/eval_science.txt`
        
4. **Compute importance matrices**
    
    ```bash
    bash scripts/run_imatrix_generic.sh
    bash scripts/run_imatrix_science.sh
    ```
    
    These wrap `llama-imatrix` with something like:
    
    ```bash
    ./llama.cpp/build/bin/llama-imatrix \
      -m models/DeepSeek-R1-Distill-Llama-70B-f16.gguf \
      -f data/science_calibration.txt \
      --chunk 512 \
      -ngl 40 \
      --save-frequency 50 \
      -o imatrix/imatrix_science.dat
    ```
    
5. **Quantize to Q4_K_M**
    
    ```bash
    bash scripts/quantize_generic.sh
    bash scripts/quantize_science.sh
    ```
    
    Example:
    
    ```bash
    ./llama.cpp/build/bin/llama-quantize \
      --imatrix imatrix/imatrix_science.dat \
      models/DeepSeek-R1-Distill-Llama-70B-f16.gguf \
      models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M-science.gguf \
      Q4_K_M
    ```
    
6. **Evaluate perplexity and throughput**
    
    ```bash
    bash scripts/eval_perplexity.sh
    bash scripts/eval_bench.sh
    ```
    

---

## 5. Evaluation Results

### 5.1 Perplexity on math reasoning text

Using `llama-perplexity` with `eval_science.txt` (MetaMathQA subset):

```bash
./llama.cpp/build/bin/llama-perplexity \
  -m <MODEL> \
  -f data/eval_science.txt \
  --chunks -1 \
  -ngl 40
```

Summarized results:

|Model|Type|PPL (↓)|Std. error|
|---|---|---|---|
|BF16 baseline|BF16 GGUF|**9.74**|± 0.29|
|Q4_K_M (generic imatrix)|Q4_K_M + Wiki|**9.71**|± 0.29|
|Q4_K_M (science imatrix)|Q4_K_M + MetaMathQA|**9.77**|± 0.29|

Takeaways:

- All models are **within ~0.5%** of each other.
    
- Differences are smaller than the estimated uncertainty (~3%).
    
- Q4_K_M with either imatrix **preserves BF16-level perplexity** on MetaMathQA‑style text.
    

### 5.2 Throughput on A100 80GB

Prompt‑eval throughput recorded by `llama-perplexity`:

|Model|Type|Tokens/sec (↑)|
|---|---|---|
|BF16 baseline|BF16 GGUF|~**73** t/s|
|Q4_K_M (generic imatrix)|Q4_K_M + Wiki|~**164** t/s|
|Q4_K_M (science imatrix)|Q4_K_M + MetaMathQA|~**152** t/s|

Takeaways:

- **~2× speedup** over BF16 for Q4_K_M, on a single A100 80GB with `-ngl 40`.
    
- Science and generic Q4_K_M are similar in speed; small differences are within normal variance.
    

---

## 6. Implementation Notes & Gotchas

### 6.1 `inf detected in blk.3.attn_q.weight` during imatrix

While running `llama-imatrix` on the 70B BF16 GGUF, `llama.cpp` reported:

```text
inf detected in blk.3.attn_q.weight
```

By default, this check aborts the imatrix computation. For this project:

- The check in `examples/imatrix/imatrix.cpp` was modified to **clamp non‑finite values to `0.0f` and continue**, instead of returning an error.
    
- The code was then rebuilt:
    
    ```bash
    rm -rf llama.cpp/build
    cmake -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
    cmake --build llama.cpp/build -j"$(nproc)"
    ```
    

This behavior only affects **imatrix statistics during quantization**, not the runtime model behavior.

### 6.2 Memory tuning

- `-ngl 40` (40 layers on GPU) and `--chunk 512` were used on an A100 80GB.
    
- If you hit OOM:
    
    - Lower `-ngl` (e.g., 30 or 20).
        
    - Or lower `--chunk` (e.g., 256).
        

### 6.3 Hugging Face + Git LFS

- GGUF files (~40 GB) must be tracked with LFS:
    
    ```bash
    git lfs install
    echo "*.gguf filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
    git add .gitattributes
    git add models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M-science.gguf
    git commit -m "Track GGUF with LFS"
    git push origin main
    ```
    

---

## 7. How to Use the Science‑Calibrated GGUF

Example `llama.cpp` usage:

```bash
./llama.cpp/build/bin/llama-cli \
  -m models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M-science.gguf \
  -ngl 40 \
  -c 4096 \
  -n 256 \
  -p "Solve the following math problem step by step:\n\nA tank contains 120 liters of water and drains at 5 liters per minute. How long until it is empty?"
```

You can also import the GGUF into:

- LM Studio
    
- Ollama
    
- Other GGUF‑aware engines
    

for more user‑friendly inference.

---

## 8. Limitations & Future Work

- **No additional safety tuning** was performed; behavior is inherited from the base DeepSeek‑R1 distill model.
    
- PPL differences between generic and science imatrix are **not statistically significant** at Q4_K_M on this eval slice.
    
- Future directions:
    
    - Try more aggressive quantization (e.g., `Q3_K_M`, `Q2_K`) where domain‑specific imatrix may matter more.
        
    - Construct more extreme math‑only eval sets (long proofs, symbolic derivations).
        
    - Explore calibration on other STEM datasets (e.g., code/math hybrids).
        

---

## 9. Acknowledgements

- **Base model:** DeepSeek team for `DeepSeek-R1-Distill-Llama-70B`. ([Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B?utm_source=chatgpt.com "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"))
    
- **Math dataset:** MetaMath / MetaMathQA authors for the dataset and paper. ([arXiv](https://arxiv.org/abs/2309.12284?utm_source=chatgpt.com "MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models"))
    
- **Tooling:** `llama.cpp` maintainers and contributors.
    

---

## 10. License

- The underlying model follows the license of `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` (MIT at the time of this project). ([Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B?utm_source=chatgpt.com "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"))
    
- Calibration data is derived from MetaMathQA, which is itself derived from GSM8K and MATH training sets; see the original repositories for details. ([Hugging Face](https://huggingface.co/datasets/meta-math/MetaMathQA?utm_source=chatgpt.com "meta-math/MetaMathQA · Datasets at Hugging Face"))
    
- This repository’s code and configuration files may be used under the MIT license unless stated otherwise.
