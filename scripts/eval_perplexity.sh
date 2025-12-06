cd ~/science-quant/llama.cpp/build

./bin/llama-perplexity \
  -m ../../models/DeepSeek-R1-Distill-Llama-70B-f16.gguf \
  -f ../../data/eval_science.txt \
  --chunks -1 \
  -ngl 40

./bin/llama-perplexity \
  -m ../../models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M-generic.gguf \
  -f ../../data/eval_science.txt \
  --chunks -1 \
  -ngl 40

./bin/llama-perplexity \
  -m ../../models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M-science.gguf \
  -f ../../data/eval_science.txt \
  --chunks -1 \
  -ngl 40


