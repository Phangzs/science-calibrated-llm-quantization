./bin/llama-bench \
  -m ../../models/DeepSeek-R1-Distill-Llama-70B-f16.gguf \
  -ngl 40

./bin/llama-bench \
  -m ../../models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M-generic.gguf \
  -ngl 40

./bin/llama-bench \
  -m ../../models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M-science.gguf \
  -ngl 40



