cd ~/science-quant/llama.cpp/build

./bin/llama-quantize \
  --imatrix ../../imatrix/imatrix_science.dat \
  ../../models/DeepSeek-R1-Distill-Llama-70B-f16.gguf \
  ../../models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M-science.gguf \
  Q4_K_M

