cd ~/science-quant/llama.cpp/build

./bin/llama-imatrix \
  -m ../../models/DeepSeek-R1-Distill-Llama-70B-f16.gguf \
  -f ../../data/generic_calibration.txt \
  --chunk 512 \
  -ngl 40 \
  --save-frequency 50 \
  -o ../../imatrix/imatrix_generic.dat

