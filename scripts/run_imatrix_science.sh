cd ~/science-quant/llama.cpp/build

./bin/llama-imatrix \
  -m ../../models/DeepSeek-R1-Distill-Llama-70B-f16.gguf \
  -f ../../data/science_calibration.txt \
  --chunk 512 \
  -ngl 40 \
  --save-frequency 50 \
  -o ../../imatrix/imatrix_science.dat

