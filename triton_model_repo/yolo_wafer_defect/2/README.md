# TensorRT FP16 Model Placeholder

This directory is for the TensorRT FP16 engine file.

## How to populate

1. Run `notebooks/tensorrt_benchmark.ipynb` on Google Colab A100
2. Download `best_fp16.engine` from the results
3. Copy it here as `model.plan`:
   ```
   cp best_fp16.engine triton_model_repo/yolo_wafer_defect/2/model.plan
   ```
4. Update `config.pbtxt` version_policy to serve both versions:
   ```
   version_policy { specific { versions: [1, 2] } }
   ```
5. Restart Triton to load version 2
