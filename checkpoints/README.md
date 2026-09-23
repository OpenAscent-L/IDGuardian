# Checkpoints

Large model files are intentionally excluded from Git. Download the official IP-Adapter SDXL Plus Face files with:

```bash
mkdir -p checkpoints/ip_adapter
hf download h94/IP-Adapter \\
  --include "models/image_encoder/*" \\
  --include "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin" \\
  --local-dir checkpoints/ip_adapter
```

The training entry point expects:

```text
checkpoints/ip_adapter/models/image_encoder
checkpoints/ip_adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin
```

Use `--image-encoder-path` and `--ip-adapter-path` to override these paths.
