# Models Directory

This folder stores pretrained checkpoints. All `.pt` / `.pth` / `.ckpt` files are excluded from Git by `.gitignore`.

## Download

| File | Dataset | Link |
|------|---------|------|
| `ffhq_10m.pt` | FFHQ | [Google Drive](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing) (from [DPS2022](https://github.com/DPS2022/diffusion-posterior-sampling)) |
| `imagenet256.pt` | ImageNet | [Google Drive](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing) (from [DPS2022](https://github.com/DPS2022/diffusion-posterior-sampling)) |
| `256x256_diffusion_uncond.pt` | ImageNet (alt.) | [OpenAI](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) |

After downloading, place the files here:

```text
models/
├── ffhq_10m.pt
└── imagenet256.pt      # or 256x256_diffusion_uncond.pt
```
