# Text-to-Video Generation using a Pretrained Open-Source Foundation Model

A Streamlit application that generates short videos directly from natural-language prompts using the pretrained **CogVideoX-2B** model through Hugging Face Diffusers.

## Aim

To use a pretrained, open-source video generation foundation model to generate short videos directly from text prompts, without collecting a dataset or training the model, and to preview and save the generated videos.

## Features

- Simple web interface built with Streamlit.
- Text prompt input and example prompts.
- Uses a pretrained CogVideoX-2B foundation model.
- Adjustable inference steps, random seed, and output FPS.
- MP4 preview and download.
- Automatically saves generated files to `outputs/`.
- No dataset collection, model training, or fine-tuning.

## Project structure

```text
text_to_video_foundation_model/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── outputs/
    └── .gitkeep
```

## Requirements

- Windows 10/11, Linux, or macOS.
- Python 3.10 or 3.11 is recommended for the broadest compatibility with the ML dependencies.
- Internet access on the first run to download model weights.
- An NVIDIA GPU with sufficient VRAM is strongly recommended. Video generation is compute- and memory-intensive; CPU generation may be extremely slow or fail. Exact memory needs depend on software versions and offloading.
- Several gigabytes of free disk space for model weights and generated videos.

## Installation on Windows (VS Code)

### 1. Extract the ZIP

Extract `text_to_video_foundation_model.zip` to a folder and open that folder in VS Code.

### 2. Create a virtual environment

In the VS Code terminal:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If Python 3.10 is not installed, install it or use a compatible Python version supported by the current PyTorch and Diffusers releases.

### 3. Install PyTorch

For an NVIDIA GPU, install a PyTorch build compatible with your CUDA driver using the official selector:
https://pytorch.org/get-started/locally/

For CPU-only testing, you can install the CPU build, but generation can be impractically slow:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install the project dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the application

```powershell
streamlit run app.py
```

Streamlit will show a local address (usually `http://localhost:8501`). Open it in your browser, enter a prompt, and click **Generate video**.

## Example prompts

- `A tiny wooden boat drifting on a calm lake at sunrise, cinematic wide shot.`
- `A red fox walking through fresh snow in a quiet pine forest, gentle camera movement.`
- `A futuristic city skyline at night with flying vehicles and glowing neon lights.`

## How it works

1. **Prompt input:** User describes the scene in natural language.
2. **Pretrained model:** `CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b")` loads the pretrained checkpoint.
3. **Inference:** The model uses the prompt and inference settings to synthesize a sequence of frames. No training occurs in this project.
4. **Video export:** Diffusers utility `export_to_video` encodes the frames into MP4.
5. **Preview and save:** Streamlit previews the file and provides a download button. The MP4 is saved under `outputs/`.

## Important notes

- The first launch downloads model files; it can take time and requires a reliable internet connection.
- The model is large. If you encounter CUDA out-of-memory errors, close other GPU applications, reduce inference steps, and verify that model offloading is enabled. You may need a GPU with more memory or a supported optimized setup.
- Model availability, licensing, and hardware requirements can change. Review the model card and license before redistribution or commercial use.
- Generated videos can contain visual artifacts and may not exactly follow every prompt.
- This repository contains application code and dependencies only; it does **not** bundle the large pretrained model weights.

## Troubleshooting

### `ModuleNotFoundError`
Activate the virtual environment and run `pip install -r requirements.txt` again.

### PyTorch cannot detect CUDA
Run:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Install the correct PyTorch build for your system from the official PyTorch installation selector.

### CUDA out of memory
Try closing other GPU programs and lowering inference steps. The current app uses CPU offloading for CUDA inference and VAE tiling. Hardware requirements still depend on the model and software stack.

### CPU-only computer or `OSError: [Errno 22] Invalid argument`
CogVideoX-2B is very large and CPU generation is generally impractical. This project uses
full-precision weights on CPU, which can require substantially more memory than is available
on a typical laptop. An Intel integrated GPU does not provide CUDA; installing a CUDA-enabled
PyTorch package cannot add CUDA support to it. Run the app on a compatible NVIDIA GPU, or use
a smaller model designed for CPU inference.

If model loading fails, expand **Technical traceback** in the app and inspect the full error.
An interrupted Hugging Face download can also leave incomplete cache files; retry with a
stable internet connection so the Hub can resume or redownload the missing files. Do not
delete the entire Hugging Face cache, since other models may use it.

### Model download fails
Check your internet connection and try again. Hugging Face may need to download multiple model files.

## References

- Hugging Face Diffusers CogVideoX guide: https://huggingface.co/docs/diffusers/main/using-diffusers/cogvideox
- CogVideoX model repository: https://huggingface.co/THUDM/CogVideoX-2b
- PyTorch installation selector: https://pytorch.org/get-started/locally/

## Academic project summary

This project demonstrates text-conditioned video synthesis using a pretrained open-source foundation model. The user supplies a natural-language prompt; the application runs model inference, exports generated frames to an MP4 file, and provides preview and download functionality. No dataset is collected and no model training or fine-tuning is performed.
