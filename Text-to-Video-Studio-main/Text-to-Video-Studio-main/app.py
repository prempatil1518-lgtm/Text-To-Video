import os
import time
from pathlib import Path

import streamlit as st
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

APP_DIR = Path(__file__).parent
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Text-to-Video Studio", page_icon="🎬", layout="wide")
st.title("🎬 Text-to-Video Studio")
st.write("Generate a short video from a text prompt using a pretrained open-source foundation model.")
st.info("This project uses pretrained CogVideoX-2B. It does not collect a dataset or train/fine-tune the model. The first run downloads model weights and can take a while.")

with st.sidebar:
    st.header("Generation settings")
    steps = st.slider("Inference steps", min_value=10, max_value=50, value=20, step=5,
                      help="More steps may improve quality but take longer.")
    seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
    fps = st.selectbox("Output frame rate (FPS)", [8, 10, 12], index=0)
    st.caption("Video generation is resource-intensive. An NVIDIA GPU is strongly recommended.")

examples = [
    "A small red hot-air balloon floating above green hills at sunrise, cinematic camera movement",
    "A golden retriever running through a field of flowers on a sunny day, natural motion",
    "A futuristic city at night with flying cars and glowing neon signs, cinematic wide shot",
    "Ocean waves gently rolling onto a sandy beach during sunset, peaceful atmosphere",
]
example = st.selectbox("Try an example prompt (optional)", ["Choose an example..."] + examples)
default_prompt = "" if example == "Choose an example..." else example
prompt = st.text_area("Describe the video you want to create", value=default_prompt, height=120,
                      placeholder="Example: A paper boat sailing along a tiny stream in a forest, soft morning light...")
generate = st.button("✨ Generate video", type="primary", use_container_width=False)

if generate:
    if not prompt.strip():
        st.error("Please enter a text prompt first.")
        st.stop()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        st.warning(
            "CUDA GPU was not detected. CogVideoX-2B is not practical on most CPU-only "
            "computers and may fail while loading or run out of memory. Use a compatible "
            "NVIDIA GPU for generation."
        )

    status = st.status("Preparing video generation...", expanded=True)
    try:
        dtype = torch.float16 if device == "cuda" else torch.float32
        status.write(f"Device selected: {device.upper()}")
        status.write("Loading pretrained CogVideoX-2B model. First-time download may be several gigabytes.")
        pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-2b",
            torch_dtype=dtype,
        )
        if device == "cuda":
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_tiling()
        else:
            pipe.to("cpu")

        status.write("Generating frames. Please keep this page open...")
        generator = torch.Generator(device="cuda" if device == "cuda" else "cpu").manual_seed(int(seed))
        result = pipe(
            prompt=prompt.strip(),
            num_videos_per_prompt=1,
            num_inference_steps=int(steps),
            num_frames=49,
            guidance_scale=6,
            generator=generator,
        )
        frames = result.frames[0]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"generated_video_{stamp}.mp4"
        export_to_video(frames, str(output_path), fps=int(fps))
        status.update(label="Video generated successfully!", state="complete", expanded=False)
        st.success(f"Saved video to: {output_path.relative_to(APP_DIR)}")
        st.video(str(output_path))
        with open(output_path, "rb") as video_file:
            st.download_button("⬇️ Download MP4", data=video_file, file_name=output_path.name,
                               mime="video/mp4")
        st.caption(f"Prompt: {prompt.strip()}")
    except Exception as exc:
        status.update(label="Generation failed", state="error", expanded=True)
        st.error(f"{type(exc).__name__}: {exc}")
        with st.expander("Technical traceback"):
            st.exception(exc)
        st.markdown("Check the README troubleshooting section for common fixes.")
    finally:
        if "pipe" in locals():
            del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

st.divider()
st.subheader("How it works")
st.markdown("""
1. You enter a text prompt.
2. The app loads a pretrained CogVideoX model through Hugging Face Diffusers.
3. The model generates video frames from the prompt.
4. Frames are exported as an MP4 file, previewed here, and saved in the `outputs/` folder.
""")
