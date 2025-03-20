import gradio as gr
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

# Load Stable Diffusion Inpainting model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe.to(device)

def inpaint(image, mask, prompt):
    image = image.convert("RGB")
    mask = mask.convert("L")  # Convert mask to grayscale
    result = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
    return result

# Create Gradio UI
demo = gr.Interface(
    fn=inpaint,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Image(type="pil", label="Draw Mask", tool="sketch"),
        gr.Textbox(label="Describe What Should Replace the Masked Area")
    ],
    outputs=gr.Image(label="Generated Image"),
    title="AI Generative Fill with Stable Diffusion",
    description="Upload an image, mask the area you want to change, and enter a prompt to generate AI-filled content."
)

demo.launch()
