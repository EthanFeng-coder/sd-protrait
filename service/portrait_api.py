from fastapi import APIRouter, HTTPException,Query
from pydantic import BaseModel
import requests
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
import io
import os
import base64

#Stable Diffusion API endpoint
SD_API_URL = "http://host.docker.internal:7860/sdapi/v1/txt2img"

os.makedirs("data/pic", exist_ok=True)
# Define the router
router = APIRouter()

# Request model for the `/portrait` endpoint
class PortraitRequest(BaseModel):
    role_id: str
    role_name: str
    role_gender: str
    role_desc: str
    role_voice: str
    role_characters: str
    role_background: str

@router.post("/portrait")
async def generate_portrait(request: PortraitRequest):
    file_name = f"data/pic/{request.role_id.strip() if request.role_id.strip() else 'default'}.png"

    # Check if the file already exists
    if os.path.exists(file_name):
        print(f"File {file_name} already exists. Returning the existing image.")
        return FileResponse(file_name, media_type="image/png")

    role_name = request.role_name if request.role_name.strip() else "Unnamed"
    role_background = request.role_background if request.role_background.strip() else "No background provided"
    role_voice = request.role_voice if request.role_voice.strip() else "Unknown"
    role_gender = request.role_gender if request.role_gender.strip() else "Unknown"
    role_desc = request.role_desc if request.role_desc.strip() else "No description provided"
    role_characters = request.role_characters if request.role_characters.strip() else "No character traits provided"

    prompt = (
        f"{role_name}, Gender: {role_gender}, Description: {role_desc}, Characters: {role_characters}, "
        f"{role_background}, Role Voice: {role_voice}. "
        f"cinematic still score_9_up, score_8_up, score_7_up, 4k epic detailed, shot on kodak, "
        f"35mm photo, sharp focus, high budget"
    )
    negative_prompt: str = (
        "worst quality, low quality, normal quality, lowers, low details, oversaturated, "
        "undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, "
        "bad art:1.4, (watermark, signature, text font, username, error, logo, words, letters, "
        "digits, autograph, trademark, name:1.2)"
    )
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "styles": [],
         "seed": -1,  # Use a random seed
    "subseed": -1,
    "subseed_strength": 0,
    "seed_resize_from_h": -1,
    "seed_resize_from_w": -1,
    "sampler_name": "DPM++ 2M",
    "scheduler": "Automatic",
    "batch_size": 1,
    "n_iter": 1,
    "steps": 20,
    "cfg_scale": 7,
    "width": 512,
    "height": 512,
    "restore_faces": True,
    "tiling": False,
    "do_not_save_samples": False,
    "do_not_save_grid": False,
    "eta": 0.0,
    "denoising_strength": 0.0,
    "override_settings": {
        "sd_model_checkpoint": "purerealismMixXL_v10.safetensors [c94accb2f7]"
    },
    "override_settings_restore_afterwards": True,
    "enable_hr": False,
    "hr_scale": 2,
    "hr_upscaler": "R-ESRGAN 4x+",
    "hr_second_pass_steps": 20,
    "sampler_index": "Euler"
    }


    # Debug: Print payload
    #print("Payload being sent to Stable Diffusion API:", payload)

    # Call the Stable Diffusion API
    try:
        response = requests.post(SD_API_URL, json=payload)
        print("Stable Diffusion API response status:", response.status_code)  # Debug
        print("Stable Diffusion API response text:", response.text)  # Debug
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Error contacting Stable Diffusion API:", e)  # Debug
        raise HTTPException(status_code=500, detail=str(e))

    # Parse the response and save the image to a file
    try:
        result = response.json()
        print("Stable Diffusion API JSON response:", result)  # Debug
        if "images" in result:
            image_data = result["images"][0]  # Base64 encoded image
            image_bytes = base64.b64decode(image_data)
            file_name = f"data/pic/{request.role_id.strip() if request.role_id.strip() else 'default'}.png"  # Use role_id or default

            # Save the image to the file
            with open(file_name, "wb") as f:
                f.write(image_bytes)

            return {"message": "Image generated successfully", "file_path": file_name}
        else:
            print("No images key in response:", result)  # Debug
            raise HTTPException(status_code=500, detail="No images generated")
    except ValueError as e:
        print("Error parsing response JSON:", e)  # Debug
        raise HTTPException(status_code=500, detail="Invalid response from Stable Diffusion API")


@router.get("/get-portrait")
async def get_portrait(role_id: str = Query(..., description="Role ID to fetch the portrait")):
    file_name = f"data/pic/{role_id.strip()}.png"

    if not os.path.exists(file_name):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(file_name, media_type="image/png")
