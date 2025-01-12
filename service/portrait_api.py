from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import httpx  # Use httpx for asynchronous HTTP requests
from fastapi.responses import FileResponse
import os
import base64
import aiofiles  # For asynchronous file handling

# Stable Diffusion API endpoint
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
        print(f"File {file_name} already exists. Returning the existing URL.")
        return {"message": "Image already exists", "file_url": f"/get-portrait?role_id={request.role_id.strip()}"}

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
        "seed": -1,
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

    try:
        # Call the Stable Diffusion API asynchronously using httpx
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            response = await client.post(SD_API_URL, json=payload)
            response.raise_for_status()
    except httpx.RequestError as exc:
        print(f"Error contacting Stable Diffusion API: {exc}")
        raise HTTPException(status_code=500, detail="Failed to contact Stable Diffusion API")

    # Process the response
    result = response.json()
    if "images" in result:
        image_data = base64.b64decode(result["images"][0])
        async with aiofiles.open(file_name, "wb") as f:
            await f.write(image_data)
        return {"message": "Image generated successfully",
                "file_url": f"/get-portrait?role_id={request.role_id.strip()}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to generate image")


@router.get("/get-portrait")
async def get_portrait(role_id: str = Query(..., description="Role ID to fetch the portrait")):
    file_name = f"data/pic/{role_id.strip()}.png"

    if not os.path.exists(file_name):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(file_name, media_type="image/png")
