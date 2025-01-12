from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import requests
from fastapi.responses import FileResponse
import os
import base64
import spacy

# Load English and Chinese language models
nlp_en = spacy.load("en_core_web_sm")
nlp_cn = spacy.load("zh_core_web_sm")

# Stable Diffusion API endpoint
SD_API_URL = "http://host.docker.internal:7860/sdapi/v1/txt2img"

os.makedirs("data/scenebg", exist_ok=True)

# Define the router
router = APIRouter()

# Updated request model for the `/scenebg` endpoint
class SceneBGRequest(BaseModel):
    scene_id: str
    scene_description: str
    scene_name: str


def detect_language_and_load_model(sentence):
    """Detects the language and loads the appropriate spaCy model."""
    if any("\u4e00" <= char <= "\u9fff" for char in sentence):
        return nlp_cn
    return nlp_en


def extract_nouns_ignoring_names_and_capitalized(sentence):
    """Extract nouns while ignoring names and capitalized words."""
    nlp = detect_language_and_load_model(sentence)
    doc = nlp(sentence)

    # Collect named entities of type PERSON
    character_names = {ent.text for ent in doc.ents if ent.label_ == "PERSON"}

    # Extract nouns
    nouns = [
        token.text for token in doc
        if token.pos_ in ["NOUN", "PROPN"]
        and token.text not in character_names  # Exclude names
        and (nlp.lang == "zh" or not token.text[0].isupper())  # Ignore capitalization for English
    ]

    return nouns


@router.post("/scenebg")
async def generate_scene_background(request: SceneBGRequest):
    """Generate a scene background based on description."""
    file_name = f"data/scenebg/{request.scene_id.strip() if request.scene_id.strip() else 'default'}.png"

    # Check if the file already exists
    if os.path.exists(file_name):
        print(f"File {file_name} already exists. Returning the existing URL.")
        return {"message": "Scene background already exists", "file_url": f"/get-scenebg?scene_id={request.scene_id.strip()}"}

    # Extract nouns from the description
    extracted_nouns = extract_nouns_ignoring_names_and_capitalized(request.scene_description)

    scene_name = request.scene_name if request.scene_name.strip() else "Unnamed scene"

    # Construct the prompt
    prompt = (
        f"{scene_name}, {extracted_nouns}, best render, no character, no people, high res, "
        f"best resolution, 4k, 8k, super great render, ((photorealistic)), great colors, "
        f"((yosemite)), ((el captain))"
    )
    negative_prompt = (
        "bad resolution, 1080p, awkward, distorted, weird, matching prompts, bad color, "
        "weird color spots, distorted color, weird color, blended color"
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
        "width": 1024,
        "height": 1024,
        "restore_faces": False,
        "tiling": False,
        "do_not_save_samples": False,
        "do_not_save_grid": False,
        "eta": 0.0,
        "denoising_strength": 0.0,
        "override_settings": {
            "sd_model_checkpoint": "landscapeRealistic_v20WarmColor.safetensors [ca2e3bd9f9]"
        },
        "override_settings_restore_afterwards": True,
        "enable_hr": False,
        "hr_scale": 2,
        "hr_upscaler": "R-ESRGAN 4x+",
        "hr_second_pass_steps": 20,
        "sampler_index": "Euler"
    }

    try:
        # Call the Stable Diffusion API
        response = requests.post(SD_API_URL, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Error contacting Stable Diffusion API:", e)
        raise HTTPException(status_code=500, detail="Failed to contact Stable Diffusion API")

    # Process the response
    result = response.json()
    if "images" in result:
        image_data = base64.b64decode(result["images"][0])
        with open(file_name, "wb") as f:
            f.write(image_data)
        return {"message": "Scene background generated successfully",
                "file_url": f"/get-scenebg?scene_id={request.scene_id.strip()}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to generate scene background")


@router.get("/get-scenebg")
async def get_scene_background(scene_id: str = Query(..., description="Scene ID to fetch the background")):
    """Fetch the generated scene background."""
    file_name = f"data/scenebg/{scene_id.strip()}.png"

    if not os.path.exists(file_name):
        raise HTTPException(status_code=404, detail="Background not found")

    return FileResponse(file_name, media_type="image/png")
