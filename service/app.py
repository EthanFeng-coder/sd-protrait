from fastapi import FastAPI
from portrait_api import router as portrait_router
# Create the FastAPI app
app = FastAPI()
app.include_router(portrait_router)
# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Univcor App!"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}
