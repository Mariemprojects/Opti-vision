from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from image_classification import main as image_classification
from Object_Detection_Model import main as object_detection
from Documents_Analyzes import main as document_analysis

app = FastAPI(title="AI Vision Service")

# ✅ Add CORS middleware here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # during dev; in prod, use your website URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers for all services
app.include_router(image_classification.router,  tags=["Image Classification"])
app.include_router(object_detection.router, tags=["Object Detection"])
app.include_router(document_analysis.router, tags=["Document Analysis"])


# In your main file (app.py)

@app.get("/")
def read_root():
    """
    A simple health check endpoint.
    """
    return {"message": "AI Vision Service is running! Check /docs for endpoints."}