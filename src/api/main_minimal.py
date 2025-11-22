from fastapi import FastAPI
import uvicorn

app = FastAPI(title="AWIS API - Minimal")

@app.get("/")
def root():
    return {"status": "running", "message": "AWIS API is working!"}

@app.get("/health")
def health():
    return {"status": "healthy", "service": "AWIS API"}

if __name__ == "__main__":
    print("="*60)
    print("🚀 Starting MINIMAL AWIS API on http://localhost:8000")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000)