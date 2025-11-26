from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from database import Base, engine
from routers import api_router

settings = get_settings()

# Ensure database tables exist
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Legend Processing Backend",
    version="0.1.0",
    description="Backend service for PDF legend extraction, icon detection, and matching.",
)

# Allow frontend origin if provided
cors_origins = [settings.frontend_base_url] if settings.frontend_base_url else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

# Debug: Print all registered routes
print("\n📋 Registered API Routes:")
for route in app.routes:
    if hasattr(route, 'methods') and hasattr(route, 'path'):
        methods = ', '.join(route.methods)
        print(f"  {methods:8} {route.path}")
print()


@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)


