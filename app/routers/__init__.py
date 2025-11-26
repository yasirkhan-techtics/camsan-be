from fastapi import APIRouter

from routers import files, icons, labels, legends, pages, projects

api_router = APIRouter()

api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(pages.router, prefix="/projects", tags=["pages"])
api_router.include_router(legends.router, prefix="/projects", tags=["legends"])
api_router.include_router(icons.router, prefix="/icons", tags=["icons"])
api_router.include_router(labels.router, prefix="/labels", tags=["labels"])
api_router.include_router(files.router, prefix="/files", tags=["files"])


