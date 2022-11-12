from fastapi import APIRouter

routes = APIRouter()

@routes.get("/health")
async def health_check():
    return {
        "status": "UP"
    }


