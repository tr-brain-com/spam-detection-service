from fastapi import FastAPI
from api.route import service, health

app = FastAPI()

app.include_router(service.routes, prefix="/service",tags=["service"])
app.include_router(health.routes, prefix="/actuator", tags=["health"])


