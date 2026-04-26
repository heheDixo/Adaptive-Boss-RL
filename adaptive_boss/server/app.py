# adaptive_boss/server/app.py
"""FastAPI app entry point for the AdaptiveBoss OpenEnv server."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server import create_fastapi_app
except ImportError:
    from core.env_server import create_fastapi_app

from models import BossAction, BossObservation
from server.environment import AdaptiveBossEnvironment


# create_fastapi_app expects a factory callable plus the Action / Observation
# classes — passing the class itself is the canonical "factory" form.
app = create_fastapi_app(
    AdaptiveBossEnvironment,
    BossAction,
    BossObservation,
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
