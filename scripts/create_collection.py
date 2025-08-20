"""Create a new AI model hub collection."""

import asyncio
import logging
import os

from dotenv import load_dotenv
from hypha_rpc import connect_to_server
from hypha_rpc.rpc import RemoteException, RemoteService

logger = logging.getLogger(__name__)


async def create_collection() -> None:
    """Create a new AI model hub collection."""
    load_dotenv()

    workspace = "ri-scale"

    server_config = {
        "server_url": "https://hypha.aicell.io",
        "workspace": workspace,
        "token": os.getenv("HYPHA_TOKEN"),
    }

    async with connect_to_server(server_config) as server:
        artifact_manager: RemoteService = await server.get_service(
            "public/artifact-manager",
        )

        model_hub_manifest = {
            "name": "AI Model Hub",
            "description": "A collection of AI models",
        }

        config = {
            "permissions": {
                "*": "r",
                "@": "r+",
            },
        }

        model_hub_alias = "ai-model-hub"

        try:
            await artifact_manager.create(
                artifact_manager=artifact_manager,
                alias=model_hub_alias,
                type="collection",
                manifest=model_hub_manifest,
                config=config,
            )
        except RemoteException:
            logger.info("Artifact already exists")
            await artifact_manager.edit(
                artifact_id=f"{workspace}/{model_hub_alias}",
                manifest=model_hub_manifest,
                config=config,
            )


if __name__ == "__main__":
    asyncio.run(create_collection())
