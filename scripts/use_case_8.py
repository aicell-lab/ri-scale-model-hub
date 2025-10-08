from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

from hypha_rpc import connect_to_server, login
from hypha_rpc.utils.schema import schema_function
from pydantic import Field

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from hypha_rpc.rpc import RemoteService


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def make_search_bioimage(
    artifact_manager: RemoteService,
) -> Callable[..., Coroutine[Any, Any, list[dict[str, Any]]]]:
    """Create a function to search the BioImage.IO archive."""
    keywords_field = Field(
        default=None,
        description="Search keywords. Leave empty to get all items.",
    )

    @schema_function
    async def search_bioimage(
        keywords: list[str] | None = keywords_field,
    ) -> list[dict[str, Any]]:
        """Search for AI models, datasets, and applications in the BioImage.IO archive."""
        items_per_page = 1000

        return await artifact_manager.list(
            parent_id="bioimage-io/bioimage.io",
            keywords=keywords,
            stage=False,
            limit=items_per_page,
            offset=0,
            pagination=True,
            _rkwargs=True,
        )

    return search_bioimage


def make_run_model_on_dataset(
    server: RemoteService,
) -> Callable[..., Coroutine[Any, Any, dict]]:
    """Create a function to run an AI model on a dataset."""

    @schema_function
    async def run_model_on_dataset(
        model_id: str = Field(..., description="ID of the AI model"),
        dataset_id: str = Field(..., description="ID of the dataset"),
    ) -> dict[str, Any]:
        """Run an AI model on a dataset and return the results."""
        model_runner_service = await server.get_service("bioimage-io/model-runner")
        artifact_manager = await server.get_service(
            "public/artifact-manager",
        )
        dataset_file = await artifact_manager.get_file(
            artifact_id=dataset_id,
            file_path="data.csv",
        )

        return await model_runner_service.infer(
            model_id=model_id,
            input=dataset_file,
        )

    return run_model_on_dataset


async def register_service(
    server: RemoteService,
    service_id: str = "bioimage_runner",
) -> None:
    """Register the BioImage.IO runner service on the server."""
    artifact_manager: RemoteService = await server.get_service(
        "public/artifact-manager",
    )

    search_bioimage = make_search_bioimage(artifact_manager)
    run_model_on_dataset = make_run_model_on_dataset(server)

    await server.register_service(
        {
            "id": service_id,
            "name": "BioImage.IO Runner",
            "config": {
                "visibility": "public",
                "require_context": False,
            },
            "description": "Service to search and run AI models from the BioImage.IO archive",
            "search_bioimage": search_bioimage,
            "run_model_on_dataset": run_model_on_dataset,
        },
    )

    workspace = server.config.workspace
    mcp_url = f"https://hypha.aicell.io/{workspace}/mcp/{service_id}/mcp"
    logger.info("Registered Hypha service at %s", mcp_url)


async def main() -> None:
    """Connect to Hypha server and register the Hypha Datasets service."""
    server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
    token = os.environ.get("HYPHA_TOKEN")
    if not token:
        token = await login({"server_url": server_url})

    async with connect_to_server({"server_url": server_url, "token": token}) as server:
        await register_service(server)  # type: ignore
        await server.serve()  # type: ignore


if __name__ == "__main__":
    asyncio.run(main())
