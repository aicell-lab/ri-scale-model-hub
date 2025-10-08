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
logger.setLevel(logging.INFO)


def dataset_from_model(model: dict[str, Any]) -> dict[str, Any]:
    """Extract dataset information from a model dictionary."""
    model_manifest = model.get("manifest", {})

    return {
        "type": "dataset",
        "name": f"Dataset for {model_manifest.get('name', 'unknown model')}",
        "tags": model_manifest.get("tags", []),
        "description": model_manifest.get("description", ""),
    }


def make_search_datasets(
    artifact_manager: RemoteService,
) -> Callable[..., Coroutine[Any, Any, list[dict[str, Any]]]]:
    """Create a function to search datasets in the Bioimage Archive."""
    keywords_field = Field(
        default=None,
        description="Search keywords. Leave empty to get all datasets.",
    )

    @schema_function
    async def search_datasets(
        keywords: list[str] | None = keywords_field,
        items_per_page: int = Field(
            default=25,
            description="Number of datasets to return per page.",
        ),
        page_num: int = Field(
            default=1,
            description="Page number to return.",
        ),
    ) -> list[dict[str, Any]]:
        """Search for datasets in the BioImage Archive.

        If you don't find any relevant results, try broadening your search or leaving
        the keywords empty to get all datasets.

        Parameters
        ----------
        keywords: list[str] | None)
            List of keywords to search for. If None,
            returns all datasets.
        items_per_page: int
            Number of items to return per page. Default is 25.
        page_num: int
            Page number to return. Default is 1.

        Returns
        -------
            list[dict[str, Any]]: List of matching datasets from the BioImage Archive.

        """
        page_offset = (page_num - 1) * items_per_page

        am_response = await artifact_manager.list(
            parent_id="bioimage-io/bioimage.io",
            keywords=keywords,
            stage=False,
            filters={"type": "model"},
            limit=items_per_page,
            offset=page_offset,
            pagination=True,
        )

        return [dataset_from_model(model) for model in am_response["items"]]

    return search_datasets


def make_search_models(
    artifact_manager: RemoteService,
) -> Callable[..., Coroutine[Any, Any, list[dict[str, Any]]]]:
    """Create a function to search models in the RI-SCALE Model Hub."""
    keywords_field = Field(
        default=None,
        description="Search keywords. Leave empty to get all items.",
    )

    @schema_function
    async def search_models(
        keywords: list[str] | None = keywords_field,
        items_per_page: int = Field(
            default=25,
            description="Number of items to return per page.",
        ),
        page_num: int = Field(
            default=1,
            description="Page number to return.",
        ),
    ) -> list[dict[str, Any]]:
        """Search for AI models in the RI-SCALE Model Hub.

        If you don't find any relevant results, try broadening your search or leaving
        the keywords empty to get all items.

        Parameters
        ----------
        keywords: list[str] | None)
            List of keywords to search for. If None,
            returns all models.
        items_per_page: int
            Number of items to return per page. Default is 25.
        page_num: int
            Page number to return. Default is 1.

        Returns
        -------
            list[dict[str, Any]]: List of matching models from the RI-SCALE Model Hub.

        """
        page_offset = (page_num - 1) * items_per_page

        am_response = await artifact_manager.list(
            parent_id="bioimage-io/bioimage.io",
            keywords=keywords,
            stage=False,
            filters={"type": "model"},
            limit=items_per_page,
            offset=page_offset,
            pagination=True,
        )

        return am_response["items"]

    return search_models


def make_run_model(
    server: RemoteService,
) -> Callable[..., Coroutine[Any, Any, dict]]:
    """Create a function to run an RI-SCALE model on a dataset."""

    @schema_function
    async def run_model(
        model_id: str = Field(..., description="ID of the RI-SCALE model"),
        dataset_id: str = Field(..., description="ID of the dataset"),
    ) -> dict[str, Any]:
        """Run an RI-SCALE model on a dataset and return the results."""
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

    return run_model


async def register_service(
    server: RemoteService,
    service_id: str = "bioimage_runner",
) -> None:
    """Register the RI-SCALE Model Hub runner service on the server."""
    artifact_manager: RemoteService = await server.get_service(
        "public/artifact-manager",
    )

    search_models = make_search_models(artifact_manager)
    search_datasets = make_search_datasets(artifact_manager)
    run_model = make_run_model(server)

    description = (
        "Service to search and run AI models from the RI-SCALE Model Hub."
        " If the user doesn't specify workflow, do:"
        " 1. search for datasets using `search_datasets()`,"
        " 2. then find a suitable model using `search_models()`,"
        " 3. then run the model using `run_model()`."
        " NOTE: do not provide a dataset as input to `run_model()`."
    )

    await server.register_service(
        {
            "id": service_id,
            "name": "RI-SCALE Model Hub Runner",
            "config": {
                "visibility": "public",
                "require_context": False,
            },
            "description": description,
            "search_datasets": search_datasets,
            "search_models": search_models,
            "run_model": run_model,
        },
    )

    workspace = server.config.workspace
    mcp_url = f"https://hypha.aicell.io/{workspace}/mcp/{service_id}/mcp"
    logger.info("Registered Hypha service at %s", mcp_url)


async def main() -> None:
    """Connect to Hypha server and register the RI-SCALE Model Hub Runner service."""
    server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
    token = os.environ.get("HYPHA_TOKEN")
    if not token:
        token = await login({"server_url": server_url})

    async with connect_to_server({"server_url": server_url, "token": token}) as server:
        await register_service(server)  # type: ignore
        await server.serve()  # type: ignore


if __name__ == "__main__":
    asyncio.run(main())
