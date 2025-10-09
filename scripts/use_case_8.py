from __future__ import annotations

import asyncio
import logging
import os
from io import BytesIO
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np
from hypha_rpc import connect_to_server, login
from hypha_rpc.utils.schema import schema_function
from pydantic import Field

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from hypha_rpc.rpc import RemoteService


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def dataset_from_model(
    model: dict[str, Any],
    artifact_manager: RemoteService,
) -> dict[str, Any]:
    """Extract dataset information from a model dictionary."""
    model_manifest = model.get("manifest", {})

    if ("inputs" in model_manifest) and ("test_tensor" in model_manifest["inputs"][0]):
        file_paths = [model_manifest["inputs"][0]["test_tensor"]["source"]]
    elif "test_inputs" in model_manifest:
        file_paths = [model_manifest["test_inputs"][0]]
    else:
        file_paths = []

    file_urls = [
        await artifact_manager.get_file(
            artifact_id=model.get("id"),
            file_path=file_path,
        )
        for file_path in file_paths
    ]

    return {
        "type": "dataset",
        "name": f"Dataset for {model_manifest.get('name', 'unknown model')}",
        "tags": model_manifest.get("tags", []),
        "description": f"Dataset of {model_manifest.get('description', '')}",
        "file_urls": file_urls,
    }


def summarize_model(model: dict[str, Any]) -> dict[str, Any]:
    """Create a summary string for a model."""
    model_manifest = model.get("manifest", {})

    return {
        "type": "model",
        "id": model.get("id", "unknown id"),
        "name": model_manifest.get("name", "unknown model"),
        "tags": model_manifest.get("tags", []),
        "description": model_manifest.get("description", ""),
    }


def make_search_datasets(
    artifact_manager: RemoteService,
) -> Callable[..., Coroutine[Any, Any, list[dict[str, Any]]]]:
    """Create a function to search datasets in the Bioimage Archive."""
    keywords_field = Field(
        default=None,
        description=(
            "Fuzzy search keywords. Results match at least one (not necessarily all)"
            " of the keywords. Leave empty to get all datasets."
        ),
    )

    @schema_function
    async def search_datasets(
        keywords: list[str] | None = keywords_field,
        mode: str = Field(
            default="OR",
            description=(
                'Search mode: "OR" (default) returns models matching any of the'
                ' keywords. "AND" returns models matching all of the keywords.'
            ),
        ),
        items_per_page: int = Field(
            default=25,
            description="Number of datasets to return per page.",
        ),
        page_num: int = Field(
            default=1,
            description="Page number to return.",
        ),
    ) -> list[dict[str, Any]]:
        """Fuzzy search for datasets in the BioImage Archive using keywords.

        If you don't find any relevant results, try broadening your search or leaving
        the keywords empty to get all datasets.

        Returns a list of datasets from the BioImage Archive that match
        any (mode: "OR") or all (mode: "AND") of the keywords, depending on mode.
        """
        page_offset = (page_num - 1) * items_per_page

        am_response = await artifact_manager.list(
            parent_id="bioimage-io/bioimage.io",
            keywords=keywords,
            stage=False,
            mode=mode,
            filters={"type": "model"},
            limit=items_per_page,
            offset=page_offset,
            pagination=True,
        )

        return [
            await dataset_from_model(model, artifact_manager)
            for model in am_response["items"]
        ]

    return search_datasets


def make_search_models(
    artifact_manager: RemoteService,
) -> Callable[..., Coroutine[Any, Any, list[dict[str, Any]]]]:
    """Create a function to search models in the RI-SCALE Model Hub."""
    keywords_field = Field(
        default=None,
        description=("Fuzzy search keywords. Leave empty to get all models."),
    )

    @schema_function
    async def search_models(
        keywords: list[str] | None = keywords_field,
        mode: str = Field(
            default="OR",
            description=(
                'Search mode: "OR" (default) returns models matching any of the'
                ' keywords. "AND" returns models matching all of the keywords.'
            ),
        ),
        items_per_page: int = Field(
            default=25,
            description="Number of items to return per page.",
        ),
        page_num: int = Field(
            default=1,
            description="Page number to return.",
        ),
    ) -> list[dict[str, Any]]:
        """Fuzzy search for models in the RI-SCALE Model Hub using keywords.

        If you don't find any relevant results, try broadening your search or leaving
        the keywords empty to get all models.

        Returns a list of models from the RI-SCALE Model Hub that match
        any (mode: "OR") or all (mode: "AND") of the keywords, depending on mode.
        """
        page_offset = (page_num - 1) * items_per_page

        am_response = await artifact_manager.list(
            parent_id="bioimage-io/bioimage.io",
            keywords=keywords,
            stage=False,
            mode=mode,
            filters={"type": "model"},
            limit=items_per_page,
            offset=page_offset,
            pagination=True,
        )

        return [summarize_model(model) for model in am_response["items"]]

    return search_models


def make_run_model(
    model_runner: RemoteService,
) -> Callable[..., Coroutine[Any, Any, dict]]:
    """Create a function to run an RI-SCALE model on a data file."""

    @schema_function
    async def run_model(
        model_id: str = Field(..., description="ID of the RI-SCALE model"),
        input_file_url: str = Field(
            ...,
            description="A file URL for input data file, e.g. 'test_input_0.npy'",
        ),
    ) -> dict[str, Any]:
        """Run an RI-SCALE model on a data file from URL and return the results."""
        model_alias = model_id.split("/")[1] if "/" in model_id else model_id

        async with httpx.AsyncClient() as client:
            response = await client.get(input_file_url)
            response.raise_for_status()
            dataset_file = np.load(BytesIO(response.content))

        output = await model_runner.infer(
            model_id=model_alias,
            inputs=dataset_file,
        )

        return {
            "input": dataset_file,
            "output": output,
            "status": "success",
        }

    return run_model


async def register_service(
    server: RemoteService,
    service_id: str = "bioimage_runner",
) -> None:
    """Register the RI-SCALE Model Hub runner service on the server."""
    artifact_manager = await server.get_service("public/artifact-manager")
    model_runner = await server.get_service("bioimage-io/model-runner")

    search_models = make_search_models(artifact_manager)
    search_datasets = make_search_datasets(artifact_manager)
    run_model = make_run_model(model_runner)

    description = (
        "Service to search and run AI models from the RI-SCALE Model Hub."
        " If the user doesn't specify workflow, do:"
        " 1. search for datasets using `search_datasets()`,"
        " 2. then find a suitable model using `search_models()`,"
        " 3. then run the model using `run_model()`."
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
