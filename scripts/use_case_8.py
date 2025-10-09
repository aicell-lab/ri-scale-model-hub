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

        The dataset dictionary has keys:
        - 'type': always 'dataset'
        - 'name': name of the dataset
        - 'tags': list of tags associated with the dataset
        - 'description': description of the dataset
        - 'file_urls': list of file URLs for the dataset
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
        The model dictionary has keys:
        - 'type': always 'model'
        - 'id': the model ID
        - 'name': name of the model
        - 'tags': list of tags associated with the model
        - 'description': description of the model
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


def squeeze_to_image(
    arr: np.ndarray,
    ref_spatial_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return a 2D view from an ndarray for display.

    If ``ref_spatial_shape`` (H, W) is provided, try to preserve those spatial
    dimensions by selecting/aggregating other axes accordingly. This helps when
    output tensors permute or add axes compared to the input.

    Heuristics (when no reference provided):
    - (H, W): return as-is
    - (C, H, W): first channel
    - (N, C, H, W): first batch, first channel
    - (N, H, W, C): first batch, mean over channels (or first if single)
    - (H, W, C): mean over channels (or single if 1)
    - (Z, H, W): max projection over Z
    For higher dims, iteratively max-project until 2D.
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        return a

    # If a reference spatial shape is given, try to select axes that match it.
    if ref_spatial_shape is not None and a.ndim >= 2:
        H, W = ref_spatial_shape

        # Find candidate axis pairs whose sizes match H and W (order-free).
        axes = list(range(a.ndim))
        best_pair: tuple[int, int] | None = None
        best_score = -1
        for i in axes:
            for j in axes:
                if i >= j:
                    continue
                si, sj = a.shape[i], a.shape[j]
                # Match in either order; allow square cases naturally
                if (si == H and sj == W) or (si == W and sj == H):
                    # Prefer trailing axes (higher indices) heuristically
                    score = i + j
                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)

        if best_pair is not None:
            i, j = best_pair
            # Reduce all other axes. Process from highest to lowest index to
            # avoid reindexing complexity while dimensions drop.
            reduce_axes = [ax for ax in axes if ax not in (i, j)]
            reduce_axes.sort(reverse=True)
            for ax in reduce_axes:
                length = a.shape[ax]
                if length <= 8:
                    # Likely channel/small categorical axis: average if >1 else squeeze
                    if length == 1:
                        a = np.take(a, 0, axis=ax)
                    else:
                        a = a.mean(axis=ax)
                else:
                    # Likely Z/Time/Batch: use max projection
                    a = a.max(axis=ax)

            # Now we should have 2D with axes corresponding to i and j (possibly swapped)
            if a.ndim == 2:
                # Ensure order (H, W)
                if a.shape == (H, W):
                    return a
                if a.shape == (W, H):
                    return a.T
                # If the shape changed slightly due to reductions, fall back
                # to default heuristics below.

    # Default heuristics
    if a.ndim == 3:
        # (C, H, W) or (Z, H, W) or (H, W, C)
        if a.shape[0] <= 4 and a.shape[0] != a.shape[-1]:
            return a[0]  # (C, H, W)
        if a.shape[-1] <= 4 and a.shape[-1] != a.shape[0]:
            return a[..., 0] if a.shape[-1] == 1 else a.mean(axis=-1)  # (H, W, C)
        return a.max(axis=0)  # (Z, H, W)

    if a.ndim == 4:
        # (N, C, H, W) or (N, H, W, C)
        if a.shape[1] <= 8 and a.shape[1] != a.shape[-1]:
            return a[0, 0]  # (N, C, H, W)
        if a.shape[-1] <= 8:
            first = a[0]  # (N, H, W, C)
            return first[..., 0] if first.shape[-1] == 1 else first.mean(axis=-1)
        return squeeze_to_image(a.max(axis=0))  # fallback
    # Generic fallback
    while a.ndim > 2:
        a = a.max(axis=0)
    return a


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize a numeric array to uint8 [0, 255] for display."""
    x = np.asarray(img)
    if x.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.uint8)
    vmin = x[finite].min()
    vmax = x[finite].max()
    if vmax <= vmin:
        return np.zeros_like(x, dtype=np.uint8)
    scaled = (x - vmin) / (vmax - vmin)
    scaled[~finite] = 0
    return np.clip((scaled * 255.0).round(), 0, 255).astype(np.uint8)


def image_from_arr(
    arr: np.ndarray,
    ref_spatial_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Display a single ndarray as an image using matplotlib.

    Squeezes to 2D and displays with suitable scaling. Returns the Figure.
    """
    img2d = squeeze_to_image(arr, ref_spatial_shape=ref_spatial_shape)
    if np.issubdtype(img2d.dtype, np.floating):
        return img2d
    return normalize_to_uint8(img2d)


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
        """Run an RI-SCALE model on a data file from URL and return the results.

        Returns a dictionary with keys:
        - 'input': the input ndarray
        - 'output': the output ndarray
        - 'input_image': a 2D image view of the input. Perfect for matplotlib imshow.
        - 'output_image': a 2D image view of the output. Perfect for matplotlib imshow.
        - 'status': 'success' if the model ran successfully
        """
        model_alias = model_id.split("/")[1] if "/" in model_id else model_id

        async with httpx.AsyncClient() as client:
            response = await client.get(input_file_url)
            response.raise_for_status()
            dataset_file = np.load(BytesIO(response.content))

        output = await model_runner.infer(
            model_id=model_alias,
            inputs=dataset_file,
        )

        input_image = image_from_arr(dataset_file)
        first_output = next(iter(output.values()))
        # Use input image spatial shape as a reference to select the correct axes
        # when squeezing the output tensor to 2D for visualization.
        output_image = image_from_arr(
            first_output,
            ref_spatial_shape=(
                int(input_image.shape[0]),
                int(input_image.shape[1]),
            ),
        )

        return {
            "input": dataset_file,
            "output": output,
            "input_image": input_image,
            "output_image": output_image,
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
