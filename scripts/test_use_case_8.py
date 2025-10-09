import asyncio
import os

from hypha_rpc import connect_to_server, login
from hypha_rpc.rpc import RemoteService

from .use_case_8 import make_run_model, make_search_datasets, make_search_models


async def test_workflow(server: RemoteService) -> None:
    artifact_manager = await server.get_service("public/artifact-manager")
    model_runner = await server.get_service("bioimage-io/model-runner")
    search_models = make_search_models(artifact_manager)
    search_datasets = make_search_datasets(artifact_manager)
    run_model = make_run_model(model_runner)

    datasets = await search_datasets(keywords=["nuclei"], items_per_page=2, page_num=1)
    dataset = datasets[0]
    print("\n=========Selected dataset========\n")
    print(dataset)

    models = await search_models(keywords=["nuclei"], items_per_page=2, page_num=1)
    model = models[0]
    print("\n=========Selected model=========\n")
    print(model)

    print("\n=========RUNNING...=========\n")
    code_run = """await model_runner.infer(
    model_id="affable-shark",
    inputs=file("test_input_0.npy"),
)"""
    print(code_run)

    print("\n=========Model run result========\n")
    result = await run_model(
        model_id=model["id"],
        input_file_url=dataset["file_urls"][0],
    )
    print(result)


async def main() -> None:
    server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
    token = os.environ.get("HYPHA_TOKEN")
    if not token:
        token = await login({"server_url": server_url})

    async with connect_to_server({"server_url": server_url, "token": token}) as server:
        await login(server)
        await test_workflow(server)  # type: ignore


if __name__ == "__main__":
    asyncio.run(main())
