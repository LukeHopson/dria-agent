from dria_agent.agent.tool import tool

try:
    import docker
except ImportError:
    raise ImportError("Please run pip install dria_agent[tools]")


@tool
def list_containers(all: bool = False) -> list:
    """
    List Docker containers.

    :param all: Include stopped containers if True.
    :return: A list of dictionaries, each containing container details:
             - 'id' (str): The container's unique identifier.
             - 'name' (str): The container's name.
             - 'status' (str): The container's status.
    """
    client = docker.client.from_env()
    containers = client.containers.list(all=all)
    return [{"id": c.id, "name": c.name, "status": c.status} for c in containers]


@tool
def create_container(image: str, name: str = None, ports: dict = None) -> dict:
    """
    Create a new Docker container.

    :param image: The name of the Docker image to use.
    :param name: Optional name for the container.
    :param ports: A dictionary mapping container ports to host ports (e.g., {"80/tcp": 8080}).
    :return: A dictionary containing container details:
             - 'id' (str): The unique identifier of the created container.
             - 'name' (str): The assigned name of the container.
    :rtype: dict
    """
    client = docker.client.from_env()
    container = client.containers.run(image, name=name, ports=ports, detach=True)
    return {"id": container.id, "name": container.name}


@tool
def stop_container(container_id: str) -> bool:
    """
    Stop a Docker container.

    :param container_id: The ID or name of the container to stop.
    :return: True if the container was successfully stopped.
    """
    client = docker.client.from_env()
    container = client.containers.get(container_id)
    container.stop()
    return True


@tool
def remove_container(container_id: str, force: bool = False) -> bool:
    """
    Remove a Docker container.

    :param container_id: Container ID or name
    :param force: Force remove running container
    :return: True if successful
    """
    client = docker.client.from_env()
    container = client.containers.get(container_id)
    container.remove(force=force)
    return True


@tool
def list_images() -> list:
    """
    List available Docker images.

    :return: A list of dictionaries, each containing image details:
             - 'id' (str): The unique identifier of the image.
             - 'tags' (List[str]): A list of tags associated with the image.
    """
    client = docker.client.from_env()
    images = client.images.list()
    return [{"id": img.id, "tags": img.tags} for img in images]


@tool
def pull_image(image_name: str, tag: str = "latest") -> dict:
    """
    Pull a Docker image from a registry.

    :param image_name: The name of the image to pull.
    :param tag: The tag of the image to pull (default is "latest").
    :return: A dictionary containing image details:
             - 'id' (str): The unique identifier of the pulled image.
             - 'tags' (List[str]): A list of tags associated with the image.
    """
    client = docker.client.from_env()
    image = client.images.pull(f"{image_name}:{tag}")
    return {"id": image.id, "tags": image.tags}


@tool
def build_image(path: str, tag: str, dockerfile: str = None) -> dict:
    """
    Build a Docker image from a Dockerfile.

    :param path: The path to the directory containing the Dockerfile.
    :param tag: The tag to assign to the built image.
    :param dockerfile: Optional path to the Dockerfile relative to the build context.
    :return: A dictionary containing image details:
             - 'id' (str): The unique identifier of the built image.
             - 'tags' (List[str]): A list of tags associated with the image.
    """
    client = docker.client.from_env()
    kwargs = {}
    if dockerfile:
        kwargs["dockerfile"] = dockerfile
    image, build_logs = client.images.build(path=path, tag=tag, **kwargs)
    return {"id": image.id, "tags": image.tags}


@tool
def get_container_logs(container_id: str, tail: int = 100) -> str:
    """
    Get container logs.

    :param container_id: Container ID or name
    :param tail: Number of lines to return from the end
    :return: Container logs
    """
    client = docker.client.from_env()
    container = client.containers.get(container_id)
    return container.logs(tail=tail).decode("utf-8")


@tool
def inspect_container(container_id: str) -> dict:
    """
    Inspect a Docker container and retrieve detailed information.

    :param container_id: The ID or name of the container to inspect.
    :return: A dictionary containing detailed container information. Dict has:
        - Id: string
        - Created: string ( ISO 8601 date)
        - Path: string
        - Args: list[string]
        - State: dict
        - Image: string
        - ResolvConfPath: string
        - HostnamePath: string
        - HostsPath: string
        - LogPath: string
        - Name: string
        - RestartCount: int
        - Driver: string
        - Platform: string
        - MountLabel: string
        - ProcessLabel: string
        - AppArmorProfile: string
        - HostConfig: dict
    """
    client = docker.client.from_env()
    container = client.containers.get(container_id)
    return container.attrs


@tool
def create_network(name: str, driver: str = "bridge") -> dict:
    """
    Create a Docker network.

    :param name: The name of the network to create.
    :param driver: The network driver to use (default is "bridge").
    :return: A dictionary containing network details:
             - 'id' (str): The unique identifier of the created network.
             - 'name' (str): The name of the created network.
    """
    client = docker.client.from_env()
    network = client.networks.create(name, driver=driver)
    return {"id": network.id, "name": network.name}


@tool
def run_container(
    image: str,
    command: str = None,
    name: str = None,
    detach: bool = True,
    ports: dict = None,
    environment: dict = None,
) -> dict:
    """
    Run a Docker container from an image.

    :param image: The Docker image to run.
    :param command: Optional command to execute in the container.
    :param name: Optional name for the container.
    :param detach: Run the container in detached mode (default is True).
    :param ports: A dictionary mapping container ports to host ports (e.g., {"80/tcp": 8080}).
    :param environment: Environment variables to set in the container.
    :return: A dictionary containing container details:
             - 'id' (str): The unique identifier of the container.
             - 'name' (str): The name of the container.
    """
    client = docker.client.from_env()
    container = client.containers.run(
        image,
        command=command,
        name=name,
        detach=detach,
        ports=ports,
        environment=environment,
    )
    return {"id": container.id, "name": container.name}


@tool
def start_container(container_id: str) -> bool:
    """
    Start an existing Docker container.

    :param container_id: The ID or name of the container to start.
    :return: True if the container was successfully started.
    """
    client = docker.client.from_env()
    container = client.containers.get(container_id)
    container.start()
    return True


@tool
def push_image(image_name: str, tag: str = "latest") -> dict:
    """
    Push a Docker image to a registry.

    :param image_name: The name of the image to push.
    :param tag: The tag of the image to push (default is "latest").
    :return: A dictionary containing the push result:
             - 'result' (str): Output logs or status message from the push operation.
    """
    client = docker.client.from_env()
    result = client.images.push(image_name, tag=tag, stream=False)
    return {"result": result}


@tool
def remove_image(image_name: str, force: bool = False) -> bool:
    """
    Remove a Docker image.

    :param image_name: The name or ID of the image to remove.
    :param force: Force removal of the image (default is False).
    :return: True if the image was successfully removed.
    """
    client = docker.client.from_env()
    client.images.remove(image=image_name, force=force)
    return True


@tool
def create_volume(name: str, driver: str = "local") -> dict:
    """
    Create a Docker volume.

    :param name: The name of the volume to create.
    :param driver: The volume driver to use (default is "local").
    :return: A dictionary containing volume details:
             - 'name' (str): The name of the created volume.
             - 'driver' (str): The volume driver used.
             - 'mountpoint' (str): The mount point of the volume.
    """
    client = docker.client.from_env()
    volume = client.volumes.create(name=name, driver=driver)
    return {
        "name": volume.name,
        "driver": volume.attrs.get("Driver"),
        "mountpoint": volume.attrs.get("Mountpoint"),
    }


@tool
def list_volumes() -> list:
    """
    List Docker volumes.

    :return: A list of dictionaries, each containing volume details:
             - 'name' (str): The name of the volume.
             - 'driver' (str): The volume driver.
             - 'mountpoint' (str): The mount point of the volume.
    :rtype: list
    """
    client = docker.client.from_env()
    volumes = client.volumes.list()
    return [
        {
            "name": v.name,
            "driver": v.attrs.get("Driver"),
            "mountpoint": v.attrs.get("Mountpoint"),
        }
        for v in volumes
    ]


@tool
def remove_volume(name: str, force: bool = False) -> bool:
    """
    Remove a Docker volume.

    :param name: The name of the volume to remove.
    :param force: Force removal of the volume (default is False).
    :return: True if the volume was successfully removed.
    """
    client = docker.client.from_env()
    volume = client.volumes.get(name)
    volume.remove(force=force)
    return True


DOCKER_TOOLS = [
    list_containers,
    create_container,
    stop_container,
    remove_container,
    list_images,
    pull_image,
    build_image,
    get_container_logs,
    inspect_container,
    create_network,
    remove_image,
    run_container,
    list_volumes,
    create_volume,
    remove_volume,
    start_container,
    push_image,
]
