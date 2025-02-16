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
    :type all: bool
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
    :type container_id: str
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
    :type image_name: str
    :param tag: The tag of the image to pull (default is "latest").
    :return: A dictionary containing image details:
             - 'id' (str): The unique identifier of the pulled image.
             - 'tags' (List[str]): A list of tags associated with the image.
    """
    client = docker.client.from_env()
    image = client.images.pull(f"{image_name}:{tag}")
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
    :type container_id: str
    :return: A dictionary containing detailed container information.
    """
    client = docker.client.from_env()
    container = client.containers.get(container_id)
    return container.attrs


@tool
def create_network(name: str, driver: str = "bridge") -> dict:
    """
    Create a Docker network.

    :param name: The name of the network to create.
    :type name: str
    :param driver: The network driver to use (default is "bridge").
    :type driver: str, optional
    :return: A dictionary containing network details:
             - 'id' (str): The unique identifier of the created network.
             - 'name' (str): The name of the created network.
    """
    client = docker.client.from_env()
    network = client.networks.create(name, driver=driver)
    return {"id": network.id, "name": network.name}


DOCKER_TOOLS = [
    list_containers,
    create_container,
    stop_container,
    remove_container,
    list_images,
    pull_image,
    get_container_logs,
    inspect_container,
    create_network,
]
