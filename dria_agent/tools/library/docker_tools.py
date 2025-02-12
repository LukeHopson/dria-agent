from dria_agent.tools.tool import tool

try:
    import docker
except ImportError:
    raise ImportError("Please run pip install dria_agent[tools]")


@tool
def list_containers(all: bool = False) -> list:
    """
    List Docker containers.

    :param all: Include stopped containers if True
    :return: List of container information
    """
    client = docker.from_client()
    containers = client.containers.list(all=all)
    return [{"id": c.id, "name": c.name, "status": c.status} for c in containers]


@tool
def create_container(image: str, name: str = None, ports: dict = None) -> dict:
    """
    Create a new Docker container.

    :param image: Docker image name
    :param name: Container name
    :param ports: Port mapping dictionary
    :return: Container information
    """
    client = docker.from_client()
    container = client.containers.run(image, name=name, ports=ports, detach=True)
    return {"id": container.id, "name": container.name}


@tool
def stop_container(container_id: str) -> bool:
    """
    Stop a Docker container.

    :param container_id: Container ID or name
    :return: True if successful
    """
    client = docker.from_client()
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
    client = docker.from_client()
    container = client.containers.get(container_id)
    container.remove(force=force)
    return True


@tool
def list_images() -> list:
    """
    List Docker images.

    :return: List of image information
    """
    client = docker.from_client()
    images = client.images.list()
    return [{"id": img.id, "tags": img.tags} for img in images]


@tool
def pull_image(image_name: str, tag: str = "latest") -> dict:
    """
    Pull a Docker image.

    :param image_name: Image name
    :param tag: Image tag
    :return: Image information
    """
    client = docker.from_client()
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
    client = docker.from_client()
    container = client.containers.get(container_id)
    return container.logs(tail=tail).decode("utf-8")


@tool
def inspect_container(container_id: str) -> dict:
    """
    Inspect a Docker container.

    :param container_id: Container ID or name
    :return: Detailed container information
    """
    client = docker.from_client()
    container = client.containers.get(container_id)
    return container.attrs


@tool
def create_network(name: str, driver: str = "bridge") -> dict:
    """
    Create a Docker network.

    :param name: Network name
    :param driver: Network driver
    :return: Network information
    """
    client = docker.from_client()
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
