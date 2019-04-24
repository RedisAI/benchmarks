import docker
client = docker.from_env()

docker_image = 'tensorwerk/raibenchmarks:tfserving-optim-cpu'
a = client.images.pull(docker_image)
b = client.containers.run(
    docker_image,
    auto_remove=True)
