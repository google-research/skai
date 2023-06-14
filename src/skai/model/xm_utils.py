def get_docker_instructions():
    return [
        "FROM python:3.10",
        "RUN pip install tensorflow",
        "RUN if ! id 1000; then useradd -m -u 1000 clouduser; fi",

        "ENV L≈ßANG=C.UTF-8",
        "RUN rm -f /etc/apt/sources.list.d/cuda.list",
        "RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -",
        "RUN apt-get update && apt-get install -y git netcat",
        "RUN python -m pip install --upgrade pip",
        "COPY skai/requirements.txt /skai/requirements.txt",
        "RUN python -m pip install -r skai/requirements.txt",
        "COPY skai/ /skai",
        "RUN chown -R 1000:root /skai && chmod -R 775 /skai",
        "WORKDIR /skai",
    ]