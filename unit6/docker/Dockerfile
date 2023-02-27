ARG PARENT_IMAGE
# ARG USE_GPU
# FROM nvidia/cuda:12.0.0-base-ubuntu22.04
FROM $PARENT_IMAGE

# install through apt
RUN \
    apt-get -y update && \
    apt-get -y install \
    python3 \
    python3-pip \
    ffmpeg \
    xvfb \
    sudo \
    wget \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install git lfs
RUN \
    wget https://github.com/git-lfs/git-lfs/releases/download/v3.3.0/git-lfs-linux-amd64-v3.3.0.tar.gz \
    && pwd && ls \
    && tar -xf git-lfs-linux-amd64-v3.3.0.tar.gz \
    && cd git-lfs-3.3.0 \
    && chmod 700 install.sh && ./install.sh

# install python dependencies
ENV CODE_DIR /root/code
ENV VENV /root/venv
COPY requirements.txt /tmp/
RUN \
    mkdir -p ${CODE_DIR} && \
    pip install -r /tmp/requirements.txt && \
    rm -rf $HOME/.cache/pip \
ENV PATH=$VENV/bin:$PATH

# capture screen output for replays
COPY entrypoint.sh /tmp/
RUN chmod +x /tmp/entrypoint.sh
ENTRYPOINT ["/tmp/entrypoint.sh"]
CMD /bin/bash
