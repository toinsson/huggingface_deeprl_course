ARG PARENT_IMAGE
FROM $PARENT_IMAGE

# bypass interaction
ENV TZ=Asia/Dubai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install through apt
RUN \
    apt-get -y update && \
    apt-get -y install \
    python3.8 \
    python3-pip \
    freeglut3-dev \
    mesa-common-dev \
    ffmpeg \
    xvfb \
    swig \
    sudo \
    wget \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# RUN which python
RUN python3 --version
# RUN which pip
RUN pip --version

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
COPY requirements_torch.txt /tmp/

RUN mkdir -p ${CODE_DIR}
RUN \
    pip install -r /tmp/requirements_torch.txt && \
    rm -rf $HOME/.cache/pip
RUN \
    pip install -r /tmp/requirements.txt && \
    rm -rf $HOME/.cache/pip \
ENV PATH=$VENV/bin:$PATH

# capture screen output for replays
COPY entrypoint.sh /tmp/
RUN chmod +x /tmp/entrypoint.sh
ENTRYPOINT ["/tmp/entrypoint.sh"]
CMD /bin/bash
