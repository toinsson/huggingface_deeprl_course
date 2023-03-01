ARG PARENT_IMAGE
FROM $PARENT_IMAGE

# bypass interaction
ENV TZ=Asia/Dubai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install python
RUN \
    apt-get -y update && \
    apt-get -y install \
    python3.8 python3-pip sudo wget git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install git lfs
RUN \
    wget https://github.com/git-lfs/git-lfs/releases/download/v3.3.0/git-lfs-linux-amd64-v3.3.0.tar.gz \
    && pwd && ls \
    && tar -xf git-lfs-linux-amd64-v3.3.0.tar.gz \
    && cd git-lfs-3.3.0 \
    && chmod 700 install.sh && ./install.sh

# install rest
RUN \
    apt-get -y update && \
    apt-get -y install \
    build-essential zlib1g-dev libsdl2-dev libjpeg-dev nasm tar libbz2-dev \
    libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev libopenal-dev timidity \
    libwildmidi-dev unzip ffmpeg libboost-all-dev liblua5.1-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install python dependencies
ENV CODE_DIR /root/code
ENV VENV /root/venv

COPY requirements.txt /tmp/

RUN mkdir -p ${CODE_DIR}
RUN \
    pip install -r /tmp/requirements.txt && \
    rm -rf $HOME/.cache/pip \
ENV PATH=$VENV/bin:$PATH

