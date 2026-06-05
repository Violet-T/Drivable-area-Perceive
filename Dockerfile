FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    lsb-release \
    locales \
    software-properties-common \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
       > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
       ros-humble-desktop \
       ros-humble-cv-bridge \
       ros-humble-image-transport \
       ros-humble-nav-msgs \
       ros-humble-sensor-msgs \
       python3-colcon-common-extensions \
       python3-pip \
       python3-rosdep \
       python3-vcstool \
       python3-opencv \
       git \
       build-essential \
       ffmpeg \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init || true && rosdep update

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url ${TORCH_INDEX_URL} \
    && python3 -m pip install -r /tmp/requirements.txt

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc \
    && echo "[ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]
