ARG BASE_IMAGE=nvidia/cudagl:11.3.0-devel
FROM $BASE_IMAGE

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME=user

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y ssh gcc g++ gdb clang rsync tar python sudo git ffmpeg ninja-build locales \
  && apt-get clean \
  && sudo rm -rf /var/lib/apt/lists/*

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion \
  && mkdir /run/sshd

RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash -m ${USER_NAME} && \
    yes password | passwd ${USER_NAME} && \
    usermod -aG sudo ${USER_NAME} && \
    echo "${USER_NAME}  ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/user && \
    chmod 440 /etc/sudoers

USER ${USER_NAME}

RUN git clone https://github.com/Midren/dotfiles /home/${USER_NAME}/.dotfiles && \
    /home/${USER_NAME}/.dotfiles/install-profile ubuntu-cli

RUN git config --global user.email "milromchuk@gmail.com" && \
    git config --global user.name "Roman Milishchuk"

USER root

RUN apt-get update \
  && apt-get install -y software-properties-common curl \
  && add-apt-repository -y ppa:deadsnakes/ppa \ 
  && DEBIAN_FRONTEND=noninteractive apt-get install -y python3.10 python3.10-dev python3.10-venv \
  && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
  && apt-get clean \
  && sudo rm -rf /var/lib/apt/lists/*

RUN sudo update-alternatives --install /usr/bin/python3 python /usr/bin/python3.10 1 \
    && sudo update-alternatives --install /usr/bin/python python3 /usr/bin/python3.10 1

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}/

RUN mkdir /home/${USER_NAME}/rl_sandbox

COPY pyproject.toml /home/${USER_NAME}/rl_sandbox/pyproject.toml
COPY rl_sandbox /home/${USER_NAME}/rl_sandbox/rl_sandbox

RUN cd /home/${USER_NAME}/rl_sandbox \
    && python3.10 -m pip install --no-cache-dir -e . \
    && rm -Rf /home/${USER_NAME}/.cache/pip


