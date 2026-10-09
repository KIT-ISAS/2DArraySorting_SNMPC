FROM tensorflow/tensorflow:2.16.1-gpu

ENV DEBIAN_FRONTEND=noninteractive

# Using apt-key is deprecated and will be removed in a future release. Likewise, the command below is not working
# anymore. There are two approaches to solve this issue:
#
# 1. (recommended by NVIDIA):
# see https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
# RUN dpkg -i cuda-keyring_1.1-1_all.deb && \
# apt-get update
# However, for me currently throws an error while executing the second to last command
#
# 2. (general recommended way to add keys):
# see https://askubuntu.com/questions/1286545/what-commands-exactly-should-replace-the-deprecated-apt-key
# RUN apt-key del 7fa2af80 && \
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
# gpg --no-default-keyring --keyring ./temp-keyring.gpg --import 3bf863cc.pub && \
# gpg --no-default-keyring --keyring ./temp-keyring.gpg --export --output /etc/apt/keyrings/3bf863cc.gpg && \
# rm temp-keyring.gpg && \
# echo 'deb [signed-by=/etc/apt/keyrings/3bf863cc.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /'  > /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list && \
# apt-get update
# However, for me currently does not solve the problem of unrerognized keys
#
# List of all keys
# https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && \
apt-get update && \
apt-get install -y --no-install-recommends python3.11-tk && \
pip3 install --upgrade pip && \
pip3 install matplotlib==3.7.0  # note that older versions, e.g., 3.2.1, may cause warning when use latex fonts, but
# newer versions may cause problems with display forwarding from gpu server, which works with 3.2.1 (thus, choose the
# version carefully). Update: 3.2.1. won't work with the above settings.

RUN pip3 install --upgrade pip && \
pip3 install tf-keras==2.16.0 && \
pip3 install --upgrade tensorflow-probability==0.24.0 && \
pip3 install scikit-learn==1.5.0 && \
pip3 install pandas==2.2.2 &&\
pip3 install pyoselm==1.0.1  # installing tf-keras is explicitly required for tfp:0.24.0 (see release notes)

RUN pip3 install --upgrade pip && \
pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 && \
pip3 install gymnasium==0.29.1 && \
pip3 install gymnasium[classic-control] && \
pip3 install moviepy==2.2.1 && \
pip3 install stable-baselines3==2.1.0

# See https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get update && \
apt-get install ffmpeg libsm6 libxext6  -y  && \
pip3 install --upgrade pip && \
pip3 install opencv-python==4.5.5.64 && \
pip3 install psutil==6.0.0

RUN pip3 install --upgrade pip && \
pip3 install particles==0.4

RUN ldconfig
