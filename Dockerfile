FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/Miniconda3-latest-Linux-x86_64.sh && \
    bash /root/Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3

ENV PATH="/root/miniconda3/bin:${PATH}"

RUN conda init bash && \
    echo "conda activate retroinfer" >> ~/.bashrc

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -n retroinfer python=3.10 -y

RUN apt-get install ca-certificates gpg wget -y

RUN test -f /usr/share/doc/kitware-archive-keyring/copyright || wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null

RUN apt-get update

RUN apt-get install kitware-archive-keyring -y

RUN apt-get install cmake -y

RUN apt-get install libopenblas-dev -y

RUN apt-get install libgflags-dev -y

SHELL ["conda", "run", "-n", "retroinfer", "/bin/bash", "-c"]

RUN conda install -y mkl

RUN conda install -c conda-forge libstdcxx-ng -y

RUN python -m pip install pip==25.0

WORKDIR /root

RUN git clone https://github.com/wennitao/RetrievalAttention.git

WORKDIR /root/RetrievalAttention

RUN ls

RUN pip install -r requirements.txt

RUN pip install flash-attn==2.7.3 --no-build-isolation

RUN pip install flashinfer-python==0.2.4 -i https://flashinfer.ai/whl/cu124/torch2.5/

RUN pip install git+https://github.com/Starmys/flash-attention.git@weighted

WORKDIR /root/RetrievalAttention/library

RUN git clone https://github.com/NVIDIA/cutlass.git

WORKDIR /root/RetrievalAttention/library/retroinfer