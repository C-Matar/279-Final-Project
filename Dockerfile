FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    g++ \
    make \
    python3 \
    python3-pip \
    libhdf5-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip3 install numpy h5py matplotlib

RUN make clean && make

CMD ["bash"]
