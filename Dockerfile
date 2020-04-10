FROM nvcr.io/nvidia/pytorch:19.06-py3

# Install COLMAP
RUN apt-get update && apt-get -y install \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libopenblas-dev

RUN git clone https://ceres-solver.googlesource.com/ceres-solver && \
    cd ceres-solver && \
    git checkout 1.14.0 && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j8 && \
    make install

RUN git clone https://github.com/colmap/colmap.git && \
    cd colmap && \
    git checkout 3.5 && \
    mkdir build && \
    cd build && \
    cmake .. -DCUDA_ARCHS="5.2 6.0 6.1 7.0 7.5+PTX" && \
    make -j8 && \
    make install

# Install xtreme-view dependencies
RUN pip install pydensecrf \
    pyquaternion \
    imageio

COPY xtreme-view xtreme-view
WORKDIR xtreme-view

