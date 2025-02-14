# Use the official Python 3.12-slim image
FROM python:3.12-slim

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    libspdlog-dev \
    libfmt-dev \
    libgtest-dev \
    python3-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install required pip packages for building (e.g. pybind11)
RUN pip install --upgrade pip pybind11

# Set the working directory (adjust if needed)
WORKDIR /manabot

# Copy the full repository into the image
COPY . /manabot

# Optionally, build the C++ extension for managym.
# (If your pyproject.toml is configured to invoke CMake automatically,
# you might be able to skip the manual build step.)
RUN cd managym && \
    rm -rf build && mkdir build && cd build && \
    cmake -DPython_ROOT_DIR=/usr/local \
          -DPython_EXECUTABLE=/usr/local/bin/python3 \
          -DCMAKE_PREFIX_PATH="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
          -G Ninja .. && \
    ninja

# Install managym and the parent package (manabot) in editable mode
RUN pip install -e managym
RUN pip install -e .

CMD ["python3", "manabot/scripts/train.py"]
