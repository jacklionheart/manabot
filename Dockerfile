# Use the official Python 3.12-slim image
FROM python:3.12-slim

# Enable shell debugging (optional)
SHELL ["/bin/bash", "-ex", "-c"]

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

# Upgrade pip and install required pip packages for building
RUN pip install --upgrade pip pybind11

# Set the working directory for our project
WORKDIR /manabot

# Copy the full repository into the image
COPY . /manabot

# Set an environment variable to force verbose makefile output
ENV CMAKE_VERBOSE_MAKEFILE=ON
ENV SKBUILD_VERBOSE=1

# Build managym and install it using editable installs.
# This will invoke scikit-build and run your CMakeLists.txt.
RUN pip install -e managym
RUN pip install -e .

# Debug: List shared libraries to see what was built.
RUN find /manabot -name "*.so" -exec echo "Found shared library: {}" \;

ENV HYDRA_FULL_ERROR=1
CMD ["python3", "manabot/scripts/train.py"]
