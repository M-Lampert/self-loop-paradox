FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
# Update packages
RUN apt update
RUN apt install -y git

# Install packages with requirements.txt
COPY requirements.txt .
RUN pip3 --disable-pip-version-check --no-cache-dir install -r requirements.txt
