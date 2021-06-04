#!/usr/bin/env bash
set -e
CLOUDTOKEN="$1"

echo_time() {
    date +"%R $*"
}

echo_time "Installing Unix dependencies."
apt-get update
apt-get install -y ffmpeg libsm6 libxext6
apt-get install -y libxrender-dev
apt-get install -y tmux

echo_time "Installing Python."
apt-get install -y python3.7 python3.7-dev
ln --force --symbolic /usr/bin/python3.7 /usr/bin/python3
wget https://bootstrap.pypa.io/get-pip.py
apt-get install -y python3-distutils
python3 get-pip.py

echo_time "Setting up pip config"
pip3 config set global.index-url 'https://pypi.apple.com/simple'
pip3 config set global.extra-index-url 'https://pypi.python.org/simple'

echo_time "Install additional libraries for NSVF"
pip3 install --upgrade awscli
pip3 install --upgrade tensorboard
pip3 install --upgrade turibolt

echo_time "Install install NSVF"
git clone https://github.com/rocksat/nsvf.pytorch.git --recursive
cd nsvf.pytorch
pip3 install torch==1.4.0 torchvision==0.5.0
pip3 install -r requirements.txt
python3 setup.py build_ext --inplace


echo_time "Setup step done"