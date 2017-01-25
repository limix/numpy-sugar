#!/usr/bin/env bash

sudo add-apt-repository -y 'deb http://llvm.org/apt/trusty/ llvm-toolchain-trusty-3.9 main' || true
wget -O - http://llvm.org/apt/llvm-snapshot.gpg.key | sudo apt-key add - || true
sudo apt-get update -y || true
sudo apt-get install -qq -y clang-3.9 zlib1g-dev libbz2-dev || true
