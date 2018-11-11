#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$DIR:$DIR/mobilenetv2
touch mobilenetv2/__init__.py
touch mobilenetv2/models/__init__.py
sed -i "s/round(.*)/int(round(inp * expand_ratio))/" mobilenetv2/models/imagenet/mobilenetv2.py