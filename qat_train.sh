#!/bin/bash

# Default values for parameters
EPOCHS=100
WORKERS=8
DEVICE=0
BATCH=8
DATA="data/visdrone.yaml"
IMG=640
CFG="models/detect/gelan-c.yaml"
WEIGHTS="runs/train/yolov9-c/weights/yolov9-c-converted.pt"
NAME="yolov9-c-qat"
HYP="hyp.scratch-high.yaml"
MIN_ITEMS=0
CLOSE_MOSAIC=15

# Parse input arguments
while [ $# -gt 0 ]; do
  case $1 in
    --epochs)
      EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    --workers)
      WORKERS="$2"
      shift
      shift
      ;;
    --device)
      DEVICE="$2"
      shift
      shift
      ;;
    --batch)
      BATCH="$2"
      shift
      shift
      ;;
    --data)
      DATA="$2"
      shift
      shift
      ;;
    --img)
      IMG="$2"
      shift
      shift
      ;;
    --cfg)
      CFG="$2"
      shift
      shift
      ;;
    --weights)
      WEIGHTS="$2"
      shift
      shift
      ;;
    --name)
      NAME="$2"
      shift
      shift
      ;;
    --hyp)
      HYP="$2"
      shift
      shift
      ;;
    --min-items)
      MIN_ITEMS="$2"
      shift
      shift
      ;;
    --close-mosaic)
      CLOSE_MOSAIC="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
done

# Execute the command with the provided or default parameters
python train_qat.py \
 --workers $WORKERS \
 --device $DEVICE \
 --batch $BATCH \
 --data $DATA \
 --img $IMG \
 --cfg $CFG \
 --weights $WEIGHTS \
 --name $NAME \
 --hyp $HYP \
 --min-items $MIN_ITEMS \
 --epochs $EPOCHS \
 --close-mosaic $CLOSE_MOSAIC
