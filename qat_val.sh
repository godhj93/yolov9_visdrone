python val_qat.py \
    --data data/visdrone.yaml \
    --img 640 \
    --batch 1 \
    --conf 0.001 \
    --iou 0.7 \
    --device 1 \
    --weights 'runs/train/yolov9-c/weights/yolov9-c-converted.pt' \
    --name yolov9_c_c_640_val \
    --task val
