
BASE_DIR="/home/2017025/fledoy01/code/contrast_generation/data/coco"
IMGS_DIR="$BASE_DIR/imgs"
ANNS_DIR="$BASE_DIR/datasets"
horovodrun -np 4 python train.py -p --imgs $IMGS_DIR --anns $ANNS_DIR 

