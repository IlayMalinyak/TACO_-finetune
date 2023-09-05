
# TACO FineTune - DETR, YOLOv8, Faster-rcnn

the repo consists of three models - DETR, YOLOv8 and Faster-rcnn. each model can be finetuned on a custom dataset. currently, the models fit to TACO dataset but this can be changed with minimal code




## Usage/Examples

to finetune DETR you can run from the command line:


```bash
python detr/main.py --coco_path <path_to_dataset> --output_dir <path_to_outputs> --num_classes <number of classes>
```
or alternatively: 

```python
import detr.detr_finetune as detr_finetune

detr_finetune.finetune()
```

and change the argument inside the function finetune() (this will call main() with the desired arguments).

among the possible arguments are:

**focal_loss** - to use focal loss instead cross entropy

**alpha_t** - weights for focal_loss/cross_entropy 

**early_stopping** - specify early early stopping

**freeze backbone** - train only transformer and classification head

and many more. the list of all the arguments can be found in detr/main.py

running faster-rcnn is done using the following code:

```python
import rcnn.faster_rcnn as rcnn_finetune

rcnn_finetune.finetune()
```

again, you can control the arguments inside fintune function

running YOLOv8 is currently done using ultralytics api.
first you need to install ultralytics: 

```python
!pip install ultralytics

```

than create a yaml file and simply call:

```python
from ultralytics import YOLO
model = YOLO("yolov8l.yaml")
model.train(data = 'datasets.yaml', epochs=120,patience=10, imgsz=864, batch=4)
```

an example can be seen in the yolo folder.


