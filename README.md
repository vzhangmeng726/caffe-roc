# caffe-roc
a simple script to extract deep feature from caffe model, and test ROC on the LFW database.

This script is for deep feature extraction and evaluating face verification performance on the LFW database, implemented by PyCaffe and bash. 

I’ve provide a deep caffemodel and trainNet in the model folder.

For feature extraction, just run the follow command:
```bash
 sh runExFea.sh

``` 

When feature extraction has done, run:

```bash
 sh runROC.sh

``` 

for performance evaluation.
