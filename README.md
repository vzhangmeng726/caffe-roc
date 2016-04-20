# caffe-roc

This is for deep feature extraction and evaluating face verification performance on the LFW database, implemented by PyCaffe and python. 

I’ve provide a deep caffemodel and trainNet in the model folder, more detail can be found at [1]

For feature extraction, just run the follow command:
```bash
 sh runExFea.sh

``` 

When feature extraction has done, run:

```bash
 sh runROC.sh

``` 



for performance evaluation.

## Reference
[1] [Deep Learning Face Representation by Joint  Identification-Verification](http://papers.nips.cc/paper/5416-deep-learning-face-representation-by-joint-identification-verification.pdf) 
