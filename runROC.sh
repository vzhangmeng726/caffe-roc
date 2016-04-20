#!bin/sh

name=whatyouwant

pairlist=filelist/lfw_pair.mat
test_data=feature/${name}.npy
thres_s=0
thres_e=1
thres_g=0.05


# run the ROC results

python testROC/test_lfw.py $pairlist $test_data $thres_s $thres_e $thres_g

echo "done"

