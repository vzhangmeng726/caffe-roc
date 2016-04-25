#!/bin/sh

name=whatyouwant

trainNet=deepid2_50x50_rgb.prototxt
caffemodel=deepid2_50x50_rgb_w0.caffemodel
meanFile=rgb_50x50_mean.npy
filelist=lfw_filelist.txt
dimension=160
imagesize=50
outputPath=$name



# extract the deep feature from data by caffmodels and trainNet that you are using

python extraction.py model/$trainNet model/caffemodel/$caffemodel model/image_mean/$meanFile filelist/$filelist $dimension $imagesize feature/${name}

echo "extractin done.."


