import sys
import numpy as np
import caffe


def readDeepNet(trainNet_path, caffemodel_path, proj_root):
    caffe.set_mode_cpu()
    net = caffe.Net(trainNet_path,
                    caffemodel_path,
                    caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(meanFile_path).mean(1).mean(1))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB, if you are working on gray model, comment this line

    return (net, transformer)


def extract_feature(net, transformer, filelist, dimension, imageSize):
    file = open(filelist)

    nan = np.empty(shape=[0, dimension])
    _label = np.empty(shape=[0, 1])
    looptimes = 0
    while 1:
        line = file.readline()
        if not line:
            break
        pass
        spaceIndex = line.find(" ")
        imagePath = line[0:spaceIndex]
        thisLabel = int(line[spaceIndex + 1:len(line)])
        net.blobs['data'].reshape(1, 3, imageSize, imageSize)
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(imagePath, color=True))
        out = net.forward()
        feat = net.blobs['fc160'].data[0] # 'fc160' is the layer name in caffemodel, edit this arguments base on your own model

        nan = np.vstack((nan, feat))
        _label = np.vstack((_label, thisLabel))
    return (nan, _label)


def print_help():
    print "argv: [1]trainNet path; [2]caffemodel path; [3]meanFile path \n"
    print "[4]filelist path; [5]dimension of feature vector; [6]imageSize; [7]output path\n"
    exit


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print_help()
    else:
        trainNet_path = sys.argv[1]
        caffemodel_path = sys.argv[2]
        meanFile_path = sys.argv[3]
        filelist = sys.argv[4]
        dimension = int(sys.argv[5])
        imageSize = int(sys.argv[6])
        (net, transformer) = readDeepNet(trainNet_path, caffemodel_path, meanFile_path)
        (feature, label) = extract_feature(net, transformer, filelist, dimension, imageSize)
        outputFileName = sys.argv[7]

        print(feature.shape)
        np.save(outputFileName, feature)

