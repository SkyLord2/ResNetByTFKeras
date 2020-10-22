# -*- coding: utf-8 -*-
# @Time : 2020/10/22 20:14
# @Author : cds
# @Site : https://github.com/SkyLord2?tab=repositories
# @Email: chengdongsheng@outlook.com
# @File : resnet.py
# @Software: PyCharm

from tensorflow.keras import layers,Model,Sequential

class BasicBlock(layers.Layer):
    expansion=1
    def __init__(self,out_channel,strides=1,downsample=None,**kwargs):
        super(BasicBlock,self).__init__(**kwargs)
        
        self.conv1 = layers.Conv2D(out_channel,kernel_size=3,strides=strides,padding="SAME",use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.conv2 = layers.Conv2D(out_channel,kernel_size=3,strides=1,padding="SAME",use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=0.9,epsilon=1e-5)

        # 下采样函数
        self.downsample = downsample
        self.relu = layers.ReLU()
        self.add = layers.Add()
    def call(self,inputs,training=False):
        identify = inputs
        if(self.downsample is not None):
            identify = self.downsample(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x,training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x,training=training)

        x = self.add([identify,x])
        x = self.relu(x)
        return x

class Bottleneck(layers.Layer):
    expansion = 4
    def __init__(self,out_channel,strides=1,downsample=None,**kwargs):
        super(Bottleneck,self).__init__(**kwargs)
        
        self.conv1 = layers.Conv2D(out_channel,kernel_size=1,use_bias=False,name="conv1")
        self.bn1 = layers.BatchNormalization(momentum=0.9,epsilon=1e-5, name="conv1/BatchNorm")

        self.conv2 = layers.Conv2D(out_channel,kernel_size=3,strides=strides,padding="SAME",use_bias=False,name="conv2")
        self.bn2 = layers.BatchNormalization(momentum=0.9,epsilon=1e-5,name="conv2/BatchNorm")

        self.conv3 = layers.Conv2D(out_channel*self.expansion,kernel_size=1,use_bias=False,name="conv3")
        self.bn3 = layers.BatchNormalization(momentum=0.9,epsilon=1e-5,name="conv3/BatchNorm")

        self.relu = layers.ReLU()
        self.downsample = downsample
        self.add = layers.Add()
    def call(self,inputs,training=False):
        identity = inputs
        if(self.downsample is not None):
            identity = self.downsample(inputs)
        
        x = self.conv1(inputs)
        x = self.bn1(x,training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x,training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x,training=training)

        x = self.add([identity,x])
        x = self.relu(x)

        return x

def _make_layer(block,in_channel,channel,block_num,name,strides=1):
    downsample = None
    if(strides!=1 or in_channel != channel*block.expansion):
        downsample = Sequential([
                                #layers.Conv2D(channel*block.expansion,kernel_size=1,padding="SAME",use_bias=False,name="conv1"),
                                layers.Conv2D(channel*block.expansion,kernel_size=1,strides=strides,use_bias=False,name="conv1"),
                                layers.BatchNormalization(momentum=0.9,epsilon=1.001e-5,name="BatchNorm")],name="shortcut")
    
    layer_list = []
    layer_list.append(block(channel,strides,downsample,name="unit_1"))

    for index in range(1,block_num):
        layer_list.append(block(channel,name="unit_"+str(index+1)))

    return Sequential(layer_list,name=name)

def _resnet(block,blocks_num,im_width=224,im_height=224,channel=3,num_classes=1000,include_top=True):
    input_image = layers.Input(shape=(im_height,im_width,channel),dtype="float32")
    x = layers.Conv2D(filters=64,kernel_size=7,strides=2,padding="SAME",use_bias=False,name="conv1")(input_image)
    x = layers.BatchNormalization(momentum=0.9,epsilon=1e-5,name="conv1/BatchNorm")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3,strides=2,padding="SAME")(x)

    print("-----------------------------block_1-------------------------------------")
    print("\ndata shape:", x.shape)
    x = _make_layer(block, x.shape[-1], 64, blocks_num[0], name="block_1")(x)
    print("-----------------------------block_2-------------------------------------")
    print("\ndata shape:", x.shape)
    x = _make_layer(block, x.shape[-1], 128, blocks_num[1], strides=2, name="block_2")(x)
    print("-----------------------------block_3-------------------------------------")
    print("\ndata shape:", x.shape)
    x = _make_layer(block, x.shape[-1], 256, blocks_num[2], strides=2, name="block_3")(x)
    print("-----------------------------block_4-------------------------------------")
    print("\ndata shape:", x.shape)
    x = _make_layer(block, x.shape[-1], 512, blocks_num[3], strides=2, name="block_4")(x)

    if(include_top):
        x = layers.GlobalAvgPool2D()(x)
        x = layers.Dense(num_classes,name="logits")(x)
        predict = layers.Softmax()(x)
    else:
        predict = x
    model = Model(inputs=input_image,outputs=predict)
    
    return model

def resnet18(im_width=224,im_height=224,channel=3,num_classes=1000):
    return _resnet(BasicBlock,[2,2,2,2],im_width, im_height,channel,num_classes)