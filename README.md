# tensorflowClassification
train a image classifier

**本session主要包含使用Tensorflow训练图像分类器，包含从自定义数据准备，到训练，到测试，到对单幅图像的测试，整个流程。**
## train eval predict
两种训练图像分类器的方法

方法一：更便于做transfer learning

方法二：一定要注意，网络采用mobilenet的时候直接训练是很难收敛的，具体表现为虽然训练的loss下降，但是在进行eval的时候约为$\frac{1}{n}$，其中n为类别数。
因此如果是训练mobilenet建议用方法一对mobilenet进行retrain。


## create_TFRecords
create_TFRecords is referencing https://github.com/yeephycho/tensorflow_input_image_by_tfrecord
