# train
epoch = 1000
num_classes = 9
batch_size = 10
device = 'cpu'  # cpu or 'cuda:0'
train_image_path = '/data2/numbers/train/'  # 每个类别一个文件夹, 类别使用数字
valid_image_path = '/data2/numbers/val/'  # 每个类别一个文件夹, 类别使用数字
num_workers = 4  # 加载数据集线程并发数
best_loss = 0.001  # 当loss小于等于该值会保存模型
save_model_iter = 500  # 每多少次保存一份模型
model_output_dir = '/data2/resnet_cls/'
resume = True  # 是否从断点处开始训练
chkpt = '/data2/resnet_cls/best_11.pth'  # 断点训练的模型
lr = 0.0001

# predict
predict_model = '/data2/resnet_cls/best_4.pth'
predict_image_path = '/data2/numbers/test'  # 每个类别一个文件夹, 类别使用数字


image_format = 'png'
