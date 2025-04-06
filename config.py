# coding:utf8
import warnings
import torch as t
import os


class DefaultConfig(object):
    model = 'ResNet34'
    env = "'default"
    # load_model_path = "checkpoints/alexnet_0401_14:38:32.pth"
    load_model_path = None
    use_gpu = True if t.cuda.is_available() else False  # Automatically detect CUDA availability
    num_workers = 2  # 加载数据时使用的线程数目
    print_freq = 10  # 打印频率

    # Get the absolute path of the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Data paths
    train_image_path = os.path.join(project_root, "data/train/train-images.gz")
    train_label_path = os.path.join(project_root, "data/train/train-labels.gz")
    test_image_path = os.path.join(project_root, "data/test/test-images.gz")
    test_label_path = os.path.join(project_root, "data/test/test-label.gz")

    # Model parameters
    image_size = 28
    num_channels = 1
    pixel_depth = 255
    train_image_nums = 60000
    test_image_nums = 10000

    # Training parameters
    seed = 42
    batch_size = 32  # Reduced batch size to use less memory
    max_epoch = 10
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-5  # L2 regularization

    # Debug and output
    debug_file = os.path.join(project_root, 'tmp/debug')
    result_file = os.path.join(project_root, 'result.csv')

    def parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        # self.device = t.device('cuda') if self.use_gpu else t.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


opt = DefaultConfig()
