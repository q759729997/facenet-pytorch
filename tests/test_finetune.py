import sys

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
# from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np

sys.path.append('./')

from models.mtcnn import MTCNN  # noqa
from models.utils import training  # noqa
from models.mtcnn import fixed_image_standardization  # noqa
from models.inception_resnet_v1 import InceptionResnetV1  # noqa


class EarlyStopCallback(object):
    """ 早停止回调类.多少个epoch没有变好就停止训练.
    """

    def __init__(self, patience=10):
        """
        :param int patience: epoch的数量
        """
        super().__init__()
        self.patience = patience
        self.wait = 0
        # epoch计数，用于后续日志输出
        self.epoch_no = 1
        self.max_metric_value = 0

    def on_valid_end(self, metric_value, metric_key='acc'):
        """
        每次执行验证集的evaluation后会调用。

        :param metric_value 指标值。
        :param str metric_key: 指标key。
        :return:
        """
        print('======epoch : {} , early stopping : {}/{}======'.format(self.epoch_no, self.wait, self.patience))
        print('metric_key : {}, metric_value : {}, max_metric_value:{}'.format(metric_key, metric_value, self.max_metric_value))
        self.epoch_no += 1
        # 判断是否超过上次指标
        is_better_eval = False
        if metric_value > self.max_metric_value:
            is_better_eval = True
            self.max_metric_value = metric_value
            self.wait = 0
        else:
            self.wait += 1
        if not is_better_eval:
            # current result is getting worse
            if self.wait >= self.patience:
                print('reach early stopping patience, stop training.')
                raise Exception("Early stopping raised.")


if __name__ == "__main__":
    data_dir = './data/test_images'
    batch_size = 2
    epochs = 20
    workers = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    mtcnn = MTCNN(image_size=160,
                  margin=0,
                  min_face_size=20,
                  thresholds=[0.6, 0.7, 0.7],
                  factor=0.709,
                  post_process=True,
                  device=device)
    # print('mtcnn:{}'.format(mtcnn))
    dataset = datasets.ImageFolder(data_dir,
                                   transform=transforms.Resize((512, 512)))
    dataset.samples = [(p, p.replace(data_dir, data_dir + '_cropped'))
                       for p, _ in dataset.samples]

    loader = DataLoader(dataset,
                        num_workers=workers,
                        batch_size=batch_size,
                        collate_fn=training.collate_pil)

    for i, (x, y) in enumerate(loader):
        mtcnn(x, save_path=y)
        print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')

    # Remove mtcnn to reduce GPU memory usage
    del mtcnn
    resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=len(dataset.class_to_idx)
    ).to(device)
    # print('resnet:{}'.format(resnet))
    # optimizer = optim.Adam(resnet.parameters(), lr=0.001)
    print([name for name, param in resnet.named_parameters()])
    # 微调时，只重新训练输出层参数
    optim_params = [param for name, param in resnet.named_parameters() if name in {'logits.weight', 'logits.bias'}]
    print('optim_params:{}'.format(optim_params))
    optimizer = optim.Adam(optim_params)
    scheduler = MultiStepLR(optimizer, [5, 10])

    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)
    img_inds = np.arange(len(dataset))
    np.random.shuffle(img_inds)
    train_inds = img_inds  # [:int(0.8 * len(img_inds))]
    val_inds = img_inds  # [int(0.8 * len(img_inds)):]

    train_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_inds)
    )
    print('train_loader:{}'.format(len(train_loader)))
    val_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_inds)
    )
    print('val_loader:{}'.format(len(val_loader)))
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {
        'fps': training.BatchTimer(),
        'acc': training.accuracy
    }
    # writer = SummaryWriter()
    # writer.iteration, writer.interval = 0, 10

    print('\n\nInitial')
    print('-' * 10)
    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device
    )
    # 早停止
    early_stop_callback = EarlyStopCallback(patience=5)

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        resnet.train()
        training.pass_epoch(
            resnet, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device
        )

        resnet.eval()
        val_loss, val_metrics = training.pass_epoch(
            resnet, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device
        )
        val_acc = val_metrics['acc']
        try:
            early_stop_callback.on_valid_end(metric_value=val_acc, metric_key='acc')
        except Exception:
            break
    torch.save(resnet, './data/test_data/test_model/vggface2_finetune.pt')
    print('dataset.classes:{}'.format(dataset.classes))
    print('dataset.class_to_idx:{}'.format(dataset.class_to_idx))
    # writer.close()
