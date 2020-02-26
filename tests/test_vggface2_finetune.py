import sys
from PIL import Image
import torch

sys.path.append('./')

from models.mtcnn import MTCNN  # noqa


if __name__ == "__main__":
    mtcnn = MTCNN()
    # print('mtcnn:{}'.format(mtcnn))
    img_path = './data/predict_images/02huangxuan/timg.jpg'
    save_path = './data/predict_images/02huangxuan/timg_cropped.jpg'
    # img_path = './data/CASIA-FaceV5/test/009_0.bmp'
    # save_path = './data/CASIA-FaceV5/test/009_0_cropped.bmp'
    resnet = torch.load('./data/test_data/test_model/vggface2_finetune.pt').eval()
    # print('resnet:{}'.format(resnet))
    img = Image.open(img_path)
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img, save_path=save_path)
    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = resnet(img_cropped.unsqueeze(0))
    # Or, if using for VGGFace2 classification
    resnet.classify = True
    img_probs = resnet(img_cropped.unsqueeze(0))
    probs = torch.nn.functional.softmax(img_probs, dim=-1)
    print('probs:{}'.format(probs))
    print('img_probs:{}'.format(probs.max(dim=1)))
    # dataset.classes:['01jay', '02huangxuan', '03liudehua', '04zhangxueyou', '05zengxiaoxian', 'angelina_jolie', 'bradley_cooper', 'kate_siegel', 'paul_rudd', 'shea_whigham']
    # dataset.class_to_idx:{'01jay': 0, '02huangxuan': 1, '03liudehua': 2, '04zhangxueyou': 3, '05zengxiaoxian': 4, 'angelina_jolie': 5, 'bradley_cooper': 6, 'kate_siegel': 7, 'paul_rudd': 8, 'shea_whigham': 9}
