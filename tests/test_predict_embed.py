import sys
from PIL import Image

sys.path.append('./')

from models.mtcnn import MTCNN  # noqa
from models.inception_resnet_v1 import InceptionResnetV1  # noqa


if __name__ == "__main__":
    mtcnn = MTCNN()
    # print('mtcnn:{}'.format(mtcnn))
    img_path = './data/test_images/angelina_jolie/1.jpg'
    save_path = './data/test_images/angelina_jolie/1_cropped.jpg'
    resnet = InceptionResnetV1(pretrained='vggface2', classify=False).eval()
    # print('resnet:{}'.format(resnet))
    img = Image.open(img_path)
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img, save_path=save_path)
    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = resnet(img_cropped.unsqueeze(0)).view(-1).detach().numpy()
    print('img_embedding len:{}：example：{}'.format(len(img_embedding), img_embedding))
    # img_embedding len:512：example：[-6.37354422e-03 -2.62405071e-02 ……
