import sys
from PIL import Image

sys.path.append('./')

from models.mtcnn import MTCNN  # noqa
from models.inception_resnet_v1 import InceptionResnetV1  # noqa
from models.utils import face_verify  # noqa


if __name__ == "__main__":
    mtcnn = MTCNN()
    resnet = InceptionResnetV1(pretrained='vggface2', classify=False).eval()
    know_imgs = [
        './data/test_data/mingxing/train/chenglong/4d017614bbd293dc.jpg',
        './data/test_data/mingxing/train/gutianle/3d6fd78e5d9b0703.jpg',
        './data/test_data/mingxing/train/huge/01e31e7e0d162357.jpg',
        './data/test_data/mingxing/train/jindong/0ad09ddfefe51c62.jpg',
    ]
    test_img = './data/test_data/mingxing/test/huge.jpg'
    known_encodings = list()
    for img_id, know_img in enumerate(know_imgs):
        save_path = './data/test_data/img_{}.jpg'.format(img_id)
        img = Image.open(know_img)
        # Get cropped and prewhitened image tensor
        img_cropped = mtcnn(img, save_path=save_path)
        # Calculate embedding (unsqueeze to add batch dimension)
        img_embedding = resnet(img_cropped.unsqueeze(0)).view(-1).detach().numpy()
        print('embed id:{}'.format(img_id))
        known_encodings.append(img_embedding)
    print('已有图像编码完成,known_encodings:{}'.format(len(known_encodings)))
    save_path = './data/test_data/img_{}.jpg'.format('test')
    img = Image.open(test_img)
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img, save_path=save_path)
    # Calculate embedding (unsqueeze to add batch dimension)
    image_to_test_encoding = resnet(img_cropped.unsqueeze(0)).view(-1).detach().numpy()
    # See how far apart the test image is from the known faces
    face_distances = face_verify.face_distance(known_encodings, image_to_test_encoding)
    print(face_distances)
    # [1.0669953  0.9328357  0.75719655 1.1956822 ]
