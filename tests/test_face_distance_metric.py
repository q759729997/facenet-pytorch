import os
import sys
from PIL import Image

sys.path.append('./')

from models.mtcnn import MTCNN  # noqa
from models.inception_resnet_v1 import InceptionResnetV1  # noqa
from models.utils import face_verify  # noqa


if __name__ == "__main__":
    """计算距离评测"""
    mtcnn = MTCNN()
    resnet = InceptionResnetV1(pretrained='vggface2', classify=False).eval()
    img_labels = ['chenglong', 'dongxuan', 'guanzhilin', 'gulinazha', 'gutianle', 'huge', 'jindong', 'jingtian', 'lilianjie', 'liming', 'linjunjie', 'liudehua', 'sunli', 'tongliya', 'yangmi', 'zhangmin', 'zhangxueyou', 'zhoujielun', 'zhourunfa', 'zhouxingchi']
    # 已有图像
    known_img_path = './data/test_data/mingxing/train/'
    known_imgs = list()
    known_img_label_dict = dict()  # 图片序号对应标签
    known_img_id = 0
    for img_label in img_labels:
        temp_img_path = os.path.join(known_img_path, img_label)
        for file in os.listdir(temp_img_path):
            temp_img_file_name = os.path.join(temp_img_path, file)
            known_imgs.append(temp_img_file_name)
            known_img_label_dict[known_img_id] = img_label
            known_img_id += 1
    print('known_imgs len:{},example:{}'.format(len(known_imgs), known_imgs[:5]))
    # 测试图像读取
    test_img_path = './data/test_data/mingxing/test/'
    test_imgs = list()
    for img_label in img_labels:
        temp_img_path = os.path.join(test_img_path, img_label + '.jpg')
        test_imgs.append(temp_img_path)
    print('test_imgs len:{},test_imgs:{}'.format(len(test_imgs), test_imgs[:5]))
    # 已有图像编码
    known_encodings = list()
    for img_id, known_img in enumerate(known_imgs):
        print(known_img)
        save_path = './data/test_data/img_{}.jpg'.format(img_id)
        img = Image.open(known_img)
        # Get cropped and prewhitened image tensor
        img_cropped = mtcnn(img, save_path=save_path)
        # Calculate embedding (unsqueeze to add batch dimension)
        img_embedding = resnet(img_cropped.unsqueeze(0)).view(-1).detach().numpy()
        print('embed id:{}'.format(img_id))
        known_encodings.append(img_embedding)
    print('已有图像编码完成')
    # 测试图像编码
    test_encodings = list()
    for img_id, known_img in enumerate(test_imgs):
        print(known_img)
        save_path = './data/test_data/img_{}.jpg'.format(img_id)
        img = Image.open(known_img)
        # Get cropped and prewhitened image tensor
        img_cropped = mtcnn(img, save_path=save_path)
        # Calculate embedding (unsqueeze to add batch dimension)
        img_embedding = resnet(img_cropped.unsqueeze(0)).view(-1).detach().numpy()
        print('embed id:{}'.format(img_id))
        test_encodings.append(img_embedding)
    print('测试图像编码完成')
    # 进行评测
    right_count = 0
    sum_count = 0
    for test_encoding, img_label in zip(test_encodings, img_labels):
        sum_count += 1
        face_distances = face_verify.face_distance(known_encodings, test_encoding)
        face_distances = face_distances.tolist()
        min_distance = min(face_distances)
        min_index = face_distances.index(min_distance)
        pred_label = known_img_label_dict.get(min_index, 'unk')
        if pred_label == img_label:
            right_count += 1
        print('pred_label:{},\t img_label:{},\t min_distance:{}'.format(pred_label, img_label, min_distance))
    print('{}/{}'.format(right_count, sum_count))
    """
    pred_label:chenglong,	 img_label:chenglong,	 min_distance:0.5847517848014832
    pred_label:dongxuan,	 img_label:dongxuan,	 min_distance:0.5635831952095032
    pred_label:guanzhilin,	 img_label:guanzhilin,	 min_distance:0.5095146298408508
    pred_label:gulinazha,	 img_label:gulinazha,	 min_distance:0.47702234983444214
    pred_label:gutianle,	 img_label:gutianle,	 min_distance:0.4202735722064972
    pred_label:huge,	 img_label:huge,	 min_distance:0.6576021313667297
    pred_label:jindong,	 img_label:jindong,	 min_distance:0.25166016817092896
    pred_label:jingtian,	 img_label:jingtian,	 min_distance:0.7095677256584167
    pred_label:lilianjie,	 img_label:lilianjie,	 min_distance:0.4656141996383667
    pred_label:liming,	 img_label:liming,	 min_distance:0.6135781407356262
    pred_label:linjunjie,	 img_label:linjunjie,	 min_distance:0.6249994039535522
    pred_label:liudehua,	 img_label:liudehua,	 min_distance:0.4157223105430603
    pred_label:sunli,	 img_label:sunli,	 min_distance:0.6081469655036926
    pred_label:tongliya,	 img_label:tongliya,	 min_distance:0.8020559549331665
    pred_label:yangmi,	 img_label:yangmi,	 min_distance:0.4146314561367035
    pred_label:zhangmin,	 img_label:zhangmin,	 min_distance:0.5783259272575378
    pred_label:zhangxueyou,	 img_label:zhangxueyou,	 min_distance:0.3920447826385498
    pred_label:zhoujielun,	 img_label:zhoujielun,	 min_distance:0.7544615268707275
    pred_label:zhourunfa,	 img_label:zhourunfa,	 min_distance:0.40853047370910645
    pred_label:zhouxingchi,	 img_label:zhouxingchi,	 min_distance:0.6933392882347107
    20/20
    """
