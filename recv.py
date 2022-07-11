import warnings
from socket import *
import base64
import time
from utils.datasets import letterbox
from utils import torch_utils, google_utils
from utils.plots import *
from utils.general import non_max_suppression, scale_coords, check_img_size
import argparse
import cv2

warnings.filterwarnings('ignore')

label_path = 'dic.txt'
'''导入数据'''
labels = {}
label_file = open(label_path, 'r', encoding="utf-8")
for line in label_file.readlines():
    line = line.strip()
    line = line.split('    ')
    labels[line[1]] = line[0]


def label_to_word(label):
    """标签转文本"""
    if isinstance(label, str):
        return labels[label]


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt', help='path to weights file')  # 训练好的模型
    parser.add_argument('--conf-thres', type=float, default=0.65, help='object confidence threshold')  # 设置置信度
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for NMS')  # 设置NMS非极大值抑制的阈值
    parser.add_argument('--img-size', type=int, default=480, help='inference size (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide conf')
    opt = parser.parse_args()

    class Yolo:
        def __init__(self):
            self.prepare()
            self.hidelb = opt.hide_labels
            self.hide = opt.hide_conf

        def prepare(self):
            global model, device, classes, colors, names  # 需要在函数内部更改函数外的变量
            imgsz = opt.img_size
            if torch.cuda.is_available():
                device = torch_utils.select_device(device='0')
            else:
                device = torch_utils.select_device(device='cpu')

            google_utils.attempt_download(opt.weights)
            model = torch.load(opt.weights, map_location=device)['model'].float()  # 读取模型

            model.to(device).eval()

            stride = int(model.stride.max())
            self.imgsz = check_img_size(imgsz, s=stride)  # check img_size

            names = model.names if hasattr(model, 'names') else model.modules.names  # hasattr用于判断对象是否含有该属性
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        def detect(self, frame):
            im0 = frame
            img = letterbox(frame, new_shape=self.imgsz)[0]

            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32)
            img /= 255.0

            img = torch.from_numpy(img).to(device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = model(img)[0]
            pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

            boxes = []
            confidences = []
            classIDs = []
            classcounts = {}
            clscount = []
            clsname = []
            clsnameID = []
            object_needed = ['Sericinus_montelus_Grey_Adult',
                             'Spilarctia_subcarnea_Adult',
                             'Micromelalopha_troglodyta_Adult',
                             'plagiodera_versicolora_Adult',
                             'Chalcophora_ japonica_Adult',
                             'Monochamus_alternatus_Hope_Adult',
                             'Apriona_germari_Adult',
                             'Drosicha_corpulenta_Adult',
                             'Erthesina_fullo_Adult',
                             'Cnidocampa_flavescens_Adult',
                             'Hyphantria_cunea_Adult',
                             'Clostera_anachoreta_Adult',
                             'Latoria_consocia_Walker_Adult',
                             'Anoplophora_chinensis_Forster_Adult',
                             'Psilogramma_menephron_Adult', ]

            for i in range(len(object_needed)):
                classcounts[object_needed[i]] = 0

            for i, det in enumerate(pred):

                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, self.score, cls in det:
                        # label = label_to_word(('%s' % (names[int(cls)])).replace(' ', ''))
                        # label = '%s' % (names[int(cls)])
                        label = label_to_word(('%s' % (names[int(cls)])))
                        if names[int(cls)] in object_needed:
                            if self.hide and self.hidelb:
                                plot_one_box(xyxy, im0, label=[], color=colors[int(cls)])

                            if self.hide and not self.hidelb:
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                            if not self.hide and self.hidelb:
                                plot_one_box(xyxy, im0, label=str(round(self.score.item(), 2)), color=colors[int(cls)])

                            if not self.hide and not self.hidelb:
                                plot_one_box(xyxy, im0, label=str(round(self.score.item(), 2)) + ' ' + label,
                                             color=colors[int(cls)])

                            classcounts[names[int(cls)]] += 1

                            boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])])
                            confidences.append(float(self.score))
                            classIDs.append(int(cls))

            for k in list(classcounts.keys()):
                if classcounts[k] == 0:
                    del classcounts[k]

            for i in classcounts.keys():
                clsname.append(i)
                clscount.append(classcounts[i])

            for k in range(len(clsname)):
                clsnameID.append(label_to_word(clsname[k]))

            return im0, (str(clsnameID) + ',' + str(clscount)).replace('[', '').replace(']', '') \
                .replace("'", '').replace(' ', '')

    print(opt)
    yolo = Yolo()
    ip = [a for a in os.popen('route print').readlines() if ' 0.0.0.0 ' in a][0].split()[-2]
    print('IP:', ip)
    PORT = 99
    print('PORT:', PORT)
    BUFSIZ = 1024 * 20
    ADDR = (ip, PORT)  # IPv4地址
    tcpSerSock = socket(AF_INET, SOCK_STREAM)
    tcpSerSock.bind(ADDR)
    tcpSerSock.listen(5)
    while True:
        rec_d = bytes([])
        print('Waiting For Connection...')
        tcpCliSock, addr = tcpSerSock.accept()
        print('...connected from:', addr)
        while True:
            data = tcpCliSock.recv(BUFSIZ)
            if not data or len(data) == 0:
                break
            else:
                rec_d = rec_d + data
        rec_d = base64.b64decode(rec_d)
        np_arr = np.frombuffer(rec_d, np.uint8)
        image = cv2.imdecode(np_arr, 1)
        res = yolo.detect(image)
        res_img = res[0]
        res_count = res[1]
        print(res_count)
        cv2.imwrite('results/res_img.jpg', res_img)

        print('发送图片线程启动')
        filepath = 'results/res_img.jpg'  # 输入需要传输的图片名 xxx.jpg
        fp = open(filepath, 'rb')  # 打开要传输的图片

        while True:
            data = fp.read(2048)  # 读入图片数据
            if not data:
                print('{0} send over...'.format(filepath))
                break
            tcpCliSock.send(data)  # 以二进制格式发送图片数据
        tcpCliSock.close()
        # if cv2.waitKey(1000) == 27:
        #     break
        time.sleep(0.01)
        tcpCliSock, addr = tcpSerSock.accept()
        tcpCliSock.send(res_count.encode())
        tcpCliSock.close()

    tcpSerSock.close()


if __name__ == "__main__":
    main()
