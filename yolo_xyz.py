# yolov5
# 导包
import torch
import cv2
from multiprocessing import Process, Manager, Value
# 下面两个是yolov5文件夹里面的代码
from utils.general import non_max_suppression
from models.experimental import attempt_load
# 串口包
import serial
# 卡尔曼
import numpy as np


class KalmanFilter2D:
    def __init__(self, dt, initial_state, initial_covariance):
        # 状态转移矩阵
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # 控制矩阵
        self.B = np.array([[0.5 * dt ** 2, 0],
                           [0, 0.5 * dt ** 2],
                           [dt, 0],
                           [0, dt]])

        # 测量矩阵
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # 状态协方差矩阵
        self.P = np.copy(initial_covariance)

        # 过程噪声协方差矩阵
        self.Q = np.array([[0.1 * dt ** 4, 0, 0.5 * dt ** 3, 0],
                           [0, 0.1 * dt ** 4, 0, 0.5 * dt ** 3],
                           [0.5 * dt ** 3, 0, 0.1 * dt ** 2, 0],
                           [0, 0.5 * dt ** 3, 0, 0.1 * dt ** 2]])

        # 测量噪声协方差矩阵
        self.R = np.array([[0.1, 0],
                           [0, 0.1]])

        # 初始状态
        self.x = np.copy(initial_state)

    def predict(self):
        # 预测状态
        self.x = np.dot(self.A, self.x)

        # 预测协方差
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        return self.x

    def update(self, z):
        # 计算卡尔曼增益
        if z is None:
            # 如果没有新的测量值，只进行预测步骤
            self.x = self.predict()
        else:
            # 如果有新的测量值，进行状态更新步骤
            S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
            self.x = self.x + np.dot(K, z - np.dot(self.H, self.x))
            self.P = np.dot((np.eye(len(self.x)) - np.dot(K, self.H)), self.P)


# 确保在进行对象检测时，边界框的位置可以与输入图像的大小相对应，以便正确地识别和定位对象。
def scale_coords(img, coords, out_shape):
    # img: 输入图像，具有(C, H, W)的形状，表示通道数、高度和宽度。
    # coords: 包含边界框坐标的数组，具有(..., 4)的形状，其中...表示可变数量的维度，而最后一个维度4表示左上角和右下角的坐标。
    # out_shape: 输出尺寸，一个包含高度、宽度和通道数的元组或列表([out_h, out_w, _])。
    img_h, img_w = img.shape[2:]
    out_h, out_w, _ = out_shape
    coords[..., 0] *= out_w / img_w
    coords[..., 1] *= out_h / img_h
    coords[..., 2] *= out_w / img_w
    coords[..., 3] *= out_h / img_h
    # 通过按比例缩放坐标值来调整边界框的位置。首先，将输入图像的高度和宽度记录为img_h和img_w。
    # 然后，获取输出尺寸的高度和宽度，分别记录为out_h和out_w。
    # 接下来，通过计算比例因子out_w / img_w和out_h / img_h，将边界框的所有坐标值乘以相应的比例因子。
    # 这样做的目的是将坐标值映射到新的输出尺寸上，以便适应不同尺寸的输入图像。
    return coords
    # 最后，返回经过缩放的坐标数组coords


def detect_objects(weights_path, output_frame):
    # 加载YOLOv5模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 检查是否可用CUDA加速。如果CUDA可用，则将设备设置为'cuda'，否则设置为'cpu'。
    model = attempt_load(weights_path, device)

    # 设置置信度阈值和IoU阈值
    model.conf = 0.7
    model.iou = 0.5

    # 设置模型为推理模式
    model.eval()

    # 打开视频流
    cap = cv2.VideoCapture(1)
    cap.set(3, 640)  # 设置分辨率
    cap.set(4, 480)

    while True:
        ret, frame = cap.read()  # 视频读入
        if not ret:
            break

        # 图像预处理
        img = frame.copy()  # 创建一个副本以进行预处理
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to(device).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = torch.nn.functional.interpolate(img, size=(320, 320), mode='bilinear', align_corners=False)
        # size = (320, 320)：表示目标输出图像的大小为320x320像素。将输入图像按照此大小进行调整。
        # mode = 'bilinear'：表示插值方法采用双线性插值。双线性插值是一种常用的插值方法，它通过对最近的四个像素进行加权平均来计算新像素值。
        # align_corners = False：表示在计算插值时不对齐角点。这个参数的具体影响取决于插值方法和具体的实现方式。
        # 进行物体检测
        outputs = model(img)
        results = non_max_suppression(outputs, conf_thres=model.conf, iou_thres=model.iou)
        # 通过非最大抑制算法对模型输出的边界框进行处理，以选择最佳的边界框并过滤掉冗余的检测结果。
        # outputs：模型的输出结果，通常是一个包含预测边界框及其置信度的数组。
        # conf_thres：表示置信度阈值，用于过滤掉置信度低于此阈值的边界框。
        # iou_thres：表示IoU（Intersection over Union）阈值，用于合并重叠度高于此阈值的边界框。
        # 算法的步骤如下：
        # 1、根据置信度阈值过滤掉置信度低于阈值的边界框。
        # 2、对剩余的边界框按照置信度进行排序，置信度高的排在前面。
        # 3、从排好序的边界框列表中选择置信度最高的边界框，并将其添加到最终的结果列表中。
        # 4、遍历剩余的边界框列表，计算当前边界框与已选择的边界框的IoU值（重叠度），如果大于IoU阈值，则将其从列表中移除。
        # 5、重复步骤3和4，直到所有边界框都被处理完毕。
        # 6、最终，non_max_suppression()函数返回经过非最大抑制处理后的边界框结果。
        if results[0] is not None and len(results[0]) > 0:
            result = results[0]

        # 绘制边界框和标签
        for result in results:  # 逐个访问并获取
            if result is not None and len(result) > 0:
                result[:, :4] = scale_coords(img, result[:, :4], frame.shape).round()
                # result[:, :4]表示选取每个边界框的前四个元素，即边界框的坐标信息。
                for *xyxy, conf, cls in result:
                    # result是一个数组或张量，每一行代表一个边界框，包含边界框的坐标、类别标签和置信度等信息。
                    # *xyxy表示将边界框的前四个元素解包为一个名为xyxy的列表。这里的xyxy表示边界框的坐标信息，通常是左上角和右下角的坐标值。
                    # conf表示边界框的置信度，通常是模型对该边界框所属类别的预测置信度。
                    # cls表示边界框的类别标签，通常是模型对该边界框所属类别的预测结果。
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    # f'{model.names[int(cls)]} {conf:.2f}'使用了格式化字符串的语法，将类别名称和置信度转换为一个字符串。
                    # 其中，{model.names[int(cls)]}表示插入类别名称，
                    # {conf: .2f}表示插入置信度，并保留两位小数。
                    # 绘制边界框
                    xyxy = [int(x) for x in xyxy]
                    # 将边界框的坐标值从浮点数转换为整数，以适应图像的像素值,将转换后的结果赋值给xyxy
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                    # 绘制标签
                    cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if 350 <= (xyxy[2] - xyxy[0]) and 350 <= (xyxy[3] - xyxy[1]):  # 需优化
                        print("框的尺寸在指定范围内")

                    x = int((xyxy[0] + xyxy[2]) / 2)
                    y = int((xyxy[1] + xyxy[3]) / 2)

                    predicted_state, updated_state = Kalman(x, y)#卡尔曼

                    print("predicted_state", predicted_state)
                    print("updated_state", updated_state)
                    print("x", x, "y", y)

                    dx = updated_state[0]
                    dy = updated_state[1]
                    cv2.circle(frame, (int(dx), int(dy)), 5, (0, 255, 0), -1)
                    cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 0), -1)

                    byte(dx, dy)

            else:

                byte(dx, dy)

        cv2.imshow('frame', frame)
        # 在适当的位置添加以下代码
        output_path = "output.mp4"  # 视频保存路径
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 视频编码格式
        fps = 30  # 视频帧率
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))  # 创建视频写入器

        # 在每一帧之后添加以下代码
        video_writer.write(frame)  # 将当前帧写入视频



        # 按q退出
        if cv2.waitKey(1) == ord('q'):
            break
    # 在结束时添加以下代码
    video_writer.release()  # 释放视频写入器
    cap.release()
    cv2.destroyAllWindows()


# 卡尔曼!启动！
def Kalman(x, y):
    # 初始化卡尔曼滤波器
    dt = 0.1
    initial_state = np.array([x, y, 0, 0])  # 初始状态 [x, y, vx, vy]
    initial_covariance = np.eye(4)  # 初始协方差矩阵
    kf = KalmanFilter2D(dt, initial_state, initial_covariance)
    # 将x和y作为测量值进行解算
    z = np.array([x, y])
    # 预测状态
    predicted_state = kf.predict()
    # 更新状态
    updated_state = kf.update(z)
    return predicted_state, updated_state


# 串口输出
def byte(dx, dy):
    # 将x和y转换为16位的整数数据
    x_int = int(dx)  # 假设x为字符串表示的整数
    y_int = int(dy)  # 假设y为字符串表示的整数
    # 将16位整数数据拆分为高八位和低八位
    x_high_byte = (x_int >> 8) & 0xFF  # 高八位
    x_low_byte = x_int & 0xFF  # 低八位
    y_high_byte = (y_int >> 8) & 0xFF  # 高八位
    y_low_byte = y_int & 0xFF  # 低八位
    z_high_byte = (x_int >> 8) & 0xFF  # 高八位
    z_low_byte = x_int & 0xFF  # 低八位
    # 通过串口发送数据帧
    ser.write(bytes([header, x_high_byte, x_low_byte, y_high_byte, y_low_byte]))


if __name__ == '__main__':
    # 串口相关
    # 设置头帧和尾帧
    header = 0x5A
    footer = 0xA5

    ser = serial.Serial("/dev/ttyUSB0", 115200)  # 打开COM17，将波特率配置为115200，其余参数使用默认值
    if ser.isOpen():  # 判断串口是否成功打开
        print("打开串口成功。")
    else:
        ser = serial.Serial("/dev/ttyUSB1", 115200)
    # 串口结束

    # 设置权重文件路径
    weights_path = '/home/bc/yolov5/yolov5-7.0/10.13.pt'

    # 使用 Manager 创建共享变量
    manager = Manager()
    output_frame = manager.dict()

    # 创建物体检测进程
    detection_process = Process(target=detect_objects, args=(weights_path, output_frame))
    detection_process.start()
    # 等待物体检测进程结束
    detection_process.join()
