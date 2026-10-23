智慧电力-电力施工绑扎不规范检测数据集，基于YOLOv11 电力施工绑扎不规范检测系统![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5459c6c062024c59ad8a9054bfb97539.png)
1799张，yolo，voc，coco三种标注方式
图像尺寸:1000*1000
类别数量:1类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d5c662a10e3d4713b51314d8a9bbd983.png)

训练集图像数量:1438; 验证集图像数量:183； 测试集图像数量:178
类别名称: 每一类图像数 ，每一类标注数
绑扎不规范: 1371,5495
image num: 1799
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3373efd963e649be88708b8972fc308c.png)

模型代码：模型训练使用yolov11n训练，30个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3f62c1a167f148a89ec4a4ec57b82ca7.png)

qt界面：运行界面采用pyqt编写，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bd7a2e3e9d4e4570a717379b1a3aa537.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/de7af6ac721741bdadd0660cab0925f4.png)
# 基于YOLOv11 电力施工绑扎不规范检测系统完整代码
## 1、数据集yaml配置 `power_binding.yaml`
```yaml
path: ./power_binding_dataset
train: images/train
val: images/val
test: images/test

names:
  0: 电力绑扎不规范
```

## 2、模型训练代码 train.py
```python
from ultralytics import YOLO

if __name__ == "__main__":
    # 加载yolov11n模型
    model = YOLO("yolov11n.pt")
    result = model.train(
        data="power_binding.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        project="power_binding_result",
        name="yolov11_power_binding",
        save=True,
        val=True
    )
    print("训练完成，最佳权重路径：power_binding_result/yolov11_power_binding/weights/best.pt")
```

## 3、图像、视频推理代码 predict.py
```python
from ultralytics import YOLO
import cv2

model = YOLO("power_binding_result/yolov11_power_binding/weights/best.pt")

def detect_image(img_path, save_path="detect_result.jpg"):
    res = model.predict(img_path, conf=0.4)
    img_draw = res[0].plot()
    cv2.imwrite(save_path, img_draw)
    return img_draw

def detect_video(video_path):
    model.predict(source=video_path, save=True, conf=0.4)

if __name__ == "__main__":
    detect_image("test.jpg")
```

## 4、PyQt5可视化界面完整代码
```python
import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
                             QTableWidget, QTableWidgetItem, QComboBox, QTextEdit)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO

class PowerDetectWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的电力施工绑扎不规范检测系统")
        self.setGeometry(50, 50, 1280, 850)
        # 加载模型
        self.model = YOLO("power_binding_result/yolov11_power_binding/weights/best.pt")
        self.result_img = None
        self.img_path = ""

        # 图片显示区域
        self.img_label = QLabel(self)
        self.img_label.setGeometry(30, 80, 720, 520)
        self.img_label.setStyleSheet("border:1px solid #333;")

        # 右侧控件
        label_file = QLabel("文件导入", self)
        label_file.setGeometry(780,80,120,30)

        self.btn_img = QPushButton("选择图片", self)
        self.btn_img.setGeometry(780,110,180,35)
        self.btn_img.clicked.connect(self.load_image)

        self.btn_video = QPushButton("选择视频文件", self)
        self.btn_video.setGeometry(780,150,180,35)

        self.btn_cam = QPushButton("开启摄像头", self)
        self.btn_cam.setGeometry(780,190,180,35)

        label_conf = QLabel("置信度", self)
        label_conf.setGeometry(780,240,80,30)
        self.combo_conf = QComboBox(self)
        self.combo_conf.addItems(["0.3","0.4","0.5","0.6","0.7"])
        self.combo_conf.setGeometry(830,240,80,30)

        label_target = QLabel("目标选择", self)
        label_target.setGeometry(780,280,80,30)
        self.combo_target = QComboBox(self)
        self.combo_target.addItems(["全部"])
        self.combo_target.setGeometry(830,280,80,30)

        label_pos = QLabel("目标位置", self)
        label_pos.setGeometry(780,330,120,30)
        self.text_pos = QTextEdit(self)
        self.text_pos.setGeometry(780,360,220,120)

        self.btn_save = QPushButton("保存", self)
        self.btn_save.setGeometry(780,500,100,40)
        self.btn_save.clicked.connect(self.save_result)

        self.btn_exit = QPushButton("退出", self)
        self.btn_exit.setGeometry(900,500,100,40)
        self.btn_exit.clicked.connect(self.close)

        # 底部结果表格
        self.table = QTableWidget(self)
        self.table.setGeometry(30,630,1020,180)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["序号","文件路径","类别","置信度","坐标位置"])

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "Image Files(*.jpg *.png *.jpeg)")
        if not fname:
            return
        self.img_path = fname
        conf_thresh = float(self.combo_conf.currentText())
        results = self.model.predict(fname, conf=conf_thresh)
        self.table.setRowCount(0)
        for r in results:
            for box in r.boxes:
                cls_name = self.model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row,0,QTableWidgetItem(str(row+1)))
                self.table.setItem(row,1,QTableWidgetItem(fname))
                self.table.setItem(row,2,QTableWidgetItem(cls_name))
                self.table.setItem(row,3,QTableWidgetItem(f"{conf:.2f}"))
                coord_str = f"[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]"
                self.table.setItem(row,4,QTableWidgetItem(coord_str))
                self.text_pos.setText(f"xmin:{int(x1)}\nymin:{int(y1)}\nxmax:{int(x2)}\nymax:{int(y2)}")
        img_draw = results[0].plot()
        self.result_img = img_draw
        img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
        h,w,c = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, c*w, QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.img_label.size(), Qt.KeepAspectRatio))

    def save_result(self):
        if self.result_img is not None:
            cv2.imwrite("power_detect_save.jpg", self.result_img)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PowerDetectWindow()
    win.show()
    sys.exit(app.exec_())
```

## 环境安装命令
```bash
pip install ultralytics opencv-python pyqt5
```

## 标签
`0: 电力绑扎不规范`

## 应用场景
1. 无人机航拍输电线路巡检，识别绝缘子绑扎不规范缺陷
2. 电力施工后图像质检，自动排查绑扎工艺缺陷
3. 电力运维、毕业设计、目标检测算法研究
4. 输电通道图像批量缺陷筛查

## PR曲线绘图代码
```python
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("power_binding_result/yolov11_power_binding/weights/best.pt")
res = model.val()
pr_data = res.plots["PR_curve"]
plt.figure(figsize=(10,7))
plt.plot(pr_data[0],pr_data[1],label=f"all classes {res.box.map50:.3f} mAP@0.5",linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("pr_curve.png",dpi=300)
plt.show()
```
