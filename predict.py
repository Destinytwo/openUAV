from ultralytics import RTDETR
import cv2
import matplotlib.pyplot as plt

# 加载预训练的 RT-DETR 模型（可以使用 'rtdetr-l.pt'、'rtdetr-x.pt' 等预训练权重）
model = RTDETR("C:\\Funny\\ultralytics-main\\ultralytics-main\\RT-DETR\\Origin\\weights\\best.pt")

# 指定待预测图片的路径
img_path = "0002.jpg"  # 修改为你的图片路径

# 使用 OpenCV 读取图片（BGR格式），然后转换为 RGB 格式
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 执行预测，conf=0.25 为置信度阈值（可按需求调整）
results = model.predict(source=img_rgb, conf=0.5)

# 查看预测结果（此处 results 返回的是一个列表，每个元素为一个 Results 对象）
result = results[0]

# 可视化预测结果，使用内置的 plot 方法显示带检测框的图片
annotated_img = result.plot(show=True)  # 此方法返回 BGR 格式的 numpy 数组

# 若需要将预测后的图片保存到本地
result.save(filename="0002A.jpg")

# 打印检测结果（类别、置信度和边框坐标）
'''for box in result.boxes:
    cls_id = int(box.cls)            # 类别索引
    conf = float(box.conf)           # 检测框置信度
    xyxy = box.xyxy[0].tolist()      # 边框坐标[xmin, ymin, xmax, ymax]
    print(f"类别: {cls_id}, 置信度: {conf:.2f}, 边框坐标: {xyxy}")'''


# 如果需要将图像转换为 RGB 后以 Matplotlib 显示
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
#plt.title("RT-DETR 预测结果")
plt.show()
