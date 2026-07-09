# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import cv2
# import os

# current_file_path = os.path.abspath(__file__)
# # print(f"[INFO] 当前运行 SAM 获得mask,文件运行路径为{current_file_path}")
# print(f"[INFO] 当前运行阶段 SAM → 生成 mask")
# print(f"[INFO] 文件路径：{current_file_path}")
# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)
    
# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

# clicked_points = []
# def onclick(event):
#     # event.xdata, event.ydata 是浮点坐标
#     if event.xdata is not None and event.ydata is not None:
#         x = int(event.xdata)
#         y = int(event.ydata)
#         clicked_points.append([x, y])
#         print(f"[CLICK] ({x}, {y})")
#         # 绘制点击点
#         ax.plot(x, y, 'ro')
#         fig.canvas.draw()
# # 读取数据
# image = cv2.imread('images/airpods.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.axis('on')
# plt.show()

# import sys
# sys.path.append("..")
# from segment_anything import sam_model_registry, SamPredictor

# sam_checkpoint = "/data/mmw/test/segment-anything/checkpoints/sam_vit_h_4b8939.pth"
# model_type = "vit_h"

# device = "cuda"
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)
# predictor = SamPredictor(sam)
# predictor.set_image(image)
# input_point = np.array([[1300, 600]])
# input_label = np.array([1])

# plt.figure(figsize=(10,10))
# plt.imshow(image)
# show_points(input_point, input_label, plt.gca())
# plt.axis('on')
# plt.show()  

# masks, scores, logits = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     multimask_output=True,
# )
# masks.shape  # (number_of_masks) x H x W

# for i, (mask, score) in enumerate(zip(masks, scores)):
#     plt.figure(figsize=(10,10))
#     plt.imshow(image)
#     show_mask(mask, plt.gca())
#     show_points(input_point, input_label, plt.gca())
#     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#     plt.axis('off')
#     plt.show()  
# plt.imsave('output/mask_1.png', masks[0])    

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import sys
import matplotlib
matplotlib.use('TkAgg')

current_file_path = os.path.abspath(__file__)
print(f"[INFO] 当前运行阶段 SAM → 生成 mask")
print(f"[INFO] 文件路径：{current_file_path}")

# =======================
# 可视化辅助函数
# =======================
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# =======================
# 读取图片
# =======================
image = cv2.imread('images/airpods.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# =======================
# 初始化 SAM 模型
# =======================
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "/data/mmw/test/segment-anything/checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image)

# =======================
# GUI + 点击事件
# =======================
clicked_points = []

def onclick(event):
    """鼠标点击获取坐标并自动运行 SAM 推理"""
    if event.xdata is None or event.ydata is None:
        return

    x = int(event.xdata)
    y = int(event.ydata)
    clicked_points.append([x, y])
    print(f"[CLICK] ({x}, {y})")

    # 在图上绘制点
    ax.plot(x, y, 'ro')
    fig.canvas.draw()

    # =============== SAM 推理部分 ==================
    input_point = np.array([[x, y]])
    input_label = np.array([1])  # 前景点
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # 显示所有 masks
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(8,8))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

    # 保存最优 mask（得分最高）
    best_id = np.argmax(scores)
    best_mask = masks[best_id]

    os.makedirs("output", exist_ok=True)
    save_path = f"output/mask_click_{x}_{y}.png"
    plt.imsave(save_path, best_mask)
    print(f"[INFO] 已保存最优 mask: {save_path}")

# =======================
# 创建 GUI 窗口
# =======================
fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(image)
ax.set_title("Click on the image to generate SAM mask")
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
