import cv2
import os
import argparse

def detect_humans_in_folder(folder_path):
    # 加载人形检测模型
    # 使用OpenCV自带的HOG+SVM人形检测器
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # 遍历文件夹中的所有图片文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为图片
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图片: {filename}")
                continue
            
            # 调整图片大小以提高检测速度（可选）
            # image = cv2.resize(image, (640, 480))
            
            # 检测人形 - 调整参数平衡假阳性和漏检
            (rects, weights) = hog.detectMultiScale(
                image, 
                winStride=(12, 12),  # 适中的步长
                padding=(20, 20),  # 适中的填充
                scale=1.1,  # 适中的缩放因子
                useMeanshiftGrouping=False  # 禁用均值漂移分组，使用非极大值抑制
            )
            
            # 过滤检测结果，平衡假阳性和漏检
            filtered_rects = []
            for (x, y, w, h), weight in zip(rects, weights):
                # 适度的置信度阈值
                if weight > 0.75:
                    # 放宽宽高比范围
                    aspect_ratio = h / w
                    if aspect_ratio > 1.2 and aspect_ratio < 3.0:
                        # 降低最小尺寸要求
                        if h > 60 and w > 25:
                            # 降低面积要求
                            area = w * h
                            if area > 1500:  # 25*60=1500
                                # 降低边缘密度阈值
                                roi = image[y:y+h, x:x+w]
                                if roi.size > 0:
                                    # 转换为灰度图
                                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                    # 计算边缘
                                    edges = cv2.Canny(gray_roi, 50, 150)
                                    # 计算边缘密度
                                    edge_density = cv2.countNonZero(edges) / (w * h)
                                    # 降低边缘密度阈值
                                    if edge_density > 0.03:
                                        filtered_rects.append((x, y, w, h))
            
            # 添加非极大值抑制(NMS)进一步过滤重叠检测
            def non_max_suppression(boxes, overlapThresh=0.4):
                if len(boxes) == 0:
                    return []
                
                if boxes.dtype.kind == "i":
                    boxes = boxes.astype("float")
                
                pick = []
                x1 = boxes[:, 0]
                y1 = boxes[:, 1]
                x2 = boxes[:, 0] + boxes[:, 2]
                y2 = boxes[:, 1] + boxes[:, 3]
                
                area = (x2 - x1 + 1) * (y2 - y1 + 1)
                idxs = np.argsort(y2)
                
                while len(idxs) > 0:
                    last = len(idxs) - 1
                    i = idxs[last]
                    pick.append(i)
                    
                    xx1 = np.maximum(x1[i], x1[idxs[:last]])
                    yy1 = np.maximum(y1[i], y1[idxs[:last]])
                    xx2 = np.minimum(x2[i], x2[idxs[:last]])
                    yy2 = np.minimum(y2[i], y2[idxs[:last]])
                    
                    w = np.maximum(0, xx2 - xx1 + 1)
                    h = np.maximum(0, yy2 - yy1 + 1)
                    
                    overlap = (w * h) / area[idxs[:last]]
                    
                    idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
                
                return boxes[pick].astype("int")
            
            # 应用非极大值抑制
            if len(filtered_rects) > 0:
                import numpy as np
                filtered_rects = non_max_suppression(np.array(filtered_rects))
            
            # 输出检测结果
            if len(filtered_rects) > 0:
                print(f"图片 {filename} 中检测到人形，数量: {len(filtered_rects)}")
                # 在图片上绘制检测框（可选）
                for (x, y, w, h) in filtered_rects:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 保存带有检测框的图片（可选）
                output_path = os.path.join(folder_path, f"detected_{filename}")
                cv2.imwrite(output_path, image)
                print(f"带有检测框的图片已保存为: detected_{filename}")
            else:
                print(f"图片 {filename} 中未检测到人形")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='检测文件夹中图片是否包含人形')
    parser.add_argument('folder', type=str, help='包含图片的文件夹路径')
    args = parser.parse_args()
    
    # 检查文件夹是否存在
    if not os.path.isdir(args.folder):
        print(f"错误: 文件夹 {args.folder} 不存在")
    else:
        # 开始检测
        detect_humans_in_folder(args.folder)
