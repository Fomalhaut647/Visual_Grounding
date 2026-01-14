import cv2
import numpy as np
# detect_ui_with_canny传入图片路径，返回检测到的边界框列表和可视化结果图像
def detect_ui_with_canny(image_path):
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print("图片未找到")
        return
    # 2. 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 高斯模糊 (去噪)
    # (5, 5) 是卷积核大小，必须是奇数。对于高清 UI 图，(3,3) 或 (5,5) 比较合适
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Canny 边缘检测
    # 50, 150 是滞后阈值 (minVal, maxVal)。
    # 低于 50 的被丢弃，高于 150 的被认为是强边缘。
    # 介于两者之间的，如果与强边缘相连则保留。
    canny_edges = cv2.Canny(blurred, 50, 150)

    # --- 关键技巧：形态学操作 ---
    # Canny 出来的线条只有 1px 宽，且可能有断点。
    # 我们使用“膨胀 (Dilate)”或“闭运算 (Close)”让边缘连接起来，形成封闭图形。
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))

    # dilate 会让线条变粗，把断开的微小缝隙连上
    processed_edges = cv2.dilate(canny_edges, morph_kernel, iterations=1) 
    
    # 5. 查找轮廓
    # 使用 processed_edges 作为输入
    contours, hierarchy = cv2.findContours(processed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Canny 检测到 {len(contours)} 个轮廓")

    # 6. 筛选与可视化 (准备给 Qwen-VL 使用)
    output_img = img.copy()
    valid_elements = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # 过滤杂讯：过滤掉太小(比如噪点)或太大(比如整个屏幕边框)的轮廓
        img_h, img_w = img.shape[:2]
        if w < 15 or h < 15 or (w * h) > (img_w * img_h * 0.1):
            continue
            
        # 过滤长宽比极端的元素 (例如纯线条)，视情况而定
        #aspect_ratio = float(w)/h
        #if aspect_ratio > 10 or aspect_ratio < 0.1:
        #    continue

        valid_elements.append((x, y, w, h))

        # 画框 (绿色)
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 标号 (红色) - 这就是 SoM 方法需要的图
        cv2.putText(output_img, str(len(valid_elements)), (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 显示对比
    #cv2.imshow("Canny Edges", canny_edges)       # 原始边缘
    #cv2.imshow("Processed Edges", processed_edges) # 膨胀后的边缘
    #cv2.imshow("Result", output_img)             # 最终结果
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    cv2.imwrite("canny_result.jpg", output_img)
    return valid_elements, output_img

# 使用
if __name__ == "__main__":
    image = "test.png"
    boxes, output_img = detect_ui_with_canny(image)
    if output_img is not None:
        cv2.show("Canny Detection Result", output_img)
        cv2.waitKey(0)