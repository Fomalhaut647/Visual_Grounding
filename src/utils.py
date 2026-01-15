import PIL.ImageDraw as ImageDraw


def add_visual_grid(image, grid_size=50):
    """
    推理增强：在图片上叠加参考网格，帮助模型更准确地定位坐标。
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # 绘制垂直线
    for x in range(0, width, grid_size):
        draw.line([(x, 0), (x, height)], fill=(200, 200, 200, 100), width=1)
        if x % (grid_size * 2) == 0:
            draw.text((x + 2, 2), str(round(x/width, 2)), fill=(150, 150, 150))

    # 绘制水平线
    for y in range(0, height, grid_size):
        draw.line([(0, y), (width, y)], fill=(200, 200, 200, 100), width=1)
        if y % (grid_size * 2) == 0:
            draw.text((2, y + 2), str(round(y/height, 2)), fill=(150, 150, 150))
            
    return image

def draw_bbox(image, bbox, label=None, color="red", line_width=3):
    """
    在图片上绘制 bbox 用于可视化。
    bbox: [xmin, ymin, xmax, ymax] 归一化坐标
    """
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size
    xmin, ymin, xmax, ymax = bbox
    
    real_bbox = [xmin * img_w, ymin * img_h, xmax * img_w, ymax * img_h]
    draw.rectangle(real_bbox, outline=color, width=line_width)
    
    if label:
        draw.text((real_bbox[0], real_bbox[1] - 20), label, fill=color)
    
    return image

