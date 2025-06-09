import numpy as np
from PIL import Image, ImageDraw

def generate_ddpcr_image(
    average_diameter_um=100,    # 平均液滴直径（微米）
    concentration=5,            # 目标分子浓度（copies/uL）
    image_size=(1532, 1024),   # 输出图像尺寸
    num_droplets=200,          # 生成液滴数量
    output_path="ddpcr_sim.png"
):
    # 初始化随机数生成器
    rng = np.random.default_rng()
    
    # 转换系数：37像素 = 100微米
    pixels_per_um = 37/100
    
    # 1. 生成液滴尺寸（正态分布，单位：像素）
    average_diameter_pixels = average_diameter_um * pixels_per_um
    diameters = rng.normal(average_diameter_pixels, average_diameter_pixels*0.1, num_droplets)
    diameters = np.clip(diameters, 1, None)  # 确保最小直径为1像素

    # 2. 计算每个液滴的分子数（泊松分布）
    # 将直径转回微米计算体积
    diameters_um = diameters / pixels_per_um
    volumes = (4/3) * np.pi * (diameters_um/2)**3 / 1e9  # 转换为微升
    lambdas = concentration * volumes
    molecular_counts = rng.poisson(lambdas)

    # 3. 生成液滴位置（带碰撞检测）
    droplets = []
    for idx in range(num_droplets):
        diameter = diameters[idx]
        radius = diameter / 2
        max_attempts = 100
        
        for _ in range(max_attempts):
            # 生成随机位置
            x = rng.uniform(radius, image_size[0] - radius)
            y = rng.uniform(radius, image_size[1] - radius)
            
            # 碰撞检测
            collision = False
            for (ox, oy, oradius) in [(d[0], d[1], d[2]) for d in droplets]:
                distance = np.hypot(x - ox, y - oy)
                if distance < (radius + oradius + 2):  # 2像素安全间距
                    collision = True
                    break
            
            if not collision:
                droplets.append((x, y, radius, molecular_counts[idx]))
                break

    # 4. 创建图像并绘制液滴
    img = Image.new('L', image_size, color=0)  # 黑色背景
    draw = ImageDraw.Draw(img)

    for x, y, radius, count in droplets:
        # 设置亮度：阴性-低亮度噪声，阳性-与拷贝数相关
        if count > 0:
            base = 200
            brightness = base + 20 * min(count, 5)  # 最大亮度限制
            noise = rng.normal(0, 10)
        else:
            base = 50
            noise = rng.normal(0, 5)
        
        brightness = int(np.clip(base + noise, 0, 255))
        
        # 绘制椭圆
        bbox = [
            x - radius,
            y - radius,
            x + radius,
            y + radius
        ]
        draw.ellipse(bbox, fill=brightness)

    # 5. 保存图像
    img.save(output_path)
    print(f"图像已保存至：{output_path}")
    img.show()

# 使用示例
generate_ddpcr_image(
    average_diameter_um=150,    # 直接指定微米单位的直径
    concentration=1650100,         # 适当提高浓度使阳性液滴可见（copies/uL）
    num_droplets=200,
    output_path="ddpcr_simulation.png"
)
