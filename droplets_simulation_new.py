import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
from datetime import datetime

def generate_ddpcr_image(
    average_diameter_um=100,    
    concentration=5,            
    image_size=(1532, 1024),   
    num_droplets=200,          
    output_path="ddpcr_sim.png",
    false_negative_rate=0.1,   # 假阴性率
    false_positive_rate=0.05,  # 假阳性率
    diameter_variance=0.1,      
    excel_path="droplets_data.xlsx"  
):
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 在文件名中添加时间戳
    output_path = output_path.rsplit('.', 1)
    output_path = f"{output_path[0]}_{timestamp}.{output_path[1]}"
    
    excel_path = excel_path.rsplit('.', 1)
    excel_path = f"{excel_path[0]}_{timestamp}.{excel_path[1]}"

    # 初始化随机数生成器
    rng = np.random.default_rng()
    
    # 转换系数：37像素 = 100微米
    pixels_per_um = 37/100
    
    # 1. 生成液滴尺寸（正态分布，单位：像素）
    average_diameter_pixels = average_diameter_um * pixels_per_um
    std_dev = average_diameter_pixels * diameter_variance  # 10%的标准差
    diameters = rng.normal(average_diameter_pixels, std_dev, num_droplets)
    diameters = np.clip(diameters, average_diameter_pixels*0.8, average_diameter_pixels*1.2)  # 限制在±20%范围内

    # 2. 计算每个液滴的分子数（泊松分布）
    diameters_um = diameters / pixels_per_um
    volumes = (4/3) * np.pi * (diameters_um/2)**3 / 1e9  # 转换为微升
    lambdas = concentration * volumes
    molecular_counts = rng.poisson(lambdas)

    # 计算动态假阴性/假阳性率（基于液滴直径）
    relative_diameters = diameters_um / average_diameter_um
    dynamic_false_negative_rates = false_negative_rate * relative_diameters
    dynamic_false_positive_rates = false_positive_rate * relative_diameters

    # 存储液滴的真实和显示状态
    droplet_states = []  # 用于存储每个液滴的状态信息
    
    # 应用假阴性和假阳性效果
    for i in range(len(molecular_counts)):
        is_false_negative = False
        is_false_positive = False
        
        if molecular_counts[i] > 0:  # 有靶标的液滴可能产生假阴性
            if rng.random() < dynamic_false_negative_rates[i]:
                is_false_negative = True
        else:  # 无靶标的液滴可能产生假阳性
            if rng.random() < dynamic_false_positive_rates[i]:
                is_false_positive = True
                
        droplet_states.append({
            'count': molecular_counts[i],
            'is_false_negative': is_false_negative,
            'is_false_positive': is_false_positive
        })

    # 3. 生成液滴位置（带碰撞检测）
    droplets = []
    for idx in range(num_droplets):
        diameter = diameters[idx]
        radius = diameter / 2
        max_attempts = 100
        
        for _ in range(max_attempts):
            x = rng.uniform(radius, image_size[0] - radius)
            y = rng.uniform(radius, image_size[1] - radius)
            
            collision = False
            for (ox, oy, oradius) in [(d[0], d[1], d[2]) for d in droplets]:
                distance = np.hypot(x - ox, y - oy)
                if distance < (radius + oradius + 2):
                    collision = True
                    break
            
            if not collision:
                droplets.append((x, y, radius, molecular_counts[idx]))
                break

    # 4. 创建图像并绘制液滴
    img = Image.new('L', image_size, color=0)
    draw = ImageDraw.Draw(img)

    # 存储液滴数据用于Excel
    droplet_data = []

    for idx, (x, y, radius, count) in enumerate(droplets):
        state = droplet_states[idx]
        
        # 根据真实状态和假阳性/假阴性状态确定亮度
        if state['is_false_negative']:  # 假阴性：有靶标但显示为阴性
            base = 50
            noise = rng.normal(0, 5)
            display_status = "假阴性"
        elif state['is_false_positive']:  # 假阳性：无靶标但显示为阳性
            base = 200
            noise = rng.normal(0, 10)
            display_status = "假阳性"
        elif count > 0:  # 真阳性
            base = 200
            brightness = base + 20 * min(count, 5)
            noise = rng.normal(0, 10)
            display_status = "真阳性"
        else:  # 真阴性
            base = 50
            noise = rng.normal(0, 5)
            display_status = "真阴性"
        
        brightness = int(np.clip(base + noise, 0, 255))
        
        bbox = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bbox, fill=brightness)

        # 收集液滴数据
        droplet_data.append({
            '直径(微米)': (radius * 2) / pixels_per_um,
            'X坐标(像素)': x,
            'Y坐标(像素)': y,
            '拷贝数': count,
            '亮度值': brightness,
            '显示状态': display_status
        })

    # 5. 保存图像
    img.save(output_path)
    print(f"图像已保存至：{output_path}")

    # 6. 导出Excel数据
    df = pd.DataFrame(droplet_data)
    df.to_excel(excel_path, index=False)
    print(f"数据已导出至：{excel_path}")

    # 7. 计算并显示统计信息
    total_drops = len(droplet_data)
    true_positive = sum(1 for d in droplet_data if d['显示状态'] == "真阳性")
    false_positive = sum(1 for d in droplet_data if d['显示状态'] == "假阳性")
    true_negative = sum(1 for d in droplet_data if d['显示状态'] == "真阴性")
    false_negative = sum(1 for d in droplet_data if d['显示状态'] == "假阴性")
    
    # 计算直径相关统计信息
    diameters = [d['直径(微米)'] for d in droplet_data]
    mean_diameter = np.mean(diameters)
    std_diameter = np.std(diameters)
    cv_diameter = std_diameter / mean_diameter
    
    print(f"\n统计信息:")
    print(f"总液滴数: {total_drops}")
    print(f"真阳性数量: {true_positive}")
    print(f"假阳性数量: {false_positive}")
    print(f"真阴性数量: {true_negative}")
    print(f"假阴性数量: {false_negative}")
    print(f"假阳性率: {false_positive/total_drops:.1%}")
    print(f"假阴性率: {false_negative/total_drops:.1%}")
    print(f"\n液滴直径统计:")
    print(f"平均直径: {mean_diameter:.1f}微米")
    print(f"直径标准差: {std_diameter:.1f}微米")
    print(f"直径变异系数: {cv_diameter:.1%}")

    # 添加统计信息到Excel
    summary_stats = {
        '统计指标': [
            '总液滴数',
            '真阳性数量',
            '假阳性数量',
            '真阴性数量',
            '假阴性数量',
            '假阳性率',
            '假阴性率',
            '平均液滴直径(微米)',
            '直径标准差(微米)',
            '直径变异系数'
        ],
        '数值': [
            f"{total_drops}",
            f"{true_positive}",
            f"{false_positive}",
            f"{true_negative}",
            f"{false_negative}",
            f"{false_positive/total_drops:.1%}",
            f"{false_negative/total_drops:.1%}",
            f"{mean_diameter:.1f}",
            f"{std_diameter:.1f}",
            f"{cv_diameter:.1%}"
        ]
    }
    
    # 将统计信息添加到新的Excel表单
    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name='液滴数据', index=False)
        pd.DataFrame(summary_stats).to_excel(writer, sheet_name='统计信息', index=False)

    img.show()

# 使用示例
if __name__ == "__main__":
    generate_ddpcr_image(
        average_diameter_um=150,      
        concentration=300,            
        num_droplets=200,
        output_path="ddpcr_simulation.png",
        false_negative_rate=0.1,     # 10%的假阴性率
        false_positive_rate=0.05,    # 5%的假阳性率
        diameter_variance=0.1,        
        excel_path="droplets_data.xlsx"
    )