import numpy as np
from scipy import stats as scipy_stats  # 重命名 stats 模块

def calculate_ddpcr_statistics(
    total_volume_ul=20.0,      # 总体积（微升）
    average_diameter_um=100,    # 平均液滴直径（微米）
    concentration=5,            # 目标分子浓度（copies/uL）
    false_negative_rate=0.1,    # 假阴性率
    false_positive_rate=0.05,   # 假阳性率
    diameter_variance=0.1,      # 直径变异系数
    confidence_level=0.95      # 置信水平
):
    """
    计算ddPCR的统计指标
    参数:
        total_volume_ul: 总体积（微升）
        average_diameter_um: 平均液滴直径（微米）
        concentration: 目标分子浓度（copies/uL）
        false_negative_rate: 基础假阴性率
        false_positive_rate: 基础假阳性率
        diameter_variance: 直径变异系数
    返回:
        包含统计指标的字典
    """
    # 初始化随机数生成器
    rng = np.random.default_rng()
    
    # 计算单个液滴的平均体积（微升）
    single_droplet_volume = (4/3) * np.pi * (average_diameter_um/2)**3 / 1e9
    
    # 估算液滴数量
    num_droplets = int(total_volume_ul / single_droplet_volume)
    
    # 生成液滴直径（正态分布，单位：微米）
    diameters = rng.normal(average_diameter_um, average_diameter_um * diameter_variance, num_droplets)
    diameters = np.clip(diameters, average_diameter_um*0.8, average_diameter_um*1.2)  # 限制在±20%范围内
    
    # 计算每个液滴的体积和分子数
    volumes = (4/3) * np.pi * (diameters/2)**3 / 1e9  # 转换为微升
    lambdas = concentration * volumes
    molecular_counts = rng.poisson(lambdas)
    
    # 计算动态假阴性/假阳性率
    relative_diameters = diameters / average_diameter_um
    dynamic_false_negative_rates = false_negative_rate * relative_diameters
    dynamic_false_positive_rates = false_positive_rate * relative_diameters
    
    # 应用假阴性和假阳性效果
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    
    for i in range(len(molecular_counts)):
        if molecular_counts[i] > 0:  # 有靶标的液滴
            if rng.random() < dynamic_false_negative_rates[i]:
                false_negative += 1  # 假阴性
            else:
                true_positive += 1   # 真阳性
        else:  # 无靶标的液滴
            if rng.random() < dynamic_false_positive_rates[i]:
                false_positive += 1  # 假阳性
            else:
                true_negative += 1   # 真阴性
    
    # 初始化所有需要的变量
    total_drops = len(molecular_counts)
    mean_diameter = np.mean(diameters)
    std_diameter = np.std(diameters)
    cv_diameter = std_diameter / mean_diameter
    actual_volume = np.sum(volumes)
    
    # 计算LOD和LOQ
    blank_lambda = -np.log(1 - false_positive_rate)
    lod_copies_per_ul = (blank_lambda + 3 * np.sqrt(blank_lambda)) / single_droplet_volume
    loq_copies_per_ul = 3.3 * lod_copies_per_ul
    
    # 计算置信区间
    total_positive = true_positive + false_positive
    if total_drops > 0:
        positive_ratio = total_positive/total_drops
        # 使用泊松分布计算浓度的置信区间
        ci_lower, ci_upper = scipy_stats.binom.interval(confidence_level, total_drops, positive_ratio)
        # 转换为每微升拷贝数
        concentration_ci_lower = (-np.log(1 - ci_lower/total_drops)) / single_droplet_volume
        concentration_ci_upper = (-np.log(1 - ci_upper/total_drops)) / single_droplet_volume
    else:
        concentration_ci_lower = concentration_ci_upper = 0

    # 打印统计结果（不使用字典引用）
    print("\n=== ddPCR统计结果 ===")
    print(f"体积统计:")
    print(f"目标总体积: {total_volume_ul:.2f}微升")
    print(f"实际总体积: {actual_volume:.2f}微升")
    print(f"\n液滴数量统计:")
    print(f"总液滴数: {total_drops}")
    print(f"真阳性数量: {true_positive}")
    print(f"假阳性数量: {false_positive}")
    print(f"真阴性数量: {true_negative}")
    print(f"假阴性数量: {false_negative}")
    print(f"\n准确度统计:")
    print(f"假阳性率: {false_positive/total_drops:.1%}" if total_drops > 0 else "假阳性率: 0%")
    print(f"假阴性率: {false_negative/total_drops:.1%}" if total_drops > 0 else "假阴性率: 0%")
    print(f"\n液滴直径统计:")
    print(f"平均直径: {mean_diameter:.1f}微米")
    print(f"直径标准差: {std_diameter:.1f}微米")
    print(f"直径变异系数: {cv_diameter:.1%}")
    print(f"\n灵敏度指标:")
    print(f"检测限(LOD): {lod_copies_per_ul:.2f} copies/uL")
    print(f"定量限(LOQ): {loq_copies_per_ul:.2f} copies/uL")
    print(f"浓度95%置信区间: {concentration_ci_lower:.2f} - {concentration_ci_upper:.2f} copies/uL")

    # 最后创建并返回统计结果字典
    stats = {
        "总体积(微升)": total_volume_ul,
        "实际体积(微升)": actual_volume,
        "总液滴数": total_drops,
        "平均直径(微米)": mean_diameter,
        "直径标准差(微米)": std_diameter,
        "直径变异系数": cv_diameter,
        "真阳性数量": true_positive,
        "假阳性数量": false_positive,
        "真阴性数量": true_negative,
        "假阴性数量": false_negative,
        "假阳性率": false_positive/total_drops if total_drops > 0 else 0,
        "假阴性率": false_negative/total_drops if total_drops > 0 else 0,
        "检测限LOD(copies/uL)": lod_copies_per_ul,
        "定量限LOQ(copies/uL)": loq_copies_per_ul,
        "浓度置信区间下限(copies/uL)": concentration_ci_lower,
        "浓度置信区间上限(copies/uL)": concentration_ci_upper,
    }
    
    return stats

# 使用示例
if __name__ == "__main__":
    stats = calculate_ddpcr_statistics(
        total_volume_ul=25.0,         # 25微升总体积
        average_diameter_um=150,       # 150微米平均直径
        concentration=3000,             # 3000 copies/uL
        false_negative_rate=0.05,       # 假阴性率
        false_positive_rate=0.01,      # 假阳性率
        diameter_variance=0.05,        # 5%直径变异系数
        confidence_level=0.95         # 95%置信水平
    )