import numpy as np
import matplotlib.pyplot as plt
from math import factorial

# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class DigitalPCRSimulator:
    def __init__(self, n_droplets):
        """
        初始化模拟器，设置液滴总数
        参数:
            n_droplets: 液滴总数
        """
        self.n_droplets = n_droplets
        self.lambda_value = None
        self.distribution = None

    def set_lambda_from_copies(self, total_copies):
        """
        根据总拷贝数计算lambda值
        参数:
            total_copies: 靶标总拷贝数
        """
        self.lambda_value = total_copies / self.n_droplets
        return self.lambda_value

    def set_lambda_from_concentration(self, concentration, droplet_volume):
        """
        根据浓度和液滴体积计算lambda值
        参数:
            concentration: 每微升拷贝数
            droplet_volume: 每个液滴的体积（微升）
        """
        self.lambda_value = concentration * droplet_volume
        return self.lambda_value

    def simulate_distribution(self):
        """
        模拟液滴中靶标的泊松分布
        返回:
            每个液滴中的拷贝数数组
        """
        if self.lambda_value is None:
            raise ValueError("模拟前必须设置Lambda值")
        
        self.distribution = np.random.poisson(self.lambda_value, self.n_droplets)
        return self.distribution

    def get_empty_ratio(self):
        """
        计算空液滴比例
        返回:
            包含（实际空液滴比例，理论空液滴比例）的元组
        """
        if self.distribution is None:
            raise ValueError("必须先运行模拟")
        
        actual_empty = np.sum(self.distribution == 0) / self.n_droplets
        theoretical_empty = np.exp(-self.lambda_value)
        return actual_empty, theoretical_empty

    def plot_distribution(self):
        """
        绘制拷贝数分布直方图，并显示数值标注
        """
        if self.distribution is None:
            raise ValueError("必须先运行模拟")
        
        plt.figure(figsize=(10, 6))
        max_copies = max(3, int(np.max(self.distribution)))
        bins = np.arange(-0.5, max_copies + 1.5, 1)
        
        # 绘制直方图并获取直方图数据
        n, bins, patches = plt.hist(self.distribution, bins=bins, density=True, 
                                   alpha=0.7, label='实际分布')
        
        # 在直方图柱子上方添加数值标注
        bin_centers = (bins[:-1] + bins[1:]) / 2
        for count, x in zip(n, bin_centers):
            if count > 0:  # 只标注非零值
                plt.text(x, count, f'{count:.3f}', ha='center', va='bottom')
        
        # 绘制理论泊松分布
        x = np.arange(0, max_copies + 1)
        pmf = np.exp(-self.lambda_value) * (self.lambda_value ** x) / np.array([factorial(i) for i in x])
        line = plt.plot(x, pmf, 'ro-', label='理论泊松分布')
        
        # 在理论分布点上添加数值标注
        for xi, yi in zip(x, pmf):
            plt.text(xi, yi, f'{yi:.3f}', ha='center', va='bottom', color='red')
        
        plt.title(f'靶标分布 (λ={self.lambda_value:.3f})')
        plt.xlabel('每个液滴的拷贝数')
        plt.ylabel('概率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 调整y轴的范围，为标注留出空间
        ymax = max(max(n), max(pmf)) * 1.2
        plt.ylim(0, ymax)
        
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 初始化模拟器，设置10000个液滴
    sim = DigitalPCRSimulator(n_droplets=10000)
    
    # 设置lambda值（例如：5000个拷贝）
    sim.set_lambda_from_copies(5000)
    
    # 运行模拟
    sim.simulate_distribution()
    
    # 获取空液滴比例
    actual, theoretical = sim.get_empty_ratio()
    print(f"空液滴比例: {actual:.3f} (实际) vs {theoretical:.3f} (理论)")
    
    # 绘制分布图
    sim.plot_distribution()
