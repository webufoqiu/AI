from random import choice
import matplotlib.pyplot as plt


class RandomWalk:
    """生成随机漫步的类"""
    def __init__(self, num_points=5000):
        """初始化随机漫步的属性"""
        self.num_points = num_points
        # 所有随机漫步都始于(0,0)
        self.x_values = [0]
        self.y_values = [0]

    def fill_walk(self):
        """计算随机漫步包含的所有点"""
        # 生成漫步包含的点，并决定每次漫步的方向
        # 不断漫步，也就是遍历列表
        while len(self.x_values) < self.num_points:
            # 给x_direction 选择一个值，结果要么是表示向右走的1，要么是表示向左走的-1
            x_direction = choice([-1, 1])
            # 随机地选择一个0~4之间的整数，决定走多远
            x_distance = choice([0, 1, 2, 3, 4])
            # 将移动方向乘以移动距离，确定沿 x 和 y 轴移动的距离
            # x_step 为正，将向右移动，为负将向左移动，而为零将垂直移动
            x_step = x_direction * x_distance
            # y轴也类似
            y_direction = choice([-1, 1])
            y_distance = choice([0, 1, 2, 3, 4])
            y_step = y_direction * y_distance
            # 拒绝原地踏步
            if x_step == 0 and y_step ==0:
                continue
            # 计算下一个点的x和y值，-1指示列表最后一个数
            next_x = self.x_values[-1] + x_step
            next_y = self.y_values[-1] + y_step
            # 附加到列表末尾
            self.x_values.append(next_x)
            self.y_values.append(next_y)



while True:
    # 创建一个RandomWalk实例，并将其包含的点都绘制出来
    rw = RandomWalk()
    rw.fill_walk()
    point_numbers = list(range(rw.num_points))
    plt.scatter(rw.x_values, rw.y_values, c=point_numbers, cmap=plt.cm.prism, edgecolors='none', s=15)
    plt.show()
    keep_running = input("Make another walk? (y/n): ")
    if keep_running == 'n':
        break


