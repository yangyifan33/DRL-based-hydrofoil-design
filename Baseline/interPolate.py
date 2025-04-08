from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import math

class calinterPlate:
    def __init__(self,CpArray):
        self.func = interploate(CpArray)
        self.CpArray = CpArray
        self.x_bigen = CpArray[np.argmin(CpArray[:,0]),0]
        self.x_end = min(CpArray[0,0],CpArray[-1,0])

    def compute_argmin(self):
        # compute the argmin of the CpArray
        x = np.linspace(self.x_bigen,self.x_end,100)
        id_min = np.argmin(self.func[1](x))
        argmin = x[id_min]
        return argmin

    def compute_min(self):
        # compute the global min and middle min of the CpArray
        x_global = np.linspace(self.x_bigen,self.x_end,100)
        x_middle = np.linspace(0.2,0.8,60)
        Cpmin_global = np.min(self.func[1](x_global))
        Cpmin_middle = np.min(self.func[1](x_middle))
        Cpmin = [Cpmin_global, Cpmin_middle]
        return Cpmin
    
    def compute_middle_parameter(self):
        # compute the k_fit of the CpArray
        x = np.linspace(0.2,0.8,100)
        Cp_distance = self.func[0](x) - self.func[1](x)
        k, _ = np.polyfit(x, Cp_distance, 1)
        parameter = k
        return parameter

    def is_injected(self):
        # compute the degree of crossover
        x = np.linspace(self.x_bigen,0.2,100)
        diff = self.func[0](x) - self.func[1](x)
        value_is_injected = np.min(diff)
        return value_is_injected
    

def interploate(CpArray):
    # interpolate the CpArray given by the panel method
    x = CpArray[:,0]
    Cp_array_up = CpArray[0:np.argmin(x)+1,:]
    Cp_array_down = CpArray[np.argmin(x):,:]

    x_up = Cp_array_up[:,0]
    Cp_up = Cp_array_up[:,1]

    x_down = Cp_array_down[:,0]
    Cp_down = Cp_array_down[:,1]

    f_up = interp1d(x_up,Cp_up) # function of the upper edge
    f_down = interp1d(x_down,Cp_down) # function of the lowwer edge
    func = [f_up, f_down]
    return func

if __name__ == "__main__":
    import panel2D as pan
    cal = False
    co = [-25, 3, -6, -5, 15]
    def caculaton(alpha, f_c):
        s1 = pan.solver()
        s1.solve(angleInDegree = alpha, camberRatio = f_c)
        I1 = calinterPlate(s1._CpArray)
        pressure_min1 = I1.compute_min()
        middle_parameter1 = I1.compute_middle_parameter()
        value_is_injected = I1.is_injected()
        reward1 = (
            co[0] * abs(1.15 - s1._Cl) + 
            co[1] * pressure_min1[0] + 
            co[2] * (pressure_min1[0] / pressure_min1[1] - 1) + 
            co[3] * abs(middle_parameter1) + 
            co[4] * value_is_injected
        )
        print(f"for {alpha}, {f_c}")
        print(reward1)
        print( abs(1.15 - s1._Cl), 0)
        print( pressure_min1[0], 1)
        print( (pressure_min1[0] / pressure_min1[1] - 1), 2)
        print( abs(middle_parameter1), 4)
        print(value_is_injected, 5)

        done = (
                    (abs(1.15 - s1._Cl) < 0.01) and  
                    pressure_min1[0] / pressure_min1[1] <= 1.05 and 
                    abs(middle_parameter1) <= 0.6 and 
                    value_is_injected >= -0.01
                    )
        print(done)


    def plot_rewardMap(clt):
        # 定义x, y的范围和步长
        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 0.1, 10)
        X, Y = np.meshgrid(x, y)
        def reward_function(x, y):
            s1 = pan.solver()
            s1.solve(angleInDegree = x, camberRatio = y)
            I1 = calinterPlate(s1._CpArray)
            pressure_min1 = I1.compute_min()
            middle_parameter1 = I1.compute_middle_parameter()
            value_is_injected = I1.is_injected()
            reward1 = (
            co[0] * abs(clt - s1._Cl) + 
            co[1] * pressure_min1[0] + 
            co[2] * (pressure_min1[0] / pressure_min1[1] - 1) + 
            co[3] * abs(middle_parameter1) + 
            co[4] * value_is_injected
        )
            reward1 = - (-reward1 /70) ** 0.5
            
            return reward1

        reward_vectorized = np.vectorize(reward_function)

        Z = reward_vectorized(X, Y)
        # C = cl_ver(X,Y)
        # 绘制热力图
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, levels = 50 ,cmap='viridis')  # 选择合适的颜色映射
        plt.colorbar(label='Reward')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Heatmap of Reward for Cl = {clt}')
        plt.savefig(f'./Heatmaps/Heatmap of Reward for Cl = {clt} .jpg')
        plt.close()

    def plot_termianl_stations(clt):
        # 定义x, y的范围和步长
        x = np.linspace(0, 5, 50)
        y = np.linspace(0, 0.1, 50)
        X, Y = np.meshgrid(x, y)

        # 定义reward函数，这里使用一个示例函数，可以根据实际需求修改
        def stop_function(x, y):
            s1 = pan.solver()
            s1.solve(angleInDegree = x, camberRatio = y)
            I1 = calinterPlate(s1._CpArray)
            pressure_min1 = I1.compute_min()
            middle_parameter1 = I1.compute_middle_parameter()
            value_is_injected = I1.is_injected()
            done = (
                    (abs(clt - s1._Cl) < 0.01) and  
                    pressure_min1[0] / pressure_min1[1] <= 1.05 and 
                    abs(middle_parameter1) <= 0.6 and
                    value_is_injected >= -0.01
                    )
            if(done): ans = 1
            else: ans = 0
            return ans
        
        reward_vectorized = np.vectorize(stop_function)
        # 计算reward值
        Z = reward_vectorized(X, Y)
        # 绘制热力图
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, levels = 1 ,cmap='coolwarm')  # 选择合适的颜色映射
        plt.colorbar(label='done')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Heatmap of Reward for partial Terminal Stations for Cl = {clt}')
        plt.show()

    import numpy as np

    def cal_rewardcoe(clt):
        # 定义 x, y 的范围和步长
        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 0.1, 10)
        X, Y = np.meshgrid(x, y)  # 生成网格点
        Rewards = [[], [], [], [], []]  # 存储每项 reward 的值

        def reward_function(x, y):
            s1 = pan.solver()
            s1.solve(angleInDegree=x, camberRatio=y)
            I1 = calinterPlate(s1._CpArray)
            pressure_min1 = I1.compute_min()
            middle_parameter1 = I1.compute_middle_parameter()
            value_is_injected = I1.is_injected()
            reward1 = [
                abs(clt - s1._Cl), 
                pressure_min1[0],
                (pressure_min1[0] / pressure_min1[1] - 1), 
                abs(middle_parameter1),
                value_is_injected
            ]
            return reward1

        # 遍历每个 (x, y) 位置，计算 reward
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                rewards = reward_function(X[i, j], Y[i, j])  # 计算 reward1
                for k in range(5):  # 存储每项 reward
                    Rewards[k].append(rewards[k])

        # 计算每一项 reward 的标准差
        std_devs = [np.std(Rewards[i]) for i in range(5)]

        return std_devs
    plot_rewardMap(0.75)
    plot_rewardMap(1.15)
    plot_rewardMap(1.5)