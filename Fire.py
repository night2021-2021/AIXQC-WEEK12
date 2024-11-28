import numpy as np
import pandas as pd
import random

class SimulatedAnnealing:
    def __init__(self, distances, initial_temperature=1000, cooling_rate=0.995, stop_temperature=1e-1, max_iter=1000):
        self.distances = distances  # 距離矩陣
        self.n_cities = len(distances)
        self.temperature = initial_temperature  # 初始溫度
        self.cooling_rate = cooling_rate  # 降溫速率
        self.stop_temperature = stop_temperature  # 終止溫度
        self.max_iter = max_iter  # 每個溫度下的最大迭代次數

    def run(self, start_city=0):
        # 初始化路徑（以起點開始）
        current_route = list(range(self.n_cities))
        random.shuffle(current_route)
        if start_city != 0:
            current_route.remove(start_city)
            current_route = [start_city] + current_route

        current_distance = self.calculate_distance(current_route)
        best_route = current_route[:]
        best_distance = current_distance

        while self.temperature > self.stop_temperature:
            for _ in range(self.max_iter):
                new_route = self.generate_neighbor(current_route)
                new_distance = self.calculate_distance(new_route)

                # 接受新解的條件
                if self.acceptance_probability(current_distance, new_distance):
                    current_route = new_route
                    current_distance = new_distance

                    # 更新最優解
                    if new_distance < best_distance:
                        best_route = new_route[:]
                        best_distance = new_distance

            # 降低溫度
            self.temperature *= self.cooling_rate
            print(f"Temperature: {self.temperature:.4f}, Best Distance: {best_distance}")

        return best_route, best_distance

    def calculate_distance(self, route):
        distance = 0
        for i in range(len(route) - 1):
            distance += self.distances[route[i]][route[i + 1]]
        distance += self.distances[route[-1]][route[0]]  # 回到起點
        return distance

    def generate_neighbor(self, route):
        # 隨機交換兩個城市的位置
        new_route = route[:]
        i, j = random.sample(range(1, self.n_cities), 2)  # 起點不參與交換
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route

    def acceptance_probability(self, current_distance, new_distance):
        if new_distance < current_distance:
            return True
        else:
            # 機率接受較差解
            return np.random.rand() < np.exp((current_distance - new_distance) / self.temperature)


if __name__ == "__main__":
    # 載入距離矩陣和行政區名稱
    file_path = "County.csv"
    df = pd.read_csv(file_path, index_col=0)
    distances = df.values  # 提取數值部分
    district_names = df.index.tolist()  # 提取行政區名稱

    # 確保距離矩陣對稱，並設置對角線為 0
    distances = (distances + distances.T) / 2
    np.fill_diagonal(distances, 0)

    # 提示使用者輸入起點行政區
    print("台北市行政區列表:")
    print(", ".join(district_names))
    start_district = input("請輸入起點行政區名稱：").strip()

    # 驗證輸入是否合法
    if start_district not in district_names:
        print("輸入的行政區名稱無效！請重新執行程式並輸入正確名稱。")
        exit()

    start_city_index = district_names.index(start_district)  # 找到起點索引

    # 初始化退火算法
    sa = SimulatedAnnealing(distances)
    best_route, best_distance = sa.run(start_city=start_city_index)

    # 將最佳路徑索引轉換為行政區名稱
    best_route_names = [district_names[i] for i in best_route]

    # 輸出結果
    print(f"\n起點行政區: {start_district}")
    print("最佳路徑 (行政區):", best_route_names)
    print("最短距離:", best_distance)
