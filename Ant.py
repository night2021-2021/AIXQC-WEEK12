import numpy as np
import pandas as pd
import random

class AntColony:
    def __init__(self, distances, n_ants, n_iterations, alpha=1, beta=2, evaporation_rate=0.5, q=10, start_city=0):
        self.distances = distances  # 距離矩陣
        self.n_cities = len(distances)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # 信息素重要程度
        self.beta = beta  # 距離重要程度
        self.evaporation_rate = evaporation_rate  # 信息素蒸發率
        self.q = q  # 信息素強度
        self.start_city = start_city  # 使用者指定的起點
        # 根據距離初始化信息素（距離越大信息素越少）
        self.pheromone = 1 / (self.distances + 1e-10)

    def run(self):
        best_route = None
        best_distance = float('inf')

        for iteration in range(self.n_iterations):
            all_routes = []
            all_distances = []

            for ant in range(self.n_ants):
                route = self.generate_route()
                distance = self.calculate_distance(route)
                all_routes.append(route)
                all_distances.append(distance)

                if distance < best_distance:
                    best_route = route
                    best_distance = distance

            self.update_pheromone(all_routes, all_distances)
            print(f"Iteration {iteration + 1}, Best Distance: {best_distance}")

        return best_route, best_distance

    def generate_route(self):
        route = [self.start_city]  # 使用指定的起點
        while len(route) < self.n_cities:
            current_city = route[-1]
            probabilities = self.calculate_probabilities(current_city, route)
            next_city = self.select_next_city(probabilities)
            route.append(next_city)
        return route

    def calculate_probabilities(self, current_city, visited):
        probabilities = []
        for next_city in range(self.n_cities):
            if next_city in visited:
                probabilities.append(0)
            else:
                pheromone = self.pheromone[current_city][next_city] ** self.alpha
                distance = self.distances[current_city][next_city] ** (-self.beta)
                probabilities.append(pheromone * distance)
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        return probabilities

    def select_next_city(self, probabilities):
        if random.random() < 0.1:  # 假設 10% 的概率選擇信息素最濃的城市
            return np.argmax(probabilities)
        else:
            cumulative_sum = np.cumsum(probabilities)
            random_value = random.random()
            for i, cumulative_probability in enumerate(cumulative_sum):
                if random_value <= cumulative_probability:
                    return i

    def calculate_distance(self, route):
        distance = 0
        for i in range(len(route) - 1):
            distance += self.distances[route[i]][route[i + 1]]
        distance += self.distances[route[-1]][route[0]]  # 回到起點
        return distance

    def update_pheromone(self, routes, distances):
        self.pheromone *= (1 - self.evaporation_rate)
        best_index = np.argmin(distances)
        best_route = routes[best_index]
        best_distance = distances[best_index]

        for i in range(len(best_route) - 1):
            self.pheromone[best_route[i]][best_route[i + 1]] += self.q / best_distance
        self.pheromone[best_route[-1]][best_route[0]] += self.q / best_distance


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

    # 初始化螞蟻演算法
    colony = AntColony(distances, n_ants=20, n_iterations=100, start_city=start_city_index)
    best_route, best_distance = colony.run()

    # 將最佳路徑索引轉換為行政區名稱
    best_route_names = [district_names[i] for i in best_route]

    # 輸出結果
    print(f"\n起點行政區: {start_district}")
    print("最佳路徑 (行政區):", best_route_names)
    print("最短距離:", best_distance)