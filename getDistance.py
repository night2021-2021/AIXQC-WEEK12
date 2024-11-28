import googlemaps
import pandas as pd
import numpy as np

# 初始化 Google Maps 客戶端
API_KEY = "AIzaSyAeFz33Bt_DRe-gVNHzrmvaKorCQ9bNZvQ"  # 替換為你的 API Key
gmaps = googlemaps.Client(key=API_KEY)

# 縣市政府地址
locations = [
    "台北市政府", "新北市政府", "桃園市政府", "台中市政府", "台南市政府", 
    "高雄市政府", "基隆市政府", "新竹市政府", "苗栗縣政府", "彰化縣政府", 
    "南投縣政府", "雲林縣政府", "嘉義縣政府", "屏東縣政府", "宜蘭縣政府", 
    "花蓮縣政府", "台東縣政府", "澎湖縣政府", "新竹縣政府", "嘉義市政府"
]

# 生成空的距離矩陣
n = len(locations)
distance_matrix = np.zeros((n, n))

# 計算距離矩陣
for i, origin in enumerate(locations):
    result = gmaps.distance_matrix(origins=[origin], destinations=locations, mode="driving")
    for j, row in enumerate(result['rows'][0]['elements']):
        if row['status'] == 'OK':
            distance_matrix[i][j] = row['distance']['value']  # 距離以公尺為單位
        else:
            distance_matrix[i][j] = None  # 無法計算距離

# 將矩陣轉為 DataFrame
df = pd.DataFrame(distance_matrix, index=locations, columns=locations)
df = df.applymap(lambda x: round(x / 1000, 2) if pd.notnull(x) else None)  # 轉為公里並保留小數點2位

# 儲存為 CSV 或顯示結果
df.to_csv("distance_matrix.csv", encoding="utf-8-sig")
print(df)
