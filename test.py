import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 기존 평점 행렬
rating_matrix = np.array([
    [5, 3, 0, 1, 4, 2],
    [4, 0, 0, 1, 3, 5],
    [1, 1, 0, 5, 2, 4],
    [0, 0, 5, 4, 1, 3],
    [0, 0, 5, 4, 2, 2],
    [0, 0, 5, 4, 2, 3],
    [4, 3, 4, 1, 5, 5],
    [3, 4, 2, 2, 5, 4],
    [2, 5, 1, 4, 3, 3],
    [1, 4, 3, 5, 2, 2],
    [5, 2, 3, 1, 4, 4],
    [4, 1, 2, 2, 3, 5],
    [3, 5, 4, 1, 2, 2],
    [2, 4, 5, 5, 3, 1],
    [1, 3, 2, 4, 2, 5],
    [5, 4, 3, 3, 5, 4],
    [4, 2, 1, 5, 2, 3],
    [3, 3, 4, 4, 4, 2],
    [2, 1, 5, 3, 5, 1],
    [1, 2, 3, 2, 4, 5],
])

# 랜덤값 포함하여 10배로 늘리기
np.random.seed(42)  # 재현성을 위해 시드 설정
num_users, num_movies = rating_matrix.shape
new_rating_matrix = np.zeros((num_users * 10, num_movies))

for i in range(num_users * 10):
    base_user = i % num_users
    random_values = np.random.randint(0, 6, num_movies)
    new_rating_matrix[i] = rating_matrix[base_user] + random_values

# PCA를 사용한 차원 축소
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(new_rating_matrix)

# k-means 클러스터링
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(reduced_data)
clusters = kmeans.labels_

# 결과 시각화
plt.figure(figsize=(10, 8))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', s=50)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-means Clustering of Movie Preferences with Randomized Data')
plt.colorbar()
plt.savefig('/home/ubuntu/project/test.png')
plt.show()
