from pathlib import Path

import numpy as np
import pandas as pd


class ML1MDataset:
    """读取 MovieLens-1M，并划分常规数据集与固定冷启动电影集。"""

    cold_start_seed = 2025
    cold_start_ratio = 0.1
    min_cold_start_ratings = 20

    def __init__(self, data_path, split_ratio=(0.8, 0.1, 0.1)):
        self.data_path = Path(data_path).resolve()
        required = ("users.dat", "movies.dat", "ratings.dat")
        for filename in required:
            if not (self.data_path / filename).is_file():
                raise FileNotFoundError(f"缺少数据文件：{self.data_path / filename}")

        self._load_data()
        self._generate_splits(split_ratio)

        # MovieLens 的编号从 1 开始。加 1 后可以直接把原编号作为 Embedding 下标。
        self.user_num = int(self.users_df.index.max()) + 1
        self.item_num = int(self.items_df.index.max()) + 1

    def _load_data(self):
        """分别读取用户、电影和评分数据。:: 是 MovieLens-1M 的字段分隔符。"""
        self.users_df = pd.read_csv(
            self.data_path / "users.dat",
            sep="::",
            names=["user_id", "gender", "age", "occupation", "zip_code"],
            engine="python",
        ).set_index("user_id")

        self.items_df = pd.read_csv(
            self.data_path / "movies.dat",
            sep="::",
            names=["movie_id", "title", "genres"],
            engine="python",
            encoding="latin-1",
        ).set_index("movie_id")

        self.interactions_df = pd.read_csv(
            self.data_path / "ratings.dat",
            sep="::",
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python",
        )

    def _generate_splits(self, split_ratio):
        """固定留出一组电影，其余评分记录按比例划分。"""
        if not np.isclose(sum(split_ratio), 1.0):
            raise ValueError("split_ratio 三项之和必须为 1")

        data = self.interactions_df.sample(frac=1, random_state=42).reset_index(drop=True)
        movie_rating_counts = data["movie_id"].value_counts()
        eligible_movies = np.sort(
            movie_rating_counts[
                movie_rating_counts >= self.min_cold_start_ratings
            ].index.to_numpy()
        )
        holdout_count = int(round(len(eligible_movies) * self.cold_start_ratio))
        if len(eligible_movies) and holdout_count == 0:
            holdout_count = 1
        if holdout_count:
            rng = np.random.RandomState(self.cold_start_seed)
            selected_movies = rng.choice(
                eligible_movies, size=holdout_count, replace=False
            )
            self.cold_start_movie_ids = tuple(
                sorted(int(movie_id) for movie_id in selected_movies)
            )
        else:
            self.cold_start_movie_ids = ()

        # 0、1、2、3 分别代表训练、验证、测试和独立冷启动测试。
        labels = np.full(len(data), 3, dtype=np.int8)
        warm_indices = np.flatnonzero(
            ~data["movie_id"].isin(self.cold_start_movie_ids).to_numpy()
        )
        train_end = int(len(warm_indices) * split_ratio[0])
        valid_end = int(len(warm_indices) * (split_ratio[0] + split_ratio[1]))
        labels[warm_indices[:train_end]] = 0
        labels[warm_indices[train_end:valid_end]] = 1
        labels[warm_indices[valid_end:]] = 2
        data["split"] = labels
        self.interactions_df = data

    def get_split_data(self, split="train", filter_new_users=True):
        """返回指定集合；验证和测试阶段默认只保留训练时见过的用户。"""
        if split is None:
            return tuple(self.get_split_data(name) for name in ("train", "validation", "test"))

        split_map = {"train": 0, "validation": 1, "test": 2, "cold_start": 3}
        if split not in split_map:
            raise ValueError("split 必须是 train、validation、test 或 cold_start")

        result = self.interactions_df[
            self.interactions_df["split"] == split_map[split]
        ].copy()
        if filter_new_users and split != "train":
            train_users = set(
                self.interactions_df.loc[self.interactions_df["split"] == 0, "user_id"]
            )
            result = result[result["user_id"].isin(train_users)]
        return result

    def get_cold_start_movie_ids(self):
        """返回固定预留的冷启动电影 ID。"""
        return self.cold_start_movie_ids

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def __len__(self):
        return len(self.interactions_df)

    def __str__(self):
        interactions = len(self.interactions_df)
        users = self.interactions_df["user_id"].nunique()
        items = self.interactions_df["movie_id"].nunique()
        sparsity = 1 - interactions / (users * items)
        return (
            "MovieLens-1M Dataset\n"
            f"Users: {users}\nMovies: {items}\n"
            f"Interactions: {interactions}\nSparsity: {sparsity:.2%}"
        )


if __name__ == "__main__":
    dataset = ML1MDataset("ml-1m")
    print(dataset)
    print("Split sizes:", [len(part) for part in dataset.get_split_data(None)])
