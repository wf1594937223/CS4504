import math

import torch


def _score_cold_candidates(model, user_ids, candidate_ids):
    """对每个用户计算其对全部冷启动候选电影的分数。"""
    user_vectors = model.user_embedding(user_ids)
    text_vectors = model.text_projector(model.text_features[candidate_ids])
    item_vectors = torch.zeros_like(text_vectors)

    batch_size = len(user_ids)
    candidate_count = len(candidate_ids)
    user_vectors = user_vectors[:, None, :].expand(-1, candidate_count, -1)
    item_vectors = item_vectors[None, :, :].expand(batch_size, -1, -1)
    text_vectors = text_vectors[None, :, :].expand(batch_size, -1, -1)
    combined_vectors = torch.cat(
        [user_vectors, item_vectors, text_vectors], dim=-1
    )
    return model.mlp(combined_vectors).squeeze(-1)


def _random_ranking_baseline(candidate_count, positive_count, k):
    """计算均匀随机排序的期望 HR@K 和 NDCG@K。"""
    miss_probability = 1.0
    for rank in range(k):
        remaining_negatives = candidate_count - positive_count - rank
        remaining_items = candidate_count - rank
        if remaining_negatives <= 0:
            miss_probability = 0.0
            break
        miss_probability *= remaining_negatives / remaining_items

    discounts = [1.0 / math.log2(rank + 2.0) for rank in range(k)]
    ideal_count = min(k, positive_count)
    ideal_dcg = sum(discounts[:ideal_count])
    expected_dcg = (positive_count / candidate_count) * sum(discounts)
    return 1.0 - miss_probability, expected_dcg / ideal_dcg


def cold_start_recommendation(
    model,
    dataset,
    train_data,
    cold_test_data,
    device="cuda",
    topk=10,
    eval_batch_size=64,
):
    print("\n" + "="*40)
    print("开始进行新电影（冷启动）评估")
    print("="*40)

    model.eval()
    # 1. 使用数据集按固定随机种子预留的电影作为冷启动候选。
    cold_movie_ids = list(dataset.get_cold_start_movie_ids())
    train_movies = set(train_data["movie_id"].unique())
    if train_movies.intersection(cold_movie_ids):
        raise ValueError("冷启动预留电影不应出现在训练集中")
    if not cold_movie_ids:
        print("训练集中没有未出现过的电影，跳过冷启动演示。")
        return

    cold_movie_set = set(cold_movie_ids)
    test_positives = {
        user_id: set(movie_ids) & cold_movie_set
        for user_id, movie_ids in cold_test_data.groupby("user_id")["movie_id"]
    }
    eligible_users = [
        (user_id, positives)
        for user_id, positives in test_positives.items()
        if positives
    ]
    candidate_ids = torch.tensor(cold_movie_ids, dtype=torch.long, device=device)

    print(f"冷启动候选电影数: {len(cold_movie_ids)}")
    print(f"测试集中有冷电影正样本的用户数: {len(eligible_users)}")
    print(f"测试冷电影正样本数: {sum(len(positives) for _, positives in eligible_users)}")

    if eligible_users:
        hit_results = []
        ndcg_results = []
        random_hr_results = []
        random_ndcg_results = []
        k = min(topk, len(cold_movie_ids))

        with torch.no_grad():
            for start in range(0, len(eligible_users), eval_batch_size):
                user_batch = eligible_users[start:start + eval_batch_size]
                user_ids = torch.tensor(
                    [user_id for user_id, _ in user_batch],
                    dtype=torch.long,
                    device=device,
                )
                batch_scores = _score_cold_candidates(
                    model, user_ids, candidate_ids
                )
                ranked_ids = candidate_ids[batch_scores.topk(k, dim=1).indices]

                for row, (_, positives) in enumerate(user_batch):
                    ranked = ranked_ids[row].tolist()
                    hits = [movie_id in positives for movie_id in ranked]
                    hit_results.append(float(any(hits)))
                    dcg = sum(
                        hit / math.log2(rank + 2.0)
                        for rank, hit in enumerate(hits)
                    )
                    ideal_dcg = sum(
                        1.0 / math.log2(rank + 2.0)
                        for rank in range(min(k, len(positives)))
                    )
                    ndcg_results.append(dcg / ideal_dcg if ideal_dcg else 0.0)

                    baseline_hr, baseline_ndcg = _random_ranking_baseline(
                        len(cold_movie_ids), len(positives), k
                    )
                    random_hr_results.append(baseline_hr)
                    random_ndcg_results.append(baseline_ndcg)

        print(f"HR@{topk}: {sum(hit_results) / len(hit_results):.4f}")
        print(f"NDCG@{topk}: {sum(ndcg_results) / len(ndcg_results):.4f}")
        print(
            f"随机排序基线 HR@{topk}: "
            f"{sum(random_hr_results) / len(random_hr_results):.4f}"
        )
        print(
            f"随机排序基线 NDCG@{topk}: "
            f"{sum(random_ndcg_results) / len(random_ndcg_results):.4f}"
        )
    else:
        print("测试集中没有训练未见电影的正样本，无法计算冷启动 HR/NDCG。")

    with torch.no_grad():
        # 展示训练交互最多的用户在全部冷电影上的 Top-K 推荐。
        user_id = int(train_data["user_id"].value_counts().index[0])
        user_tensor = torch.tensor([user_id], dtype=torch.long, device=device)
        scores = _score_cold_candidates(model, user_tensor, candidate_ids)[0]
        k = min(topk, len(cold_movie_ids))
        top_indices = scores.topk(k).indices.cpu().tolist()

    print(f"\n用户 {user_id} 的冷启动 Top-{k} 推荐:")
    for rank, index in enumerate(top_indices, start=1):
        movie_id = cold_movie_ids[index]
        title = dataset.items_df.loc[movie_id, "title"]
        print(f"{rank:2}. ID={movie_id}: {title} (score={scores[index].item():.4f})")

if __name__ == "__main__":
    # 这段代码假设你已经运行了 main.py 并保存了模型
    from dataset import ML1MDataset
    from recommender import HybridRecommender

    # 初始化数据和模型
    dataset = ML1MDataset("ml-1m")
    train_data = dataset.get_split_data("train")
    cold_test_data = dataset.get_split_data("cold_start")
    text_features = torch.load("movie_features.pt", map_location="cpu")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridRecommender(dataset.get_user_num(), dataset.get_item_num(), text_features, embed_dim=32).to(device)
    
    # 加载训练好的权重（如果你没保存，可以跳过这步，但跑出来的结果是随机的）
    try:
        model.load_state_dict(torch.load("checkpoint.pth", map_location=device))
        print("成功加载已训练好的模型权重。")
    except:
        print("警告：未找到 checkpoint.pth，将使用随机权重进行演示。")
        
    cold_start_recommendation(
        model, dataset, train_data, cold_test_data, device
    )
