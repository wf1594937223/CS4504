import torch

def cold_start_recommendation(model, dataset, train_data, device="cuda"):
    print("\n" + "="*40)
    print("开始进行新电影（冷启动）推荐演示")
    print("="*40)

    model.eval()
    # 1. 找几部在训练集中完全没有出现过的“新电影”
    # 获取所有电影ID
    all_movies = set(dataset.items_df.index)
    # 获取训练集中出现过的电影ID
    train_movies = set(train_data["movie_id"].unique())
    # 找出新电影（注意排除掉一些无效ID）
    new_movie_ids = sorted(all_movies - train_movies)
    
    # 固定排序后取 3 部，保证每次演示结果可复现。
    demo_new_movies = new_movie_ids[:3]
    if not demo_new_movies:
        print("训练集中没有未出现过的电影，跳过冷启动演示。")
        return
    
    # 找出一个在训练集里看片很多的老用户做推荐
    user_id = train_data["user_id"].value_counts().index[0] 
    print(f"目标用户 ID: {user_id} （看过 {train_data['user_id'].value_counts()[user_id]} 部电影）")

    with torch.no_grad():
        # 2. 直接使用已缓存的 GPT-2 标题与类型特征，避免重复加载模型。
        new_item_ids_tensor = torch.tensor(demo_new_movies, device=device)
        new_text_features_768 = model.text_features[new_item_ids_tensor]

        # 3. 构造新电影的完整特征并打分。
        # 新电影没有训练交互，ID 向量置零，仅保留文本侧的电影信息。
        new_item_vecs = torch.zeros(
            len(demo_new_movies),
            model.item_embedding.embedding_dim,
            device=device,
            dtype=model.user_embedding.weight.dtype,
        )
        
        # 文本特征经过投影层降维
        new_text_vecs = model.text_projector(new_text_features_768)
        
        # 用户特征
        user_vec = model.user_embedding(torch.tensor([user_id], device=device))
        
        # 拼接并送入 MLP 计算分数
        # 注意：需要把 user_vec 复制到每部新电影。
        user_vecs_expanded = user_vec.expand(len(demo_new_movies), -1)
        
        combined_vecs = torch.cat([user_vecs_expanded, new_item_vecs, new_text_vecs], dim=-1)
        scores = model.mlp(combined_vecs).squeeze(-1)

        # 4. 打印结果。
        print("\n--- 新电影推荐得分 ---")
        for i, mid in enumerate(demo_new_movies):
            title = dataset.items_df.loc[mid, 'title']
            score = scores[i].item()
            print(f"新电影: ID={mid}, 名称='{title}'")
            print(f"  -> 模型预测得分: {score:.4f} (由于没有交互记录，全凭GPT-2文本特征打分)")

if __name__ == "__main__":
    # 这段代码假设你已经运行了 main.py 并保存了模型
    from dataset import ML1MDataset
    from recommender import HybridRecommender

    # 初始化数据和模型
    dataset = ML1MDataset("ml-1m")
    train_data = dataset.get_split_data("train")
    text_features = torch.load("movie_features.pt", map_location="cpu")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridRecommender(dataset.get_user_num(), dataset.get_item_num(), text_features, embed_dim=32).to(device)
    
    # 加载训练好的权重（如果你没保存，可以跳过这步，但跑出来的结果是随机的）
    try:
        model.load_state_dict(torch.load("checkpoint.pth", map_location=device))
        print("成功加载已训练好的模型权重。")
    except:
        print("警告：未找到 checkpoint.pth，将使用随机权重进行演示。")
        
    cold_start_recommendation(model, dataset, train_data, device)
