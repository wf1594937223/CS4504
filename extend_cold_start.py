import torch
import pandas as pd
from pathlib import Path
from transformers import GPT2Model, GPT2Tokenizer

def cold_start_recommendation(model, dataset, train_data, device="cuda"):
    print("\n" + "="*40)
    print("开始进行新电影（冷启动）推荐演示")
    print("="*40)

    model.eval()
    model_source, local_only = "gpt2_model", True
    if not Path(model_source).is_dir():
        model_source, local_only = "gpt2", False
        
    # 1. 加载 GPT-2 用于实时提取新电影特征
    tokenizer = GPT2Tokenizer.from_pretrained(model_source, local_files_only=local_only)
    tokenizer.pad_token = tokenizer.eos_token
    gpt2 = GPT2Model.from_pretrained(model_source, local_files_only=local_only).to(device).eval()

    # 2. 找几部在训练集中完全没有出现过的“新电影”
    # 获取所有电影ID
    all_movies = set(dataset.items_df.index)
    # 获取训练集中出现过的电影ID
    train_movies = set(train_data["movie_id"].unique())
    # 找出新电影（注意排除掉一些无效ID）
    new_movie_ids = list(all_movies - train_movies)
    
    # 随便挑 3 部新电影做演示
    demo_new_movies = new_movie_ids[:3] 
    
    # 找出一个在训练集里看片很多的老用户做推荐
    user_id = train_data["user_id"].value_counts().index[0] 
    print(f"目标用户 ID: {user_id} （看过 {train_data['user_id'].value_counts()[user_id]} 部电影）")

    with torch.no_grad():
        # 3. 提取新电影的文本特征 (TODO 1 的掩码平均池化在这里复用)
        new_texts = []
        for mid in demo_new_movies:
            title = dataset.items_df.loc[mid, 'title']
            genres = dataset.items_df.loc[mid, 'genres'].replace('|', ', ')
            new_texts.append(f"Title: {title}. Genres: {genres}.")
            
        inputs = tokenizer(new_texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        hidden = gpt2(**inputs).last_hidden_state
        mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        sum_hidden = torch.sum(hidden * mask_expanded, dim=1)
        valid_counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        new_text_features_768 = sum_hidden / valid_counts # [3, 768]

        # 4. 构造新电影的完整 96 维特征并打分
        # 对于新电影，其 ID Embedding 要么是随机的（如果在模型初始化矩阵中），要么置零
        # 我们直接用 model.item_embedding 查表，或者用全0代替（代表没有协同过滤信号）
        new_item_ids_tensor = torch.tensor(demo_new_movies, device=device)
        
        # 模拟：ID 向量使用随机值（因为模型没训练过它们，相当于冷启动）
        # 如果你想更严谨，可以改成 torch.zeros(...)
        new_item_vecs = model.item_embedding(new_item_ids_tensor) # [3, 32]
        
        # 文本特征经过投影层降维
        new_text_vecs = model.text_projector(new_text_features_768) # [3, 32]
        
        # 用户特征
        user_vec = model.user_embedding(torch.tensor([user_id], device=device)) # [1, 32]
        
        # 拼接并送入 MLP 计算分数
        # 注意：需要把 user_vec 复制 3 次，以对应 3 部新电影
        user_vecs_expanded = user_vec.expand(3, -1)
        
        combined_vecs = torch.cat([user_vecs_expanded, new_item_vecs, new_text_vecs], dim=-1) # [3, 96]
        scores = model.mlp(combined_vecs).squeeze(-1) # [3]

        # 5. 打印结果
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
    import torch

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
