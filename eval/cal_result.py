import pandas as pd
from typing import List

def calculate_metrics(csv_path: str):
    """
    计算precision@k和MRR指标
    :param csv_path: 结果文件路径
    :return: 指标字典
    """
    df = pd.read_csv(csv_path)
    df['added_lib'] = df['added_lib'].str.strip().str.lower()
    for i in range(1, 11):
        df[f'rec{i}'] = df[f'rec{i}'].str.strip().str.lower()

    metrics = {
        'precision@1': 0,
        'precision@3': 0,
        'precision@5': 0,
        'precision@10': 0,
        'mrr': 0
    }
    
    total_samples = len(df)
    
    for _, row in df.iterrows():
        gt = row['added_lib']
        recs = [row[f'rec{i}'] for i in range(1, 11)]
        
        # 计算precision@k
        found = [False] * 10
        for i in range(10):
            if recs[i] == gt:
                found[i] = True
              
                for j in range(i, 10):
                    found[j] = True
                break
        
        # 累加各precision@k
        metrics['precision@1'] += int(found[0])
        metrics['precision@3'] += int(any(found[:3]))
        metrics['precision@5'] += int(any(found[:5]))
        metrics['precision@10'] += int(any(found))
        
        # 计算MRR,即倒数排名的平均值
        for rank in range(10):
            if recs[rank] == gt:
                metrics['mrr'] += 1 / (rank + 1)
                break
    
    # 计算平均值
    for k in [1, 3, 5, 10]:
        metrics[f'precision@{k}'] = round(metrics[f'precision@{k}'] / total_samples, 4)
    metrics['mrr'] = round(metrics['mrr'] / total_samples, 4)
    
    return metrics

def print_metrics(metrics: dict):
    """格式化打印指标结果"""
    print("Evaluation Metrics:")
    print("-" * 30)
    print(f"Precision@1 : {metrics['precision@1']:.4f}")
    print(f"Precision@3 : {metrics['precision@3']:.4f}")
    print(f"Precision@5 : {metrics['precision@5']:.4f}")
    print(f"Precision@10: {metrics['precision@10']:.4f}")
    print(f"MRR         : {metrics['mrr']:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    # 使用示例
    result_path = "xxxx"  # 替换为实际路径
    metrics = calculate_metrics(result_path)
    print_metrics(metrics)
