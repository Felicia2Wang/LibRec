import sqlite3
import csv
import os
import re

from rank_bm25 import BM25Okapi
import openai
from typing import List, Dict

DB_PATH = "xxx"
OUTPUT_CSV = "xxx" 
BATCH_SIZE = 4                 

SYSTEM_PROMPT = (
    "You are a software migration recommendation assistant with expertise in suggesting replacement libraries "
    "Your task:\n"
    "1. For each case, analyze the source library.\n"
    "2. Recommend EXACTLY 10 potential target libraries sorted by descending probability of suitability.\n\n"
    "Output requirements:\n"
    "- For each case, first print \"Case {rowid}:\" on its own line.\n"
    "- Then strictly follow a numbered list format:\n"
    "  1. First_recommendation\n"
    "  2. Second_recommendation\n"
    "  ...\n"
    "  10. Tenth_recommendation\n"
    "- Do not include any explanations, comments or additional text outside these blocks.\n"
    "- Do not repeat the same recommendation in different cases.\n"
    "- Do not output the probability.\n"
)

USER_PROMPT_TEMPLATE = (
    "Case {rowid}\n"
    "Source Library: {source_library}\n"
    "\n"
)
def load_tables(db_path: str):
    """从 SQLite 中读取两张表，并返回历史记录列表与待测列表"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # 历史检索库
    cur.execute("""
        SELECT rowid, removed_lib, masked_migration_reason, reason_type, added_lib, migration_reason
        FROM new_retrieval
    """)
    retrievals = [dict(zip(
        ["rowid", "removed_lib", "masked_migration_reason", "reason_type", "added_lib", "migration_reason"],
        row
    )) for row in cur.fetchall()]

    # 待测记录表，现在包含 masked_migration_purpose 与 reason_type
    cur.execute("""
        SELECT rowid,
               removed_lib   AS source_library,
               added_lib,
               masked_migration_reason,
               reason_type
        FROM new_test
    """)
    tests = [dict(zip(
        ["rowid", "source_library", "added_lib","masked_migration_reason", "reason_type"],
        row
    )) for row in cur.fetchall()]

    conn.close()
    return retrievals, tests

def build_bm25(retrievals: List[Dict]):
    """基于 new_retrieval 的 removed_lib\migration_reason 和 reason_type 三列构建 BM25 索引"""
    corpus = [
        " ".join([r["removed_lib"], r["migration_reason"], r["reason_type"]])
        for r in retrievals
    ]
    tokenized = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized

def get_topk_records(bm25: BM25Okapi, tokenized: List[List[str]], retrievals: List[Dict],
                     test_item: Dict, k: int = 3):
    """检索与一条测试记录最相似的 Top-k 历史案例"""
    query = " ".join([test_item["source_library"], test_item["masked_migration_reason"], test_item["reason_type"]])
    scores = bm25.get_scores(query.split())
    topk_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [retrievals[i] for i in topk_idxs]

def make_batch_messages(batch_tests: List[Dict]):
    """为一个批次的测试记录构造 single system + combined user 消息"""
    user_blocks = []
    for test in batch_tests:
        
        user_blocks.append(
            USER_PROMPT_TEMPLATE.format(
                rowid=test["rowid"],
                source_library=test["source_library"],
            )
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "\n".join(user_blocks)}
    ]

def parse_response(text: str) -> Dict[int, List[str]]:
    """
    解析模型返回，将每个 Case {rowid}: 后的 10 条推荐提取成列表
    返回映射 { rowid: [rec1, ..., rec10], ... }
    """
    cases = {}
    parts = re.split(r"Case\s+(\d+):", text)
    # parts[0] 是前导内容，之后成对 (rowid, block)
    for idx in range(1, len(parts), 2):
        rowid = int(parts[idx])
        block = parts[idx+1]
        recs = []
        for line in block.strip().splitlines():
            m = re.match(r"\s*\d+\.\s*(\S+)", line)
            if m:
                recs.append(m.group(1))
        cases[rowid] = recs
    return cases

# ========================
# 主流程
# ========================
def main():
    retrievals, tests = load_tables(DB_PATH)
    

    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    try:
        client = openai.OpenAI(
            api_key="xxx",
            base_url="xxx",
        )
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rowid", "removed_lib", "added_lib"] + [f"rec{i}" for i in range(1, 11)])

            # 分批调用 OpenAI
            for i in range(0, len(tests), BATCH_SIZE):
                batch = tests[i : i + BATCH_SIZE]
                messages = make_batch_messages(batch)

                resp = client.chat.completions.create(
                    model="xxx",
                    messages=messages,
                    max_tokens=1000,  # 增加token限制
                    temperature=0.7,
                    top_p=0.95,
                )
                output = resp.choices[0].message.content

                batch_results = parse_response(output)
                print(f"Batch {i // BATCH_SIZE + 1} processed, results: {batch_results}")
                for test in batch:
                    rowid = test["rowid"]
                    recs = batch_results.get(rowid, [])
                    recs += [""] * (10 - len(recs))
                    
                    writer.writerow([rowid, test["source_library"], test["added_lib"]] + recs)
    except Exception as e:
        print(f"Recommendation failed: {str(e)}")
        raise
    print(f"完成，已将推荐结果写入 `{OUTPUT_CSV}`.")

if __name__ == "__main__":
    main()
