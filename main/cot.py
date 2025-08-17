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
    "for a given source library based on functional compatibility, migration intent with mask, intent type, and retrieved similar historical precedents.\n\n"
    "Before producing the final list, internally follow this concise reasoning guide (do NOT print these steps):\n"
    "  1. Core‑Functionality Focus: Briefly extract the main functionality of the Source Library (e.g., “HTTP client”, “image processing”).\n"
    "  2. Purpose‑Type Alignment: Restate the masked migration purpose and intent type in one sentence (e.g., “performance optimization”, “compatibility”).\n"
    "  3. Historical‑Case Comparison: For each retrieved case, score functional similarity, intent relevance, and community activity; identify the top 2 strongest signals.\n"
    "  4. Candidate Generation: Based on the core functionality and intent, select at least 12 potential target libraries that satisfy steps 1–2.\n"
    "  5. Candidate Filtering & Ranking: Score these 12 candidates on  compatibility, maintenance status, and performance metrics; then take the top 10.\n"
    "  6. Diversity Check: Ensure the final 10 cover different technical approaches (e.g., sync vs. async, C‑based vs. pure‑Python) and contain no duplicates.\n"
    
    "Your task:\n"
    "1. For each case, analyze the source library, the  intent type and its migration intent of which the target library has been masked.\n"
    "2. Review the provided retrieved similar migration records for that case.\n"
    "3. Recommend EXACTLY 10 potential target libraries sorted by descending probability of suitability.\n\n"
    "Output requirements:\n"
    "- For each case, first print \"Case {rowid}:\" on its own line.\n"
    "- Then strictly follow a numbered list format:\n"
    "  1. First_recommendation\n"
    "  2. Second_recommendation\n"
    "  ...\n"
    "  10. Tenth_recommendation\n"
    "- Do not include any explanations, comments or additional text outside these blocks.\n"
)

USER_PROMPT_TEMPLATE = (
    "Case {rowid}\n"
    "Source Library: {source_library}\n"
    "Masked Migration Purpose: {masked_migration_purpose}\n"
    "Intent Type: {reason_type}\n"
    "Relevant Historical Cases:\n"
    "1. Source: {hist1_source} → Target: {hist1_target} | Intent: {hist1_intent} | Intent Type: {hist1_type}\n"
    "2. Source: {hist2_source} → Target: {hist2_target} | Intent: {hist2_intent} | Intent Type: {hist2_type}\n"
    "3. Source: {hist3_source} → Target: {hist3_target} | Intent: {hist3_intent} | Intent Type: {hist3_type}\n"
    "\n"
)


def load_tables(db_path: str):
    """从 SQLite 中读取两张表，并返回历史记录列表与待测列表"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT rowid, removed_lib, masked_migration_reason, reason_type, added_lib, migration_reason
        FROM new_retrieval
    """)
    retrievals = [dict(zip(
        ["rowid", "removed_lib", "masked_migration_reason", "reason_type", "added_lib", "migration_reason"],
        row
    )) for row in cur.fetchall()]

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

def make_batch_messages(batch_tests: List[Dict], retrievals: List[Dict], bm25, tokenized):
    """为一个批次的测试记录构造 single system + combined user 消息"""
    user_blocks = []
    for test in batch_tests:
        hist = get_topk_records(bm25, tokenized, retrievals, test, k=3)
        user_blocks.append(
            USER_PROMPT_TEMPLATE.format(
                rowid=test["rowid"],
                source_library=test["source_library"],
                masked_migration_purpose=test["masked_migration_reason"],
                reason_type=test["reason_type"],
                hist1_source=hist[0]["removed_lib"],
                hist1_target=hist[0]["added_lib"],
                hist1_intent=hist[0]["migration_reason"],
                hist1_type=hist[0]["reason_type"],
                hist2_source=hist[1]["removed_lib"],
                hist2_target=hist[1]["added_lib"],
                hist2_intent=hist[1]["migration_reason"],
                hist2_type=hist[1]["reason_type"],
                hist3_source=hist[2]["removed_lib"],
                hist3_target=hist[2]["added_lib"],
                hist3_intent=hist[2]["migration_reason"],
                hist3_type=hist[2]["reason_type"],
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

def main():
    retrievals, tests = load_tables(DB_PATH)
    bm25, tokenized = build_bm25(retrievals)
    # 检查现有输出，确定断点
    processed_ids = set()
    file_exists = os.path.exists(OUTPUT_CSV)
    if file_exists:
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row and row[0].isdigit():
                    processed_ids.add(int(row[0]))

    # 过滤待处理测试集
    if processed_ids:
        remaining_tests = [t for t in tests if t["rowid"] not in processed_ids]
    else:
        remaining_tests = tests

    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)

   
    mode = "a" if file_exists else "w"

    try:
       
        client = openai.OpenAI(
            api_key="xxx",
            base_url="xxx",
        )
        
        with open(OUTPUT_CSV, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rowid", "removed_lib", "added_lib"] + [f"rec{i}" for i in range(1, 11)])

            for i in range(0, len(remaining_tests), BATCH_SIZE):
                batch = remaining_tests[i : i + BATCH_SIZE]
                messages = make_batch_messages(batch, retrievals, bm25, tokenized)

                resp = client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    max_tokens=1000,  
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
