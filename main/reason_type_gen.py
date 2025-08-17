import sqlite3
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

def type_gen(migration_resaon):
    system_prompt = """
You are an expert in software engineering and library migration. Given only the list of migration purposes, your task is to classify the reason(s) for migration strictly according to the predefined categories listed below.

Migration Reason Standard:
1. Source Library Issues
   - Not Maintained/Outdated: The source library is no longer maintained or has become outdated.
   - Bug/Defect Issues: The source library contains unresolved bugs or defects OR the source library can not be downloaded or installed successfully because of its own issues.
   - Security Vulnerability: The source library has known security vulnerabilities OR The source library has security-related issues such as thread insafety and memory leaks.
2. Target Library Advantages
   - Usability: The new library is easier to use or integrate.
   - Enhanced Features: The new library offers additional or improved functionalities.
   - Performance: The new library provides better performance or resource efficiency.
   - Activity:Although the source library is under maintenance, the project still chooses to use a more recent, well-maintained dependency.
   - Popularity: The target library is more widely used or complies with industrial standards/ecosystem best practices.
   - Size/Complexity: The new library is simpler, lighter, or more streamlined.
3. Project Specific Reasons
   - Integration: Migration is required for compatibility or better integration with other components.
   - Simplification: The goal is to simplify the codebase or reduce dependency overhead.
   - License: Migration is driven by licensing issues.
   - Organizational Influence: Migration is influenced by internal policies or strategic decisions.
4. Other: Use this category if none of the above subcategories apply or if the reason is unclear.

Important:
- You may select multiple subcategories if applicable.
- Only use the exact subcategory names from the standard above (e.g., "Not Maintained/Outdated", "Enhanced Features", etc.).
- Do NOT return parent categories like "Source Library Issues" or "Target Library Advantages".
- Do NOT invent new categories or modify existing ones.
- Output only the JSON format, and nothing else.

Output format(Output only the following JSON format (and nothing else):
{
  "migration reason type": ["<Category Name 1>", "<Category Name 2>", ...]
}

Example:
    Input:
    the list of migration purposes: ["Improve model performance","Simplify training loop"]
    Output:
    {
      "migration reason type": ["Performance","Simplification"]
    }
"""
    user_prompt = f"""
## New Analysis Reques
Input:
{{
  "the list of migration purposes": {migration_resaon}
}}
Output:
{{
  "migration reason type": ["<Category Name 1>", "<Category Name 2>", ...]
}}
Please output your final result and includes only the key "migration reason type".
"""

    try:
        client = OpenAI(
            api_key="xxx", 
            base_url="xxx",
        )
        response = client.chat.completions.create(
            model="gpt-4o",  # 唯一需要修改的地方
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
            n=1,  
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {str(e)}")
        return None

def parse_response(response):
    """解析大模型响应"""
    try:
        data = json.loads(response)
        return data.get("migration reason type", "error")
    except json.JSONDecodeError:
        print("Invalid JSON response")
        return "error"

def process_row(row):
    """
    处理单条记录：调用 API 得到结果，并返回更新所需信息
    """
    repo_owner, repo_name, commit_sha, commit_message, removed_lib, added_lib,migration_reason = row
    response = type_gen(migration_reason)
    print(f"Response for {repo_owner}/{repo_name}@{commit_sha}: {response}")
    if response:
        migration_type = parse_response(response)
        # 将结果转换成字符串，便于保存到数据库
        migration_type_str = str(migration_type)
        return (repo_owner, repo_name, commit_sha, commit_message, migration_type_str)
    return None

def main():

    db_path = 'xxx'
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # 确保字段存在
        cursor.execute("PRAGMA table_info(purpose_gen_success)")
        columns = [col[1] for col in cursor.fetchall()]
        for col in ['reason_type']:
            if col not in columns:
                cursor.execute(f"ALTER TABLE purpose_gen_success ADD COLUMN {col} TEXT")
                conn.commit()

        cursor.execute("""
            xxx
        """)
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return


    results = []
    max_workers = 5  # 可根据实际需求和API限流情况进行调整
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {executor.submit(process_row, row): row for row in rows}
        for future in tqdm(as_completed(future_to_row), total=len(rows), desc="Processing"):
            res = future.result()
            if res:
                results.append(res)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for repo_owner, repo_name, commit_sha, commit_message, migration_type_str in results:
            try:
                cursor.execute("""
                    UPDATE purpose_gen_success
                    SET reason_type = ?
                    WHERE repo_owner = ? AND repo_name = ? AND commit_sha = ? AND commit_message = ? 
                """, (migration_type_str, repo_owner, repo_name, commit_sha, commit_message))
                conn.commit()
                print(f"Updated: {repo_owner}/{repo_name}@{commit_sha}")
            except sqlite3.Error as e:
                print(f"Database error while updating {repo_owner}/{repo_name}@{commit_sha}: {e}")
                conn.rollback()
    except Exception as e:
        print(f"Error updating database: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()
