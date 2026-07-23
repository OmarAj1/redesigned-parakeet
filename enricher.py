import sqlite3
import json
import asyncio
import aiohttp
import os
import sys
import time

# ==========================================
# 1. GEMINI CLOUD CONFIGURATION
# ==========================================
DB_FILE = "MasterUnifiedDB.db"

# Pull the API key from the GitHub Actions environment securely
GEMINI_API_KEY = os.environ.get("GROQ_API_KEY") # Keep this as GROQ_API_KEY if that is what you named your GitHub Secret
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

BATCH_SIZE = 100
REQUEST_DELAY = 4.1  # 4.1 seconds ensures we strictly hit no more than ~14.6 requests per minute

# ==========================================
# 2. STRICT DATABASE SETUP & CLEANING
# ==========================================
def enforce_strict_database_mode():
    print("[*] Enforcing strict direct-write mode on SQLite...", flush=True)
    try:
        conn = sqlite3.connect(DB_FILE, timeout=60.0)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)") 
        conn.execute("PRAGMA journal_mode = DELETE")    
        conn.execute("PRAGMA synchronous = FULL")       
        conn.commit()
        conn.close()
        print("[✓] Hidden files disabled. Database is in direct-write mode.", flush=True)
    except Exception as e:
        print(f"[!] Error enforcing strict mode: {e}")

def remove_duplicates():
    print("[*] Scanning for duplicate entries...", flush=True)
    try:
        conn = sqlite3.connect(DB_FILE, timeout=60.0)
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM UnifiedIngredients 
            WHERE id NOT IN (
                SELECT MIN(id) 
                FROM UnifiedIngredients 
                GROUP BY LOWER(TRIM(name))
            )
        ''')
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            print(f"🧹 SUCCESS: Found and deleted {deleted_count} duplicate items!")
        else:
            print("🧹 Database is perfectly clean. No duplicates found.")
    except Exception as e:
        print(f"[!] Error during deduplication: {e}")

# ==========================================
# 3. DATA FETCHING & SAVING
# ==========================================
def fetch_pending_ingredients():
    conn = sqlite3.connect(DB_FILE, timeout=60.0)
    cursor = conn.cursor()
    query = """
    SELECT id, name, description, identification_code, toxicity_or_safety, 
           category, dietary_info, plain_english_name, purpose, health_risks, 
           risk_level, dietary_safety 
    FROM UnifiedIngredients 
    WHERE description IS NULL OR TRIM(description) = '' 
       OR identification_code IS NULL OR TRIM(identification_code) = '' 
       OR toxicity_or_safety IS NULL OR TRIM(toxicity_or_safety) = '' 
       OR category IS NULL OR TRIM(category) = '' 
       OR dietary_info IS NULL OR TRIM(dietary_info) = '' 
       OR plain_english_name IS NULL OR TRIM(plain_english_name) = '' 
       OR purpose IS NULL OR TRIM(purpose) = '' 
       OR health_risks IS NULL OR TRIM(health_risks) = '' 
       OR risk_level IS NULL OR TRIM(risk_level) = '' 
       OR dietary_safety IS NULL OR TRIM(dietary_safety) = ''
    """
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
    except Exception as e:
        print(f"   [!] Error selecting pending rows: {e}")
        rows = []
    conn.close()
    return rows

def save_enriched_results(results_tuples):
    if not results_tuples: return
    print(f"[!] DIRECT WRITE: Saving batch of {len(results_tuples)} items to DB...")
    conn = sqlite3.connect(DB_FILE, timeout=15.0)
    conn.execute("PRAGMA journal_mode = DELETE;")
    conn.execute("PRAGMA synchronous = FULL;")
    cursor = conn.cursor()
    try:
        cursor.executemany('''
            UPDATE UnifiedIngredients 
            SET description = ?, identification_code = ?, toxicity_or_safety = ?, category = ?, 
                dietary_info = ?, plain_english_name = ?, purpose = ?, health_risks = ?, 
                risk_level = ?, dietary_safety = ?, is_enriched = 1
            WHERE id = ?
        ''', results_tuples)
        conn.commit()
        print("✅ Batch saved perfectly!")
    except Exception as e:
        print(f"❌ Error saving batch: {e}")
    finally:
        conn.close()

# ==========================================
# 4. AI PROMPT & BATCH PROCESSING
# ==========================================
def get_batch_prompt(batch_data):
    return f"""
    Analyze the following JSON array of food ingredients. 
    Input Data: {json.dumps(batch_data, indent=2)}
    
    SPECIAL INSTRUCTION FOR LONG NAMES: 
    If an ingredient name is longer than 5 words, deduce the core short ingredient name and place it in "plain_english_name". Use the remaining extra information from the name to deeply flesh out the "description".
    
    Your task is to provide a complete JSON object updating and filling missing (null or empty) values for ALL the keys to guarantee NO NULLS.
    
    You MUST return a JSON object with a single key "results", which contains a list of the enriched objects.
    Each object in the "results" array MUST have the exact following structure, including the original 'id':
    {{
      "id": <exact_integer_id_from_input>,
      "description": "Short description of what this is",
      "identification_code": "e.g. E-number or INS number if applicable, otherwise 'None'",
      "toxicity_or_safety": "General safety profile",
      "category": "e.g. Preservative, Colorant, Sweetener, etc.",
      "dietary_info": "e.g. Vegan, Halal, Kosher, Gluten-free",
      "plain_english_name": "What average people call this (short)",
      "purpose": "Why manufacturers use it",
      "health_risks": "Health risks or 'Generally recognized as safe'.",
      "risk_level": "Categorize risk exactly as: 'Low', 'Moderate', 'High', or 'Unknown'",
      "dietary_safety": "e.g., vegan, halal, gluten-free"
    }}
    """

def sanitize_and_map_results(ai_results, original_rows):
    row_map = {row[0]: row for row in original_rows}
    mapped_tuples = []
    
    for item in ai_results.get("results", []):
        item_id = item.get("id")
        if item_id not in row_map:
            continue
            
        row_id, name, desc, ident, tox, cat, diet_info, plain_english, purpose, risks, risk_level, dietary = row_map[item_id]
        
        def safe_extract(key, fallback):
            val = item.get(key, fallback)
            if isinstance(val, (dict, list)): return json.dumps(val)
            if val is None or str(val).strip() == "":
                return fallback if (fallback is not None and str(fallback).strip() != "") else "Not specified"
            return str(val).strip()

        mapped_tuples.append((
            safe_extract("description", desc), safe_extract("identification_code", ident),
            safe_extract("toxicity_or_safety", tox), safe_extract("category", cat),
            safe_extract("dietary_info", diet_info), safe_extract("plain_english_name", plain_english),
            safe_extract("purpose", purpose), safe_extract("health_risks", risks),
            safe_extract("risk_level", risk_level), safe_extract("dietary_safety", dietary), row_id
        ))
        
    return mapped_tuples

# ==========================================
# 5. MASTER EXECUTION LOOP
# ==========================================
async def enrich_data_async():
    if not GEMINI_API_KEY:
        print("\n[!] ERROR: Missing API Key. Ensure GROQ_API_KEY is set in your Actions Secrets.")
        sys.exit(1)
        
    rows = fetch_pending_ingredients()
    total_pending = len(rows)
    print(f"\nFound {total_pending} ingredients missing data.")
    
    if total_pending == 0:
        print("Everything is fully enriched! 0 Nulls remain.")
        return

    batches = [rows[i:i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    
    async with aiohttp.ClientSession() as session:
        headers = {"Content-Type": "application/json"}
        
        for index, batch in enumerate(batches):
            start_time = time.time()
            
            prompt_data = [{"id": r[0], "name": r[1], "current_data": {
                "description": r[2], "identification_code": r[3], "category": r[5], "plain_english_name": r[7]
            }} for r in batch]
            
            payload = {
                "contents": [{"parts": [{"text": get_batch_prompt(prompt_data)}]}],
                "generationConfig": {"response_mime_type": "application/json"}
            }
            
            print(f"☁️ Processing Batch {index + 1}/{len(batches)} ({len(batch)} items)...")
            
            for attempt in range(3):
                try:
                    async with session.post(GEMINI_URL, headers=headers, json=payload, timeout=120) as response:
                        if response.status == 200:
                            res_json = await response.json()
                            raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
                            ai_results = json.loads(raw_text)
                            
                            mapped_data = sanitize_and_map_results(ai_results, batch)
                            save_enriched_results(mapped_data)
                            break
                        elif response.status == 429:
                            print("   [!] Rate limit hit. Backing off for 15 seconds...")
                            await asyncio.sleep(15) 
                        else:
                            print(f"   [!] API Error {response.status}: {await response.text()}")
                            await asyncio.sleep(2)
                except Exception as e:
                    print(f"   [!] Request failed on attempt {attempt + 1}: {e}")
                    await asyncio.sleep(2)
            
            elapsed = time.time() - start_time
            if elapsed < REQUEST_DELAY:
                await asyncio.sleep(REQUEST_DELAY - elapsed)

if __name__ == "__main__":
    print("\n=================================================================")
    print("      🚀 GEMINI FLASH BATCH ENRICHER (GITHUB ACTIONS) 🚀")
    print("=================================================================")
        
    enforce_strict_database_mode()
    remove_duplicates() 
    
    print("\nInitiating Cloud Enrichment Phase...")
    asyncio.run(enrich_data_async())
