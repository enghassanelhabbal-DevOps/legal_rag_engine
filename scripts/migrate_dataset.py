import json
import re
from pathlib import Path

def normalize_arabic(text: str) -> str:
    # Basic normalization for search: remove tatweel, normalize alef/yeh/teh marbuta, remove tashkeel
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text) # remove tashkeel
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ـ', '', text) # tatweel
    return text

def parse_metadata(title: str):
    # Example: "المادة 1 - قانون الإجراءات الجنائية"
    # Or: "المادة 8 مكرراً - قانون الإجراءات الجنائية"
    parts = title.split(" - ", 1)
    if len(parts) == 2:
        article_str, law_name = parts
        article_id = article_str.replace("المادة", "").strip()
        return article_id, law_name.strip()
    return "UNKNOWN", title.strip()

def migrate(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
        
    new_docs = []
    for doc in docs:
        doc_id = str(doc.get("id", ""))
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        title = metadata.get("title", "")
        source = metadata.get("source", "Egyptian Legal RAG Dataset")
        
        article_id, law_name = parse_metadata(title)
        
        normalized = normalize_arabic(content)
        
        # Embedding text includes law name and article to give dense vectors strong semantic context
        embedding_text = f"{title}: {content}"
        
        new_doc = {
            "document_id": doc_id,
            "jurisdiction": "EG",
            "law_id": law_name, # using law name as a placeholder for internal law_id
            "law_name": law_name,
            "article_id": article_id,
            "raw_text": content,
            "normalized_text": normalized,
            "embedding_text": embedding_text,
            "version_id": None,
            "source": source
        }
        new_docs.append(new_doc)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_docs, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully migrated {len(new_docs)} documents to {output_path}")

if __name__ == "__main__":
    import sys
    migrate(sys.argv[1], sys.argv[2])
