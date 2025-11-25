import pandas as pd
from typing import Dict, List, Tuple, Set

df_ingredients = pd.read_excel("final_ingredients_dataset_canon.xlsx")   # 같은 폴더에 있을 경우

import re
import torch
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# DB2 성분 리스트 + 임베딩
df_ingredients['성분명'] = df_ingredients['성분명'].astype(str).str.replace(r"[^가-힣]", "", regex=True)
ingredisnts_names = df_ingredients['성분명'].tolist()
ingredisnts_embs = model.encode(ingredisnts_names, convert_to_tensor=True)

# 전성분 문자열 → 후보 분리
# 성분명에서 한글만 추출해서 DB 매칭 정확도를 높이기 위한 전처리 코드
def clean_and_split(text):
    text = re.sub(r"[^가-힣, ]", " ", text)
    parts = re.split(r",", text)
    cleaned = [re.sub(r"[^가-힣]", "", p).strip() for p in parts]
    return [c for c in cleaned if len(c) >= 2]

# SBERT 성분 교정
def correct_ingredients(text, threshold=0.55):
    cand = clean_and_split(text)
    emb = model.encode(cand, convert_to_tensor=True)
    sims = util.cos_sim(emb, ingredisnts_embs)

    corrected = []
    for i, w in enumerate(cand):
        best_idx = torch.argmax(sims[i]).item()
        best_score = sims[i][best_idx].item()
        if best_score >= threshold:
            corrected.append(ingredisnts_names[best_idx])
    return list(set(corrected))


# ["미백", "진정", "피부장벽강화", "보습"]
# ✔ 성분 전체에서 효능만 추출한 중복 없는 리스트
def map_effects(corrected_list):
    effects = []

    for ing in corrected_list:
        row = df_ingredients[df_ingredients['성분명'] == ing]

        if len(row) > 0:
            effect_str = row.iloc[0]['효과별']
            eff_list = [e.strip() for e in str(effect_str).split(",")]

            effects.extend(eff_list)

    return list(set(effects))  # 중복 제거
