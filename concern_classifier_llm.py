# SBERT + LLM 로딩
import pandas as pd
from typing import Dict, List, Tuple, Set

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "EleutherAI/polyglot-ko-1.3b"

# # 이건 cpu 성능 좋은 경우만..
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# llm = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     device_map="cpu"
# )

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id)

llm = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

llm = llm.to(device)



def ask_llm(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = llm.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=80,
        do_sample=True,
        temperature=0.3
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


df_concerns = pd.read_excel("skin_concerns_canon.xlsx")   # 같은 폴더에 있을 경우
# Define text and label columns for DB1
text_col = '소비자 언어 (리뷰)'
label_col = '피부고민'
concsern_labels =df_concerns[label_col].tolist()

label_reviews = {lbl: [] for lbl in concsern_labels}
for _, row in df_concerns.iterrows():
    label_reviews[row["피부고민"]].append(str(row["소비자 언어 (리뷰)"]))

label_vectors = {}

for lbl in concsern_labels:
    texts = label_reviews[lbl][:120]
    embs = model.encode(texts, convert_to_numpy=True)
    label_vectors[lbl] = embs.mean(axis=0)



# 전문가 규칙

expert_rules = {
    "수분부족": ["속당김","건조","푸석","당김"],
    "미백/잡티": ["흉터","자국","잡티","기미"],
    "피지/블랙헤드": ["기름","번들","유분","피지"],
    "여드름": ["트러블","뾰루지","염증"],
    "모공": ["모공","구멍"],
    "각질": ["각질","일어남"]
}

# SBERT 기반 예측 + 전문가 규칙
import numpy as np
from scipy.special import softmax
from collections import Counter

def predict_concern_expert(review, top_k=2):

    refined = ask_llm(f"피부 고민을 한 문장으로 요약해줘: {review}")

    emb = model.encode([refined], convert_to_numpy=True)[0]

    sims = []
    for lbl in concsern_labels:
        v = label_vectors[lbl]
        sims.append(np.dot(emb, v) / (np.linalg.norm(emb)*np.linalg.norm(v)))

    sims = np.array(sims)
    probs = softmax(sims)
    score_dict = {concsern_labels[i]: float(probs[i]) for i in range(len(concsern_labels))}

    review_l = review.lower()
    OVERRIDE = 3.0

    for concern, kws in expert_rules.items():
        for kw in kws:
            if kw in review_l:
                score_dict[concern] += OVERRIDE

    total = sum(score_dict.values())
    norm = {k: v/total for k,v in score_dict.items()}

    final = sorted(norm, key=lambda x: norm[x], reverse=True)[:top_k]
    return final, {lbl: norm[lbl] for lbl in final}

# ensemble_predict (다수결)
def ensemble_predict(review_text):

    votes = []

    try:
        a, _ = predict_concern_expert(review_text)
        votes += a
    except:
        pass

    try:
        refined = ask_llm(f"피부 고민을 한 문장으로 다시 요약해줘: {review_text}")
        b, _ = predict_concern_expert(refined)
        votes += b
    except:
        pass

    try:
        c, _ = predict_concern_expert(review_text)
        votes += c
    except:
        pass

    count = Counter(votes)
    if not count:
        return None, {}

    # 1등만 primary로
    primary, primary_cnt = count.most_common(1)[0]

    total = sum(count.values())
    probs = {lbl: cnt/total for lbl, cnt in count.items()}

    return primary, probs