

# import pandas as pd

# from ingredients_loader import df_ingredients, correct_ingredients
# from concern_classifier_llm import df_concerns, label_col, text_col, ensemble_predict
# from skin_type_loader import df_types


# # 1) 고민 → 효능 리스트
# def get_concern_effects(concern_name: str):
#     """
#     concern_name: 예) '수분부족', '피지/블랙헤드'
#     df_concerns: skin_concerns.xlsx
#       - label_col: '피부고민'
#       - text_col: '소비자 언어 (리뷰)'
#       - 나머지 컬럼: 효능들
#     """
#     row = df_concerns[df_concerns[label_col] == concern_name].iloc[0]
#     effects = row.drop([label_col, text_col]).dropna().tolist()
#     return effects   # 예: ["보습", "장벽강화"]


# # 2) 피부타입 → 효능 리스트
# def get_skin_type_effects(type_name: str):
#     """
#     type_name: '지성', '복합성', '건성', '민감성'
#     df_types: skin_types.xlsx
#       - '피부타입' + 여러 효능 컬럼들
#     """
#     row = df_types[df_types['피부타입'] == type_name].iloc[0]
#     effects = row.drop("피부타입").dropna().tolist()
#     return effects   # 예: ["피지조절", "모공관리", "진정"]


# # 3) 공통 일치도 계산
# #    일치도 = (성분효능 ∩ 타겟효능) 개수 / 성분의 전체효과 개수
# def calc_match_score(ingredients, effect_list):
#     matched = []
#     scores = []

#     for ing in ingredients:
#         rows = df_ingredients[df_ingredients['성분명'] == ing]

#         if len(rows) == 0:
#             matched.append({
#                 "성분": ing,
#                 "성분효능": [],
#                 "일치효능": [],
#                 "전체효과개수": 0,
#                 "일치도": 0
#             })
#             scores.append(0)
#             continue

#         ing_effects = rows['효과별'].dropna().unique().tolist()
#         total = len(ing_effects)

#         if total == 0:
#             matched.append({
#                 "성분": ing,
#                 "성분효능": [],
#                 "일치효능": [],
#                 "전체효과개수": 0,
#                 "일치도": 0
#             })
#             scores.append(0)
#             continue

#         intersection = list(set(ing_effects) & set(effect_list))
#         score = len(intersection) / total

#         matched.append({
#             "성분": ing,
#             "성분효능": ing_effects,
#             "일치효능": intersection,
#             "전체효과개수": total,
#             "일치도": round(score, 3)
#         })

#         scores.append(score)

#     df = pd.DataFrame(matched)
#     mean_score = sum(scores) / len(scores) if scores else 0

#     return mean_score, df


# # (선택) 피부타입용 래퍼
# def calc_type_match_score(ingredients, skin_type_name: str):
#     type_effects = get_skin_type_effects(skin_type_name)
#     return calc_match_score(ingredients, type_effects)

# # 고민 문장 → ensemble_predict로 고민 라벨 뽑기

# # 전성분 문자열 → correct_ingredients로 성분 교정
# # 나중에 여기서 보조성분 액티브성분 구분하고, 

# # 고민 라벨 → 고민 효능 리스트 (get_concern_effects)

# # 피부타입 번호 → "지성"/"건성" 같은 이름으로 매핑

# # 그 이름으로 타입 효능 리스트 (get_skin_type_effects)

# # 각 효능 리스트와 성분 효능을 비교해서 일치도 계산 (calc_match_score)

# # 고민/피부타입 일치도 가중합해서 최종 점수



# # eview, ingredients, skin_num = 이건 사용자한테 입력 받는 값

# # 4) full_pipeline
# def full_pipeline(review, ingredients, skin_num):
#     """
#     review: 사용자 고민 텍스트 (문장)
#     ingredients: 전성분 (문자열 또는 리스트)
#     skin_num: 1=지성, 2=복합성, 3=건성, 4=민감성
#     """

#     # 0) 전성분 타입 정리 (문자열로 통일)
#     if isinstance(ingredients, list):
#         ingredients_text = ",".join(ingredients)
#     else:
#         ingredients_text = ingredients

#     # 1) 고민 예측
#     primary, probs = ensemble_predict(review)

#     # 2) 성분 교정
#     corrected = correct_ingredients(ingredients_text)

#     # 3) 고민 → 효능 리스트
#     concern_effects = get_concern_effects(primary)

#     # 4) 피부타입 번호 → 이름
#     skin_map = {1: "지성", 2: "복합성", 3: "건성", 4: "민감성"}
#     stype = skin_map[skin_num]

#     # 5) 고민/피부타입 기준 일치도 계산
#     concern_score, concern_df = calc_match_score(corrected, concern_effects)
#     type_score, type_df = calc_type_match_score(corrected, stype)

#     # 6) 최종 점수 (가중합)
#     final = (concern_score * 0.65 + type_score * 0.35) * 100

#     return {
#         "예측고민": primary,
#         "고민확률": probs,
#         "성분": corrected,
#         "피부타입": stype,
#         "피부타입일치도": type_score,
#         "고민일치도": concern_score,
#         "최종점수": round(final, 2),
#         "고민별_매칭표": concern_df,
#         "피부타입별_매칭표": type_df,
#     }


import pandas as pd

from ingredients_loader import df_ingredients, correct_ingredients
from concern_classifier_llm import df_concerns, label_col, text_col, ensemble_predict
from skin_type_loader import df_types

# scoring.py 상단 어딘가
import pandas as pd
from pathlib import Path

# --- 고민 -> 효능 매핑용 엑셀 로드 ---
SKIN_CONCERN_FILE = Path("skin_concerns_canon.xlsx")  # 파일 위치에 맞게 조정

df_concern_map = pd.read_excel(SKIN_CONCERN_FILE)

def build_concern_map(df: pd.DataFrame):
    """
    skin_concerns.xlsx 에서
    '피부 고민' -> ['효능1', '효능2', ...] 딕셔너리로 변환
    """
    # 필요한 컬럼만 사용 + 공백/결측 처리
    df = df[['피부고민', '효능']].dropna()
    df['피부고민'] = df['피부고민'].astype(str).str.strip()
    df['효능'] = df['효능'].astype(str)

    tmp = {}

    for _, row in df.iterrows():
        label = row['피부고민']
        # "안티아크네, 피지 조절" 이런 식으로 들어있을 수도 있으니까 split
        effects = [e.strip() for e in row['효능'].split(",") if e.strip()]

        if label not in tmp:
            tmp[label] = set()
        tmp[label].update(effects)

    # set → list로 변환
    return {label: sorted(effs) for label, effs in tmp.items()}

# 전역 딕셔너리
CONCERN_TO_EFFECTS = build_concern_map(df_concern_map)

def get_concern_effects(label: str):
    """
    고민 라벨(예: '여드름') -> 효능 리스트 반환
    """
    return CONCERN_TO_EFFECTS.get(label, [])



# # 1) 고민 → 효능 리스트
# def get_concern_effects(concern_name: str):
#     """
#     concern_name: 예) '수분부족', '피지/블랙헤드'
#     df_concerns: skin_concerns.xlsx
#       - label_col: '피부고민'
#       - text_col: '소비자 언어 (리뷰)'
#       - 나머지 컬럼: 효능들
#     """
#     row = df_concerns[df_concerns[label_col] == concern_name].iloc[0]
#     effects = row.drop([label_col, text_col]).dropna().tolist()
#     return effects   # 예: ["보습", "장벽강화"]

# def get_concern_effects(concern_name):
#     """
#     concern_name:
#       - "수분부족" 같은 문자열이거나
#       - ["수분부족", "각질"] 이런 리스트일 수도 있음
#     """
#     # 🔹 ensemble_predict가 리스트를 줄 수도 있으니, 그럴 땐 첫 번째만 사용
#     if isinstance(concern_name, (list, tuple)):
#         if not concern_name:   # 빈 리스트면 안전장치
#             raise ValueError("예측된 고민 라벨이 비어 있습니다.")
#         concern_name = concern_name[0]

#     row = df_concerns[df_concerns[label_col] == concern_name].iloc[0]
#     effects = row.drop([label_col, text_col]).dropna().tolist()
#     return effects


# 2) 피부타입 → 효능 리스트
def get_skin_type_effects(type_name: str):
    """
    type_name: '지성', '복합성', '건성', '민감성'
    df_types: skin_types.xlsx
      - '피부타입' + 여러 효능 컬럼들
    """
    row = df_types[df_types['피부타입'] == type_name].iloc[0]
    effects = row.drop("피부타입").dropna().tolist()
    return effects   # 예: ["피지조절", "모공관리", "진정"]


# 3) 공통 일치도 계산
#    일치도 = (성분효능 ∩ 타겟효능) 개수 / 성분의 전체효과 개수
def calc_match_score(ingredients, effect_list):
    """
    ingredients: 성분명 리스트
    effect_list: '안티아크네', '진정' 같은 효능 문자열 리스트
    """
    import pandas as pd
    import numpy as np

    # 0) effect_list 타입 정리 (문자열/Series/set 들어와도 방어)
    if effect_list is None:
        effect_list = []
    elif isinstance(effect_list, str):
        effect_list = [e.strip() for e in effect_list.split(",") if e.strip()]
    elif isinstance(effect_list, (set, tuple, np.ndarray, pd.Series)):
        effect_list = [str(e).strip() for e in list(effect_list) if pd.notna(e)]
    elif isinstance(effect_list, list):
        effect_list = [str(e).strip() for e in effect_list if pd.notna(e)]
    else:
        effect_list = [str(effect_list).strip()]

    effect_set = set(effect_list)

    matched = []
    scores = []

    for ing in ingredients:
        rows = df_ingredients[df_ingredients['성분명'] == ing]

        if len(rows) == 0:
            matched.append({
                "성분": ing,
                "성분효능": [],
                "일치효능": [],
                "전체효과개수": 0,
                "일치도": 0.0
            })
            scores.append(0.0)
            continue

        # '효과별' 컬럼 모아서 파싱
        raw_effects = rows['효과별'].dropna().unique().tolist()

        ing_effects = []
        for v in raw_effects:
            # "미백,보습,진정" → ["미백", "보습", "진정"]
            parts = [e.strip() for e in str(v).split(",") if e.strip()]
            ing_effects.extend(parts)   # ❗ for 안으로 이동

        ing_effects = list(set(ing_effects))  # 중복 제거
        total = len(ing_effects)

        if total == 0:
            matched.append({
                "성분": ing,
                "성분효능": [],
                "일치효능": [],
                "전체효과개수": 0,
                "일치도": 0.0
            })
            scores.append(0.0)
            continue

        intersection = list(set(ing_effects) & effect_set)
        score = len(intersection) / total

        matched.append({
            "성분": ing,
            "성분효능": ing_effects,
            "일치효능": intersection,
            "전체효과개수": total,
            "일치도": round(score, 3)
        })
        scores.append(score)

    df = pd.DataFrame(matched)
    mean_score = sum(scores) / len(scores) if scores else 0.0

    return mean_score, df



# 4) 피부타입 일치도용 래퍼
def calc_type_match_score(ingredients, skin_type_name: str):
    """
    skin_type_name: '지성', '복합성', '건성', '민감성'
    """
    type_effects = get_skin_type_effects(skin_type_name)
    return calc_match_score(ingredients, type_effects)


# 5) full_pipeline
def full_pipeline(review, ingredients, skin_num):
    """
    review: 사용자 고민 텍스트 (문장)
    ingredients: 전성분 (문자열 또는 리스트)
    skin_num: 1=지성, 2=복합성, 3=건성, 4=민감성
    """

    # 0) 전성분 타입 정리 (문자열로 통일)
    if isinstance(ingredients, list):
        ingredients_text = ",".join(ingredients)
    else:
        ingredients_text = ingredients

    # 1) 고민 예측
    primary, probs = ensemble_predict(review)

    # 2) 성분 교정
    corrected = correct_ingredients(ingredients_text)

    # 3) 고민 → 효능 리스트
    concern_effects = get_concern_effects(primary)

    
    print("🔍 예측 고민:", primary)
    print("🔍 고민 기준 효능:", concern_effects)

    # 4) 피부타입 번호 → 이름
    skin_map = {1: "지성", 2: "복합성", 3: "건성", 4: "민감성"}
    stype = skin_map[skin_num]


    # 5) 고민/피부타입 기준 일치도 계산 calc_match_score
    concern_score, concern_df = calc_match_score(corrected, concern_effects)
    type_score, type_df = calc_type_match_score(corrected, stype)

    # 6) 최종 점수 (가중합)
    final = (concern_score * 0.65 + type_score * 0.35) * 100

    return {
        "예측고민": primary,
        "고민확률": probs,
        "성분": corrected,
        "피부타입": stype,
        "고민일치도": concern_score,
        "피부타입일치도": type_score,
        "최종점수": round(final, 2),
        "고민별_매칭표": concern_df,
        "피부타입별_매칭표": type_df,
    }
