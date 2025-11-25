# # main.py
# from pathlib import Path

# from db_loader import load_all_db
# from embedding_model import load_sbert_model, embed_single_text
# from concern_classifier import ConcernClassifier
# from ingredient_matcher import (
#     parse_ingredient_text,
#     match_ingredients_with_type,
# )
# from scoring import calc_concern_match_score, calc_final_score


# # ===== 1) 전역 리소스 로드 (프로그램 시작 시 1번만) =====

# # 파일 경로는 네 환경에 맞게 수정!
# DB1_PATH = Path("skin_concern.xlsx")
# DB2_PATH = Path("test1118_ingredients_effects.xlsx")
# DB3_PATH = Path("skin_types.xlsx")


# print("📁 DB 로딩 중...")
# db1, db2, skin_type_db = load_all_db(DB1_PATH, DB2_PATH, DB3_PATH)
# print(f"✅ DB1: {len(db1)}행, DB2: {len(db2)}행, DB3: {len(skin_type_db)}행")

# print("🧠 SBERT 모델 로딩 중...")
# sbert_model = load_sbert_model()

# print("🤖 고민 분류기 학습 중...")
# # DB1의 텍스트 & 라벨 컬럼 이름은 실제 엑셀에 맞게 수정!
# text_col = "소비자 언어 (리뷰)"
# label_col = "피부고민 라벨"

# X_embs = sbert_model.encode(
#     db1[text_col].astype(str).tolist(),
#     convert_to_numpy=True,
# )
# y_labels = db1[label_col]

# concern_clf = ConcernClassifier()
# train_acc, test_acc = concern_clf.fit(X_embs, y_labels)
# print(f"✅ 고민 분류기 학습 완료 (train_acc={train_acc:.3f}, test_acc={test_acc:.3f})")


# # ===== 2) 전체 파이프라인 함수 =====

# def full_pipeline(
#     review_text: str,
#     ingredient_text: str,
#     user_skin_type: str,
#     user_target_concern: str | None = None,
# ):
#     """
#     하나의 제품에 대해:
#     - 리뷰 텍스트 → 고민 분류 모델로 predicted_concern 산출
#     - 전성분 → 성분/효능/타입 일치도 계산
#     - 고민 & 타입 점수 → 최종 점수 계산
#     결과를 dict로 반환.
#     """
#     # 1) 고민 예측
#     review_emb = embed_single_text(sbert_model, review_text)
#     predicted_concern = concern_clf.predict_label(review_emb)

#     # 2) 성분 파싱 & 타입 매칭
#     ingredients = parse_ingredient_text(ingredient_text)
#     matched_info, avg_type_score = match_ingredients_with_type(
#         ingredients=ingredients,
#         db2=db2,
#         skin_type_db=skin_type_db,
#         user_skin_type=user_skin_type,
#     )

#     # 3) 고민 점수 & 최종 점수
#     concern_score = 0.0
#     if user_target_concern:
#         concern_score = calc_concern_match_score(
#             predicted_concern=predicted_concern,
#             target_concern=user_target_concern,
#         )

#     final_score = calc_final_score(concern_score, avg_type_score)

#     result = {
#         "predicted_concern": predicted_concern,
#         "concern_score": concern_score,
#         "avg_type_score": avg_type_score,
#         "final_score": final_score,
#         "ingredients_detail": matched_info,
#     }
#     return result


# # ===== 3) 콘솔에서 테스트용 함수 =====

# def run_cli():
#     print("\n===== 스킨케어 적합도 테스트 =====")
#     user_skin_type = input("① 피부 타입 (지성/복합성/건성/민감성 등): ").strip()
#     user_concern_text = input("② 현재 피부 고민(예: 트러블, 미백 등): ").strip()
#     review_text = input("③ 사용자가 남긴 리뷰(간단한 문장): ").strip()
#     ingredient_text = input("④ 전성분을 쉼표(,)로 구분해서 입력: ").strip()

#     result = full_pipeline(
#         review_text=review_text,
#         ingredient_text=ingredient_text,
#         user_skin_type=user_skin_type,
#         user_target_concern=user_concern_text or None,
#     )

#     print("\n===== 결과 =====")
#     print(f"- 예측된 고민 라벨: {result['predicted_concern']}")
#     print(f"- 고민 일치도 점수: {result['concern_score']:.2f}")
#     print(f"- 피부타입 일치도(평균): {result['avg_type_score']:.2f}")
#     print(f"- 최종 점수: {result['final_score']:.2f}")

#     print("\n[성분별 상세 정보]")
#     for info in result["ingredients_detail"]:
#         print(f"  · {info['성분명']} | 타입일치도={info['타입일치도']:.2f}")
#         if info["효능"]:
#             print(f"    효능: {', '.join(info['효능'])}")


# if __name__ == "__main__":
#     run_cli()


# # main.py (임시 디버깅 버전)

# # import sys, inspect
# # import embedding_model

# # print("🔍 실제로 불러온 embedding_model 경로:")
# # print("   ", embedding_model.__file__)
# # print()

# # print("🔍 embedding_model 안에 들어있는 이름 목록 중 일부:")
# # names = [n for n in dir(embedding_model) if "load" in n or "embed" in n or "SBERT" in n]
# # print("   ", names)
# # print()

# # print("🔍 embedding_model 소스 코드:")
# # print("----------------------------------------")
# # print(inspect.getsource(embedding_model))
# # print("----------------------------------------")

# # sys.exit(0)


# 하 이거는 11/20에 만든 파일
# from scoring import full_pipeline, calc_final_score


# print("===== 피부타입 선택 =====")
# print("1) 지성  2) 복합성  3) 건성  4) 민감성")
# skin_type = int(input("번호 입력: "))

# review_text = input("\n고민 입력: ")
# ingredient_text = input("\n전성분 입력: ")

# result = full_pipeline(review_text, ingredient_text, skin_type)

# print("\n======= 결과 =======")
# print("🔥 예측 고민 1:", result["예측고민1"])
# print("🔥 예측 고민 2:", result["예측고민2"])
# print("🔥 효능 리스트:", result["효능"])
# print("💧 고민 일치도:", f"{result['고민일치도']*100:.2f}%")
# print("💊 교정된 성분:", result["성분"])
# print("✨ 피부타입:", result["피부타입"])
# print("🔬 피부타입 일치도:", f"{result['피부타입일치도']*100:.2f}%")

# # 최종 점수 계산
# final_score = calc_final_score(result['고민일치도'], result['피부타입일치도'])
# print("🎯 최종 점수:", result["최종점수"])


# print("\n📄 성분별 일치도 표:")
# print(result["고민매칭표"].to_string())

# print("\n📄 피부타입 매칭표:")
# print(result["타입매칭표"].to_string())

from scoring import full_pipeline

print("===== 피부타입 선택 =====")
print("1) 지성  2) 복합성  3) 건성  4) 민감성")
skin = int(input("번호 입력: "))

review = input("\n고민 입력: ")
ings = input("\n전성분 입력: ")

r = full_pipeline(review, ings, skin)

print("\n======= 결과 =======")
for k,v in r.items():
    print(f"{k} : {v}")