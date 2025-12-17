import streamlit as st
import agent_logic
import base64
from openai import OpenAI

client = OpenAI(api_key=agent_logic.API_KEY)

HIDE_Press_Enter_to_apply = """
<style>
[data-testid="InputInstructions"] {
    display: none !important;
}

.stChatMessage {
    border-radius: 20px;
    border: 1px solid #E0E0E0;
}

div.stButton > button {
    background-color: #FF9A07;
    color: white;
    border-radius: 10px;
    border: none;
}

div.stButton > button:hover {
    background-color: #E07B00;
    color: white;
}

.streamlit-expanderHeader {
    background-color: #FFFFFF;
    border-radius: 10px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    margin-bottom: 10px;
}

</style>
"""
st.markdown(HIDE_Press_Enter_to_apply, unsafe_allow_html=True)


# -------------------------------
# 전역 설정 및 세션 상태 초기화
# -------------------------------

st.set_page_config(page_title="자취생을 위한 엄마 밥선생", page_icon="🍲")

if "active_feature" not in st.session_state:
    st.session_state.active_feature = "recommend"  # recommend / info / chat

if "fridge_ingredients" not in st.session_state:
    st.session_state.fridge_ingredients = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "우리 딸/아들 왔어? 밥은 챙겨 먹었니? 뭐 해줄까?",
        }
    ]

if "agent_loaded" not in st.session_state:
    try:
        db, llm = agent_logic.get_agent_components()
        st.session_state.db = db
        st.session_state.llm = llm
        st.session_state.agent_loaded = True
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
        st.stop()

if "recent_recipe_names" not in st.session_state:
    st.session_state.recent_recipe_names = []

if "dish_image_cache" not in st.session_state:
    st.session_state.dish_image_cache = {}

# -------------------------------
# 냉장고 UI (사이드바)
# -------------------------------

def add_fridge_items():
    """냉장고 재료 추가 버튼 콜백: 재료 추가 후 입력창 비우기"""
    raw_input = st.session_state.get("fridge_input", "")
    new_ings = [x.strip() for x in raw_input.split(",") if x.strip()]
    if new_ings:
        for p in new_ings:
            if p not in st.session_state.fridge_ingredients:
                st.session_state.fridge_ingredients.append(p)
    # 입력창 비우기
    st.session_state.fridge_input = ""

def render_fridge_sidebar():
    st.subheader("우리 집 냉장고")

    #재료 입력
    st.text_input(
        "냉장고에 있는 재료 추가 (쉼표로 여러 개 입력 가능)",
        key="fridge_input",
        placeholder="예: 계란, 고구마, 밥, 치킨",
    )

    st.button("재료 추가", key="btn_add_fridge", on_click=add_fridge_items)


    # 현재 재료 목록
    if st.session_state.fridge_ingredients:
        st.markdown("**🍚현재 냉장고 재료🍚**")
        for idx, ing in enumerate(list(st.session_state.fridge_ingredients)):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(
                    f"""
                    <div style="
                        background-color: #FFCC80;
                        color: #5D4037;
                        padding: 5px 10px;
                        border-radius: 15px;
                        display: inline-block;
                        margin: 2px;
                        font-size: 1.0rem;
                    ">
                        {ing}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            with c2:
                if st.button("×", key=f"del_ing_{idx}"):
                    st.session_state.fridge_ingredients.remove(ing)
                    # 삭제 후 즉시 화면 갱신
                    st.rerun()
    else:
        st.caption("아직 등록된 재료가 없네! 계란, 밥, 참기름 이런 식으로 추가해줘.")


#이미지url 반환
def get_or_generate_dish_image(dish_name: str) -> str:
    if not dish_name:
        return None

    # 세션 캐시 먼저 확인
    cache = st.session_state.get("dish_image_cache", {})
    if dish_name in cache:
        return cache[dish_name]

    prompt = f"{dish_name} 한식 요리 음식 사진, realistic food photography, top-down view"

    try: 
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
        b64 = result.data[0].b64_json
        image_bytes = base64.b64decode(b64)

        cache[dish_name] = image_bytes
        st.session_state.dish_image_cache = cache

        return image_bytes
    except Exception as e:
        print("이미지 생성 실패:", repr(e))
        return None


# -------------------------------
# 기능 1: 요리 추천
# -------------------------------

def render_feature_recommend():
    st.header("🍳요리 추천")
    st.write("원하는 조건을 골라주면, 엄마가 오늘 먹을 메뉴를 하나 골라줄게.")

    cond_prev = st.session_state.get("recommendation_conditions", {})

    # --- 재료 조건 ---
    with st.expander("재료 조건 설정", expanded=True):
        # 이전 상태 불러오기 (기본값: 냉장고 기준 사용)
        ingredient_mode = cond_prev.get("ingredient_mode", "fridge")

        # 체크박스 라벨만 변경
        use_ingredients = st.checkbox(
            "냉장고 속 재료 위주로 쓸래",
            value=cond_prev.get("use_ingredients", True),
            key="use_ingredients_chk",
        )

        if use_ingredients:
            # 재료 조건을 쓰는 경우 → 무조건 냉장고 위주 모드
            ingredient_mode = "fridge"
            st.caption("→ 냉장고 속 재료 위주로 추천해줄게.")
        else:
            # 체크 해제 → 재료 조건을 사용하지 않음
            ingredient_mode = "ignore"

    # --- 영양 조건 ---
    with st.expander("영양(칼로리 / 탄·단·지) 조건 설정", expanded=False):
        use_nutrition = st.checkbox(
            "영양도 중요해! (칼로리 / 탄수화물 · 단백질 · 지방)",
            value=cond_prev.get("use_nutrition", False),
            key="use_nutrition_chk",
        )

        calorie_pref = None
        macro_ratio = None
        calorie_raw = int(cond_prev.get("calorie_raw", 50))
        # 이전에 저장된 탄/단/지 비율 있으면 그대로 사용, 없으면 기본값 (40, 30, 30)
        macro_raw = cond_prev.get("macro_raw", (40, 30, 30))
        if not isinstance(macro_raw, (list, tuple)) or len(macro_raw) != 3:
            macro_raw = (40, 30, 30)
        carb_default, protein_default, fat_default = [int(x) for x in macro_raw]

        # use_nutrition이 False일 때도 참조할 수 있도록 기본 raw값 세팅
        carb_raw, protein_raw, fat_raw = carb_default, protein_default, fat_default

        if use_nutrition:
            # 🔹 칼로리 슬라이더
            calorie_raw = st.slider(
                "칼로리 선호도 (왼쪽: 적은 편, 오른쪽: 많은 편)",
                min_value=0,
                max_value=100,
                value=calorie_raw,
                key="calorie_slider",
            )
            if calorie_raw < 33:
                calorie_pref = "낮음"
            elif calorie_raw < 66:
                calorie_pref = "보통"
            else:
                calorie_pref = "높음"
            st.caption(f"→ 칼로리가 '{calorie_pref}' 정도면 좋겠다는 뜻이구나.")

            st.markdown("#### 탄수화물 / 단백질 / 지방 조절")

            # 🔹 탄 / 단 / 지 각각 슬라이더 하나씩 (0~100)
            carb_raw = st.slider(
                "탄수화물",
                min_value=0,
                max_value=100,
                value=carb_default,
                key="carb_slider",
            )
            protein_raw = st.slider(
                "단백질",
                min_value=0,
                max_value=100,
                value=protein_default,
                key="protein_slider",
            )
            fat_raw = st.slider(
                "지방",
                min_value=0,
                max_value=100,
                value=fat_default,
                key="fat_slider",
            )

            # 세 값 합 기준으로 정규화 → 10단위 비율 (예: 4:3:3)
            total_raw = carb_raw + protein_raw + fat_raw or 1
            macro_ratio = (
                max(round(carb_raw / total_raw * 10), 1),
                max(round(protein_raw / total_raw * 10), 1),
                max(round(fat_raw / total_raw * 10), 1),
            )

    # --- 음식 종류 조건 ---
    with st.expander("음식 종류 (식사 / 반찬 / 디저트 / 안주)", expanded=False):
        use_type = st.checkbox(
            "음식 종류도 신경 쓸래",
            value=cond_prev.get("use_type", False),
            key="use_type_chk",
        )

        dish_type = cond_prev.get("dish_type", "식사")

        if use_type:
            dish_type = st.radio(
                "어떤 느낌의 음식을 원해?",
                ("식사", "반찬", "디저트", "안주"),
                index=("식사", "반찬", "디저트", "안주").index(dish_type),
                key="dish_type_radio",
            )
        else:
            dish_type = None

    conditions = {
        "use_ingredients": use_ingredients,
        "ingredient_mode": ingredient_mode,
        "use_nutrition": use_nutrition,
        "calorie_pref": calorie_pref,
        "macro_ratio": macro_ratio,
        "use_type": use_type,
        "dish_type": dish_type,
        "calorie_raw": calorie_raw,
        "macro_raw": (carb_raw, protein_raw, fat_raw),
    }

    recommend_clicked = st.button("요리 추천받기", key="btn_recommend")

    if recommend_clicked:
        st.session_state.recommendation_conditions = conditions
        # 최근 추천된 요리 리스트를 forbidden_names로 넘김
        forbidden = st.session_state.get("recent_recipe_names", [])
        cond_with_forbidden = {**conditions, "forbidden_names": forbidden}

        with st.spinner("엄마가 레시피 노트를 뒤적이는 중..."):
            rec = agent_logic.recommend_recipe(
                st.session_state.db,
                st.session_state.llm,
                st.session_state.fridge_ingredients,
                cond_with_forbidden,
            )

        st.session_state.last_recommendation = rec
        st.session_state.selected_dish_name = rec["name"]
        st.session_state.selected_dish_origin = "feature1"
        with st.spinner("사진 불러오는 중.."):
            img_bytes = get_or_generate_dish_image(rec["name"])
        st.session_state.last_recommendation_image = img_bytes

        # 최근 추천 리스트 업데이트 (중복 제거 + 최대 10개 유지)
        names = st.session_state.get("recent_recipe_names", [])
        # 같은 이름이 이미 있으면 제거하고, 제일 뒤에 다시 넣기
        names = [n for n in names if n != rec["name"]] + [rec["name"]]
        if len(names) > 10:
            names = names[-10:]
        st.session_state.recent_recipe_names = names

    if "last_recommendation" in st.session_state:
        rec = st.session_state.last_recommendation
        st.markdown("---")
        st.subheader("엄마 추천 메뉴")

        st.markdown(f"### 오늘은 **{rec['name']}** 어떠니?")
        st.write(rec["reason"])

        #이미지
        img_bytes = get_or_generate_dish_image(rec["name"])
        if img_bytes:
            st.image(
                img_bytes,
                caption=f"{rec['name']} 예시 이미지",
                width='stretch'
            )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("이 요리 자세히 보기", key="btn_go_to_info"):
                st.session_state.active_feature = "info"
                st.rerun()
        with c2:
            if st.button("다른 요리 추천받기", key="btn_recommend_again"):
                # 직전에 사용한 조건 다시 가져오기
                base_cond = st.session_state.recommendation_conditions
                forbidden = st.session_state.get("recent_recipe_names", [])
                cond_with_forbidden = {**base_cond, "forbidden_names": forbidden}

                with st.spinner("엄마가 다른 요리도 떠올리는 중..."):
                    rec = agent_logic.recommend_recipe(
                        st.session_state.db,
                        st.session_state.llm,
                        st.session_state.fridge_ingredients,
                        cond_with_forbidden,
                    )

                # 새로 추천된 요리를 화면에 보여주도록 상태 업데이트
                st.session_state.last_recommendation = rec
                st.session_state.selected_dish_name = rec["name"]
                st.session_state.selected_dish_origin = "feature1"
                with st.spinner("사진 불러오는 중.."):
                    img_bytes = get_or_generate_dish_image(rec["name"])
                st.session_state.last_recommendation_image = img_bytes

                # 최근 추천 리스트 업데이트 (중복 제거 + 최대 10개 유지)
                names = st.session_state.get("recent_recipe_names", [])
                names = [n for n in names if n != rec["name"]] + [rec["name"]]
                if len(names) > 10:
                    names = names[-10:]
                st.session_state.recent_recipe_names = names

                st.rerun()


# -------------------------------
# 기능 2: 특정 음식 정보 검색
# -------------------------------

def search_recipe():
    """레시피 검색 버튼 콜백: 검색 후 입력창 비우기"""
    dish_name = st.session_state.get("info_dish_input", "").strip()
    if not dish_name:
        return

    with st.spinner("엄마가 레시피 노트를 뒤적이는 중..."):
        details = agent_logic.get_recipe_details(
            st.session_state.db,
            st.session_state.llm,
            dish_name,
        )
    st.session_state.recipe_details = details
    st.session_state.selected_dish_name = details["final_name"]

    # 입력창 비우기
    st.session_state.info_dish_input = ""

def render_feature_info():
    st.header("📜레시피 검색")
    st.write("궁금한 요리 이름을 입력하면 엄마가 알려줄게.")

    default_name = st.session_state.get("selected_dish_name", "")

    if "info_dish_input" not in st.session_state:
        st.session_state.info_dish_input = default_name

    st.text_input(
        "어떤 음식이 궁금해?",
        key="info_dish_input",
        placeholder="예: 김치볶음밥, 밤 티라미수",
    )

    st.button("🔍 레시피 검색", key="btn_search_recipe", on_click=search_recipe)


    if "recipe_details" not in st.session_state:
        return

    details = st.session_state.recipe_details

    if details.get("not_food"):
        st.markdown("---")
        st.subheader(f"'{details['final_name']}'라는 이름의 음식을 찾지 못했어 😥")
        st.warning(
            f"'{details['final_name']}'는(은) 음식 이름이 아닐 수도 있고, "
            "엄마가 잘 모르는 이름일 수도 있어.\n\n"
            "다른 음식 이름으로 검색해 보거나, 철자를 한 번만 더 확인해 줄래?"
        )
        return

    st.markdown("---")
    st.subheader(f"'{details['final_name']}' 레시피")

    img_bytes = get_or_generate_dish_image(details["final_name"])
    if img_bytes:
        st.image(
            img_bytes,
            caption=f"{details['final_name']} 예시 이미지",
            width='stretch'
        )

    if details.get("from_recipes_txt"):
        st.caption("이 레시피는 엄마의 레시피 노트 (recipes.txt)를 참고해서 정리했어.")
    else:
        st.caption("엄마가 일반적인 요리 지식을 참고해서 정리했어.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 재료")
        if details["ingredients"]:
            for ing in details["ingredients"]:
                st.markdown(f"- {ing}")
        else:
            st.write("재료 정보가 부족해")

    with col2:
        st.markdown("### 조리 방법")
        if details["steps"]:
            for _, step in enumerate(details["steps"], start=1):
                st.markdown(f"{step}")
        else:
            st.write("조리 단계를 자세히 나누기 어려워서, 요약만 보여줄게.")
            st.write(details["summary"])

    #추가 정보 : 재료 비교 + 영양
    st.markdown("---")
    st.markdown("### 추가 정보")

    # 냉장고 재료
    fridge = st.session_state.get("fridge_ingredients", [])
    recipe_ings = details["ingredients"]

    in_fridge = []
    not_in_fridge = []

    fridge_lower = [x.lower() for x in fridge]

    for ing in recipe_ings:
        ing_lower = ing.lower()
        if any((f in ing_lower) or (ing_lower in f) for f in fridge_lower):
            in_fridge.append(ing)
        else:
            not_in_fridge.append(ing)

    st.markdown("**냉장고에 있는 재료**")
    st.write(", ".join(in_fridge) if in_fridge else "없음")

    st.markdown("**냉장고에 없는 재료**")
    st.write(", ".join(not_in_fridge) if not_in_fridge else "없음")

    # 영양
    nut = details.get("nutrition")
    if nut:
        st.markdown("**영양 정보 (대략적인 추정)**")
        c, p, f = nut.get("macro_ratio", (0, 0, 0))
        calorie_level = nut.get("calorie_level", "알 수 없음")
        st.write(f"- 칼로리: {calorie_level} 수준으로 추정")
        st.write(f"- 탄수화물:단백질:지방 ≈ {c}:{p}:{f}")


# -------------------------------
# 기능 3: 엄마와의 대화 (챗봇)
# -------------------------------

def render_feature_chat():
    st.header("📣엄마와의 대화")
    st.caption("엄마랑 대화하자! 요리, 재료, 생활 고민 뭐든지 물어봐.")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="user_avatar.png"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="mom_avatar.png"): 
                st.write(msg["content"])

    prompt = st.chat_input("엄마, 양파를 썰고 있는데 눈물이 너무 나와. 어떻게 해야 해? ㅠㅠ")

    if prompt:
        st.chat_message("user", avatar="user_avatar.png").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        chat_history = []
        for msg in st.session_state.messages[:-1]:
            role = "자식" if msg["role"] == "user" else "엄마"
            chat_history.append((role, msg["content"]))

        with st.spinner("엄마가 고민 중..."):
            response_text, retrieved_doc = agent_logic.generate_response(
                st.session_state.db,
                st.session_state.llm,
                prompt,
                chat_history,
            )
            
        with st.chat_message("assistant", avatar="mom_avatar.png"):
            st.write(response_text)
            
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )

        with st.expander("엄마가 참고한 레시피 노트 보기"):
            st.info(retrieved_doc)


# -------------------------------
# 사이드바: 기능 선택 + 냉장고
# -------------------------------

with st.sidebar:
    st.title("엄마 밥선생")
    st.markdown("아래 기능 중에 골라!")

    if st.button("🍳요리 추천", key="nav_recommend"):
        st.session_state.active_feature = "recommend"
        st.rerun()

    if st.button("📜레시피 검색", key="nav_info"):
        st.session_state.active_feature = "info"
        st.rerun()

    if st.button("📣엄마와 대화", key="nav_chat"):
        st.session_state.active_feature = "chat"
        st.rerun()

    st.markdown("---")
    render_fridge_sidebar()


# -------------------------------
# 메인 영역: 현재 기능 렌더링
# -------------------------------

st.title("🍲자취생을 위한 엄마 밥선생")
st.caption("엄마! 냉장고에 있는 걸로 뭐 해먹을까?")

feature = st.session_state.active_feature

if feature == "recommend":
    render_feature_recommend()
elif feature == "info":
    render_feature_info()
else:
    render_feature_chat()
