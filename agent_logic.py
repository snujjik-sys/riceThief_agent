from dotenv import load_dotenv
import os
import json
import re
from typing import List, Tuple, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

"""# 1. 에이전트 구성 요소 준비 (LLM, Vector DB)"""

def get_agent_components():
    """
    LLM과 Vector DB를 준비해서 반환하는 함수입니다.
    DB 폴더가 없으면 recipes.txt를 이용해 새로 생성합니다.
    """

    # 1. 임베딩 모델 준비
    embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

    db_directory = "./chroma_db"

    # 2. DB 준비
    if os.path.exists(db_directory):
        # (A) DB 폴더가 있으면 -> 바로 로드 (빠름)
        print("[System] 기존 ChromaDB를 로드합니다.")
        db = Chroma(
            persist_directory=db_directory,
            embedding_function=embed_model
        )
    else:
        # DB 폴더가 없으면 -> 텍스트 파일로 새로 생성
        print("[System] DB 폴더가 없어서 recipes.txt로 새로 만듭니다.")
        if os.path.exists("recipes.txt"):
            with open("recipes.txt", "r", encoding="utf-8") as f:
                # 빈 줄 제거하고 리스트로 변환
                recipe_data = [line.strip() for line in f.read().splitlines() if line.strip()]

            db = Chroma.from_texts(
                texts=recipe_data,
                embedding=embed_model,
                persist_directory=db_directory
            )
            print("[System] 새 ChromaDB 생성 완료!")
        else:
            # 파일도 없으면 에러 발생
            raise FileNotFoundError("recipes.txt 파일도 없고 chroma_db 폴더도 없습니다!")

    # 3. LLM 모델 준비
    llm = ChatOpenAI(
        openai_api_key=API_KEY,
        model_name='gpt-5-mini'
    )

    return db, llm

"""# 2. RAG 에이전트 실행 (Multi-turn 구현)"""

def generate_response(db, llm, user_input, chat_history):
    """
    [기능 추가] 사용자 질문과 '대화 기록(history)'을 함께 받아 답변을 생성합니다.
    """

    # 1. 검색 (Retrieval)
    print(f"[System] 검색 중: {user_input}")
    results = db.similarity_search(user_input, k=2)

    if results:
        retrieved_knowledge = "\n".join([x.page_content for x in results])
    else:
        retrieved_knowledge = "관련된 레시피를 찾지 못했습니다."


    # 2. 프롬프트 증강 (Augmentation) - Multi-turn 적용
    # 이전 대화 내용을 텍스트로 예쁘게 정리
    history_text = ""
    if chat_history:
        history_text = "\n".join([f"- {role}: {msg}" for role, msg in chat_history])
    else:
        history_text = "(없음)"

    system_prompt = (
        "너는 자취하는 자식에게 요리를 알려주는 다정한 '엄마'야. "
        "말투는 항상 따뜻하고 친근하게 해줘. "
        "1. [참고 자료]에 있는 레시피를 우선적으로 소개해야 해. "
        "2. [대화 기록]을 읽고 이전 대화의 흐름을 파악해서 자연스럽게 이어가. (예: '아까 말한 계란으로...') "
        "3. 만약 [참고 자료]의 요리가 질문과 관련이 있으면, '엄마 노트에 이런 게 있단다~' 하고 먼저 알려줘."
        "4. 만약 참고 자료에 없으면 '엄마 노트엔 없는데~' 하고 너의 일반적인 지식을 덧붙여서 친절하게 알려줘."
    )

    # 이전 대화 기록을 텍스트로 변환
    history_text = "\n".join([f"{role}: {msg}" for role, msg in chat_history])

    augmented_prompt = f"""
    [참고 자료 - 엄마의 레시피 노트]
    {retrieved_knowledge}

    [대화 기록 (이전 대화 맥락)]
    {history_text}

    [현재 자식의 질문]
    {user_input}

    (위 내용을 종합해서 엄마처럼 대답해줘)
    """

    # 3. 답변 생성 (Generation)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=augmented_prompt)
    ]

    response = llm.invoke(messages)

    # UI에 표시하기 위해 '답변 내용'과 '참고한 문서 내용'을 같이 반환
    return response.content, retrieved_knowledge

"""# 3. 공통 LLM 헬퍼"""


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    LLM이 JSON 앞뒤로 설명을 붙였을 때를 대비해서,
    문자열에서 JSON 오브젝트 부분만 뽑아서 dict로 변환하는 유틸 함수.
    """
    text = text.strip()
    # 이미 순수 JSON일 수도 있음
    try:
        return json.loads(text)
    except Exception:
        pass

    # 중괄호 블록만 추출
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    # 실패 시 raw 텍스트를 그대로 반환
    return {"raw": text}


def _call_llm_json(llm: ChatOpenAI, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """
    System/User 프롬프트를 주고, JSON 한 줄을 반환하도록 시킨 뒤
    실제로 JSON을 파싱해서 dict로 돌려주는 헬퍼.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    text = response.content if hasattr(response, "content") else str(response)
    return _extract_json_from_text(text)


"""# 4. 기능 1: 조건 기반 요리 추천"""


def recommend_recipe(
    db,
    llm: ChatOpenAI,
    fridge_ingredients: List[str],
    conditions: Dict[str, Any],
) -> Dict[str, Any]:
    """
    기능 1에서 사용되는 "요리 추천" 로직.

    - fridge_ingredients: 냉장고 속 재료 목록
    - conditions:
        {
          "use_ingredients": bool,
          "ingredient_mode": "fridge" | "ignore",
          "use_nutrition": bool,
          "calorie_pref": "낮음" | "보통" | "높음" | None,
          "macro_ratio": (c, p, f) or None,
          "use_type": bool,
          "dish_type": "식사" | "반찬" | "디저트" | "안주" | None
        }

    반환값 예시:
        {
          "name": "해장라면",
          "reason": "추천 이유 한국어 설명",
          "use_recipes_txt": True/False
        }
    """

    # 최근 추천되어서 당분간 피하고 싶은 요리들
    forbidden_names = conditions.get("forbidden_names", []) or []

    # 1) 검색 쿼리 구성
    if conditions.get("use_ingredients") and conditions.get("ingredient_mode") == "fridge" and fridge_ingredients:
        ing_text = " ".join(fridge_ingredients)
        query = f"{ing_text}로 만들 수 있는 집밥 요리"
    else:
        query = "자취생이 좋아할만한 한국 집밥 요리"

    if conditions.get("use_type") and conditions.get("dish_type"):
        query = f"{conditions['dish_type']}용 {query}"

    # 2) 레시피 후보군 검색 (recipes.txt 기반 RAG)
    results = db.similarity_search(query, k=8)
    candidates_text = ""

    for i, doc in enumerate(results):
        candidates_text += f"\n[후보 {i+1}]\n{doc.page_content}\n"

    # 3) 조건 설명 텍스트로 정리
    cond_lines = []

    # 재료 조건
    if conditions.get("use_ingredients"):
        if conditions.get("ingredient_mode") == "fridge":
            cond_lines.append(
                f"- 냉장고에 있는 재료를 최대한 많이 활용하는 요리면 좋겠어: "
                f"{', '.join(fridge_ingredients) if fridge_ingredients else '재료 정보 없음'}"
            )
        else:
            cond_lines.append("- 재료는 냉장고와 크게 상관 없어도 괜찮아.")
    else:
        cond_lines.append("- 재료 조건은 따로 신경 쓰지 않아도 돼.")

    # 영양 조건
    if conditions.get("use_nutrition"):
        cal = conditions.get("calorie_pref")
        ratio = conditions.get("macro_ratio")
        if cal:
            cond_lines.append(f"- 칼로리는 다른 음식과 비교했을 때 '{cal}' 수준이면 좋겠어.")
        if ratio:
            c, p, f = ratio
            cond_lines.append(f"- 탄수화물:단백질:지방 비율은 대략 {c}:{p}:{f}에 가까우면 좋겠어.")
            cond_lines.append(
                "  (완벽히 일치하지 않아도 되지만, 예를 들어 2:1:4처럼 지방이 너무 높은 비율은 피해줘.)"
            )
    else:
        cond_lines.append("- 영양(칼로리/탄단지) 조건은 크게 상관 없어.")

    # 종류 조건
    if conditions.get("use_type") and conditions.get("dish_type"):
        cond_lines.append(f"- 음식 종류는 '{conditions['dish_type']}'에 해당하면 좋겠어.")
    else:
        cond_lines.append("- 식사/반찬/디저트/안주 구분은 크게 상관 없어.")

    # 최근에 추천된 요리들 (가능하면 피하기)
    if forbidden_names:
        cond_lines.append(
            "- 아래 요리들은 최근에 여러 번 추천했으니까, 오늘은 가능하면 빼 줘: "
            + ", ".join(forbidden_names)
        )

    cond_text = "\n".join(cond_lines)

    # 4) LLM에게 최종 요리 1개를 고르게 함
    system_prompt = (
        "너는 자취하는 자식에게 요리를 추천해주는 다정한 '엄마'야. "
        "아래 조건을 최대한 존중해서 요리 하나만 골라줘. "
        "조건과 완전히 일치할 필요는 없지만, 재료/영양/종류에서 너무 어긋나는 후보는 피해야 해."
        "또한 최근에 이미 여러 번 추천된 요리는 가능하면 다시 추천하지 마."
    )

    user_prompt = f"""
[자식이 말한 조건]
{cond_text}

[냉장고 속 재료]
{', '.join(fridge_ingredients) if fridge_ingredients else '입력된 재료 없음'}

[엄마의 레시피 노트 후보들 (recipes.txt 기반)]
{candidates_text if candidates_text.strip() else '(관련된 후보가 거의 없어. 필요하면 엄마의 일반적인 요리 지식을 써도 돼.)'}

[최근에 이미 여러 번 추천된 요리들]
{', '.join(forbidden_names) if forbidden_names else '없음'}

요리 이름은 순수한 이름만 써줘. 예를 들어
- "엄마표 '해장라면'" 이라고 되어 있어도 최종 이름은 "해장라면" 처럼.

반드시 아래 JSON 형식으로, 한 줄로만 답해:

{{
  "name": "추천할 요리 이름",
  "reason": "이 요리를 추천하는 이유를 자식에게 말하듯 한국어로 2~3문장으로 설명",
  "use_recipes_txt": true
}}

use_recipes_txt 는 위의 레시피 노트 후보들 중 하나를 기반으로 고른 요리라면 true,
완전히 새로 지은 요리라면 false 로 설정해.
"""

    data = _call_llm_json(llm, system_prompt, user_prompt)

    name = (data.get("name") or "추천 요리").strip()
    reason = (data.get("reason") or "엄마 감으로 골라봤어.").strip()
    use_recipes_txt = bool(data.get("use_recipes_txt", True))

    return {
        "name": name,
        "reason": reason,
        "use_recipes_txt": use_recipes_txt,
    }


"""# 5. 기능 2: 특정 음식 정보 + 영양 분석"""

#칼로리, 탄단지
def _analyze_nutrition(llm: ChatOpenAI, dish_name: str, raw_text: str) -> Dict[str, Any]:
    system_prompt = (
        "너는 한국 가정식 요리의 영양을 대략 추정해주는 영양사야. "
        "정확한 수치가 아니라 '대략적인 경향'을 알려주는 것이 목표야."
    )

    user_prompt = f"""
요리 이름: {dish_name}

[레시피 설명]
{raw_text}

1. 이 음식의 칼로리 수준을 일반적인 한 끼 식사와 비교했을 때
   "낮음", "보통", "높음" 중 하나로 판단해줘.
2. 탄수화물:단백질:지방 비율을 대략적인 정수 비율로 추정해서
   "x@y@z" 형태로만 표현해줘. (예: "4@3@3")

반드시 아래 JSON 형식 한 줄로만 답해:

{{
  "calorie_level": "낮음 또는 보통 또는 높음",
  "macro_ratio": "x@y@z"
}}
"""

    data = _call_llm_json(llm, system_prompt, user_prompt)

    calorie_level = data.get("calorie_level", "보통")
    ratio_str = data.get("macro_ratio", "4@3@3")

    try:
        x_str, y_str, z_str = ratio_str.split("@")
        ratio = (int(x_str), int(y_str), int(z_str))
    except Exception:
        ratio = (4, 3, 3)

    return {
        "calorie_level": calorie_level,
        "macro_ratio": ratio,
    }

#레시피 검색
def get_recipe_details(db, llm: ChatOpenAI, dish_name: str) -> Dict[str, Any]:
    # 0) dish_name이 실제 음식/요리/음료 이름인지 판별
    clf_system_prompt = (
        "너는 입력 문자열이 음식/요리/음료 이름인지 판별하는 도우미야. "
        "한국어/외국어 모두 가능하지만, 의미 없는 자모 나열이나 사람/장소/자동차/회사 이름 등은 음식이 아니라고 판단해. "
        "조금 이상한 철자라도 음식 같으면 true 로 봐도 좋아."
    )

    clf_user_prompt = f"""
입력: "{dish_name}"

다음 형식의 JSON 한 줄로만 답해:

{{
  "is_food": true 또는 false,
  "normalized_name": "정리된 음식 이름 (알 수 없으면 빈 문자열)"
}}
"""

    try:
        clf = _call_llm_json(llm, clf_system_prompt, clf_user_prompt)
        is_food = bool(clf.get("is_food", True))
        normalized_name = (clf.get("normalized_name") or dish_name).strip()
    except Exception:
        # 분류에 실패하면 일단 음식이라고 가정하고 진행
        is_food = True
        normalized_name = dish_name

    # ⚠️ 음식이 아니라고 판별된 경우: 레시피를 만들지 않고 바로 리턴
    if not is_food:
        final_name = normalized_name or dish_name
        msg = (
            f"'{final_name}'는(은) 음식 이름으로는 엄마가 잘 모르겠어. "
            "혹시 음식 이름이 맞는지, 또는 철자를 조금 다르게 써 볼래?"
        )
        return {
            "final_name": final_name,
            "from_recipes_txt": False,
            "ingredients": [],
            "steps": [],
            "summary": msg,
            "nutrition": None,
            "source_text": "",
            "not_food": True,
        }

    # 여기서부터는 '음식/요리/음료'라고 판정된 경우만 진행
    query_name = normalized_name

    # 1) recipes.txt 기반으로 관련 레시피 검색 (유사도 점수까지 사용)
    try:
        scored_results = db.similarity_search_with_relevance_scores(
            query_name,
            k=5,
            score_threshold=0.0,  # threshold는 아래에서 수동으로 적용
        )
    except Exception:
        # 사용하는 버전에 따라 이 메서드가 없을 수도 있으니 fallback
        docs_only = db.similarity_search(query_name, k=3)
        scored_results = [(doc, 1.0) for doc in docs_only]

    # relevance score는 [0, 1] 범위 (1에 가까울수록 더 유사)
    # 이 threshold보다 낮으면 "그다지 비슷하지 않다"고 보고 엄마 노트로 쓰지 않음
    SIM_THRESHOLD = 0.1

    filtered_docs = [
        doc for (doc, score) in scored_results
        if score is not None and score >= SIM_THRESHOLD
    ]

    from_recipes_txt = bool(filtered_docs)

    if from_recipes_txt:
        # 충분히 유사한 레시피가 있는 경우에만 엄마 노트 내용을 LLM에 제공
        combined_text = "\n\n".join([doc.page_content for doc in filtered_docs])
        source_text = filtered_docs[0].page_content
    else:
        # 유사한 레시피가 하나도 없으면, 엄마 노트는 없는 것처럼 처리
        combined_text = "(관련된 레시피를 엄마 노트에서 찾지 못했어.)"
        source_text = query_name

    # 2) LLM에게 재료/조리법 정리 요청
    system_prompt = (
        "너는 자취생에게 요리를 알려주는 다정한 '엄마'야. "
        "가능하면 [엄마의 레시피 노트에서 찾은 내용]을 우선 활용하고, "
        "그 내용이 충분하지 않거나 거의 없으면 너의 일반 요리 지식을 보충해서 레시피를 완성해. "
        "만약 [엄마의 레시피 노트에서 찾은 내용]이 사실상 "
        "'관련된 레시피를 엄마 노트에서 찾지 못했어.' 같은 안내 문구뿐이라면, "
        "엄마의 레시피 노트는 없는 것처럼 생각하고 일반 요리 지식으로 레시피를 만들어줘."
    )

    user_prompt = f"""
요리 이름: {query_name}

[엄마의 레시피 노트에서 찾은 내용]
{combined_text}

위 정보를 바탕으로, 자식이 이해하기 쉽게 아래 정보를 정리해줘.

- 최종 요리 이름 (예: "~덮밥", "~볶음밥" 형태 등)
- 재료 목록 (한국어 재료명과 단위를 대략적으로 포함)
- 단계별 조리 방법 (1, 2, 3 ... 순서)
- 레시피 전체 내용을 2~4문장 정도로 요약

반드시 아래 JSON 형식 한 줄로만 답해:

{{
  "final_name": "최종 요리 이름",
  "from_recipes_txt": {str(from_recipes_txt).lower()},
  "ingredients": ["재료1", "재료2", "..."],
  "steps": ["1단계 설명", "2단계 설명", "..."],
  "raw_text": "레시피 전체를 2~4문장으로 요약한 설명"
}}
"""

    data = _call_llm_json(llm, system_prompt, user_prompt)

    final_name = (data.get("final_name") or query_name).strip()

    ingredients = data.get("ingredients") or []
    if isinstance(ingredients, str):
        ingredients = [ingredients]
    ingredients = [str(x).strip() for x in ingredients if str(x).strip()]

    steps = data.get("steps") or []
    if isinstance(steps, str):
        steps = [steps]
    steps = [str(x).strip() for x in steps if str(x).strip()]

    summary = (data.get("raw_text") or combined_text).strip()

    # 3) 영양 분석 추가 (엄마 노트가 없더라도, LLM 결과 텍스트 기반으로 분석)
    nutrition = _analyze_nutrition(llm, final_name, combined_text)

    return {
        "final_name": final_name,
        "from_recipes_txt": bool(data.get("from_recipes_txt", from_recipes_txt)),
        "ingredients": ingredients,
        "steps": steps,
        "summary": summary,
        "nutrition": nutrition,
        "source_text": source_text,
        "not_food": False,
    }