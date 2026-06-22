from collections import defaultdict

import pandas as pd
import requests
import streamlit as st


API_BASE = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="NLP Graph-RAG Teaching Assistant",
    page_icon="",
    layout="wide",
)


def post_json(path: str, payload: dict):
    response = requests.post(f"{API_BASE}{path}", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


def get_json(path: str):
    response = requests.get(f"{API_BASE}{path}", timeout=60)
    response.raise_for_status()
    return response.json()


def upload_course_file(uploaded_file):
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/pdf",
        )
    }
    response = requests.post(f"{API_BASE}/api/parsing/upload", files=files, timeout=180)
    response.raise_for_status()
    return response.json()


def show_error(exc: Exception):
    st.error(f"请求失败：{exc}")
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        st.code(exc.response.text)


def set_followup_question(concept: str):
    st.session_state["qa_question_input"] = f"请解释一下「{concept}」这个知识点。"
    st.session_state["selected_concept_card"] = concept
    st.session_state["active_tab"] = "智能问答"
    st.session_state["followup_message"] = f"已选择「{concept}」，智能问答输入框已更新。"


def set_compare_pair(concept_a: str, concept_b: str):
    st.session_state["compare_concept_a"] = concept_a
    st.session_state["compare_concept_b"] = concept_b
    st.session_state["active_tab"] = "概念对比"


def set_quiz_concept(concept: str):
    st.session_state["quiz_concept_input"] = concept
    st.session_state["quiz_concept_message"] = f"已选择「{concept}」，自动出题知识点已更新。"


def evidence_text(level: str, summary: str) -> tuple[str, str]:
    mapping = {
        "high": ("证据充分", "本次回答同时检索到了课程资料和较丰富的知识图谱关系，有足够的外部支撑。"),
        "medium": ("证据较充分", "本次回答有一定课程资料或知识图谱依据，但外部支撑还可以进一步补充。"),
        "low": ("证据不足", "本次回答可用的外部依据较少，建议谨慎参考或补充课程资料。"),
    }
    title, default_detail = mapping.get(level, mapping["low"])
    return title, summary or default_detail


def render_sources(sources: list[str]):
    if not sources:
        st.info("未检索到课程资料来源")
        return

    source_rows = [
        {"序号": index + 1, "课程资料片段": source}
        for index, source in enumerate(sources)
    ]
    st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)


def render_review_concepts(concepts: list[str], label: str = "建议复习", key_prefix: str = ""):
    if not concepts:
        st.info("暂无复习建议")
        return

    st.markdown(f"**{label}**")
    unique_concepts = list(dict.fromkeys(concepts))
    for row_start in range(0, len(unique_concepts), 4):
        cols = st.columns(4, gap="small")
        for col_offset, concept in enumerate(unique_concepts[row_start:row_start + 4]):
            index = row_start + col_offset
            with cols[col_offset]:
                if st.button(
                    f"› {concept}",
                    key=f"review_concept_{key_prefix}_{label}_{index}_{concept}",
                    help=f"点击后在智能问答中解释：{concept}",
                    use_container_width=True,
                    on_click=set_followup_question,
                    args=(str(concept),),
                ):
                    pass


def render_concept_card(concept: str, triples: list[dict]):
    related = [
        triple
        for triple in triples
        if triple.get("head") == concept or triple.get("tail") == concept
    ]

    st.markdown("**概念名片**")
    st.info(f"当前选择：{concept}")

    if related:
        grouped = defaultdict(list)
        for triple in related:
            if triple.get("head") == concept:
                grouped[triple["relation"]].append(triple["tail"])
            else:
                grouped[f"被关联为：{triple['relation']}"].append(triple["head"])

        for relation, values in grouped.items():
            st.write(f"**{relation}**：{'、'.join(list(dict.fromkeys(values)))}")
    else:
        st.write("当前关系集合中没有更多关联信息。")


def render_kg_relations(triples: list[dict], clickable: bool = True):
    if not triples:
        st.info("未检索到相关知识图谱关系")
        return

    st.markdown("**关系摘要**")
    grouped = defaultdict(list)
    for triple in triples:
        grouped[(triple["head"], triple["relation"])].append(triple["tail"])

    for group_index, ((head, relation), tails) in enumerate(grouped.items()):
        unique_tails = list(dict.fromkeys(tails))
        st.markdown(f"**{head}** 的 **{relation}**：")
        if clickable:
            for row_start in range(0, len(unique_tails), 4):
                cols = st.columns(4, gap="small")
                for col_offset, tail in enumerate(unique_tails[row_start:row_start + 4]):
                    index = row_start + col_offset
                    with cols[col_offset]:
                        if st.button(
                            f"› {tail}",
                            key=f"kg_tail_{group_index}_{index}_{head}_{relation}_{tail}",
                            help=f"点击后在智能问答中解释：{tail}",
                            use_container_width=True,
                            on_click=set_followup_question,
                            args=(str(tail),),
                        ):
                            pass
        else:
            st.write("、".join(unique_tails))

    st.markdown("**图谱可视化**")
    st.caption("保留全部关系，缩小显示区域；如图较密集，可结合下方表格查看完整关系。")

    edge_colors = {
        "先修知识": "#d95f02",
        "相关概念": "#1b9e77",
        "包含": "#7570b3",
    }

    graph_lines = ["digraph {"]
    graph_lines.append('  graph [rankdir="LR", size="7.5,4.2", ratio="compress", margin=0.04];')
    graph_lines.append('  node [shape=box, style="rounded,filled", fontsize=10, margin=0.08, fillcolor="#eef6ff", color="#8ab6d6"];')
    graph_lines.append('  edge [fontsize=9, arrowsize=0.65];')
    for triple in triples:
        head = str(triple["head"]).replace('"', '\\"')
        relation = str(triple["relation"]).replace('"', '\\"')
        tail = str(triple["tail"]).replace('"', '\\"')
        color = edge_colors.get(str(triple["relation"]), "#666666")
        graph_lines.append(f'  "{head}" -> "{tail}" [label="{relation}", color="{color}", fontcolor="{color}"];')
    graph_lines.append("}")
    with st.container(height=520):
        st.graphviz_chart("\n".join(graph_lines), use_container_width=True)

    st.caption("边颜色：先修知识=橙色，相关概念=绿色，包含=紫色，其他关系=灰色。")

    selected_concept = st.session_state.get("selected_concept_card")
    if selected_concept:
        render_concept_card(selected_concept, triples)

    st.markdown("**三元组表格**")
    st.dataframe(pd.DataFrame(triples), use_container_width=True, hide_index=True)


def render_learning_path(data: dict):
    path = data.get("path", [])
    edges = data.get("edges", [])
    follow_ups = data.get("follow_ups", [])

    if not path:
        st.warning("暂未找到学习路径")
        return

    st.success(" -> ".join(path))

    graph_lines = ["digraph {"]
    graph_lines.append('  graph [rankdir="LR", margin=0.05];')
    graph_lines.append('  node [shape=box, style="rounded,filled", fontsize=12, margin=0.12, fillcolor="#eef6ff", color="#8ab6d6"];')
    graph_lines.append('  edge [fontsize=10, arrowsize=0.7, color="#d95f02", fontcolor="#d95f02"];')
    for step, node in enumerate(path, start=1):
        safe_node = str(node).replace('"', '\\"')
        graph_lines.append(f'  "{safe_node}" [label="{step}. {safe_node}"];')
    for edge in edges:
        head = str(edge.get("head", "")).replace('"', '\\"')
        tail = str(edge.get("tail", "")).replace('"', '\\"')
        relation = str(edge.get("relation", "")).replace('"', '\\"')
        graph_lines.append(f'  "{head}" -> "{tail}" [label="{relation}"];')
    graph_lines.append("}")

    st.markdown("**路径可视化**")
    st.graphviz_chart("\n".join(graph_lines), use_container_width=True)

    if edges:
        st.markdown("**路径关系表**")
        edge_rows = [
            {
                "序号": index + 1,
                "前置知识": edge.get("head", ""),
                "关系": edge.get("relation", ""),
                "目标知识": edge.get("tail", ""),
            }
            for index, edge in enumerate(edges)
        ]
        st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)

    st.markdown("**学完后可继续探索路径中的前序知识点**")
    if follow_ups:
        unique_follow_ups = list(dict.fromkeys(follow_ups))
        for row_start in range(0, len(unique_follow_ups), 4):
            cols = st.columns(4, gap="small")
            for col_offset, concept in enumerate(unique_follow_ups[row_start:row_start + 4]):
                index = row_start + col_offset
                with cols[col_offset]:
                    st.button(
                        f"› {concept}",
                        key=f"learning_followup_{index}_{concept}",
                        help=f"点击后在智能问答中解释：{concept}",
                        use_container_width=True,
                        on_click=set_followup_question,
                        args=(str(concept),),
                    )
    else:
        st.info("当前目标知识点暂无明确前序知识点。")


def render_concept_list(concepts: list[str], empty_text: str):
    if concepts:
        rows = [{"序号": index + 1, "知识点": concept} for index, concept in enumerate(concepts)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info(empty_text)


def render_coverage(data: dict):
    asked = data.get("asked_entities", [])
    missing = data.get("missing_entities", [])
    recommended = data.get("recommended_entities", [])
    coverage_rate = data.get("coverage_rate", 0)

    col1, col2, col3 = st.columns(3)
    col1.metric("知识覆盖率", f"{coverage_rate:.1%}")
    col2.metric("已覆盖知识点", len(asked))
    col3.metric("未覆盖知识点", len(missing))

    st.markdown("**已覆盖知识点**")
    render_concept_list(asked, "还没有记录到已覆盖知识点。请先在智能问答、自动出题、概念对比或学习路径中使用一些知识点。")

    st.markdown("**建议优先补充**")
    render_concept_list(recommended, "暂无推荐补充知识点。")


st.title("NLP 课程 Graph-RAG 智能教学助手")
st.caption("融合课程资料检索、知识图谱、大模型问答、自动出题与学习分析")

with st.sidebar:
    st.header("后端配置")
    api_base = st.text_input("FastAPI 地址", value=API_BASE)
    if api_base:
        API_BASE = api_base.rstrip("/")
    st.info("使用前请先启动 FastAPI 后端，并完成课程资料解析与索引构建。")

    st.divider()
    st.header("课程资料准备")
    uploaded_course_file = st.file_uploader("上传课程 PDF", type=["pdf"])

    if st.button("上传并解析课程资料", disabled=uploaded_course_file is None):
        try:
            parse_result = upload_course_file(uploaded_course_file)
            st.success(
                f"解析完成：{parse_result.get('filename', uploaded_course_file.name)}，"
                f"共生成 {parse_result.get('chunk_count', 0)} 个文本片段。"
            )
        except Exception as exc:
            show_error(exc)

    if st.button("构建/更新检索索引"):
        try:
            index_result = post_json("/api/search/build_index", {})
            if "error" in index_result:
                st.warning(index_result["error"])
            else:
                st.success(
                    f"索引构建完成：{index_result.get('chunk_count', 0)} 个片段，"
                    f"词表大小 {index_result.get('vocab_size', 0)}。"
                )
        except Exception as exc:
            show_error(exc)


tab_labels = ["智能问答", "自动出题与答题", "概念对比", "学习路径", "学习报告"]
if st.session_state.get("active_tab") not in tab_labels:
    st.session_state["active_tab"] = "智能问答"

active_tab = st.radio(
    "功能导航",
    tab_labels,
    key="active_tab",
    horizontal=True,
    label_visibility="collapsed",
)


if active_tab == "智能问答":
    st.subheader("Graph-RAG 智能问答")
    if "qa_question_input" not in st.session_state:
        st.session_state["qa_question_input"] = "词嵌入是什么？Word2Vec 和它有什么关系？"

    followup_message = st.session_state.pop("followup_message", None)
    if followup_message:
        st.success(followup_message)

    with st.expander("基于学习历史推荐的知识点", expanded=False):
        col_refresh, col_hint = st.columns([1, 4])
        with col_refresh:
            refresh_recommendations = st.button("刷新推荐", key="refresh_qa_recommendations")
        with col_hint:
            st.caption("推荐会结合智能问答历史、错题本记录、知识图谱关联和知识覆盖度。")

        if refresh_recommendations or "qa_recommended_concepts" not in st.session_state:
            try:
                st.session_state["qa_recommended_concepts"] = get_json("/api/analytics/recommendations?limit=8")
            except Exception as exc:
                show_error(exc)

        recommended = st.session_state.get("qa_recommended_concepts", {}).get("concepts", [])
        if recommended:
            concepts = [item.get("concept", "") for item in recommended]
            render_review_concepts(concepts, label="推荐追问", key_prefix="qa_history")
            if st.checkbox("查看推荐依据", key="show_qa_recommendation_reasons"):
                rows = [
                    {
                        "知识点": item.get("concept", ""),
                        "推荐依据": item.get("reason", ""),
                        "来源": item.get("source", ""),
                    }
                    for item in recommended
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("暂无推荐知识点。可以先进行问答、出题或答题后再刷新。")

    question = st.text_area(
        "请输入你的问题",
        height=100,
        key="qa_question_input",
    )

    should_run_qa = st.button("生成回答", type="primary")

    if should_run_qa:
        try:
            data = post_json("/api/rag/ask", {"question": question})
            st.markdown("### 回答")
            st.write(data.get("answer", ""))

            st.markdown("### 证据充分性")
            evidence_title, evidence_detail = evidence_text(
                data.get("evidence_level", "low"),
                data.get("evidence_summary", ""),
            )
            st.success(f"{evidence_title}：{evidence_detail}")
            st.caption(
                f"课程资料来源 {len(data.get('sources', []))} 个；"
                f"知识图谱关系 {len(data.get('related_triples', []))} 条。"
            )

            with st.expander("课程资料来源"):
                render_sources(data.get("sources", []))

            with st.expander("知识图谱关系"):
                render_kg_relations(data.get("related_triples", []))
        except Exception as exc:
            show_error(exc)


if active_tab == "自动出题与答题":
    st.subheader("自动出题")
    if "quiz_concept_input" not in st.session_state:
        st.session_state["quiz_concept_input"] = "词嵌入"

    concept_message = st.session_state.pop("quiz_concept_message", None)
    if concept_message:
        st.success(concept_message)

    concept = st.text_input("知识点", key="quiz_concept_input")

    with st.expander("基于学习历史推荐的出题知识点", expanded=False):
        col_refresh, col_hint = st.columns([1, 4])
        with col_refresh:
            refresh_quiz_recommendations = st.button("刷新推荐", key="refresh_quiz_recommendations")
        with col_hint:
            st.caption("推荐会结合错题本、智能问答历史、知识图谱关联和知识覆盖度。")

        if refresh_quiz_recommendations or "quiz_recommended_concepts" not in st.session_state:
            try:
                st.session_state["quiz_recommended_concepts"] = get_json("/api/analytics/recommendations?limit=8")
            except Exception as exc:
                show_error(exc)

        quiz_recommended = st.session_state.get("quiz_recommended_concepts", {}).get("concepts", [])
        if quiz_recommended:
            unique_concepts = list(dict.fromkeys([item.get("concept", "") for item in quiz_recommended if item.get("concept", "")]))
            for row_start in range(0, len(unique_concepts), 4):
                cols = st.columns(4, gap="small")
                for col_offset, recommended_concept in enumerate(unique_concepts[row_start:row_start + 4]):
                    index = row_start + col_offset
                    with cols[col_offset]:
                        st.button(
                            f"› {recommended_concept}",
                            key=f"quiz_recommend_{row_start}_{index}_{recommended_concept}",
                            help=f"点击后用「{recommended_concept}」作为出题知识点",
                            use_container_width=True,
                            on_click=set_quiz_concept,
                            args=(str(recommended_concept),),
                        )
            if st.checkbox("查看推荐依据", key="show_quiz_recommendation_reasons"):
                rows = [
                    {
                        "知识点": item.get("concept", ""),
                        "推荐依据": item.get("reason", ""),
                        "来源": item.get("source", ""),
                    }
                    for item in quiz_recommended
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("暂无推荐知识点。可以先进行问答、出题或答题后再刷新。")

    col1, col2 = st.columns(2)
    with col1:
        num_questions = st.number_input("题目数量", min_value=1, max_value=10, value=3)
    with col2:
        difficulty = st.selectbox("难度", ["easy", "medium", "hard"], index=1)

    question_types = st.multiselect(
        "题型",
        [
            "choice",
            "multiple_choice",
            "true_false",
            "fill",
            "short_answer",
            "compare",
            "application",
        ],
        default=["choice", "fill", "short_answer"],
    )

    if st.button("生成题目", type="primary"):
        try:
            payload = {
                "concept": concept,
                "num_questions": int(num_questions),
                "question_types": question_types,
                "difficulty": difficulty,
            }
            quiz_data = post_json("/api/quiz/generate", payload)
            st.session_state["latest_quiz"] = quiz_data
            st.success("题目生成成功")
        except Exception as exc:
            show_error(exc)

    latest_quiz = st.session_state.get("latest_quiz")
    if latest_quiz:
        st.markdown("### 生成结果")
        optimize_message = st.session_state.pop("quiz_optimize_message", None)
        if optimize_message:
            st.success(optimize_message)
        questions = latest_quiz.get("questions", [])
        for index, item in enumerate(questions):
            question_id = item.get("id", index + 1)
            question_type = item.get("type", "short_answer")
            submit_key = f"quiz_submit_result_{question_id}_{index}"

            with st.expander(f"题目 {question_id}｜{question_type}", expanded=index == 0):
                st.markdown("**题干**")
                st.write(item.get("question", ""))

                options = item.get("options")
                answer_key = f"student_answer_{question_id}_{index}"

                if question_type == "choice" and options:
                    student_answer = st.radio(
                        "请选择答案",
                        options,
                        key=answer_key,
                    )
                elif question_type == "multiple_choice" and options:
                    selected = st.multiselect(
                        "请选择所有正确答案",
                        options,
                        key=answer_key,
                    )
                    student_answer = "；".join(selected)
                elif question_type == "true_false":
                    tf_options = options or ["正确", "错误"]
                    student_answer = st.radio(
                        "请选择判断结果",
                        tf_options,
                        key=answer_key,
                    )
                else:
                    student_answer = st.text_area(
                        "请输入你的答案",
                        key=answer_key,
                        height=90,
                    )

                if st.button("提交本题并查看反馈", key=f"submit_quiz_{question_id}_{index}"):
                    try:
                        submit_payload = {
                            "concept": latest_quiz.get("concept", concept),
                            "question": item.get("question", ""),
                            "student_answer": student_answer,
                            "reference_answer": item.get("answer", ""),
                            "explanation": item.get("explanation", ""),
                        }
                        st.session_state[submit_key] = post_json("/api/quiz/submit", submit_payload)
                    except Exception as exc:
                        show_error(exc)

                result = st.session_state.get(submit_key)
                if result:
                    st.markdown("**答题反馈**")
                    col1, col2 = st.columns(2)
                    col1.metric("得分", result.get("score", 0))
                    col2.metric("是否正确", "是" if result.get("is_correct") else "否")
                    st.write(result.get("feedback", ""))
                    render_review_concepts(
                        result.get("suggested_review", []),
                        key_prefix=f"quiz_{question_id}_{index}",
                    )
                    st.markdown("**参考答案**")
                    st.write(item.get("answer", ""))
                    st.markdown("**解析**")
                    st.write(item.get("explanation", ""))
                    render_sources(item.get("source_chunk_ids", []))

        st.divider()
        st.subheader("题目质量自检")
        question_labels = [
            f"题目 {item.get('id', idx + 1)}｜{item.get('type', 'short_answer')}"
            for idx, item in enumerate(questions)
        ]
        if question_labels:
            selected_label = st.selectbox("选择要评估的题目", question_labels)
            selected_index = question_labels.index(selected_label)
            selected_question = questions[selected_index]
            evaluation_key = f"quiz_eval_result_{selected_index}_{selected_question.get('id', selected_index + 1)}"

            if st.button("评估所选题目质量"):
                try:
                    eval_payload = {
                        "concept": latest_quiz.get("concept", concept),
                        "type": selected_question.get("type", "short_answer"),
                        "question": selected_question.get("question", ""),
                        "options": selected_question.get("options"),
                        "answer": selected_question.get("answer", ""),
                        "explanation": selected_question.get("explanation", ""),
                        "difficulty": difficulty,
                    }
                    st.session_state[evaluation_key] = post_json("/api/quiz/evaluate", eval_payload)
                except Exception as exc:
                    show_error(exc)

            result = st.session_state.get(evaluation_key)
            if result:
                st.markdown("**质量自检结果**")
                score_comments = result.get("score_comments", {})
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "指标": "总体质量",
                                "分数": result.get("overall_score", 0),
                                "评价": score_comments.get("overall_score", ""),
                            },
                            {
                                "指标": "题干清晰度",
                                "分数": result.get("clarity_score", 0),
                                "评价": score_comments.get("clarity_score", ""),
                            },
                            {
                                "指标": "答案正确性",
                                "分数": result.get("correctness_score", 0),
                                "评价": score_comments.get("correctness_score", ""),
                            },
                            {
                                "指标": "难度匹配度",
                                "分数": result.get("difficulty_match_score", 0),
                                "评价": score_comments.get("difficulty_match_score", ""),
                            },
                            {
                                "指标": "依据充分性",
                                "分数": result.get("evidence_score", 0),
                                "评价": score_comments.get("evidence_score", ""),
                            },
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown("**发现的问题**")
                problems = result.get("problems", [])
                if problems:
                    for problem in problems:
                        st.write(f"- {problem}")
                else:
                    st.success("暂无明显问题。")

                st.markdown("**修改建议**")
                suggestions = result.get("suggestions", [])
                if suggestions:
                    for suggestion in suggestions:
                        st.write(f"- {suggestion}")
                else:
                    st.info("暂无必须修改的地方，可以保留当前题目。")

                if st.button("根据自检结果自动优化并替换当前题目", key=f"optimize_quiz_{selected_index}"):
                    try:
                        optimize_payload = {
                            "concept": latest_quiz.get("concept", concept),
                            "type": selected_question.get("type", "short_answer"),
                            "question": selected_question.get("question", ""),
                            "options": selected_question.get("options"),
                            "answer": selected_question.get("answer", ""),
                            "explanation": selected_question.get("explanation", ""),
                            "difficulty": difficulty,
                            "problems": problems,
                            "suggestions": suggestions,
                            "score_comments": score_comments,
                        }
                        optimized = post_json("/api/quiz/optimize", optimize_payload)
                        optimized_question = optimized.get("optimized_question", {})
                        optimized_question["id"] = selected_question.get("id", selected_index + 1)
                        latest_quiz["questions"][selected_index] = optimized_question
                        st.session_state["latest_quiz"] = latest_quiz
                        st.session_state.pop(evaluation_key, None)
                        notes = optimized.get("optimization_notes", [])
                        if notes:
                            st.session_state["quiz_optimize_message"] = "；".join(notes)
                        else:
                            st.session_state["quiz_optimize_message"] = "已根据自检结果优化并替换当前题目。"
                        st.rerun()
                    except Exception as exc:
                        show_error(exc)


if active_tab == "概念对比":
    st.subheader("易混概念对比")
    st.caption("这里更适合做“辨析训练”：系统会突出二者区别、易混原因、判断规则和一道辨析小题。")

    st.session_state.setdefault("compare_concept_a", "CBOW")
    st.session_state.setdefault("compare_concept_b", "Skip-gram")

    with st.expander("推荐易混概念", expanded=True):
        col_refresh, col_hint = st.columns([1, 4])
        with col_refresh:
            refresh_compare_pairs = st.button("刷新推荐")
        with col_hint:
            st.caption("推荐会优先结合学习报告/以往问答记录，再补充知识图谱中的相关概念。")

        if refresh_compare_pairs or "compare_recommendations" not in st.session_state:
            try:
                st.session_state["compare_recommendations"] = get_json("/api/compare/recommend?limit=8")
            except Exception as exc:
                show_error(exc)

        recommendations = st.session_state.get("compare_recommendations", {}).get("pairs", [])
        if recommendations:
            for pair_index, pair in enumerate(recommendations):
                label = f"{pair.get('concept_a', '')} vs {pair.get('concept_b', '')}"
                reason = pair.get("reason", "")
                source = "学习记录" if pair.get("source") == "history" else "知识图谱"
                cols = st.columns([2, 5, 1])
                cols[0].button(
                    label,
                    key=f"compare_pair_{pair_index}_{label}",
                    use_container_width=True,
                    on_click=set_compare_pair,
                    args=(pair.get("concept_a", ""), pair.get("concept_b", "")),
                )
                cols[1].write(reason)
                cols[2].caption(source)
        else:
            st.info("暂时没有可推荐的概念对，可以先手动输入两个概念。")

    col1, col2 = st.columns(2)
    with col1:
        concept_a = st.text_input("概念 A", key="compare_concept_a")
    with col2:
        concept_b = st.text_input("概念 B", key="compare_concept_b")

    if st.button("生成概念对比", type="primary"):
        try:
            data = post_json(
                "/api/compare/concepts",
                {"concept_a": concept_a, "concept_b": concept_b},
            )
            st.markdown("### 对比结果")
            st.write(data.get("answer", ""))
            with st.expander("课程资料来源"):
                render_sources(data.get("source_chunk_ids", []))
            with st.expander("知识图谱关系"):
                render_kg_relations(data.get("related_triples", []))
        except Exception as exc:
            show_error(exc)


if active_tab == "学习路径":
    st.subheader("学习路径推荐")
    st.caption("这里专注于“按什么顺序学”：展示先修路径、路径关系和学完后的下一步探索。")
    target_entity = st.text_input("目标知识点", value="Word2Vec")

    if st.button("生成学习路径", type="primary"):
        try:
            data = post_json("/api/learning/path", {"target_entity": target_entity})
            st.markdown("### 推荐路径")
            render_learning_path(data)
        except Exception as exc:
            show_error(exc)


if active_tab == "学习报告":
    st.subheader("学习报告与知识覆盖度")
    st.write("系统会统计你在智能问答、自动出题、概念对比和学习路径中使用过的知识点。")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("查看覆盖度"):
            try:
                data = get_json("/api/analytics/coverage")
                render_coverage(data)
            except Exception as exc:
                show_error(exc)

    with col2:
        if st.button("生成学习报告", type="primary"):
            try:
                data = get_json("/api/analytics/report")
                st.metric("知识覆盖率", f"{data.get('coverage_rate', 0):.1%}")
                st.write(data.get("summary", ""))
                st.markdown("**已覆盖知识点**")
                render_concept_list(data.get("mastered_concepts", []), "还没有记录到已覆盖知识点。")
                st.markdown("**薄弱/未覆盖知识点**")
                render_concept_list(data.get("weak_concepts", []), "暂无薄弱/未覆盖知识点。")
                st.markdown("**下一步建议**")
                steps = data.get("recommended_next_steps", [])
                if steps:
                    for step in steps:
                        st.write("-", step)
                else:
                    st.info("暂无下一步建议。")

                st.markdown("**错题本记录**")
                mistakes = get_json("/api/analytics/mistakes")
                records = mistakes.get("records", [])
                if records:
                    rows = [
                        {
                            "序号": item.get("id", index + 1),
                            "知识点": item.get("concept", ""),
                            "得分": item.get("score", 0),
                            "题目": item.get("question", ""),
                            "建议复习": "、".join(item.get("suggested_review", [])),
                        }
                        for index, item in enumerate(records)
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("暂无错题记录。答题得分较低时，系统会自动加入错题本。")
            except Exception as exc:
                show_error(exc)
