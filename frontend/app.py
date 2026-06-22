from collections import defaultdict
from html import escape
import math
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components


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


def set_learning_path_target(concept: str):
    st.session_state["learning_path_target_input"] = concept
    st.session_state["learning_path_message"] = f"已选择「{concept}」，学习路径目标知识点已更新。"


def recommendation_source_text(source: str) -> str:
    mapping = {
        "mistake_book": "错题本",
        "knowledge_graph": "知识图谱关联",
        "question_history": "问答历史",
        "coverage": "覆盖度补充",
        "history": "学习记录",
    }
    return mapping.get(source, source or "未知来源")


RELATION_COLORS = {
    "先修知识": "#F59E0B",
    "相关概念": "#34D399",
    "包含": "#A78BFA",
    "应用场景": "#60A5FA",
    "属于": "#F472B6",
    "容易混淆": "#F87171",
}


QUESTION_TYPE_LABELS = {
    "choice": "单选题",
    "multiple_choice": "多选题",
    "true_false": "判断题",
    "fill": "填空题",
    "short_answer": "简答题",
    "compare": "概念对比题",
    "application": "应用分析题",
}

QUESTION_TYPE_VALUES = {label: value for value, label in QUESTION_TYPE_LABELS.items()}


def relation_color(relation: str) -> str:
    return RELATION_COLORS.get(relation, "#93C5FD")


def render_relation_chip(concept: str, relation: str):
    color = relation_color(relation)
    href = f"?qa_concept={quote(str(concept))}"
    st.markdown(
        f"""
        <a href="{href}" target="_self" style="
            display:block;
            width:100%;
            box-sizing:border-box;
            margin:0 0 10px 0;
            padding:9px 13px;
            border-radius:999px;
            border:1.5px solid {color};
            color:{color};
            background:rgba(255,255,255,0.02);
            text-align:center;
            text-decoration:none;
            font-size:15px;
            line-height:1.25;
            box-shadow:0 1px 2px rgba(15, 23, 42, 0.08);
        ">› {escape(str(concept))}</a>
        """,
        unsafe_allow_html=True,
    )


def clear_quiz_runtime_state():
    prefixes = (
        "quiz_submit_result_",
        "quiz_eval_result_",
        "student_answer_",
    )
    for key in list(st.session_state.keys()):
        if key.startswith(prefixes):
            del st.session_state[key]
    st.session_state.pop("quiz_optimize_message", None)


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


def render_dark_relation_graph(triples: list[dict], height: int = 520, focus_nodes: set[str] | None = None):
    if not triples:
        st.info("暂无可展示的知识图谱关系。")
        return

    focus_nodes = focus_nodes or set()

    nodes = sorted(
        {
            node
            for triple in triples
            for node in [triple.get("head", ""), triple.get("tail", "")]
            if node
        }
    )
    node_relation_colors = {}
    for triple in triples:
        color = relation_color(str(triple.get("relation", "")))
        for node in [triple.get("head", ""), triple.get("tail", "")]:
            if node and node not in focus_nodes and node not in node_relation_colors:
                node_relation_colors[node] = color

    def node_color(node: str) -> dict:
        if node in focus_nodes:
            return {"background": "#1E3A8A", "border": "#93C5FD"}
        color = node_relation_colors.get(node, "#64748B")
        return {"background": color, "border": color}

    try:
        from pyvis.network import Network

        net = Network(
            height=f"{height}px",
            width="100%",
            directed=True,
            bgcolor="#0B1220",
            font_color="#F8FAFC",
            cdn_resources="in_line",
        )
        net.barnes_hut(
            gravity=-22000,
            central_gravity=0.2,
            spring_length=145,
            spring_strength=0.045,
            damping=0.88,
        )

        for node in nodes:
            net.add_node(
                node,
                label=node,
                title=node,
                color=node_color(node),
                size=27 if node in focus_nodes else 22,
                borderWidth=2,
            )

        for triple in triples:
            head = triple.get("head", "")
            tail = triple.get("tail", "")
            relation = triple.get("relation", "")
            if head and tail:
                color = relation_color(relation)
                net.add_edge(head, tail, label=relation, title=relation, color=color, arrows="to")

        net.set_options(
            """
            {
              "nodes": {
                "shape": "dot",
                "font": {"size": 18, "face": "Microsoft YaHei", "color": "#F8FAFC", "strokeWidth": 3, "strokeColor": "#0B1220"}
              },
              "edges": {
                "font": {"size": 12, "align": "middle", "color": "#DBEAFE", "strokeWidth": 3, "strokeColor": "#0B1220"},
                "smooth": {"type": "dynamic"}
              },
              "interaction": {
                "hover": true,
                "dragNodes": true,
                "zoomView": true
              },
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 160}
              }
            }
            """
        )
        components.html(net.generate_html(), height=height + 30, scrolling=False)
        return
    except Exception:
        pass

    graph_lines = ["digraph {"]
    graph_lines.append('  graph [rankdir="LR", size="7.5,4.2", ratio="compress", margin=0.04, bgcolor="#0B1220"];')
    graph_lines.append('  node [shape=box, style="rounded,filled", fontsize=10, margin=0.08, fontcolor="#F8FAFC", penwidth=1.5];')
    graph_lines.append('  edge [fontsize=9, arrowsize=0.65, color="#93C5FD", fontcolor="#DBEAFE"];')
    for node in nodes:
        safe_node = str(node).replace('"', '\\"')
        colors = node_color(node)
        graph_lines.append(
            f'  "{safe_node}" [fillcolor="{colors["background"]}", color="{colors["border"]}"];'
        )
    for triple in triples:
        head = str(triple.get("head", "")).replace('"', '\\"')
        relation = str(triple.get("relation", "")).replace('"', '\\"')
        tail = str(triple.get("tail", "")).replace('"', '\\"')
        color = relation_color(str(triple.get("relation", "")))
        if head and tail:
            graph_lines.append(f'  "{head}" -> "{tail}" [label="{relation}", color="{color}", fontcolor="#DBEAFE"];')
    graph_lines.append("}")
    with st.container(height=height):
        st.graphviz_chart("\n".join(graph_lines), use_container_width=True)


def render_kg_relations(triples: list[dict], clickable: bool = True):
    if not triples:
        st.info("未检索到相关知识图谱关系")
        return

    st.markdown("**关系摘要**")
    grouped = defaultdict(list)
    for triple in triples:
        grouped[(triple["head"], triple["relation"])].append(triple["tail"])
    heads = {triple.get("head", "") for triple in triples}
    tails = {triple.get("tail", "") for triple in triples}
    focus_nodes = {node for node in heads & tails if node}
    if not focus_nodes and grouped:
        focus_nodes = {next(iter(grouped.keys()))[0]}

    for group_index, ((head, relation), tails) in enumerate(grouped.items()):
        unique_tails = list(dict.fromkeys(tails))
        st.markdown(f"**{head}** 的 **{relation}**：")
        if clickable:
            for row_start in range(0, len(unique_tails), 4):
                cols = st.columns(4, gap="small")
                for col_offset, tail in enumerate(unique_tails[row_start:row_start + 4]):
                    index = row_start + col_offset
                    with cols[col_offset]:
                        render_relation_chip(tail, relation)
        else:
            st.write("、".join(unique_tails))

    st.markdown("**图谱可视化**")
    render_dark_relation_graph(triples, height=520, focus_nodes=focus_nodes)

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


def render_chip_cloud(
    title: str,
    concepts: list[str],
    color: str,
    border: str,
    text_color: str,
    empty_text: str,
):
    st.markdown(f"**{title}**")
    unique_concepts = list(dict.fromkeys([str(concept) for concept in concepts if concept]))
    if not unique_concepts:
        st.info(empty_text)
        return

    chips = []
    for concept in unique_concepts:
        href = f"?qa_concept={quote(concept)}"
        chips.append(
            f"""
            <a href="{href}" target="_self" style="
                display:inline-block;
                margin:0 10px 12px 0;
                padding:8px 14px;
                border-radius:999px;
                border:1.5px solid {border};
                background:{color};
                color:{text_color};
                font-size:15px;
                line-height:1.2;
                text-decoration:none;
                white-space:nowrap;
                box-shadow:0 1px 2px rgba(15, 23, 42, 0.08);
            ">› {escape(concept)}</a>
            """
        )
    st.markdown("".join(chips), unsafe_allow_html=True)


def render_coverage_tag_cloud(asked: list[str], weak: list[str], recommended: list[str]):
    st.markdown("**知识点标签云**")
    st.caption("绿色表示已经触达，红色表示需要复习，黄色表示建议下一步补充。")
    col1, col2, col3 = st.columns(3)
    with col1:
        render_chip_cloud("已覆盖", asked, "#DCFCE7", "#22C55E", "#166534", "暂无已覆盖知识点。")
    with col2:
        render_chip_cloud("薄弱", weak, "#FEE2E2", "#EF4444", "#991B1B", "暂无薄弱知识点。")
    with col3:
        render_chip_cloud("推荐补充", recommended, "#FEF3C7", "#F59E0B", "#92400E", "暂无推荐补充知识点。")


def render_coverage(data: dict):
    asked = data.get("asked_entities", [])
    missing = data.get("missing_entities", [])
    weak = data.get("weak_entities", [])
    recommended = data.get("recommended_entities", [])
    coverage_rate = data.get("coverage_rate", 0)

    col1, col2, col3 = st.columns(3)
    col1.metric("知识覆盖率", f"{coverage_rate:.1%}")
    col2.metric("已覆盖知识点", len(asked))
    col3.metric("未覆盖知识点", len(missing))

    st.markdown("**知识图谱覆盖高亮**")
    render_coverage_graph(asked, missing, weak, recommended)

    render_coverage_tag_cloud(asked, weak, recommended)


def render_coverage_graph(asked: list[str], missing: list[str], weak: list[str], recommended: list[str]):
    try:
        triples_data = get_json("/api/kg/triples")
    except Exception as exc:
        show_error(exc)
        return

    triples = triples_data.get("triples", [])
    if not triples:
        st.info("暂无可展示的知识图谱关系。")
        return

    asked_set = set(asked)
    weak_set = set(weak)
    recommended_set = set(recommended)
    missing_set = set(missing)
    focus_nodes = set(asked) | set(weak) | set(recommended)

    if focus_nodes:
        visible_triples = [
            triple
            for triple in triples
            if triple.get("head") in focus_nodes or triple.get("tail") in focus_nodes
        ]
    else:
        visible_triples = triples[:24]

    visible_triples = visible_triples[:60]
    nodes = set()
    for triple in visible_triples:
        nodes.add(triple.get("head", ""))
        nodes.add(triple.get("tail", ""))

    def node_style(name: str) -> tuple[str, str]:
        if name in weak_set:
            return "#991B1B", "#FCA5A5"
        if name in asked_set:
            return "#166534", "#BBF7D0"
        if name in recommended_set:
            return "#92400E", "#FCD34D"
        if name in missing_set:
            return "#475569", "#94A3B8"
        return "#334155", "#CBD5E1"

    st.caption("绿色=已学/已触达，红色=薄弱，黄色=建议优先补充，灰色=未覆盖。")

    try:
        from pyvis.network import Network

        net = Network(
            height="560px",
            width="100%",
            directed=True,
            bgcolor="#0B1220",
            font_color="#F8FAFC",
            cdn_resources="in_line",
        )
        net.barnes_hut(
            gravity=-28000,
            central_gravity=0.18,
            spring_length=150,
            spring_strength=0.04,
            damping=0.88,
        )

        for node in sorted(n for n in nodes if n):
            fill, border = node_style(node)
            if node in weak_set:
                status = "薄弱知识点"
                size = 28
            elif node in recommended_set:
                status = "建议优先补充"
                size = 25
            elif node in asked_set:
                status = "已覆盖"
                size = 23
            else:
                status = "未覆盖/关联节点"
                size = 18
            net.add_node(
                node,
                label=node,
                title=f"{node}<br>{status}",
                color={"background": fill, "border": border},
                size=size,
                borderWidth=2,
            )

        for triple in visible_triples:
            head = triple.get("head", "")
            tail = triple.get("tail", "")
            relation = triple.get("relation", "")
            if head and tail:
                net.add_edge(head, tail, label=relation, title=relation, color="#93C5FD", arrows="to")

        net.set_options(
            """
            {
              "nodes": {
                "shape": "dot",
                "font": {"size": 18, "face": "Microsoft YaHei", "color": "#F8FAFC", "strokeWidth": 3, "strokeColor": "#0B1220"}
              },
              "edges": {
                "font": {"size": 12, "align": "middle", "color": "#DBEAFE", "strokeWidth": 3, "strokeColor": "#0B1220"},
                "smooth": {"type": "dynamic"}
              },
              "interaction": {
                "hover": true,
                "dragNodes": true,
                "zoomView": true
              },
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 180}
              }
            }
            """
        )
        components.html(net.generate_html(), height=590, scrolling=False)
        return
    except Exception:
        st.caption("当前环境未能加载 pyvis 交互图，已自动切换为分层静态图。")

    graph_lines = ["digraph {"]
    graph_lines.append('  graph [rankdir="LR", size="9,5", ratio="compress", margin=0.04, bgcolor="#0B1220"];')
    graph_lines.append('  node [shape=box, style="rounded,filled", fontsize=10, margin=0.08, fontcolor="#F8FAFC"];')
    graph_lines.append('  edge [fontsize=8, arrowsize=0.55, color="#93C5FD", fontcolor="#DBEAFE"];')

    grouped_nodes = {
        "weak": [node for node in nodes if node in weak_set],
        "asked": [node for node in nodes if node in asked_set and node not in weak_set],
        "recommended": [node for node in nodes if node in recommended_set and node not in weak_set],
        "missing": [
            node
            for node in nodes
            if node not in weak_set and node not in asked_set and node not in recommended_set
        ],
    }

    for node in sorted(n for n in nodes if n):
        fill, border = node_style(node)
        safe_node = str(node).replace('"', '\\"')
        graph_lines.append(f'  "{safe_node}" [fillcolor="{fill}", color="{border}", penwidth=1.5];')

    for group_name, group_nodes in grouped_nodes.items():
        if group_nodes:
            same_rank = "; ".join(f'"{str(node).replace(chr(34), chr(92) + chr(34))}"' for node in group_nodes)
            graph_lines.append(f"  {{ rank=same; {same_rank}; }}")

    for triple in visible_triples:
        head = str(triple.get("head", "")).replace('"', '\\"')
        tail = str(triple.get("tail", "")).replace('"', '\\"')
        relation = str(triple.get("relation", "")).replace('"', '\\"')
        if head and tail:
            graph_lines.append(f'  "{head}" -> "{tail}" [label="{relation}"];')

    graph_lines.append("}")
    with st.container(height=560):
        st.graphviz_chart("\n".join(graph_lines), use_container_width=True)


def render_radar_chart(items: list[tuple[str, float]]):
    if not items:
        st.info("暂无足够学习记录生成能力画像。")
        return

    size = 470
    center = size / 2
    radius = 125
    red = "#FF4B4B"
    grid = "#E5E7EB"
    fill = "#D1D5DB"
    text = "#2F3440"

    angles = [
        -math.pi / 2 + 2 * math.pi * index / len(items)
        for index in range(len(items))
    ]

    def point(angle: float, value: float):
        return (
            center + math.cos(angle) * radius * value,
            center + math.sin(angle) * radius * value,
        )

    grid_parts = []
    for level in [0.25, 0.5, 0.75, 1.0]:
        pts = [point(angle, level) for angle in angles]
        points_attr = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        grid_parts.append(
            f'<polygon points="{points_attr}" fill="none" stroke="{grid}" stroke-width="1" />'
        )

    axis_parts = []
    label_parts = []
    for (label, score), angle in zip(items, angles):
        x, y = point(angle, 1.0)
        axis_parts.append(
            f'<line x1="{center}" y1="{center}" x2="{x:.1f}" y2="{y:.1f}" stroke="{grid}" stroke-width="1" />'
        )
        lx, ly = point(angle, 1.32)
        anchor = "middle"
        if lx < center - 24:
            anchor = "end"
        elif lx > center + 24:
            anchor = "start"
        label_parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" dominant-baseline="middle" fill="{text}" font-size="15" font-weight="600">{escape(label)}</text>'
        )

    values = [max(0.0, min(1.0, score)) for _, score in items]
    value_points = [point(angle, value) for angle, value in zip(angles, values)]
    polygon_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in value_points)
    dots = "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="#FFFFFF" stroke="{red}" stroke-width="2.5" />'
        for x, y in value_points
    )

    score_labels = []
    for (label, score), angle in zip(items, angles):
        lx, ly = point(angle, 0.78)
        score_labels.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" dominant-baseline="middle" fill="#6B7280" font-size="12">{score:.0%}</text>'
        )

    svg = f"""
    <div style="display:flex;justify-content:center;">
      <svg viewBox="0 0 {size} {size}" width="100%" style="max-width:450px;">
        {''.join(grid_parts)}
        {''.join(axis_parts)}
        <polygon points="{polygon_points}" fill="{fill}" opacity="0.45" stroke="{red}" stroke-width="3" />
        <polyline points="{polygon_points} {polygon_points.split()[0]}" fill="none" stroke="{red}" stroke-width="3" />
        {dots}
        {''.join(score_labels)}
        {''.join(label_parts)}
      </svg>
    </div>
    """
    components.html(svg, height=430, scrolling=False)


def render_reason_pie_chart(reason_counts: dict[str, int]):
    if not reason_counts:
        st.success("暂未发现明显薄弱知识点。")
        return

    sorted_items = sorted(reason_counts.items(), key=lambda item: item[1], reverse=True)
    total = sum(value for _, value in sorted_items)
    if total <= 0:
        st.success("暂未发现明显薄弱知识点。")
        return

    size = 320
    center = size / 2
    radius = 104
    colors = ["#FF4B4B", "#F87171", "#FCA5A5", "#D1D5DB", "#E5E7EB"]
    start_angle = -math.pi / 2
    paths = []
    legend = []

    for index, (label, value) in enumerate(sorted_items):
        angle = 2 * math.pi * value / total
        end_angle = start_angle + angle
        x1 = center + math.cos(start_angle) * radius
        y1 = center + math.sin(start_angle) * radius
        x2 = center + math.cos(end_angle) * radius
        y2 = center + math.sin(end_angle) * radius
        large_arc = 1 if angle > math.pi else 0
        color = colors[min(index, len(colors) - 1)]
        paths.append(
            f'<path d="M {center} {center} L {x1:.1f} {y1:.1f} A {radius} {radius} 0 {large_arc} 1 {x2:.1f} {y2:.1f} Z" fill="{color}" />'
        )
        percent = value / total
        legend.append(
            f"""
            <div style="display:flex;align-items:center;gap:7px;margin:6px 0;white-space:nowrap;font-size:13px;">
              <span style="width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;flex:0 0 auto;"></span>
              <span style="font-weight:600;color:#2F3440;">{escape(label)}</span>
              <span style="color:#6B7280;">{value} 项 · {percent:.0%}</span>
            </div>
            """
        )
        start_angle = end_angle

    svg = f"""
    <div style="display:grid;grid-template-columns:minmax(210px, 300px) 1fr;gap:16px;align-items:center;">
      <svg viewBox="0 0 {size} {size}" width="100%" style="max-width:300px;">
        {''.join(paths)}
        <circle cx="{center}" cy="{center}" r="62" fill="#FFFFFF" />
        <text x="{center}" y="{center - 4}" text-anchor="middle" fill="#2F3440" font-size="24" font-weight="700">{total}</text>
        <text x="{center}" y="{center + 22}" text-anchor="middle" fill="#6B7280" font-size="13">薄弱项</text>
      </svg>
      <div>{''.join(legend)}</div>
    </div>
    """
    components.html(svg, height=340, scrolling=False)


def ability_profile_text(ability_scores: list[tuple[str, float, str]]) -> str:
    if not ability_scores:
        return "当前学习记录还不够充分，暂时无法形成稳定画像。"

    sorted_scores = sorted(ability_scores, key=lambda item: item[1])
    weakest = sorted_scores[0]
    strongest = sorted_scores[-1]
    return (
        f"当前学习者在「{strongest[0]}」上表现相对更好，说明这一维度已有一定基础；"
        f"「{weakest[0]}」仍是短板，建议后续围绕这一维度补充学习记录和练习反馈。"
    )


def weakness_reason_text(reason_counts: dict[str, int]) -> str:
    if not reason_counts:
        return "当前暂未形成明显薄弱原因，建议继续通过问答和自动出题积累学习记录。"

    label, count = max(reason_counts.items(), key=lambda item: item[1])
    total = sum(reason_counts.values())
    percent = count / total if total else 0
    suggestions = {
        "错题关联": "主要薄弱来自答题反馈，建议优先回看错题解析，再围绕相关概念做针对练习。",
        "未覆盖": "主要薄弱来自课程地图空白，建议选择未覆盖知识点生成学习路径，先扩展知识面。",
        "前序缺口": "主要薄弱来自前置知识不足，建议先补基础概念，再学习目标概念。",
        "关联扩展": "主要薄弱来自相关概念扩展不足，建议通过概念对比和追问补齐关系理解。",
    }
    return f"薄弱原因中「{label}」占比最高（{percent:.0%}）。{suggestions.get(label, '建议结合知识图谱关系继续复习。')}"


def render_learning_report(data: dict):
    st.caption("学习报告更像个人诊断：它结合覆盖记录、错题本和知识图谱关系，判断“学得怎么样”和“下一步怎么补”。")

    col1, col2, col3, col4 = st.columns(4)
    profile = data.get("activity_profile", {})
    col1.metric("知识覆盖率", f"{data.get('coverage_rate', 0):.1%}")
    col2.metric("已触达", profile.get("已触达知识点", data.get("asked_count", 0)))
    col3.metric("错题记录", profile.get("错题记录", 0))
    col4.metric("错题关联概念", profile.get("错题关联概念", 0))

    st.markdown("**诊断摘要**")
    st.info(data.get("summary", "暂无诊断摘要。"))

    touched_count = profile.get("已触达知识点", data.get("asked_count", 0))
    weak_count = len(data.get("weak_concepts", []))
    mistake_count = profile.get("错题记录", 0)
    mistake_related_count = profile.get("错题关联概念", 0)
    coverage_score = min(1.0, max(0.0, float(data.get("coverage_rate", 0))))
    stability_score = 1.0 if mistake_count == 0 else max(0.05, 1 - mistake_count / max(touched_count + mistake_count, 1))
    weak_control_score = 1.0 if weak_count == 0 else max(0.05, 1 - weak_count / max(data.get("total_count", weak_count), 1))
    continuity_score = min(1.0, touched_count / max(data.get("total_count", touched_count), 1))

    ability_scores = [
        ("知识覆盖", coverage_score, "已触达课程知识图谱中的知识点比例"),
        ("答题稳定", stability_score, "错题越少，答题稳定性越高"),
        ("薄弱控制", weak_control_score, "薄弱/未覆盖概念越少，控制度越高"),
        ("学习连续", continuity_score, "学习记录覆盖越多，连续性越高"),
    ]
    weak_reasons = data.get("weak_concept_reasons", [])

    ability_col, reason_col = st.columns(2)
    with ability_col:
        st.markdown("**能力画像**")
        render_radar_chart([(name, score) for name, score, _ in ability_scores])
        st.info(ability_profile_text(ability_scores))
        st.caption("学习连续表示学习记录在知识图谱中的覆盖连续性：触达的知识点越多、越能形成稳定学习轨迹，分数越高。")

    with reason_col:
        st.markdown("**薄弱原因分布**")
        if weak_reasons:
            reason_counts = defaultdict(int)
            for item in weak_reasons:
                reason = item.get("原因", "")
                suggestion = item.get("建议", "")
                if "错题" in reason or "得分" in reason:
                    reason_counts["错题关联"] += 1
                elif "尚未覆盖" in reason:
                    reason_counts["未覆盖"] += 1
                elif "先复习" in suggestion or "学习前" in suggestion:
                    reason_counts["前序缺口"] += 1
                else:
                    reason_counts["关联扩展"] += 1

            render_reason_pie_chart(dict(reason_counts))
            st.info(weakness_reason_text(dict(reason_counts)))
            st.caption("薄弱原因按错题关联、未覆盖、前序缺口和关联扩展进行归因统计。")
        else:
            st.success("暂未发现明显薄弱知识点。")

    st.markdown("**推荐依据说明**")
    st.write(data.get("recommendation_basis", "推荐会综合错题、学习历史、知识图谱关系和未覆盖知识点。"))

    st.markdown("**下一步学习计划**")
    prereq_items = []
    weak_practice_items = []
    expansion_items = []
    for item in weak_reasons:
        concept = item.get("知识点", "")
        reason = item.get("原因", "")
        suggestion = item.get("建议", "")
        if not concept:
            continue
        if "错题" in reason or "得分" in reason:
            weak_practice_items.append(concept)
        elif "尚未覆盖" in reason:
            expansion_items.append(concept)
        elif "先复习" in suggestion or "学习前" in suggestion:
            prereq_items.append(concept)

    fallback_steps = data.get("recommended_next_steps", [])
    prereq_text = (
        f"优先回看：{'、'.join(list(dict.fromkeys(prereq_items))[:3])}。先补前序关系，再进入题目练习。"
        if prereq_items
        else (fallback_steps[0] if fallback_steps else "优先回看错题相关概念的前序知识，先把基础链条补齐。")
    )
    practice_text = (
        f"围绕：{'、'.join(list(dict.fromkeys(weak_practice_items))[:3])} 生成 2-3 道中等难度题，重点看反馈和解析。"
        if weak_practice_items
        else "当前错题型薄弱点不明显；可以选择最近答错或不确定的概念做一次针对性自测。"
    )
    expansion_text = (
        f"选择未覆盖概念：{'、'.join(list(dict.fromkeys(expansion_items))[:3])}，生成学习路径继续扩展课程地图。"
        if expansion_items
        else "当前未覆盖推荐较少；可以从知识覆盖度页面选择黄色推荐补充节点继续拓展。"
    )

    cards = [
        ("1. 先补基础", prereq_text, "#EFF6FF", "#2563EB"),
        ("2. 再练薄弱", practice_text, "#FEF2F2", "#DC2626"),
        ("3. 最后拓展", expansion_text, "#FFFBEB", "#D97706"),
    ]
    cols = st.columns(3)
    for col, (title, content, bg, color) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div style="
                    min-height:170px;
                    padding:16px 18px;
                    border-radius:10px;
                    border:1px solid {color};
                    background:{bg};
                ">
                    <div style="font-weight:700;color:{color};font-size:17px;margin-bottom:10px;">{escape(title)}</div>
                    <div style="color:#1f2937;font-size:15px;line-height:1.55;">{escape(content)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("查看已覆盖知识点明细"):
        render_concept_list(data.get("mastered_concepts", []), "还没有记录到已覆盖知识点。")


st.title("NLP 课程 Graph-RAG 智能教学助手")
st.caption("融合课程资料检索、知识图谱、大模型问答、自动出题与学习分析")

with st.sidebar:
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


tab_labels = ["智能问答", "自动出题与答题", "概念对比", "学习路径", "知识覆盖度", "学习诊断报告"]
qa_concept = st.query_params.get("qa_concept")
if qa_concept:
    st.session_state["qa_question_input"] = f"请解释一下「{qa_concept}」这个知识点。"
    st.session_state["selected_concept_card"] = qa_concept
    st.session_state["active_tab"] = "智能问答"
    st.query_params.clear()

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
                        "来源": recommendation_source_text(item.get("source", "")),
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
                        "来源": recommendation_source_text(item.get("source", "")),
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

    question_type_labels = st.multiselect(
        "题型",
        list(QUESTION_TYPE_VALUES.keys()),
        default=["单选题", "填空题", "简答题"],
    )
    question_types = [QUESTION_TYPE_VALUES[label] for label in question_type_labels]
    st.caption("题型表示允许系统从这些类型中生成题目；题目数量决定总共生成几道题，并不是一个题型或一个词条固定对应一道题。")

    if st.button("生成题目", type="primary"):
        try:
            clear_quiz_runtime_state()
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
        if st.button("清空答题反馈和质量自检结果", key="clear_quiz_runtime_state"):
            clear_quiz_runtime_state()
            st.rerun()
        optimize_message = st.session_state.pop("quiz_optimize_message", None)
        if optimize_message:
            st.success(optimize_message)
        questions = latest_quiz.get("questions", [])
        for index, item in enumerate(questions):
            question_id = item.get("id", index + 1)
            question_type = item.get("type", "short_answer")
            question_type_label = QUESTION_TYPE_LABELS.get(question_type, question_type)
            submit_key = f"quiz_submit_result_{question_id}_{index}"

            with st.expander(f"题目 {question_id}｜{question_type_label}", expanded=index == 0):
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
            f"题目 {item.get('id', idx + 1)}｜{QUESTION_TYPE_LABELS.get(item.get('type', 'short_answer'), item.get('type', 'short_answer'))}"
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
    if "learning_path_target_input" not in st.session_state:
        st.session_state["learning_path_target_input"] = "Word2Vec"

    learning_path_message = st.session_state.pop("learning_path_message", None)
    if learning_path_message:
        st.success(learning_path_message)

    target_entity = st.text_input("目标知识点", key="learning_path_target_input")

    with st.expander("基于学习历史推荐的路径目标", expanded=False):
        col_refresh, col_hint = st.columns([1, 4])
        with col_refresh:
            refresh_path_recommendations = st.button("刷新推荐", key="refresh_path_recommendations")
        with col_hint:
            st.caption("推荐会结合错题本、问答历史、知识图谱关联和知识覆盖度，帮助你选择下一条学习路径的目标。")

        if refresh_path_recommendations or "path_recommended_concepts" not in st.session_state:
            try:
                st.session_state["path_recommended_concepts"] = get_json("/api/analytics/recommendations?limit=8")
            except Exception as exc:
                show_error(exc)

        path_recommended = st.session_state.get("path_recommended_concepts", {}).get("concepts", [])
        if path_recommended:
            unique_concepts = list(dict.fromkeys([item.get("concept", "") for item in path_recommended if item.get("concept", "")]))
            for row_start in range(0, len(unique_concepts), 4):
                cols = st.columns(4, gap="small")
                for col_offset, recommended_concept in enumerate(unique_concepts[row_start:row_start + 4]):
                    index = row_start + col_offset
                    with cols[col_offset]:
                        st.button(
                            f"› {recommended_concept}",
                            key=f"path_recommend_{row_start}_{index}_{recommended_concept}",
                            help=f"点击后用「{recommended_concept}」作为学习路径目标",
                            use_container_width=True,
                            on_click=set_learning_path_target,
                            args=(str(recommended_concept),),
                        )
            if st.checkbox("查看推荐依据", key="show_path_recommendation_reasons"):
                rows = [
                    {
                        "知识点": item.get("concept", ""),
                        "推荐依据": item.get("reason", ""),
                        "来源": recommendation_source_text(item.get("source", "")),
                    }
                    for item in path_recommended
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("暂无推荐知识点。可以先进行问答、出题或答题后再刷新。")

    if st.button("生成学习路径", type="primary"):
        try:
            data = post_json("/api/learning/path", {"target_entity": target_entity})
            st.markdown("### 推荐路径")
            render_learning_path(data)
        except Exception as exc:
            show_error(exc)


if active_tab == "知识覆盖度":
    st.subheader("知识覆盖度")
    st.write("系统会统计你在智能问答、自动出题、概念对比和学习路径中使用过的知识点。")

    st.caption("这里更像课程地图：查看哪些知识点已经触达、哪些还没覆盖，以及图谱节点状态。")
    if st.button("刷新覆盖度", key="refresh_coverage"):
        try:
            st.session_state["coverage_data"] = get_json("/api/analytics/coverage")
        except Exception as exc:
            show_error(exc)

    coverage_data = st.session_state.get("coverage_data")
    if coverage_data:
        render_coverage(coverage_data)
    else:
        st.info("点击“刷新覆盖度”后查看课程知识图谱覆盖情况。")


if active_tab == "学习诊断报告":
    st.subheader("学习诊断报告")
    st.caption("这里更像个人诊断：结合覆盖记录、错题本和知识图谱关系，生成学习建议。")
    if st.button("生成/刷新学习报告", type="primary", key="refresh_learning_report"):
        try:
            st.session_state["learning_report_data"] = get_json("/api/analytics/report")
            st.session_state["mistakes_data"] = get_json("/api/analytics/mistakes")
        except Exception as exc:
            show_error(exc)

    report_data = st.session_state.get("learning_report_data")
    if report_data:
        render_learning_report(report_data)

        st.markdown("**错题本记录**")
        mistakes = st.session_state.get("mistakes_data", {})
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
    else:
        st.info("点击“生成/刷新学习报告”后查看个人学习诊断。")
