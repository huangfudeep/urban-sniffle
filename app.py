import streamlit as st
from db_utils import (
    init_db, get_all_qa, add_qa, delete_qa,
    search_similar, generate_answer, record_unanswered,
    get_unanswered, resolve_unanswered, ignore_unanswered,
    load_json_to_kb
)
import os
from dotenv import load_dotenv
if not os.path.exists("data"):
    os.mkdir("data")
if not os.path.exists("data/chroma"):
    os.mkdir("data/chroma")


load_dotenv()
init_db()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

if "role" not in st.session_state:
    st.session_state.role = None

# ========== 身份选择 ==========
if st.session_state.role is None:
    st.title("知识库AI助手")
    st.subheader("请选择登录身份")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎓 学生端（仅提问）", use_container_width=True):
            st.session_state.role = "student"
            st.rerun()
    with col2:
        if st.button("🔐 管理员端", use_container_width=True):
            st.session_state.role = "admin_login"
            st.rerun()

# ========== 管理员登录 ==========
elif st.session_state.role == "admin_login":
    st.title("管理员登录")
    pwd = st.text_input("输入管理员密码", type="password")
    if st.button("登录"):
        if pwd == ADMIN_PASSWORD:
            st.session_state.role = "admin"
            st.rerun()
        else:
            st.error("密码错误！")
    if st.button("返回"):
        st.session_state.role = None
        st.rerun()

# ========== 学生端 ==========
elif st.session_state.role == "student":
    st.title("🎓 学生问答端")
    question = st.text_input("在这里输入你的问题：")
    if st.button("提交提问") and question.strip():
        similar_list = search_similar(question)
        if similar_list:
            # 命中知识库，返回库里答案
            ans = similar_list[0]["answer"]
            st.markdown("### 回答")
            st.markdown(ans)
        else:
            # 无匹配，调用大模型，加上提示文案，存入待处理
            raw_ans = generate_answer(question, [], allow_free=True)
            record_unanswered(question)
            st.markdown("### 回答")
            st.markdown(raw_ans)
            st.info("此为网络检索内容，具体行为请微信询问/等明日管理员更新答案再来")

    st.sidebar.info("学生端：仅可提问查看答案")
    if st.sidebar.button("退出"):
        st.session_state.role = None
        st.rerun()

# ========== 管理员后台 ==========
elif st.session_state.role == "admin":
    st.title("🔐 管理员后台")
    tab1, tab2, tab3 = st.tabs(["问答测试", "待回答问题处理", "知识库管理"])

    with tab1:
        st.subheader("问答测试")
        q = st.text_input("测试提问")
        if st.button("提问测试") and q.strip():
            res = search_similar(q)
            if res:
                st.info("✅ 该问题已存在于知识库中，不展示答案")
            else:
                st.warning("⚠️ 知识库无匹配，将调用大模型临时回答")
                temp_ans = generate_answer(q, [], allow_free=True)
                st.markdown(temp_ans)
                st.info("此为网络检索内容，具体行为请微信询问/等明日管理员更新答案再来")

    with tab2:
        st.subheader("学生提问、尚未入库的问题列表")
        pending_list = get_unanswered(status="pending")
        if not pending_list:
            st.success("暂无待处理问题")
        else:
            for item in pending_list:
                with st.expander(f"【待处理】{item['question']}"):
                    st.write(f"提问时间：{item['created_at']}")
                    new_ans = st.text_area("填写答案，提交后加入知识库", key=f"ans_{item['id']}")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✅ 保存答案入库", key=f"save_{item['id']}") and new_ans.strip():
                            resolve_unanswered(item["id"], new_ans)
                            st.success("已保存到知识库！")
                            st.rerun()
                    with c2:
                        if st.button("❌ 忽略该问题", key=f"ignore_{item['id']}"):
                            ignore_unanswered(item["id"])
                            st.info("已忽略，不再展示")
                            st.rerun()

    with tab3:
        st.subheader("新增知识库条目")
        q_add = st.text_input("问题")
        a_add = st.text_area("答案")
        if st.button("添加到知识库") and q_add and a_add:
            add_qa(q_add, a_add)
            st.success("添加成功")
            st.rerun()

        st.divider()
        # ========== 一键导入sample按钮 ==========
        if st.button("📥 一键导入 sample_qa.json"):
            try:
                cnt = load_json_to_kb("sample_qa.json")
                st.success(f"导入完成！新增 {cnt} 条问答，已存在的自动跳过")
                st.rerun()
            except Exception as e:
                st.error(f"导入失败：{str(e)}")
        # =======================================

        st.subheader("现有知识库")
        all_kb = get_all_qa()
        for kb_item in all_kb:
            with st.expander(f"{kb_item['id']}. {kb_item['question']}"):
                st.write(kb_item["answer"])
                if st.button(f"删除 {kb_item['id']}", key=f"delkb_{kb_item['id']}"):
                    delete_qa(kb_item["id"])
                    st.rerun()

    st.sidebar.info("管理员端：处理待回答问题、维护知识库、批量导入题库")
    if st.sidebar.button("退出登录"):
        st.session_state.role = None
        st.rerun()

