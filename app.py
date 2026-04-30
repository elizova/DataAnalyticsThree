import pandas as pd
import streamlit as st

from agent.client import get_client, init_client
from agent.runner import run_agent

if "client_initialized" not in st.session_state:
    st.session_state.client_initialized = False

if not st.session_state.client_initialized:
    init_client(st.secrets["OPENROUTER_API_KEY"])
    st.session_state.client_initialized = True
else:
    try:
        get_client()
    except RuntimeError:
        init_client(st.secrets["OPENROUTER_API_KEY"])

st.set_page_config(page_title="AI Аналитик", layout="wide")
st.title("AI аналитик данных")

if "files_data" not in st.session_state:
    st.session_state.files_data = {}
if "current_file" not in st.session_state:
    st.session_state.current_file = None

uploaded_file = st.file_uploader(
    "Загрузите CSV или Excel", type=["csv", "xlsx"]
)
if uploaded_file is not None:
    file_name = uploaded_file.name
    if file_name not in st.session_state.files_data:
        try:
            if file_name.endswith(".csv"):
                encodings = [
                    "utf-8",
                    "cp1251",
                    "windows-1251",
                    "latin1",
                    "iso-8859-1",
                ]
                df = None
                for enc in encodings:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=enc)
                        break
                    except UnicodeDecodeError:
                        continue
                if df is None:
                    st.error("Неправильная кодировка")
                    st.stop()
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Ошибка: {e}")
            st.stop()
        st.session_state.files_data[file_name] = {
            "df": df,
            "report": None,
            "images": [],
            "debug": "",
            "analysis_done": False,
        }
        st.session_state.current_file = file_name

if st.session_state.files_data:
    file_names = list(st.session_state.files_data.keys())
    if st.session_state.current_file not in file_names:
        st.session_state.current_file = file_names[0]
    selected_file = st.selectbox(
        "Выберите файл",
        file_names,
        index=file_names.index(st.session_state.current_file),
        key="file_selector",
    )

    st.session_state.current_file = selected_file
else:
    st.info("Загрузите файл")
    st.stop()

current = st.session_state.files_data[st.session_state.current_file]
df = current["df"]

st.dataframe(df.head())

user_instruction = st.text_area(
    "Дополнительные инструкции (необязательно)", key="instr"
)

if st.button("Запустить анализ", key="run"):
    if df.empty:
        st.error("Датасет пуст")
    else:
        with st.spinner("Агент проводит анализ.."):
            try:
                report, images, debug = run_agent(df, user_instruction)
                current["report"] = report
                current["images"] = images
                current["debug"] = debug
                current["analysis_done"] = True
            except Exception as e:
                st.error(f"Ошибка: {e}")
                current["analysis_done"] = False

if current.get("analysis_done"):
    st.subheader("Отчёт")
    st.markdown(current["report"])
    if current["images"]:
        st.subheader("Графики")
        for idx, img in enumerate(current["images"]):
            st.image(
                f"data:image/png;base64,{img}",
                caption=f"График {idx+1}",
                use_container_width=True,
            )
    with st.expander("Отладка"):
        st.text(current.get("debug", "Нет данных"))
