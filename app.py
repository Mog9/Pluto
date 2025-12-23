import streamlit as st
import pandas as pd
from models import train_models, get_regression_models, get_classification_models
from plots import plot_regression_scatter, plot_confusion_matrix, plot_model_comparison
import chardet
import io

st.set_page_config(page_title="Pluto", layout="wide", initial_sidebar_state="collapsed", page_icon="pluto.png")
st.markdown("""
    <style>
        [data-testid="stSidebar"], [data-testid="stSidebarNav"] {display: none;}
        body {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: white;
        }
        .intro-container {
            text-align: center;
            padding-top: 5%;
        }
        .title {
            font-size: 3em;
            font-weight: bold;
        }
        .subtitle {
            font-size: 1.2em;
            margin-top: -10px;
        }
        .features {
            margin-top: 30px;
            font-size: 1.1em;
        }
    </style>
""", unsafe_allow_html=True)


if "show_main" not in st.session_state:
    st.session_state.show_main = False


if not st.session_state.show_main:
    st.markdown("<div class='intro-container'>", unsafe_allow_html=True)
    st.image("pluto.png", width=190)
    st.markdown("<div class='title'>Pluto, Upload. Click. Train.</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Turn your CSV into trained ML models in minutes - right in your browser.</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='features'>
        Fast & Simple - No coding required<br>
        Multiple ML Algorithms - Classification & Regression<br>
        Instant Visualizations - Get performance plots instantly
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.write("")
    if st.button("Start Training ", key="start", help="Begin model training"):
        st.session_state.show_main = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


else:
    with st.container(): 
        col1, col2, col3 = st.columns([1, 2, 1]) 
        with col2:
            st.write("#### Upload Dataset:")
            uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    with col2:
        if 'prev_file' in st.session_state and uploaded_file is None:
            keys_to_clear = [
                'df', 'target_column', 'task_type', 'selected_models',
                'selected_features', 'show_plot', 'compare_plot', 'run_models'
            ]
            for key in keys_to_clear:
                st.session_state.pop(key, None)
            st.session_state.pop('prev_file', None)

        if uploaded_file:


            uploaded_file.seek(0, 2)
            size_mb = uploaded_file.tell() / (1024 * 1024)
            uploaded_file.seek(0)

            if size_mb > 3:
                st.error("File too large! Please upload a file under 3 MB.")
            else:

                rawdata = uploaded_file.read()
                uploaded_file.seek(0)
                result = chardet.detect(rawdata)
                encoding = result['encoding'] or "utf-8"

                try:

                    text = rawdata.decode(encoding, errors='replace')
                    df = pd.read_csv(io.StringIO(text))


                    df.columns = (
                        df.columns
                        .str.replace('\ufeff', '', regex=True)
                        .str.strip()
                        .str.replace(' ', '_', regex=False)
                    )


                    df = df.dropna(axis=1, how='all')
                    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


                    if df.shape[1] < 2:
                        st.error("Dataset must have at least 2 columns.")
                        st.stop()

                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    st.stop()

                st.write("#### Cleaned Data Preview:")
                st.dataframe(df.head(25))

                #for no. of nulls dropped
                nulls_dropped = df.isnull().sum().sum() 
                st.write(f"**Total Null Values Dropped:** {nulls_dropped}") 

                st.session_state['df'] = df
                st.session_state['prev_file'] = uploaded_file

                #for showing shape of X
                if 'target_column' in st.session_state:
                    X = df.drop(columns=[st.session_state['target_column']], errors='ignore')
                else:
                    X = df.copy()

                st.write(f"**Shape of dataset:** {X.shape}")


    with col2:
        if 'df' in st.session_state:
            df = st.session_state['df']
            st.write("#### Select Target Column:")  
            target_column = st.selectbox("Select Target Column (y)", ["-select-"] + list(df.columns))

            if target_column != "-select-":
                st.session_state['target_column'] = target_column
                if df[target_column].dtype == "object" or len(df[target_column].unique()) <= 15:
                    task_type = "Classification"
                else:
                    task_type = "Regression"
                st.session_state['task_type'] = task_type
                st.info(f"Task Type: {task_type}")

                available_features = [col for col in df.columns if col != target_column]
                selected_features = st.multiselect("Select Feature Columns (X)",
                                                   options=available_features,
                                                   default=available_features)
                if selected_features:
                    st.session_state['selected_features'] = selected_features
                else:
                    st.session_state.pop('selected_features', None)
            else:
                st.session_state.pop('target_column', None)
                st.session_state.pop('task_type', None)
                st.session_state.pop('selected_models', None)
                st.session_state.pop('show_plot', None)
                st.session_state.pop('compare_plot', None)
                st.session_state.pop('run_models', None)

    with col2:
        if 'task_type' in st.session_state:
            task_type = st.session_state['task_type']
            if task_type == "Classification":
                model_options = list(get_classification_models().keys())
            else:
                model_options = list(get_regression_models().keys())
            selected_models = st.multiselect("Select Models", model_options)

            if "-select-" not in selected_models:
                st.session_state['selected_models'] = selected_models
            else:
                st.session_state.pop('selected_models', None)

            if selected_models:
                st.write("#### Optional Settings:")
                st.session_state['show_plot'] = st.checkbox("Show individual model plots")
                st.session_state['compare_plot'] = st.checkbox("Show model with combined plot")

            if st.button("Run Models"):
                if 'target_column' not in st.session_state or not selected_models:
                    st.warning("Please select a target column and at least one model.")
                else:
                    st.session_state['run_models'] = True
                    with st.spinner("Training model(s)..."):
                        results = train_models(
                            df,
                            st.session_state['target_column'],
                            st.session_state['selected_features'],
                            st.session_state['selected_models'],
                            st.session_state['task_type']
                        )

    if st.session_state.get('run_models'):
        with st.spinner("Training model(s)..."):
            results = train_models(
                st.session_state['df'],
                st.session_state['target_column'],
                st.session_state['selected_features'],
                st.session_state['selected_models'],
                st.session_state['task_type']
            )

        task_type = st.session_state['task_type']
        num_models = len(results)
        i = 0
        while i < num_models:
            cols_needed = min(4, num_models - i)
            cols = st.columns([1] * cols_needed)
            for j in range(cols_needed):
                idx = i + j
                result = results[idx]
                with cols[j]:
                    st.markdown(f"### {result['model_key']}")
                    for metric_name, score in result["metrics"].items():
                        st.markdown(f"**{metric_name}:** {round(score, 4)}")
                    if st.session_state.get("show_plot"):
                        if task_type == "Classification":
                            plot_confusion_matrix(result["y_true"], result["y_pred"], result['model_key'])
                        else:
                            plot_regression_scatter(result["y_true"], result["y_pred"], result['model_key'])
            i += cols_needed

        if st.session_state.get("compare_plot"):
            st.markdown("---")
            st.markdown("### Combined Model Comparison")
            center = st.columns([1, 3, 1])
            with center[1]:
                plot_model_comparison(results, task_type)