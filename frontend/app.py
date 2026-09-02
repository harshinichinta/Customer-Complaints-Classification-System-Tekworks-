import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

BACKEND_URL = os.environ.get('BACKEND_URL', 'http://127.0.0.1:8000')


def predict_complaint(text: str):
    url = BACKEND_URL.rstrip('/') + '/predict'
    try:
        resp = requests.post(url, json={'complaint': text}, timeout=10)
        if resp.status_code == 200:
            return resp.json(), None
        else:
            return None, f"API error {resp.status_code}: {resp.text}"
    except Exception as e:
        return None, str(e)


def main():
    st.set_page_config(page_title='BankComplaint AI', layout='wide')

    # Sidebar
    st.markdown("""
    <style>
    .sidebar .sidebar-content {background: linear-gradient(180deg, #4c6ef5 0%, #7c3aed 100%); color: white}
    </style>
    """, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("# BankComplaint AI")
        st.markdown("**AI-Powered Complaint Classification System**")
        page = st.radio('', ['Home', 'Complaint Form', 'Dashboard', 'About'])

    # Top header
    st.markdown(
        "<div style='background: linear-gradient(90deg,#7c3aed,#4c6ef5); padding:20px; border-radius:8px; color:white'>"
        "<h1 style='margin:0'>BankComplaint AI</h1>"
        "<p style='margin:0'>AI-Powered Complaint Classification System</p>"
        "</div>", unsafe_allow_html=True)

    # HOME
    if page == 'Home':
        st.markdown('## Customer Complaints and Classification System')
        st.write('An AI-powered system that automatically classifies customer complaints into appropriate categories using machine learning.')
        # Load dataset and model results
        total = 'N/A'
        categories = 'N/A'
        model_acc = None
        model_count = 0
        try:
            df = pd.read_csv('data/complaints.csv')
            total = int(df.shape[0])
            if 'product' in df.columns:
                categories = int(df['product'].nunique())
        except Exception:
            pass

        # model metrics
        try:
            metrics_path = 'models/results.json'
            if os.path.exists(metrics_path):
                metrics = pd.read_json(metrics_path)
                # metrics is a dict, read properly
                import json as _json
                with open(metrics_path, 'r') as f:
                    mj = _json.load(f)
                results = mj.get('results', {})
                model_count = len(results)
                best = mj.get('best_model')
                if best and results.get(best):
                    model_acc = results[best].get('accuracy')
        except Exception:
            pass

        cols = st.columns(4)
        cols[0].metric('Total complaints', total)
        cols[1].metric('Categories', categories)
        cols[2].metric('Model accuracy', f"{model_acc:.3f}" if model_acc is not None else 'N/A')
        cols[3].metric('Models evaluated', model_count)

    # COMPLAINT FORM
    elif page == 'Complaint Form':
        st.header('Complaint Classification')
        st.write('Enter the complaint text below and click Classify Complaint.')
        placeholder = 'My new credit card has not arrived yet.'
        complaint = st.text_area('Customer Complaint', value='', placeholder=placeholder, height=250)
        c1, c2 = st.columns([1,3])
        with c1:
            if st.button('Classify Complaint'):
                if not complaint or not complaint.strip():
                    st.error('Please enter a complaint before classifying.')
                else:
                    with st.spinner('Classifying...'):
                        result, err = predict_complaint(complaint)
                    if err:
                        st.error('Unable to get prediction. ' + str(err))
                    else:
                        # Display result card
                        pred = result.get('prediction')
                        conf = result.get('confidence')
                        st.success('Prediction complete')
                        card_html = f"""
                        <div style='border-radius:8px; padding:16px; background:linear-gradient(90deg,#f8fafc,#eef2ff);'>
                          <h3>Prediction Result</h3>
                          <p><strong>Predicted Category:</strong> {pred}</p>
                          <p><strong>Complaint:</strong> {complaint}</p>
                        """
                        if conf is not None:
                            card_html += f"<p><strong>Confidence:</strong> {conf:.2f}</p>"
                        card_html += "</div>"
                        st.markdown(card_html, unsafe_allow_html=True)
        with c2:
            st.info('Tip: enter a clear description of the complaint for best results.')

    # DASHBOARD
    elif page == 'Dashboard':
        st.header('Dashboard')
        try:
            df = pd.read_csv('data/complaints.csv')
            st.subheader('Dataset Overview')
            st.metric('Total complaints', int(df.shape[0]))
            if 'product' in df.columns:
                vc = df['product'].value_counts()
                st.plotly_chart(px.bar(vc.reset_index().rename(columns={'index':'category','product':'count'}).head(20), x='category', y='count', title='Category distribution'), use_container_width=True)
        except Exception as e:
            st.error('Failed to load dataset: ' + str(e))

        st.subheader('Model Performance')
        try:
            resp = requests.get(BACKEND_URL.rstrip('/') + '/metrics', timeout=5)
            if resp.status_code == 200:
                mj = resp.json()
                results = mj.get('results', {})
                rows = []
                for m, vals in results.items():
                    rows.append({'model': m, 'accuracy': vals.get('accuracy'), 'f1_weighted': vals.get('f1_weighted')})
                if rows:
                    st.dataframe(pd.DataFrame(rows).sort_values('f1_weighted', ascending=False))
                best = mj.get('best_model')
                st.write('Best model:', best)
            else:
                st.info('Metrics not available from backend')
        except Exception as e:
            st.info('Could not fetch metrics from backend: ' + str(e))

    # ABOUT
    elif page == 'About':
        st.header('About')
        st.markdown('''
        **Project:** Customer Complaints and Classification System

        **Technology stack:** Python, scikit-learn, FastAPI, Streamlit, Plotly

        **Description:** This project classifies customer complaints into categories using a trained ML model. The frontend sends complaints to the FastAPI backend, which loads saved model artifacts and returns predictions.
        ''')


if __name__ == '__main__':
    main()
