import streamlit as st
import pandas as pd

from src.loaders import load_calls_from_file
from src.regex_rules import (
    detect_profanity_by_role_regex,
    detect_privacy_violation_regex,
)
from src.ml_models import (
    get_profanity_model,
    get_sensitive_model,
    get_verification_model,
    ml_detect_profanity_by_role,
    ml_detect_privacy_violation,
)
from src.metrics import compute_overtalk_and_silence
from src.visualizations import metrics_chart
from src.utils import extract_call_id_from_name

st.set_page_config(page_title="Debt Collection Compliance Checker", page_icon="", layout="centered")

st.title("Debt Collection Compliance Checker")
st.caption("Analyze calls for profanity, privacy/compliance, and call quality metrics (overtalk & silence).")

with st.sidebar:
    st.header("Analysis Settings")
    approach = st.selectbox(
        "Approach",
        ["Pattern Matching", "Machine Learning"],
        index=0,
    )
    entity = st.selectbox(
        "Entity",
        ["Profanity Detection", "Privacy and Compliance Violation"],
        index=0,
    )
    uploaded = st.file_uploader(
        "Upload a call file (YAML or JSON)",
        type=["yaml", "yml", "json"],
        help="Each file contains utterances with speaker, text, stime, etime.",
    )
    analyze = st.button("Analyze")

st.markdown("----")

def comparative_analysis(calls, entity: str):
    # Always compute both approaches for comparison
    if entity == "Profanity Detection":
        pm_agent, pm_borr = detect_profanity_by_role_regex(calls)
        prof_model = get_profanity_model()
        ml_agent, ml_borr = ml_detect_profanity_by_role(calls, prof_model)

        st.subheader("Comparative Analysis (Profanity)")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Pattern Matching - Agent", "Yes" if pm_agent else "No")
            st.metric("Pattern Matching - Borrower", "Yes" if pm_borr else "No")
        with c2:
            st.metric("Machine Learning - Agent", "Yes" if ml_agent else "No")
            st.metric("Machine Learning - Borrower", "Yes" if ml_borr else "No")

        agree = (pm_agent == ml_agent) and (pm_borr == ml_borr)
        st.info(f"Agreement: {'Yes' if agree else 'No'}")
        if agree:
            st.success(
                "Recommendation: For profanity detection, prefer Pattern Matching for transparent rules and predictable precision. "
                "Use the ML approach as a secondary check when you expand labeled data to handle paraphrases and obfuscations."
            )
        else:
            st.warning(
                "The approaches disagree. Use Pattern Matching for immediate decisions, and retrain the ML model with more examples "
                "to improve alignment on your data."
            )
    else:
        vio_pm = detect_privacy_violation_regex(calls)
        sens_model = get_sensitive_model()
        ver_model = get_verification_model()
        vio_ml = ml_detect_privacy_violation(calls, sens_model, ver_model)

        st.subheader("Comparative Analysis (Privacy & Compliance)")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Pattern Matching", "Yes" if vio_pm else "No")
        with c2:
            st.metric("Machine Learning", "Yes" if vio_ml else "No")

        agree = (vio_pm == vio_ml)
        st.info(f"Agreement: {'Yes' if agree else 'No'}")
        if agree:
            st.success(
                "Recommendation: Use ML (with call-order rules) when phrasing varies widely. "
                "Regex remains a solid baseline for explicit scripts and auditing."
            )
        else:
            st.warning(
                "The approaches disagree. Prefer regex for strict compliance gates and iterate on the ML training data to close the gap."
            )

if uploaded and analyze:
    try:
        calls = load_calls_from_file(uploaded, uploaded.name)
        if not calls:
            st.error("No utterances found in the file.")
        else:
            call_id = extract_call_id_from_name(uploaded.name)
            st.subheader(f"Call ID: {call_id}")

            # Selected approach result for the selected entity
            if approach == "LLM":
                st.warning("LLM approach is not implemented in this app. Please use Pattern Matching or Machine Learning.")
            else:
                if entity == "Profanity Detection":
                    if approach == "Pattern Matching":
                        agent_flag, borrower_flag = detect_profanity_by_role_regex(calls)
                    else:
                        prof_model = get_profanity_model()
                        agent_flag, borrower_flag = ml_detect_profanity_by_role(calls, prof_model)

                    st.markdown("### Profanity Detection Result (Selected Approach)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Agent used profanity", "Yes" if agent_flag else "No")
                    with col2:
                        st.metric("Borrower used profanity", "Yes" if borrower_flag else "No")

                else:
                    if approach == "Pattern Matching":
                        violation = detect_privacy_violation_regex(calls)
                    else:
                        sens_model = get_sensitive_model()
                        ver_model = get_verification_model()
                        violation = ml_detect_privacy_violation(calls, sens_model, ver_model)

                    st.markdown("### Privacy & Compliance Result (Selected Approach)")
                    st.metric(
                        "Sensitive info shared without prior verification",
                        "Yes" if violation else "No",
                    )

            st.markdown("----")

            # Comparative Analysis (always shown for the selected entity)
            comparative_analysis(calls, entity)

            st.markdown("----")

            # Q3: Call Quality Metrics
            st.subheader("Call Quality Metrics (per call)")
            overtalk_pct, silence_pct = compute_overtalk_and_silence(calls)
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.metric("Overtalk %", f"{overtalk_pct:.2f}%")
            with mcol2:
                st.metric("Silence %", f"{silence_pct:.2f}%")

            chart_df = pd.DataFrame(
                [{"metric": "Overtalk %", "value": overtalk_pct}, {"metric": "Silence %", "value": silence_pct}]
            )
            st.altair_chart(metrics_chart(chart_df), use_container_width=True)

            # Show a preview table
            with st.expander("Preview utterances"):
                st.dataframe(
                    pd.DataFrame(calls)[["speaker", "text", "stime", "etime"]],
                    use_container_width=True,
                    hide_index=True,
                )

    except Exception as e:
        st.error(f"Failed to process the file: {e}")

elif not uploaded:
    st.info("Upload a YAML or JSON call file in the sidebar to begin.")
