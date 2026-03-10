import streamlit as st
import os
from dotenv import load_dotenv

from tools.ocr_tool import extract_text_from_image, is_low_confidence
from tools.asr_tool import transcribe_audio
from agents.parser_agent import get_parser_agent
from agents.solver_workflow import get_math_work_flow
from memory.memory_store import save_attempt, retrieve_similar, update_feedback, save_correction_rule, apply_correction_rules

load_dotenv()

st.set_page_config(
    page_title="Math Mentor — AI Planet",
    page_icon="🧮",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stButton>button { border-radius: 8px; font-weight: 600; }
.confidence-high { color: #22c55e; font-weight: bold; }
.confidence-low  { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🧮 Reliable Multimodal Math Mentor")
st.caption("AI Planet Assignment · RAG + Agents + HITL + Memory")

# ── Load models once ─────────────────────────────────────────────────────────
@st.cache_resource
def load_parser():
    return get_parser_agent()

@st.cache_resource
def load_workflow():
    return get_math_work_flow()

parser_agent = load_parser()
workflow     = load_workflow()

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("parsed_problem",   None),
    ("final_state",      None),
    ("memory_record_id", None),
    ("extracted_text",   ""),
    ("ocr_confidence",   None),
    ("transcript",       ""),
    ("asr_unclear",      False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helper: confidence badge ──────────────────────────────────────────────────
def _conf_badge(score: float) -> str:
    pct = int(score * 100)
    cls = "confidence-low" if score < 0.75 else "confidence-high"
    return f'<span class="{cls}">OCR Confidence: {pct}%</span>'


# ── Core orchestration ────────────────────────────────────────────────────────
def process_problem(raw_text: str, input_mode: str = "text", original_input: str = ""):
    """Full pipeline: parse → memory hints → multi-agent workflow → save."""
    st.session_state.final_state = None

    # ── Step 1: Parser Agent ──────────────────────────────────────────────
    with st.spinner("🔍 Parser Agent — structuring the problem…"):
        try:
            parsed_result = parser_agent.invoke({"input_text": raw_text})
            result_dict   = (parsed_result.dict()
                             if hasattr(parsed_result, "dict")
                             else parsed_result)
            st.session_state.parsed_problem = result_dict

            with st.expander("📋 Parsed Problem Structure"):
                st.json(result_dict)

            if result_dict.get("needs_clarification"):
                st.warning(
                    "⚠️ **HITL triggered** — the parser detected ambiguity. "
                    "Please edit and re-submit your input."
                )
                return
        except Exception as e:
            st.error(f"Parser error: {e}")
            return

    # ── Step 1b: Memory — show similar past solutions ─────────────────────
    similar = retrieve_similar(result_dict.get("topic", ""))
    if similar:
        with st.expander(f"💡 Memory — {len(similar)} similar solved problem(s) found"):
            for s in similar:
                st.markdown(f"**Problem:** {s['parsed_problem'].get('problem_text', '')}")
                st.markdown(f"**Explanation:** {s['final_explanation']}")
                st.divider()

    # ── Step 2: Multi-Agent LangGraph Workflow ────────────────────────────
    with st.spinner("🤖 Running multi-agent workflow…"):
        try:
            final_value = {}
            for output in workflow.stream({"parsed_problem": result_dict,
                                           "retry_count": 0,
                                           "agent_trace": [],
                                           "similar_problems": similar}):
                for node_name, value in output.items():
                    st.toast(f"Agent finished: **{node_name}**", icon="✅")
                    final_value = value

            st.session_state.final_state = final_value

            record_id = save_attempt(
                parsed_problem      = result_dict,
                rag_context         = final_value.get("rag_context", ""),
                solution_draft      = final_value.get("solution_draft", ""),
                verification_passed = final_value.get("verification_passed", False),
                final_explanation   = final_value.get("final_explanation", ""),
                original_input      = original_input,
                input_mode          = input_mode,
            )
            st.session_state.memory_record_id = record_id

        except Exception as e:
            st.error(f"Workflow error: {e}")


# ── Results panel ─────────────────────────────────────────────────────────────
def display_results():
    if not st.session_state.final_state:
        return

    state = st.session_state.final_state
    st.divider()
    st.header("📊 Results")

    # -- Agent Trace --
    trace = state.get("agent_trace", [])
    if trace:
        with st.expander("🔎 Agent Execution Trace"):
            for step in trace:
                st.markdown(step)

    # -- Confidence Indicator --
    ver_passed = state.get("verification_passed", False)
    retries    = state.get("retry_count", 0)
    hitl       = state.get("needs_human_review", False)

    if hitl:
        conf_label, conf_val, conf_color = "Low (HITL Required)", 20, "inverse"
    elif retries > 0:
        conf_label, conf_val, conf_color = "Medium (1 retry)", 60, "normal"
    else:
        conf_label, conf_val, conf_color = "High", 95, "off"

    col_c1, col_c2 = st.columns([1, 3])
    with col_c1:
        st.metric("🎯 Confidence", conf_label)
    with col_c2:
        st.progress(conf_val, text=f"Solver confidence: {conf_val}%")

    # -- RAG Context Panel --
    rag = state.get("rag_context", "")
    if rag:
        with st.expander("📚 Retrieved Knowledge Base Context (RAG)"):
            st.info(rag)

    # -- Calc result (if Python tool ran) --
    calc = state.get("calc_result", "")
    if calc:
        with st.expander("🐍 Python Calculator Result"):
            st.code(calc)

    # -- Solver draft + Verifier status --
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧮 Solver Draft")
        st.info(state.get("solution_draft", "No draft."))
    with col2:
        st.subheader("✔️ Verification")
        if ver_passed:
            st.success("**VERIFIED CORRECT** ✅")
        elif hitl:
            st.error("**Escalated to Human Review** 🙋")
            st.write(state.get("verification_feedback", ""))
        else:
            st.error("**INCORRECT** ❌")
            st.write(state.get("verification_feedback", ""))

    # -- Final explanation --
    st.subheader("🎓 Step-by-Step Explanation")
    if state.get("final_explanation"):
        st.success(state["final_explanation"])
    elif hitl:
        st.warning(
            "The verifier could not confirm this answer after 2 attempts. "
            "Please review the Solver Draft above and correct manually."
        )

    # -- Feedback (HITL + Memory) --
    st.divider()
    st.subheader("📝 Feedback & Re-Check")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        if st.button("✅ Correct", key="btn_correct", use_container_width=True):
            if st.session_state.memory_record_id:
                update_feedback(st.session_state.memory_record_id, "correct")
            st.success("Marked as correct — saved to memory!")
    with col_b:
        comment = st.text_input("Describe the error or hint for retry:", key="fb_comment")
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("❌ Incorrect (Just Log)", key="btn_incorrect", use_container_width=True):
                fb = f"incorrect:{comment}" if comment else "incorrect"
                if st.session_state.memory_record_id:
                    update_feedback(st.session_state.memory_record_id, fb)
                st.warning("Logged as incorrect — will improve future attempts!")
        with col_b2:
            if st.button("🔁 Re-Calculate (HITL)", key="btn_recalc", type="primary", use_container_width=True):
                if not comment:
                    st.error("Please provide a hint or correction in the text box for the retry.")
                else:
                    st.info("Triggering explicit re-check with your feedback...")
                    # Update LangGraph state manually and rerun
                    last_state = st.session_state.final_state
                    if last_state:
                         # Append user feedback to the current verification feedback
                         current_fb = last_state.get("verification_feedback", "")
                         new_fb = f"{current_fb}\n\nUSER MANUAL HINT: {comment}"
                         last_state["verification_feedback"] = new_fb
                         # Force it to go back to the solver
                         last_state["verification_passed"] = False
                         
                         with st.spinner("🤖 Re-running solver with your feedback..."):
                             # Re-run starting from solver with the modified state
                             final_value = {}
                             for output in workflow.stream(last_state):
                                 for node_name, value in output.items():
                                     st.toast(f"Agent finished: **{node_name}**", icon="✅")
                                     final_value = value
                             
                             st.session_state.final_state = final_value
                             
                             # Save the new attempt
                             record_id = save_attempt(
                                 parsed_problem      = final_value.get("parsed_problem", {}),
                                 rag_context         = final_value.get("rag_context", ""),
                                 solution_draft      = final_value.get("solution_draft", ""),
                                 verification_passed = final_value.get("verification_passed", False),
                                 final_explanation   = final_value.get("final_explanation", ""),
                                 original_input      = "recalc_from_feedback",
                                 input_mode          = "text"
                             )
                             st.session_state.memory_record_id = record_id
                             st.rerun()


# ── Input mode selector ───────────────────────────────────────────────────────
st.divider()
input_mode = st.radio("**Select Input Mode**",
                      ["✏️ Text", "🖼️ Image", "🎙️ Audio"], horizontal=True)

# ── Text ──────────────────────────────────────────────────────────────────────
if input_mode == "✏️ Text":
    user_text = st.text_area("Type your math problem here:", height=120,
                              placeholder="e.g. Find the roots of x^2 - 5x + 6 = 0")
    if st.button("🚀 Solve", key="solve_text"):
        if user_text.strip():
            process_problem(user_text.strip(), input_mode="text", original_input=user_text.strip())
        else:
            st.error("Please enter a math problem.")

# ── Image ORC ────────────────────────────────────────────────────────────────
elif input_mode == "🖼️ Image":
    uploaded_image = st.file_uploader("Upload a math problem image (JPG/PNG)",
                                       type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded image", use_container_width=True)

        if st.button("🔍 Extract Text via OCR", key="ocr_btn"):
            with st.spinner("Running OCR…"):
                uploaded_image.seek(0)
                text, conf = extract_text_from_image(uploaded_image)
                st.session_state.extracted_text  = text
                st.session_state.ocr_confidence  = conf

        if st.session_state.extracted_text:
            text_to_show = apply_correction_rules(st.session_state.extracted_text, "image")
            
            # Show confidence badge
            if st.session_state.ocr_confidence is not None:
                st.markdown(_conf_badge(st.session_state.ocr_confidence),
                            unsafe_allow_html=True)
                if is_low_confidence(st.session_state.ocr_confidence):
                    st.warning(
                        "⚠️ **HITL triggered** — OCR confidence is low. "
                        "Please review and correct the text below."
                    )

            edited = st.text_area("Extracted Text (edit if needed):",
                                   value=text_to_show, height=150)
            if st.button("🚀 Solve Image Problem", key="solve_image"):
                save_correction_rule(st.session_state.extracted_text, edited, "image")
                process_problem(edited, input_mode="image", original_input=st.session_state.extracted_text)

# ── Audio ASR ────────────────────────────────────────────────────────────────
elif input_mode == "🎙️ Audio":
    st.info("You can either upload an audio file or record directly using your microphone.")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_audio = st.file_uploader("Upload audio question (WAV/MP3/M4A)",
                                           type=["wav", "mp3", "m4a"])
    with col2:
        recorded_audio = st.audio_input("Or record your question")
        
    audio_source = recorded_audio or uploaded_audio

    if audio_source:
        st.audio(audio_source)

        if st.button("🎙️ Transcribe Audio", key="asr_btn"):
            with st.spinner("Transcribing with Whisper…"):
                with open("temp_audio.wav", "wb") as f:
                    f.write(audio_source.getbuffer())
                text, unclear = transcribe_audio("temp_audio.wav")
                os.remove("temp_audio.wav")
                st.session_state.transcript  = text
                st.session_state.asr_unclear = unclear

        if st.session_state.transcript:
            trans_to_show = apply_correction_rules(st.session_state.transcript, "audio")
            
            if st.session_state.asr_unclear:
                st.warning(
                    "⚠️ **HITL triggered** — transcription confidence is low. "
                    "Please review and correct below."
                )
            edited_t = st.text_area("Transcript (edit if needed):",
                                     value=trans_to_show, height=150)
            if st.button("🚀 Solve Audio Problem", key="solve_audio"):
                save_correction_rule(st.session_state.transcript, edited_t, "audio")
                process_problem(edited_t, input_mode="audio", original_input=st.session_state.transcript)

# Always render results
display_results()
