import os
import io
import contextlib
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
from tools.rag_tool import retrieve_context

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

# ── Shared Agent State ───────────────────────────────────────────────────────
class MathState(TypedDict):
    parsed_problem:      dict
    strategy:            str            # Set by Intent Router
    rag_context:         str            # Retrieved KB chunks
    solution_draft:      str            # Solver output
    calc_result:         str            # Python calculator result
    verification_passed: bool
    verification_feedback: str
    final_explanation:   str
    needs_human_review:  bool
    retry_count:         int            # Verifier retry counter
    agent_trace:         List[str]      # Human-readable execution trace
    similar_problems:    List[dict]     # Past similar solved problems from memory


# ── 1. Intent Router Agent ───────────────────────────────────────────────────
def router_node(state: MathState) -> MathState:
    """
    Classify the problem type and decide the solving strategy.
    Runs BEFORE the Solver so it can tailor the approach.
    """
    parsed = state["parsed_problem"]
    topic  = parsed.get("topic", "").lower()

    # Rule-based fast-path (avoids an extra LLM call for clear cases)
    strategy_map = {
        "algebra":       "symbolic_algebra",
        "probability":   "probability_analysis",
        "calculus":      "calculus_technique",
        "linear_algebra": "matrix_operations",
    }
    strategy = strategy_map.get(topic)

    if not strategy:
        # Fall back to LLM for ambiguous topics
        prompt = (
            f"You are a Math Router. Given the topic '{topic}' and problem:\n"
            f"{parsed.get('problem_text', '')}\n\n"
            "Choose ONE strategy from: symbolic_algebra, probability_analysis, "
            "calculus_technique, matrix_operations, general_math.\n"
            "Reply with ONLY the strategy name."
        )
        strategy = llm.invoke([SystemMessage(content=prompt)]).content.strip().lower()

    trace = state.get("agent_trace", [])
    trace.append(f"🔀 **Intent Router** → strategy: `{strategy}`")
    return {"strategy": strategy, "agent_trace": trace}


# ── Python Calculator Tool ───────────────────────────────────────────────────
def run_python_calc(code: str) -> str:
    """Sandbox-execute a small Python math snippet and capture stdout."""
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {"__builtins__": {"print": print, "round": round,
                                          "abs": abs, "max": max, "min": min}})  # noqa: S102
        return stdout.getvalue().strip() or "(no print output)"
    except Exception as e:
        return f"Calculator error: {e}"


# ── 2. Solver Agent ──────────────────────────────────────────────────────────
def solver_node(state: MathState) -> MathState:
    parsed   = state["parsed_problem"]
    strategy = state.get("strategy", "general_math")
    feedback = state.get("verification_feedback", "")
    retry    = state.get("retry_count", 0)
    similar  = state.get("similar_problems", [])

    feedback_prompt = (
        f"\n\n⚠️ Previous attempt was incorrect.\nVerifier feedback: {feedback}\n"
        "Please re-think and fix the error."
    ) if feedback else ""

    # RAG retrieval
    query       = f"{parsed.get('topic', '')} {parsed.get('problem_text', '')}"
    rag_context = retrieve_context(query)

    similar_prompt = ""
    if similar:
        similar_prompt = "\nPAST SIMILAR SOLUTIONS (Use these to guide your reasoning):\n"
        for s in similar:
            past_prob = s.get('parsed_problem', {}).get('problem_text', 'Unknown')
            past_sol = s.get('final_explanation', 'No explanation')
            similar_prompt += f"- Past Problem: {past_prob}\n  Solution: {past_sol}\n"

    prompt = f"""You are an Expert Math Problem Solver using strategy: {strategy}.

RELEVANT FORMULAS from knowledge base:
{rag_context}
{similar_prompt}
Problem: {parsed.get('problem_text')}
Topic:   {parsed.get('topic')}
Variables:   {parsed.get('variables', [])}
Constraints: {parsed.get('constraints', [])}
{feedback_prompt}

Instructions:
1. Show every step clearly.
2. If you need arithmetic verification, output a Python code block starting with ```python
3. End with "FINAL ANSWER: <value>"
"""
    response    = llm.invoke([SystemMessage(content=prompt)]).content

    # Extract and run any embedded Python calculator block
    calc_result = ""
    import re
    py_blocks   = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
    for block in py_blocks:
        calc_result = run_python_calc(block.strip())

    trace = state.get("agent_trace", [])
    trace.append(f"🧮 **Solver** (retry #{retry}) → draft generated")
    return {
        "solution_draft": response,
        "rag_context":    rag_context,
        "calc_result":    calc_result,
        "agent_trace":    trace,
    }


# ── 3. Verifier Agent ────────────────────────────────────────────────────────
def verifier_node(state: MathState) -> MathState:
    parsed    = state["parsed_problem"]
    solution  = state["solution_draft"]
    calc_res  = state.get("calc_result", "")
    retry     = state.get("retry_count", 0)

    calc_note = f"\nPython calculator result: {calc_res}" if calc_res else ""

    prompt = f"""You are a strict Math Verifier.

Problem: {parsed.get('problem_text')}
Constraints: {parsed.get('constraints', [])}
Proposed solution:
{solution}
{calc_note}

Check:
1. Calculation correctness
2. Logical validity
3. Constraint satisfaction
4. Units / domain

Reply STRICTLY with "CORRECT" OR "INCORRECT: <specific error description>".
"""
    response = llm.invoke([SystemMessage(content=prompt)]).content.strip()
    passed   = response.startswith("CORRECT")

    trace = state.get("agent_trace", [])
    trace.append(f"✔️ **Verifier** → {'CORRECT ✅' if passed else 'INCORRECT ❌'}")

    if passed:
        return {
            "verification_passed":   True,
            "verification_feedback": "",
            "needs_human_review":    False,
            "retry_count":           retry,
            "agent_trace":           trace,
        }
    else:
        return {
            "verification_passed":   False,
            "verification_feedback": response,
            "needs_human_review":    retry + 1 >= 2,   # escalate after 2 failures
            "retry_count":           retry + 1,
            "agent_trace":           trace,
        }


# ── 4. Explainer Agent ───────────────────────────────────────────────────────
def explainer_node(state: MathState) -> MathState:
    parsed   = state["parsed_problem"]
    solution = state["solution_draft"]

    prompt = f"""You are an empathetic Math Tutor for JEE students.
Take the rigorous solution and explain it step-by-step in a friendly way.
Use **bold** for key concepts. Use numbered steps. Include the final answer prominently.

Topic: {parsed.get('topic')}
Solution to explain:
{solution}
"""
    explanation = llm.invoke([SystemMessage(content=prompt)]).content
    trace       = state.get("agent_trace", [])
    trace.append("📖 **Explainer** → student-friendly explanation generated")
    return {"final_explanation": explanation, "agent_trace": trace}


# ── Routing Logic ────────────────────────────────────────────────────────────
def route_after_verifier(state: MathState) -> Literal["explainer", "solver", "hitl"]:
    if state["verification_passed"]:
        return "explainer"
    if state.get("retry_count", 0) >= 2:
        return "hitl"      # escalate to human after 2 retries
    return "solver"


# ── HITL sink node ───────────────────────────────────────────────────────────
def hitl_node(state: MathState) -> MathState:
    """Terminal node: flags the problem for human review."""
    trace = state.get("agent_trace", [])
    trace.append("🙋 **HITL** → escalated to human review after 2 failed verifications")
    return {"needs_human_review": True, "agent_trace": trace}


# ── Build LangGraph ──────────────────────────────────────────────────────────
def get_math_work_flow():
    g = StateGraph(MathState)

    g.add_node("router",    router_node)
    g.add_node("solver",    solver_node)
    g.add_node("verifier",  verifier_node)
    g.add_node("explainer", explainer_node)
    g.add_node("hitl",      hitl_node)

    g.set_entry_point("router")
    g.add_edge("router",   "solver")
    g.add_edge("solver",   "verifier")
    g.add_edge("hitl",     END)
    g.add_edge("explainer", END)

    g.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"explainer": "explainer", "solver": "solver", "hitl": "hitl"},
    )

    return g.compile()