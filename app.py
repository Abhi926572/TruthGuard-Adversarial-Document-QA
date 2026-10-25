import streamlit as st
from truthguard.agents import TruthGuardSystem

st.set_page_config(page_title="TruthGuard", page_icon="🛡️", layout="wide")
st.title("🛡️ TruthGuard")
st.caption("Adversarial verification for inconsistent document Q&A")

@st.cache_resource
def load_system():
    return TruthGuardSystem("data/documents.json")

system = load_system()

with st.sidebar:
    st.header("Multi-Agent Pipeline")
    st.markdown(
        "1. **Retriever** → relevant passages\n"
        "2. **Synthesizer** → candidate answer\n"
        "3. **Critic** → attacks the answer\n"
        "4. **Resolution** → provenance + conflict handling"
    )
    st.divider()
    st.write(f"Documents indexed: **{len(system.documents)}**")

examples = [
    "Can an employee work remotely on Friday?",
    "How many days of parental leave are available?",
    "What is the expense approval limit?"
]

question = st.text_input("Ask a question", placeholder=examples[0])

if not question:
    st.info("Enter a question or click an example.")
    cols = st.columns(3)
    for i, q in enumerate(examples):
        if cols[i].button(q, use_container_width=True):
            question = q

if question:
    result = system.answer(question)

    st.subheader("Answer")
    if result["status"] == "resolved":
        st.success(result["answer"])
    elif result["status"] == "conflict":
        st.warning(result["answer"])
    else:
        st.error(result["answer"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Retrieved passages", len(result["retrieved"]))
    c2.metric("Conflicts", len(result["conflicts"]))
    c3.metric("Unsupported claims", len(result["unsupported_claims"]))

    st.subheader("Reasoning")
    st.write(result["reasoning"])

    if result["conflicts"]:
        st.subheader("⚠️ Detected conflicts")
        for conflict in result["conflicts"]:
            st.markdown(f"**{conflict['type'].title()}** — {conflict['explanation']}")

    if result["unsupported_claims"]:
        st.subheader("🚩 Unsupported claims")
        for claim in result["unsupported_claims"]:
            st.write(f"- {claim}")

    st.subheader("Source provenance")
    for item in result["retrieved"]:
        with st.expander(
            f"{item['title']} · {item['date']} · relevance {item['score']:.3f}"
        ):
            st.write(item["text"])
            st.caption(
                f"Version: {item['version']} | Status: {item['status']} | Source ID: {item['id']}"
            )
