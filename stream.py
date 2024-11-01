import streamlit as st
from assignment import run_crew  # Ensure to replace with your actual module name

# Streamlit UI setup
st.title("Market Research & Use Case Generation Agent")

# Input field
topic = st.text_input("Enter a company or industry name:", value=" ")

# Run CrewAI when the button is clicked
if st.button("Generate Report"):
    with st.spinner("Running analysis..."):
        result = run_crew(topic)

    # Display Markdown content from the proposal task
    st.subheader("Final Proposal Document")
    # Assuming the proposal_task has been executed and you have a markdown output
    proposal_markdown = result.tasks_output[-1].raw  # Get the raw output of the final proposal task
    st.markdown(proposal_markdown)  # Render Markdown
