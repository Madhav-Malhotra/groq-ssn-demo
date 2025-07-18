import streamlit as st
from constants import Constants
from backend import VisualisationUtils, SchedulingUtils

# Context
st.title("Software Scheduled Networking (SSN) Demo")
st.write(
    """
    Experiment with the graph and data transfer settings below to see 
    different static schedules to minimise latency or maximise throughput.
    """
)

# Graph inputs
st.subheader("Inputs")
adjacency_list = st.text_area(
    "Adjacency List",
    value=str(Constants.DEFAULT_GRAPH),
    help="Enter the adjacency list of the graph, where each tuple represents an edge between two nodes.",
)
data_transfers = st.text_area(
    "Data Transfers",
    value=str(Constants.DEFAULT_DATA_TRANSFERS),
    help="Enter a list of data transfers where each tuple (Px, Cy) represents a data transfer from producer x to consumer y",
)
objective = st.selectbox(
    "Objective",
    options=["Minimise Latency", "Maximise Throughput"],
    help="Select the objective for the scheduling algorithm.",
)
submit = st.button("Submit")

# Display outputs
if submit and adjacency_list and data_transfers:
    st.subheader("Graph")
    try:
        v = VisualisationUtils()
        s = SchedulingUtils()

        # Render the graph to create a schedule for
        graph = v.adjacency_to_graph(adjacency_list)
        v.draw_graph(graph, "original.png")
        st.image("original.png", caption="Original Graph")

        # Create the schedule

    except Exception as e:
        st.error(f"Andrew thought of an edge case. There goes by shot at the job 😭")

# Error message
elif submit and not (adjacency_list and data_transfers):
    st.write(
        "Please enter an adjacency list of edges and a list of data transfers to schedule."
    )
