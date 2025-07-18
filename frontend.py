import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
from constants import Constants

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
# Submit button
submit = st.button("Submit")

if submit and adjacency_list and data_transfers:
    st.subheader("Graph")
# Error message
elif submit and not (adjacency_list and data_transfers):
    st.write(
        "Please enter an adjacency list of edges and a list of data transfers to schedule."
    )
