import inspect
import logging
import os
import queue
import shutil
import threading
import time
from logging import LogRecord
from logging.handlers import QueueHandler
from os import write

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from torch.utils.tensorboard import SummaryWriter
from torchview import draw_graph

from idinn.dual_controller import DualSourcingNeuralController
from idinn.demand import CustomDemand, UniformDemand
from idinn.sourcing_model import DualSourcingModel

from utils import tflog2pandas
from io import StringIO

st.set_page_config(layout="wide")

if 'log_queue' not in st.session_state:
    st.session_state['log_queue'] = queue.Queue()

queue_handler = QueueHandler(st.session_state['log_queue'])
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
queue_handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(queue_handler)

# Color mapping for log levels
LOG_COLORS = {
    'DEBUG': 'blue',
    'INFO': 'green',
    'WARNING': 'orange',
    'ERROR': 'red',
    'CRITICAL': 'purple'
}


@st.fragment(run_every="1s")
def write_logs():
    process_log_queue()
    messages = list(map(lambda log: log.msg,st.session_state.logs[-100:]))
    prev_log = None
    for log in messages:  # Show last 100 logs
        try:
            parts = log.split(' - ')
            if len(parts) >= 3:
                timestamp, level, message = parts[0], parts[1], ' - '.join(parts[2:])
                color = LOG_COLORS.get(level, 'gray')
                if log != prev_log:
                    st.markdown(f"`{timestamp}` :{color}[**{level}**] {message}")
                prev_log = log
        except Exception as e:
            st.write(f"Error parsing log: {str(e)}")
            time.sleep(0.1)

if torch.cuda.is_available():
    torch.set_default_device('cuda')
    torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

    def update_log_display(log_placeholder):
        """Render logs with color coding in markdown"""
        with log_placeholder.container():
           write_logs()

    def process_log_queue():
        """Check queue for new logs and update session state"""
        while True:
            try:
                log = st.session_state['log_queue'].get_nowait()
                st.session_state.logs.append(log)
            except queue.Empty:
                break

# Session State Initialization
for key in ["training", "trainingnow", "demand_generator", "dual_sourcing_model", "demand_controller"]:
    if key not in st.session_state:
        st.session_state[key] = 0 if key == "training" else False if key == "trainingnow" else None
if 'logs' not in st.session_state:
    st.session_state['logs'] = []

# Page Header
st.header("Inventory Dynamics–Informed Neural Networks")
st.markdown(
    "Welcome to Inventory Dynamics–Informed Neural Networks! This application generates ordering policies from expedited and regular suppliers. Use the sidebar on the left to select a demand model, choose your preferred solver, and view the results after fitting."
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Demand Generation", "Dual Sourcing Model", "NN Controller", "Result"])

# Demand Generation Tab
with tab1:
    c1, c2 = st.columns([1, 3])

    with c1:
        submitted = submitted2 = False
        st.subheader("Demand Generation")

        demand_type = st.radio(label="Please choose demand type", options=["Uniform", "File"])
        if demand_type == "Uniform":
            with st.form(key="uniform_demand"):
                low = st.number_input("Minimum demand", value=1, step=1, format="%i", max_value=(1 << 53) - 1)
                high = st.number_input("Maximum demand", value=4, step=1, format="%i", min_value=0)
                cg1, cg2 = st.columns([1, 2])
                with cg1:
                    pass
                with cg2:
                    submitted = st.form_submit_button("Generate")

                if submitted and high >= low:
                    st.success(
                        f"Successfully generated uniform demand within range: [{np.floor(low)}, {np.floor(high)}]!")
                    st.session_state["training"] = 0
                    st.session_state["demand_generator"] = UniformDemand(low, high)
                elif submitted:
                    st.error("Please resubmit and make sure that maximum demand is greater or equal to minimum demand.")
                    st.session_state["training"] = 0

        elif demand_type == "File":
            with st.form(key="uniform_demand"):
                uploaded_file = st.file_uploader(
                    label="Please upload a single column file with demand values. Each row represents a timestep and each element represents a demand value.")
                cgg1, cgg2 = st.columns([1, 2])
                with cgg1:
                    submitted2 = st.form_submit_button("Generate")
                with cgg2:
                    if submitted2:
                        try:
                            df = pd.read_csv(uploaded_file)
                            st.success(f"Successfully uploaded file and contains {df.shape[0]} demand points!")
                            st.session_state["demand_generator"] = CustomDemand(torch.tensor(df.iloc[:, 0].values))
                            st.session_state["training"] = 0
                        except Exception:
                            st.error("Could not parse file! Please try again!")

    with c2:
        if st.session_state["demand_generator"] is not None:
            all_demands = [st.session_state["demand_generator"].sample(1).item() for _ in range(100)]
            c2c1, c2c2 = st.columns(2)
            with c2c1:
                fig = px.line(y=all_demands, line_shape="hv").update_layout(xaxis_title="Periods", yaxis_title="Demand",
                                                                            title="Typical Demand Trajectory")
                st.plotly_chart(fig, use_container_width=True)
            with c2c2:
                fig = px.histogram(x=all_demands).update_layout(xaxis_title="Demand", yaxis_title="Frequency",
                                                                title="Demand Distribution")
                st.plotly_chart(fig, use_container_width=True)

# Dual Sourcing Model Tab
with tab2:
    st.subheader("Dual Sourcing Model")
    with st.form(key="Dual Sourcing Model"):
        c1t2, c2t2 = st.columns(2)
        with c1t2:
            regular_lead_time = np.int32(
                st.number_input("Regular lead time:", value=2, min_value=0, format="%i", step=1))
            expedited_lead_time = np.int32(
                st.number_input("Expedited lead time:", value=0, min_value=0, format="%i", step=1))
            batch_size = np.int32(
                st.number_input("Minibatch size for demand trajectories:", value=16, min_value=0, format="%i", step=1))
            init_inventory = np.int32(st.number_input("Initial inventory:", value=6, min_value=0, format="%i", step=1))

        with c2t2:
            regular_order_cost = np.int32(
                st.number_input("Regular order cost:", value=0, min_value=0, format="%i", step=1))
            expedited_order_cost = np.int32(
                st.number_input("Expedited order cost:", value=20, min_value=0, format="%i", step=1))
            holding_cost = np.int32(st.number_input("Holding cost:", value=5, min_value=0, format="%i", step=1))
            shortage_cost = np.int32(st.number_input("Shortage cost:", value=495, min_value=0, format="%i", step=1))

        model_params = {
            "regular_lead_time": regular_lead_time,
            "expedited_lead_time": expedited_lead_time,
            "regular_order_cost": regular_order_cost,
            "expedited_order_cost": expedited_order_cost,
            "holding_cost": holding_cost,
            "shortage_cost": shortage_cost,
            "batch_size": batch_size,
            "init_inventory": init_inventory,
            "demand_generator": st.session_state["demand_generator"]
        }

        cc1, cc2 = st.columns([1, 4])
        with cc1:
            pressed2 = st.form_submit_button("Create Sourcing Model")
        with cc2:
            if pressed2:
                st.session_state["training"] = 0
                st.session_state["dual_sourcing_model"] = DualSourcingModel(**model_params)
                st.success("Successfully created sourcing model!")

# Controller Definition Tab
with tab3:
    st.subheader("Controller Definition")
    c1, c2 = st.columns(2)
    with c1:
        activation_map = {
            "CELU": torch.nn.CELU,
            "ReLU": torch.nn.ReLU,
            "LeakyReLU": torch.nn.LeakyReLU,
            "ELU": torch.nn.ELU,
            "Tanh": torch.nn.Tanh,
            "Sigmoid": torch.nn.Sigmoid,
            "SiLU": torch.nn.SiLU,
            "GELU": torch.nn.GELU,
            "TahnShrink": torch.nn.Tanhshrink,
        }
        activation_id = st.selectbox("Activation function of hidden layers:", options=list(activation_map.keys()))
        activation_signature = inspect.signature(activation_map[activation_id])
        kwargs = {v.name: st.number_input(f"Value for activation function's parameter: {v.name}",
                                          value=(v.default if v.default is not None else 0.0)) for k, v in
                  activation_signature.parameters.items() if v.annotation in [float, int]}

        layer_sizes = st.text_area(label="Layer sizes:", value="128, 64, 32, 16, 8, 4",
                                   help="A comma separated list of integers, indicating neurons per layer, starting from the first hidden layer (leftmost)")
        sourcing_periods = st.number_input("Number of training sourcing periods:", value=50, min_value=1, format="%i",
                                           step=1)
        validation_sourcing_periods = st.number_input("Number of validation sourcing periods:", value=1000, min_value=1,
                                                      format="%i", step=1)
        epochs = st.number_input("Number of training epochs:", value=2000, min_value=1, format="%i", step=1)
        seed = st.number_input("Seed:", value=1234, min_value=1, format="%i", step=1)
        try:
            layer_sizes = list(map(int, layer_sizes.split(",")))
        except Exception:
            st.error("Provided input cannot be parsed to layers.")

        st.session_state["demand_controller"] = DualSourcingNeuralController(hidden_layers=layer_sizes,
                                                                             activation=activation_map[activation_id](
                                                                                 **kwargs))

    with c2:
        x = torch.linspace(-10, 10, 100)
        fig = px.line(x=x.cpu().numpy(), y=activation_map[activation_id](**kwargs)(x).cpu().numpy(),
                      title=f"Activation shape: {activation_id}")
        st.plotly_chart(fig, use_container_width=True)

        if st.session_state["demand_controller"] and st.session_state["dual_sourcing_model"]:
            st.info(
                "Graph plot for minibatch size of 4 (for illustration purposes). Click top right corner to enlarge!")
            st.session_state["demand_controller"].init_layers(regular_lead_time=regular_lead_time,
                                                              expedited_lead_time=expedited_lead_time)
            pre_input_tensors = torch.rand([4, 1]), torch.rand([4, max(regular_lead_time, 1)]), torch.rand(
                [4, max(regular_lead_time, 1)])

            input_tensor = st.session_state["demand_controller"].prepare_inputs(*pre_input_tensors,
                                                                                sourcing_model=st.session_state[
                                                                                    "dual_sourcing_model"])

            # input_sizes = [torch.Size([4, 1]), torch.Size([4, max(regular_lead_time, 1)]), torch.Size([4, max(regular_lead_time, 1)])]
            input_sizes = [input_tensor.shape]
            model_graph = draw_graph(st.session_state["demand_controller"].model, input_size=input_sizes)
            model_graph.visual_graph.attr("graph", rankdir="LR")
            st.graphviz_chart(model_graph.visual_graph)
        elif not st.session_state["dual_sourcing_model"]:
            st.warning("Please define a dual sourcing model to generate NN architecture graph!")

    log_placeholder = st.empty()

    def click():
        st.session_state["training"] += 1
        if os.path.exists("runs/dual_sourcing_model"):
            shutil.rmtree("runs/dual_sourcing_model")
        st.session_state["trainingnow"] = True

        def fit_model(controller,
                      sourcing_model,
                      sourcing_periods,
                      validation_sourcing_periods,
                      epochs,
                      seed
                      ):
            if torch.cuda.is_available():
                torch.set_default_device('cuda')
                torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

            controller.fit(
                sourcing_model=sourcing_model,
                sourcing_periods=sourcing_periods,
                validation_sourcing_periods=validation_sourcing_periods,
                epochs=epochs,
                tensorboard_writer=SummaryWriter("runs/dual_sourcing_model"),
                seed=seed,
            )
            return controller
        update_log_display(log_placeholder)
        fit_model(*[st.session_state["demand_controller"],
                                                 st.session_state["dual_sourcing_model"],
                                                 sourcing_periods,
                                                 validation_sourcing_periods,
                                                 epochs,
                                                 seed
                                                 ])
        # st.session_state["demand_controller"].fit(
        #     sourcing_model=st.session_state["dual_sourcing_model"],
        #     sourcing_periods=sourcing_periods,
        #     validation_sourcing_periods=validation_sourcing_periods,
        #     epochs=epochs,
        #     tensorboard_writer=SummaryWriter("runs/dual_sourcing_model"),
        #     seed=seed,
        # )
        st.session_state["trainingnow"] = False
        st.success("Training complete!")

    def training_fragment():
        cf1, cf2 = st.columns([1, 4])
        with cf1:
            pressed = st.button("Fit Controller")
        with cf2:
            if pressed:
                click()

    with st.container() as button_cont:
        training_fragment()
    # Display logs in a markdown window

# Results Tab
with tab4:
    if all(st.session_state[key] for key in ["demand_controller", "dual_sourcing_model", "demand_generator"]):
        if os.path.exists("runs/dual_sourcing_model") and st.session_state["training"] > 0:
            try:
                t4c1, t4c2 = st.columns(2)
                with t4c1:
                    tsb_df = tflog2pandas("runs/dual_sourcing_model")
                    print(tsb_df)
                    fig = px.line(tsb_df, x="step", y="value", color="metric", line_shape="hv",
                                  title="Learning Curves").update_layout(yaxis_title="Avg. Cost per Period",
                                                                         xaxis_title="Epoch")
                    st.plotly_chart(fig)
                with t4c2:
                    past_inventories, past_regular_orders, past_expedited_orders = st.session_state[
                        "demand_controller"].simulate(sourcing_model=st.session_state["dual_sourcing_model"],
                                                      sourcing_periods=sourcing_periods)
                    df_past = pd.DataFrame({"Inventory": past_inventories, "Regular Orders": past_regular_orders,
                                            "Expedited Orders": past_expedited_orders})
                    fig = px.line(df_past, line_shape="hv").update_layout(xaxis_title="Periods", yaxis_title="Quantity",
                                                                          title="Sample Optimization Trajectory")
                    st.plotly_chart(fig)
            except Exception as e:
                st.warning("No available model, please make sure you submit the previous steps!")
                st.error(e)
    else:
        if not st.session_state["demand_generator"]:
            st.warning("No demand generator is chosen for the current configuration, please define one.")
        if not st.session_state["dual_sourcing_model"]:
            st.warning("No sourcing model is defined for the current configuration, please define one.")
        if not st.session_state["demand_controller"]:
            st.warning(
                "No model chosen or trained for the current configuration, please go to the previous step and fit a model.")
