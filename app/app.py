import inspect
import os
import shutil
import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch.nn
from idinn.controller import DualSourcingNeuralController
from idinn.demand import CustomDemand, UniformDemand
from idinn.sourcing_model import DualSourcingModel
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
from torchview import draw_graph

st.set_page_config(layout="wide")

def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


if "training" not in st.session_state:
    st.session_state["training"] = 0
if "trainingnow" not in st.session_state:
    st.session_state["trainingnow"] = False
if "demand_generator" not in st.session_state:
    st.session_state["demand_generator"] = None
if "dual_sourcing_model" not in st.session_state:
    st.session_state["dual_sourcing_model"] = None
if "demand_controller" not in st.session_state:
    st.session_state["demand_controller"] = None

st.header("Inventory Dynamics–Informed Neural Networks ")
st.markdown(
    "Welcome to Inventory Informed Neural Networks. This application generates policies to order"
    "from expendited and regular suppliers! From the sidebar to your left, you can select a demand model, "
    "then your prefered solver and finally see the results after fitting."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Demand Generation", "Dual Sourcing Model", "NN Controller", "Results"]
)

with tab1:
    c1, c2 = st.columns([1, 3])

    with c1:
        submitted = submitted2 = False
        st.subheader("Demand Generation")

        demand_type = st.radio(
            label="Please choose demand type", options=["Uniform", "File"]
        )
        if demand_type == "Uniform":
            with st.form(key="uniform_demand"):
                high = (1 << 53) - 1
                low = st.number_input(
                    "Minimum demand",
                    value=0,
                    placeholder="Type an integer...",
                    step=1,
                    format="%i",
                    max_value=high,
                )
                high = st.number_input(
                    "Maximum demand",
                    value=4,
                    placeholder="Type an integer...",
                    step=1,
                    format="%i",
                    min_value=0,
                )
                cg1, cg2 = st.columns([1, 2])
                with cg1:
                    submitted = st.form_submit_button("Generate")
                with cg2:
                    if submitted and high >= low:
                        st.success(
                            "Successfully generated uniform demand within range: ["
                            + str(np.floor(low))
                            + ", "
                            + str(np.floor(high))
                            + "]"
                        )
                        st.session_state["training"] = 0
                        st.session_state["demand_generator"] = UniformDemand(low, high)
                    elif submitted:
                        st.error(
                            "Please resubmit and make sure that maximum demand is greater or equal to minimum demand."
                        )
                        st.session_state["training"] = 0

        elif demand_type == "File":
            with st.form(key="uniform_demand"):
                uploaded_file = st.file_uploader(
                    label="Please upload a single column file with demand values. Each row represents a timestep and each"
                    "element represents a demand value."
                )
                cgg1, cgg2 = st.columns([1, 2])
                with cgg1:
                    submitted2 = st.form_submit_button("Generate")
                with cgg2:
                    if submitted2:
                        try:
                            df = pd.read_csv(uploaded_file)
                            st.success(
                                "File successfuly uploaded and contains "
                                + str(df.shape[0])
                                + " demand points:"
                            )
                            # st.table(df)
                            st.session_state["demand_generator"] = CustomDemand(
                                torch.tensor(df.iloc[:, 0].values)
                            )
                            st.session_state["training"] = 0
                        except Exception:
                            st.error("Could not parse file! Please try again!")
    with c2:
        if st.session_state["demand_generator"] is not None:
            all_demands = []
            for i in range(100):
                all_demands.append(
                    st.session_state["demand_generator"].sample(1).item()
                )
            c2c1, c2c2 = st.columns(2)
            with c2c1:
                fig = px.line(y=all_demands).update_layout(
                    xaxis_title="Periods",
                    yaxis_title="Demand",
                    title="Typical Demand Trajectory",
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2c2:
                fig = px.histogram(x=all_demands).update_layout(
                    xaxis_title="Demand",
                    yaxis_title="Frequency",
                    title="Demand Distribution",
                )
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Dual Sourcing Model")
    with st.form(key="Dual Sourcing Model"):
        c1t2, c2t2 = st.columns(2)
        with c1t2:
            regular_lead_time = np.int32(
                st.number_input(
                    "Regular lead time:", value=2, min_value=0, format="%i", step=1
                )
            )
            expedited_lead_time = np.int32(
                st.number_input(
                    "Expedited lead time:", value=0, min_value=0, format="%i", step=1
                )
            )
            batch_size = np.int32(
                st.number_input(
                    "Minibatch/sample size for demand trajectories:",
                    value=16,
                    min_value=0,
                    format="%i",
                    step=1,
                )
            )
            init_inventory = np.int32(
                st.number_input(
                    "Initial inventory:", value=6, min_value=0, format="%i", step=1
                )
            )

        with c2t2:
            regular_order_cost = np.int32(
                st.number_input(
                    "Regular order cost:", value=0, min_value=0, format="%i", step=1
                )
            )
            expedited_order_cost = np.int32(
                st.number_input(
                    "Expedited order cost:", value=20, min_value=0, format="%i", step=1
                )
            )
            holding_cost = np.int32(
                st.number_input(
                    "Holding cost:", value=5, min_value=0, format="%i", step=1
                )
            )
            shortage_cost = np.int32(
                st.number_input(
                    "Shortage cost:", value=495, min_value=0, format="%i", step=1
                )
            )

        model_params = dict(
            regular_lead_time=regular_lead_time,
            expedited_lead_time=expedited_lead_time,
            regular_order_cost=regular_order_cost,
            expedited_order_cost=expedited_order_cost,
            holding_cost=holding_cost,
            shortage_cost=shortage_cost,
            batch_size=batch_size,
            init_inventory=init_inventory,
        )
        model_params["demand_generator"] = st.session_state["demand_generator"]

        cc1, cc2 = st.columns([1, 4])
        with cc1:
            pressed2 = st.form_submit_button("Create Sourcing Model")
        with cc2:
            if pressed2:
                st.session_state["training"] = 0
                st.session_state["dual_sourcing_model"] = DualSourcingModel(
                    **model_params
                )
                st.success("Successfully create model!")

with tab3:
    st.subheader("Controller Definition")
    c1, c2 = st.columns(2)
    with c1:
        activation_map = {
            "ReLU": torch.nn.ReLU,
            "ELU": torch.nn.ELU,
            "CELU": torch.nn.CELU,
            "Tanh": torch.nn.Tanh,
            "Sigmoid": torch.nn.Sigmoid,
            "SiLU": torch.nn.SiLU,
            "GELU": torch.nn.GELU,
            "TahnShrink": torch.nn.Tanhshrink,
        }
        activation_id = st.selectbox(
            "Please select activation for hidden layers:",
            options=list(activation_map.keys()),
        )
        activation_signature = inspect.signature(activation_map[activation_id])
        kwargs = {}
        for k, v in activation_signature.parameters.items():
            if v.annotation == float or v.annotation == int:
                default = 0.0
                if v.default is not None:
                    default = v.default
                kwargs[v.name] = st.number_input(
                    "Please choose a value for the parameter: " + v.name, value=default
                )

        layer_sizes = st.text_area(
            label="Layer sizes: A comma seperated list of integers,"
            " indicating neurons per layer, starting from the first hidden layer (leftmost)",
            value="5,5,5",
        )
        sourcing_periods = st.number_input(
            "Number of training sourcing periods:",
            value=100,
            min_value=1,
            format="%i",
            step=1,
        )
        validation_sourcing_periods = st.number_input(
            "Number of validation sourcing periods:",
            value=10,
            min_value=1,
            format="%i",
            step=1,
        )
        epochs = st.number_input(
            "Number training epochs:", value=10, min_value=1, format="%i", step=1
        )
        seed = st.number_input("Seed:", value=4, min_value=1, format="%i", step=1)
        if layer_sizes is not None and len(layer_sizes) > 0:
            try:
                layer_sizes = list(map(lambda x: int(x), layer_sizes.split(",")))
            except Exception:
                st.error("Provided input cannot be parsed to layers.")
        st.session_state["demand_controller"] = DualSourcingNeuralController(
            hidden_layers=layer_sizes,
            activation=activation_map[activation_id](**kwargs),
        )
    with c2:
        x = torch.linspace(-10, 10, 100)
        fig = px.line(
            x=x.cpu().numpy(),
            y=activation_map[activation_id](**kwargs)(x).cpu().numpy(),
            title="Activation shape: " + activation_id,
        )
        st.plotly_chart(fig, use_container_width=True)
        # device='meta' -> no memory is consumed for visualization
        if (
            st.session_state["demand_controller"] is not None
            and st.session_state["dual_sourcing_model"] is not None
        ):
            st.info(
                "Graph plot for minibatch size: 4, click top right corner to enlrage!"
            )
            st.session_state["demand_controller"].init_layers(
                regular_lead_time=regular_lead_time,
                expedited_lead_time=expedited_lead_time,
            )
            inv_size = torch.Size([4, 1])
            input_sizes = [inv_size]
            input_sizes.append(torch.Size([4, max(regular_lead_time, 1)]))
            input_sizes.append(torch.Size([4, max(regular_lead_time, 1)]))
            model_graph = draw_graph(
                st.session_state["demand_controller"], input_size=input_sizes
            )
            g = model_graph.visual_graph
            g.attr("graph", rankdir="LR")
            st.graphviz_chart(g)
        elif st.session_state["dual_sourcing_model"] is None:
            st.warning(
                "Please define a dual sourcing model to generate NN architecture graph!"
            )

    def click(progress_bar):
        st.session_state["training"] += 1
        if os.path.exists("runs/dual_sourcing_model"):
            shutil.rmtree("runs/dual_sourcing_model")

        def progress_update(i):
            progress_bar.progress(
                i / float(epochs), "Training Epochs: " + str(i) + "/" + str(epochs)
            )

        st.session_state["trainingnow"] = True

        st.session_state["demand_controller"].fit(
            sourcing_model=st.session_state["dual_sourcing_model"],
            sourcing_periods=sourcing_periods,
            validation_sourcing_periods=validation_sourcing_periods,
            epochs=epochs,
            tensorboard_writer=SummaryWriter("runs/dual_sourcing_model"),
            seed=seed,
            progress_update=progress_update,
        )
        st.session_state["trainingnow"] = False
        st.success("Training Complete!")

    cf1, cf2 = st.columns([1, 4])
    with cf1:
        pressed = st.button("Fit Controller")
    with cf2:
        if pressed:
            progress_bar = st.progress(0.0, "Training Epochs: ")
            click(progress_bar)

with tab4:
    if (
        st.session_state["demand_controller"] is not None
        and st.session_state["dual_sourcing_model"] is not None
        and st.session_state["demand_generator"] is not None
    ):
        if (
            os.path.exists("runs/dual_sourcing_model")
            and st.session_state["training"] > 0
        ):
            try:
                t4c1, t4c2 = st.columns(2)
                with t4c1:
                    tsb_df = tflog2pandas("runs/dual_sourcing_model").pivot(
                        index="step", columns="metric", values="value"
                    )
                    tsb_df.columns = list(
                        map(
                            lambda x: x.replace("Avg. cost per period/", ""),
                            tsb_df.columns,
                        )
                    )
                    st.plotly_chart(
                        px.line(tsb_df, title="Learning Curves").update_layout(
                            yaxis_title="Avg. Cost per Period", xaxis_title="Epoch"
                        )
                    )
                with t4c2:
                    (
                        past_inventories,
                        past_regular_orders,
                        past_expedited_orders,
                    ) = st.session_state["demand_controller"].simulate(
                        sourcing_model=st.session_state["dual_sourcing_model"],
                        sourcing_periods=sourcing_periods,
                    )
                    df_past = pd.DataFrame(
                        {
                            "Inventory": past_inventories,
                            "Regular Orders": past_regular_orders,
                            "Expedited Orders": past_expedited_orders,
                        },
                    )
                    fig = px.line(df_past).update_layout(
                        xaxis_title="Periods",
                        yaxis_title="# Units",
                        title="Sample Optimization Trajectory",
                    )
                    st.plotly_chart(fig)
            except Exception:
                st.warning(
                    "No available model, please make sure you submit the previous steps!"
                )
    else:
        if st.session_state["demand_generator"] is None:
            st.warning(
                "No demand generator is chosen trained for current configuration, please define one."
            )
        if st.session_state["dual_sourcing_model"] is None:
            st.warning(
                "No sourcing model is for current configuration, please define one."
            )
        if st.session_state["demand_controller"] is None:
            st.warning(
                "No model chosen trained for current configuration, please go to previous step and fit a model."
            )
