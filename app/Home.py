import os
import shutil
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import torch.nn
import plotly.express as px
from idinn.demand import UniformDemand, CustomDemand
from idinn.controller import DualSourcingNeuralController
from idinn.sourcing_model import DualSourcingModel
import inspect
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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

if 'training' not in st.session_state:
    st.session_state['training']=0
if 'trainingnow' not in st.session_state:
    st.session_state['trainingnow']=False

if 'demand_generator' not in st.session_state:
    st.session_state['demand_generator'] = None
if 'dual_sourcing_model' not in st.session_state:
    st.session_state['dual_sourcing_model'] = None
st.header('Inventory Informed Neural Networks')
st.markdown('Welcome to Inventory Informed Neural Networks. This application generates policies to order'
            'from expendited and regular suppliers! From the sidebar to your left, you can select a demand model, '
            'then your prefered solver and finally see the results after fitting.')

tab1, tab2, tab3, tab4 = st.tabs(["Demand Generation", "Dual Sourcing Model", "NN Controller", "Results"])
with tab1:
    submitted = submitted2 = False
    st.subheader('Demand Generation')

    demand_type = st.radio(label='Please choose demand type', options=['Uniform', 'File'])
    if demand_type == 'Uniform':
        with st.form(key='uniform_demand'):
            high = np.iinfo(np.integer).max
            low = st.number_input("Insert the minimum demand", value=0, placeholder="Type an integer...", step=1,
                                  format='%i', max_value=high)
            high = st.number_input("Insert the maximum demand", value=4, placeholder="Type an integer...", step=1,
                                   format='%i', min_value=0)
            submitted = st.form_submit_button('Generate')
            if submitted and high >= low:
                st.success("Successfully generated uniform demand within range: [" + str(np.floor(low)) + ", " + str(
                    np.floor(high)) + "]")
            elif submitted:
                st.error("Please resubmit and make sure that maximum demand is greater or equal to minimum demand.")

            st.session_state['demand_generator'] = UniformDemand(low, high)

    elif demand_type == "File":
        with st.form(key='uniform_demand'):
            uploaded_file = st.file_uploader(
                label='Please upload a single column file with demand values. Each row represents a timestep and each'
                      'element represents a demand value.')
            submitted2 = st.form_submit_button('Generate')
            if submitted2:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success('File successfuly uploaded and contrains ' + str(df.shape[0]) + ' demand points:')
                    # st.table(df)
                    st.session_state['demand_generator'] = CustomDemand(torch.tensor(df.iloc[:, 0].values))

                except Exception:
                    st.warning('Could not parse file! Please try again!')
    if st.session_state['demand_generator'] is not None:
        all_demands = []
        for i in range(100):
            all_demands.append(st.session_state['demand_generator'].sample(1).item())
        fig = px.line(y=all_demands).update_layout(xaxis_title="Periods", yaxis_title="Demand", title="Typical Demand Trajectory")
        st.plotly_chart(fig)

with tab2:
    st.subheader('Dual Sourcing Model')

    with st.form(key="Dual Sourcing Model"):
        model_params = dict(
        regular_lead_time = np.int32(
            st.number_input('Please add the regular lead time:', value=2, min_value=0, format='%i', step=1)),
        expedited_lead_time = np.int32(
            st.number_input('Please add the expedited lead time:', value=0, min_value=0, format='%i', step=1)),
        regular_order_cost = np.int32(
            st.number_input('Please add the regular order cost:', value=0, min_value=0, format='%i', step=1)),
        expedited_order_cost = np.int32(
            st.number_input('Please add the expedited order cost:', value=20, min_value=0, format='%i', step=1)),
        holding_cost = np.int32(
            st.number_input('Please add the holding cost:', value=5, min_value=0, format='%i', step=1)),
        shortage_cost = np.int32(
            st.number_input('Please add the shortage cost:', value=495, min_value=0, format='%i', step=1)),
        batch_size = np.int32(
            st.number_input('Please add the minibatch/sample size for demand trajectories:', value=16, min_value=0,
                            format='%i', step=1)),
        init_inventory = np.int32(
            st.number_input('Please add the initial inventory:', value=6, min_value=0, format='%i', step=1)),
        )
        model_params['demand_generator']=st.session_state['demand_generator']
        pressed2 = st.form_submit_button('Create Sourcing Model')

        if pressed2:
            st.session_state['dual_sourcing_model'] = DualSourcingModel(
                **model_params
            )
            st.success('Successfully create model!')


with tab3:
    st.subheader('Controller Definition')
    c1, c2 = st.columns(2)
    with c1:
        activation_map = {
            'ReLU': torch.nn.ReLU,
            'ELU': torch.nn.ELU,
            'CELU': torch.nn.CELU,
            'Tanh': torch.nn.Tanh,
            'Sigmoid': torch.nn.Sigmoid,
            'SiLU': torch.nn.SiLU,
            'GELU': torch.nn.GELU,
            'TahnShrink': torch.nn.Tanhshrink
        }
        activation_id = st.selectbox("Please select activation for hidden layers:", options=list(activation_map.keys()))
        activation_signature = inspect.signature(activation_map[activation_id])
        kwargs = {}
        for k, v in activation_signature.parameters.items():
            if v.annotation == float or v.annotation == int:
                default = 0.0
                if v.default is not None:
                    default = v.default
                kwargs[v.name] = st.number_input("Please choose a value for the parameter: " + v.name, value=default)

        layer_sizes = st.text_area(label="Layer sizes: A comma seperated list of integers,"
                                         " indicating neurons per layer, starting from the first hidden layer (leftmost)",
                                   value='5,5,5')
        sourcing_periods = st.number_input('Number of training sourcing periods:', value=100, min_value=1, format='%i', step=1)
        validation_sourcing_periods = st.number_input('Number of validation sourcing periods:', value=10, min_value=1, format='%i', step=1)
        epochs = st.number_input('Number training epochs:', value=10, min_value=1, format='%i', step=1)
        seed  = st.number_input('Seed:', value=4, min_value=1, format='%i', step=1)
        if layer_sizes is not None and len(layer_sizes) > 0:
            try:
                layer_sizes = list(map(lambda x: int(x), layer_sizes.split(',')))
            except Exception:
                st.error('Provided input cannot be parsed to layers.')
    with c2:
        x = torch.linspace(-10, 10, 100)
        fig = px.line(x=x.cpu().numpy(), y=activation_map[activation_id](**kwargs)(x).cpu().numpy(),
                      title='Activation shape: ' + activation_id)
        st.plotly_chart(fig, use_container_width=True)

    controller = DualSourcingNeuralController(hidden_layers=layer_sizes, activation=activation_map[activation_id](**kwargs))


    def click(progress_bar):
        st.session_state['training'] += 1
        if os.path.exists('runs/dual_sourcing_model'):
            shutil.rmtree('runs/dual_sourcing_model')

        def progress_update(i):
            progress_bar.progress(i / float(epochs), "Training Epochs: " + str(i) + '/' + str(epochs))
        st.session_state['trainingnow'] = True

        controller.train(
            sourcing_model=st.session_state['dual_sourcing_model'],
            sourcing_periods=sourcing_periods,
            validation_sourcing_periods=validation_sourcing_periods,
            epochs=epochs,
            tensorboard_writer=SummaryWriter("runs/dual_sourcing_model"),
            seed=seed,
            progress_update=progress_update
        )
        st.session_state['trainingnow'] = False
        st.success('Training Complete!')


    pressed = st.button('Fit Controller')

    if pressed:
        progress_bar = st.progress(0.0, "Training Epochs: ")
        click(progress_bar)

with tab4:
    if os.path.exists("runs/dual_sourcing_model") and st.session_state['training'] > 0:
        tsb_df = tflog2pandas("runs/dual_sourcing_model").pivot(index='step', columns='metric', values='value')
        tsb_df.columns = list(map(lambda x: x.replace('Avg. cost per period/', ''), tsb_df.columns))
        st.plotly_chart(px.line(tsb_df, title='Learning Curves').
        update_layout(
            yaxis_title="Avg. Cost per Period", xaxis_title="Epoch"
        )
        )
        past_inventories, past_regular_orders, past_expedited_orders = controller.simulate(
                sourcing_model=st.session_state['dual_sourcing_model'], sourcing_periods=sourcing_periods
            )
        df_past = pd.DataFrame({'Inventory' : past_inventories,
                                'Regular Orders' : past_regular_orders,
                                'Expedited Orders': past_expedited_orders}
                               )
        fig = px.line(df_past).update_layout(xaxis_title='Periods', yaxis_title='# Units')
        st.plotly_chart(fig)