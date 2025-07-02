import streamlit as st
import numpy as np
import pandas as pd
import altair as alt


def init_population(nb_ind, nb_gen):
    df = pd.DataFrame(index=np.arange(nb_gen), columns=np.arange(nb_ind))
    df = df.fillna(0.0)
    df.iloc[0, :] = 1.0/(nb_ind)

    return df

def simulate_generation(freq, nb_ind):
    """
    Simulate a single generation of genetic drift.
    """
    new_freq = np.zeros_like(freq)
    for i in range(len(freq)):
        # Binomial sampling
        new_freq[i] = np.random.binomial(nb_ind, freq[i]) / nb_ind
    return new_freq


def drift(df, nb_ind):
    for index,row in df.iterrows():
        if index == 0:
            continue
        df.iloc[index, :] = simulate_generation(df.iloc[index-1, :].values, nb_ind)
    
    return df


st.set_page_config(
    page_title="Pop Gen Simulator",
    page_icon=":dna:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Pop Gen Simulator")

st.write("This is a simple population genetics simulator built with Streamlit.")

col1, col2 = st.columns([1,4])

col1.header("Parameters")


nb_ind = col1.number_input(
    "Effective population size",
    min_value=1,
    max_value=1000
)

nb_gen = col1.number_input(
    "Number of generations",
    min_value=1,
    max_value=100000,
    value=1000
)



col2.header("Genetic Drift Simulation Results")

pop = init_population(nb_ind, nb_gen)
pop = drift(pop, nb_ind)


pop.index.name = "generation"
data = pop.reset_index().melt('generation')



chart = alt.Chart(data).mark_line().encode(
    x='generation',
    y=alt.Y('value').scale(domain=(0, 1)),
    color=alt.Color('variable:N').legend(None),
)

col2.altair_chart(chart, use_container_width=True)