import os
import streamlit as st

import draw
import utils.format
import utils.backend
import altair as alt
import pandas as pd
alt.data_transformers.enable('csv')


def gen_page_content(df):
    ''' create Streamlit page
        :param df:  pandas DataFrame containing ad data '''

    # if we've processed all clusters, we show a static end page
    if st.session_state.is_stop:
        st.header("You've finished all examples from this dataset. Thank you!")
        st.balloons()
        return

    # feature generation
    subdf = utils.backend.get_subdf(df, st.session_state.cluster)
    header_stats, micro_cluster_features = utils.backend.feature_extract(
        subdf)

    utils.format.write_border(header_stats)

    mcs = sorted(['c{}'.format(val)
                 for _, val in utils.format.SUBSCRIPT_DICT.items()])

    left_col, _, mid_col, _, right_col = st.columns((1, 0.05, 1, 0.05, 1))

    # strip plot with heatmap
    with left_col:
        st.header('**# Ads over time**: one row is one micro-cluster')
        micro_cluster_selector = st.multiselect(
            'Pick a subset of micro-clusters to inspect', mcs)

        top_n_params, chart_params = utils.backend.BY_CLUSTER_PARAMS
        top_df, top_map = utils.backend.top_n(micro_cluster_features, **top_n_params)
        c1 = draw.strip_plot(top_df, micro_cluster_selector, **chart_params)
        st.write(c1, use_container_width=True)

    # display features over time, aggregated forall clusters
    with mid_col:
        st.header('**Metadata over time** of meta-cluster')
        c2 = draw.stream_chart(top_df, micro_cluster_selector)
        st.write(c2, use_container_width=True)

    # show map of ad locations
    with right_col:
        st.header('**Geographical spread of ads**')
        date_range = pd.date_range(min(subdf.days), max(
            subdf.days)).strftime(utils.format.DATE)

        c3 = draw.map(subdf, top_map, micro_cluster_selector,
                      (date_range[0], date_range[-1]))
        st.write(c3, use_container_width=True)

    # template / ad text visualization
    if len(micro_cluster_selector):
        st.header(
            '**Ad text** organized by micro-cluster, for {}'.format(
                ', '.join(micro_cluster_selector)))
    else:
        st.header(
            '**Template text:** select a micro-cluster to see actual ads')

    left_col, _, right_col = st.columns((4, 0.05, 1.2))
    with left_col:
        is_infoshield = True

        start_path = '../InfoShield/results/'
        if not os.path.exists(start_path):
            st.warning(
                'We cannot find InfoShield results for this data, so only the ad text is displayed.')
            is_infoshield = False

        if len(micro_cluster_selector):
            labels = {v: k for k, v in top_map.items(
            ) if v in micro_cluster_selector}
        else:
            labels = {v: k for k, v in top_map.items()}
        draw.templates(
            start_path, subdf, labels, is_infoshield)

    # labeling table
    labels = []
    classes = ('Trafficking', 'Spam', 'Scam', 'Massage parlor', 'Benign')
    options = ('1: Very unlikely', '2: Unlikely',
               '3: Unsure', '4: Likely', '5: Very likely')

    with right_col:
        with st.form(key='labeling'):
            for index, cluster_type in enumerate(classes):
                st.write(
                    f'<p class="label_button">{cluster_type}</p>', unsafe_allow_html=True)
                labels.append(
                    st.select_slider(
                        cluster_type,
                        options,
                        key=str(index),
                        value=options[0],
                        label_visibility='hidden'))

            st.form_submit_button('Next meta-cluster',
                                  on_click=utils.backend.write_labels, args=([filename]))

    st.session_state.labels = {class_: int(label.split(
        ':')[0]) for class_, label in zip(classes, labels)}


# Generate content for app
st.set_page_config(layout='wide', page_title='Meta-Clustering Classification')


file_path = './data/synthetic_data.csv'

with st.spinner('Processing data...'):
    filename = file_path
    columns = ['image_id', 'phone', 'email', 'social']

    try:
        df = utils.backend.read_csv(filename)
    except:
        st.write('Please generate synthetic data first!')

    if 'is_first' not in st.session_state:
        st.session_state.is_first = True
        st.session_state.index = 0
        st.session_state.is_stop = False
        st.session_state.labels = {}
        graph = utils.backend.construct_metaclusters(
            utils.backend.filename_stub(filename), df, columns)
        st.session_state.gen_clusters = utils.backend.gen_ccs(graph)
        st.session_state.cluster = next(st.session_state.gen_clusters)
        st.session_state.is_first = False


page_opts = ('Landing page', 'Labeling page')

gen_page_content(df)
