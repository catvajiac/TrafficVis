from collections import defaultdict

import functools
import networkx as nx
import numpy as np
import os
import pickle as pkl
import pandas as pd
import streamlit as st
import types

import utils.format


# Global params

BY_CLUSTER_PARAMS = ({
    'groupby': 'micro-clusters',
    'sortby':  'ads'
}, {
    'y': 'ads',
    'facet': 'micro-clusters:N',
    'tooltip': ['days', 'ads'],
})


def write_border(stats):
    ''' writes the top border for the interface
        :param stats: dictionary of name: (count, unique_count) for each stat
    '''
    stats_html = '\n'.join(utils.format.header_stats(name, *tup) for name, tup in stats.items())
    
    # read css file
    with open('config/main.css') as f:
        css = ''.join(f.readlines())

    # create custom banner for top of interface
    header_html = utils.format.header_container(stats_html)

    st.write(f'<style>{css}</style>', unsafe_allow_html=True)
    st.write(header_html, unsafe_allow_html=True)


# Generic utils
@st.cache(suppress_st_warning=True)
def read_csv(filename, keep_cols=[], rename_cols={}):
    ''' read csv into Pandas DataFrame
        :param filename:    location of csv file
        :param keep_cols:   if specified, only keep those csv columns
        :param rename_cols: if specified, dictionary of rename mappings for csv columns
        :return:            Pandas DataFrame '''
    df = pd.read_csv(filename)
    if keep_cols:
        df = df[keep_cols]

    if rename_cols:
        df = df.rename(columns=rename_cols)

    return df


@st.cache(hash_funcs={types.GeneratorType: id}, show_spinner=False, suppress_st_warning=True)
def get_subdf(df, cluster, date_col='day_posted'):
    ''' get subset of DataFrame based on state.cluster, do location & date processing
        :param df:          DataFrame to take subset of
        :param state:       SessionState object, state.cluster shows subset to take
        :param date_col:    name of DataFrame column containing date
        :return:            subset of DataFrame with nicely formatted locatino & date'''
    subdf = df[df['LSH label'].isin(cluster)].copy()

    subdf = gen_locations(subdf)

    location_tups = subdf[['city_id', 'country_id']].values
    subdf['location'] = [utils.format.location(*tup) for tup in location_tups]

    subdf[date_col] = pd.to_datetime(
        subdf[date_col], infer_datetime_format=True)
    subdf[date_col] = subdf[date_col].dt.normalize()

    subdf = subdf.rename(
        columns={date_col: 'days', 'LSH label': 'micro-clusters'}
    ).drop(
        columns=['site_id', 'city_id', 'state_id',
                 'country_id', 'category_id', 'date_posted'],
        errors='ignore'
    )
    return subdf


def pre_process_df(df, filename, date_col='day_posted', use_cache=True):
    clean_filename = './data/{}-cleaned.csv'.format(
        os.path.splitext(os.path.basename(filename))[0])
    if use_cache and os.path.exists(clean_filename):
        copy_df = pd.read_csv(clean_filename)
        copy_df['days'] = pd.to_datetime(
            copy_df.days, infer_datetime_format=True).dt.normalize()
        return copy_df

    copy_df = df.copy()

    location_tups = df[['city_id', 'country_id']].values
    copy_df['location'] = [utils.format.location(*tup) for tup in location_tups]

    days = pd.to_datetime(
        copy_df[date_col], infer_datetime_format=True).dt.normalize()
    copy_df['days'] = days.dt.tz_localize('UTC', ambiguous=True)

    copy_df = gen_locations(copy_df)

    copy_df.to_csv(clean_filename, index=False)

    return copy_df


@st.cache(suppress_st_warning=True)  # (show_spinner=False)
def extract_field(series):
    ''' extract values from Pandas Series, where some entries represent multiple values
        :param series:  Pandas Series to get values from
        :return:        Numpy 1D array of all values (with repetitions)
        '''
    series = series.dropna()
    if not len(series):
        return series

    return np.concatenate(series.apply(lambda val: str(val).split(';')).values)


@st.cache(suppress_st_warning=True)
def filename_stub(filename):
    ''' strip path and extension from filename
        :param filename:    string of filename
        :return:            stub of filename '''
    return os.path.basename(filename).split('.')[0]


@st.cache(show_spinner=False, suppress_st_warning=True)
def basic_stats(df, cluster_label='micro-clusters'):
    ''' get basic meta-cluster level stats, not based on time
        :param df:      Pandas DataFrame representing ads from one meta-cluster
        :param cols:    columns of DataFrame containing relevant metadata
        :return:        DataFrame with metadata counts '''

    cols = ('phone', 'email', 'social', 'image_id')
    metadata = {utils.format.stat(col): [len(extract_field(df[col])), len(
        set(extract_field(df[col])))] for col in cols}
    metadata['ads'] = [len(df), '--']
    metadata['micro-clusters'] = [len(df[cluster_label].unique()), '--']
    metadata['locations'] = [len(df.location.unique()), '--']

    return metadata


@st.cache(suppress_st_warning=True)
def top_n(df, groupby, sortby, n=10):
    ''' get the top n groups from a DataFrame
        :param df:      Pandas DataFrame for one meta-cluster
        :param groupby: column from DataFrame to create groups before aggregation
        :param sortby:  column from DataFrame to aggregate & sort by
        :param n:       number of groups to return
        :return         DataFrame containing data from top n groups'''

    def to_subscript(num): return ''.join(
        [utils.format.SUBSCRIPT_DICT[s] for s in str(num)])

    df = df.reset_index()

    top_n = df.groupby(
        groupby
    ).agg(
        {sortby: 'sum', groupby: 'first'}
    ).sort_values(
        by=sortby,
        ascending=False
    ).index.values[:n]

    to_map = {num: 'c{}'.format(to_subscript(index))
              for index, num in enumerate(top_n)}

    top_df = df[df[groupby].isin(top_n)].copy()
    top_df[groupby] = top_df[groupby].map(to_map)
    return top_df, to_map


# Location data related functions
@st.cache(suppress_st_warning=True)
def get_center_scale(lat, lon):
    ''' get centering and scale parameters for map display
        :param lat: list of latitudes
        :param lon: list of longitudes
        :return:    center (midpoint) and scaling '''
    default_scale = 500
    def midpoint(lst): return (max(lst) + min(lst)) / 2

    def scale(lst, const): return const*2 / (max(lst) -
                                             min(lst)) if max(lst) - min(lst) else default_scale

    scale_lat = scale(lat, 90)
    scale_lon = scale(lon, 180)

    center = midpoint(lon), midpoint(lat)

    return center, min([scale_lat*50, scale_lon*50, default_scale])


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def gen_locations(df):
    ''' generate latitude and longitude coordinates given city_id
        :param df:  Pandas DataFrame with column "city_id" to get coordinates from
        :return:    DataFrame with latitude and longitude data '''
    cities_df = read_csv('~/grad_projects/data/aht_data/metadata/cities.csv',
                         keep_cols=['id', 'xcoord', 'ycoord'],
                         rename_cols={'xcoord': 'lat', 'ycoord': 'lon'})

    return pd.merge(df, cities_df, left_on='city_id', right_on='id', sort=False)




@st.cache(suppress_st_warning=True)
def aggregate_locations(df):
    ''' get location counts from DataFrame
        :param df:  Pandas DataFrame with column "location"
        :return     DataFrame with location count data '''

    return df.groupby(
        ['location', 'micro-clusters'],
        as_index=False
    ).agg({
        'ad_id': 'count',
        'lat': 'mean',
        'lon': 'mean'
    }).rename(
        columns={'ad_id': 'count'}
    )


# Date related
@st.cache(suppress_st_warning=True)
def extract_field_dates(df, col_name, date_col):
    ''' extract values from Pandas DataFrame with date, where one row represents multiple values
        :param df:          Pandas DataFrame to get data from
        :param col_name:    column from DataFrame with values to extract
        :param date_col:    column from DataFrame with time data
        :return:            DataFrame with each value's count, by day '''

    df = df.dropna()
    if not len(df):
        return pd.DataFrame(columns=['metadata', 'day_posted', 'count', 'type'])

    def get_data(row): return [(val, row[date_col])
                               for val in str(row[col_name]).split(';')]

    def concat_reduce(data): return functools.reduce(lambda x, y: x + y, data)

    # expand fields that have lists, so each is a row in df
    meta_df = pd.DataFrame(concat_reduce(
        df.apply(get_data, axis=1)), columns=df.columns)
    # aggregate by count
    meta_df = meta_df.groupby([col_name, date_col], as_index=False).size()
    # prettify df columns
    meta_df['type'] = col_name
    meta_df = meta_df.rename(columns={'size': 'count', col_name: 'metadata'})
    return meta_df

    # return pd.DataFrame(np.concatenate(df.apply(get_data, axis=1)), columns=df.columns)


@st.cache(show_spinner=False, suppress_st_warning=True)
def cluster_feature_extract(df, cluster_label='micro-clusters', date_col='days', loc_col='location'):
    ''' extract important time-based features for a particular cluster
        :param df:          Pandas DataFrame representing one meta-cluster
        :param date_col:    column from DataFrame representing time data
        :param loc_col:     column from DataFrame representing location (city id)
        :return tuple of DataFrames (1) by cluster, (2) for entire meta-cluster, (3) for metadata'''
    def total(series):
        return len(extract_field(series))

    agg_dict = {name: total for name in utils.format.FEATURE_RENAMING.keys()}
    agg_dict[loc_col] = lambda series: len(series.unique())

    keep_mc = df.groupby(
        cluster_label
    ).agg('count')

    df = df[df['micro-clusters'].isin(keep_mc.index.values[:10])]

    return df.groupby(
        [date_col, cluster_label],
        sort=False
    ).agg(
        agg_dict
    ).rename(
        columns=utils.format.FEATURE_RENAMING
    )

    # problem: cluster_feature_extract df doesn't have micro-clusters in it, which we are using for topn. Do we need that as arg for topn? Maybe...?


@st.cache(show_spinner=False, suppress_st_warning=True)
def feature_extract(df, cluster_label='micro-clusters', date_col='days', loc_col='location'):
    micro_cluster_features = cluster_feature_extract(
        df, cluster_label, date_col, loc_col)
    header_stats = basic_stats(df)

    return header_stats, micro_cluster_features


# Graph related utils
# @st.cache#(show_spinner=False)
def construct_metaclusters(filename, df, cols, cluster_label='LSH label'):
    ''' construct metadata graph from dataframe already split into clusters
        :param filename:        name of pandas dataframe (used for pkl filename)
        :param df:              pandas dataframe containing ad info
        :param cols:            subset of df.columns to link clusters by
        :param cluster_label:   column from df.columns containing cluster label
        :return                 nx graph, where each connected component is a meta-cluster
    '''

    path = './data/pkl_files'
    if not os.path.exists(path):
        os.mkdir(path)

    pkl_filename = f'{path}/{filename}.pkl'

    # don't recalculate if it's been done before
    if os.path.exists(pkl_filename):
        return pkl.load(open(pkl_filename, 'rb'))

    metadata_dict = defaultdict(list)
    metadata_graph = nx.Graph()

    for cluster_id, cluster_df in df.groupby(cluster_label):
        # only relevant if *_full_LSH_labels file is given (from InfoShield)
        # this would mean that it actually does not fall into any cluster
        if cluster_id == -1:
            continue

        metadata_graph.add_node(cluster_id, num_ads=len(cluster_df))

        for name in cols:
            metadata_graph.nodes[cluster_id][name] = extract_field(
                cluster_df[name])

            for elem in metadata_graph.nodes[cluster_id][name]:
                edges = [(cluster_id, node) for node in metadata_dict[elem]]
                metadata_graph.add_edges_from(edges, type=name)
                metadata_dict[elem].append(cluster_id)

    pkl.dump(metadata_graph, open(pkl_filename, 'wb'))
    return metadata_graph


def gen_ccs(graph):
    ''' return generator for connected components, sorted by size
        :param graph:   nx Graph
        :return         generator of connected components
    '''

    components = nx.connected_components(graph)
    for component in components:
        print('# clusters', len(component))
        yield component


def write_labels(filename):
    ''' write labels to csv file '''
    label_filename = f'{os.path.splitext(filename)[0]}-meta-labels.csv'

    # read previous csv if it's created, else create it
    if os.path.exists(label_filename):
        label_df = read_csv(label_filename)
    else:
        label_df = pd.DataFrame(
            columns=['meta_cluster_label', 'cluster_labels'] + list(st.session_state.labels.keys()))

    # add new row to csv
    row = st.session_state.labels.copy()
    row['meta_cluster_label'] = st.session_state.index
    row['cluster_labels'] = np.array(st.session_state.cluster)

    try:
        st.session_state.cluster = next(st.session_state.gen_clusters)
        st.session_state.index += 1
    except StopIteration:
        st.session_state.is_stop = True

    st.session_state.labels = {k: 0 for k in st.session_state.labels}

    label_df = label_df.append(row, ignore_index=True)

    label_df.to_csv(label_filename, index=False)