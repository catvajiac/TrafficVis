import functools
from altair.utils.core import infer_dtype
import networkx as nx
import numpy as np
import os
import sys
import pandas as pd
import pickle as pkl
import streamlit as st
import types

from annotated_text import annotated_text, annotation
from collections import defaultdict


# Params

DATE_FORMAT = "%e %b %y"

BIG_FONT_SIZE = 24
SMALL_FONT_SIZE = 20

SUBSCRIPT_DICT = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
                  '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'}


# FEATURE_COLS = ('ad_id', 'email', 'image_id', 'phone', 'social')


FEATURE_RENAMING = {
    'ad_id': 'ads',
    'phone': 'phones',
    'image_id': 'images',
    'email': 'emails',
    'social': 'social accts',
}

# all colors from category10 colorscheme
LOCATION_COLOR = '#2ca02cff'    # green
AD_COLOR = '#1f77b4ff'          # blue
CLUSTER_COLOR = '#ff7f0eff'     # orange
IMAGE_COLOR = '#17becfff'       # cyan
EMAIL_COLOR = '#d62728ff'       # red
PHONE_COLOR = '#bcbd22ff'       # yellow
SOCIAL_COLOR = '#9467bdff'      # purple

STAT_TO_COLOR = {
    'phones': PHONE_COLOR,
    'emails': EMAIL_COLOR,
    'ads': AD_COLOR,
    'micro-clusters': CLUSTER_COLOR,
    'images': IMAGE_COLOR,
    'location': LOCATION_COLOR,
    'social accts': SOCIAL_COLOR
}

STAT_TO_HEADER_COLOR = {
    'phones': '#d6d81b',
    'emails': '#ff090a',
    'ads': '#1d9bff',
    'micro-clusters': CLUSTER_COLOR,
    'locations': '#2abc29',
    'images': '#2cd6eb',
    'social accts': '#ac57ff'
}


BY_CLUSTER_PARAMS = ({
    'groupby': 'micro-clusters',
    'sortby':  'ads'
}, {
    'y': 'ads',
    'facet': 'micro-clusters:N',
    'tooltip': ['days', 'ads'],
    'show_labels': False
})


BUTTON_STYLE = '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>'


def write_border(stats):
    stats_str = ''
    template = '<div class=stat, style="color: {color};">{text}</div>'
    for name, (count, unique) in stats.items():
        if name == 'image':
            name = 'images'

        if name == 'socials':
            name = 'social accts'

        text = '''<div class="stat">
            <p class='stat_name'>{}</p>
            <p class='stat_number'>{}</p>
            '''.format(name.upper(), count)

        if unique == '--' or not count and not unique:
            text += "</div>"
            stats_str += template.format(
                color=STAT_TO_HEADER_COLOR[name], text=text)
            continue

        text += '<p class="stat_unique">{} unique</p></div>'.format(unique)

        stats_str += template.format(
            color=STAT_TO_HEADER_COLOR[name], text=text)

    # inject custom banner at top of visualization
    # note: have to double braces when using .format()
    st.write('''
    <style>
        p {{
            font-size: 18px;
            margin: 0;
        }}

        iframe {{
            padding: 10px -10px 2px 0px;
            background-color: #f9f9f9;
        }}

        .header {{
            padding: 10px 16px 10px 16px;
            background: #353535;
            color: #f1f1f1;
            position: absolute;
            top: -100px;
            width: 101%;
            font-size: 30px;
        }}

        .label_button {{
            margin-bottom: -60px;
            margin-top: 0px;
            font-weight: bold;
        }}

        #ht {{
            font-size: 20px;
            margin: 0px 0px -15px 0px;
            padding: 0px;
            color: #bbbbbb;
        }}

        #title {{
            position: float;
            float: left;
        }}

        .stat {{
            margin-left: 5%;
            position: float;
            float: left;
            text-align: center;
        }}

        .stat_name {{
            font-size: 20px;
            margin: 2px 0px 0px 0px;

        }}

        .stat_number {{
            font-weight: bold;
            font-size: 28px;
            margin: -10px;
        }}

        .stat_unique {{
            color: #bbbbbb;
            margin-top: -10px;
            margin-bottom: -5px;
        }}

        h1 {{
            color: #f1f1f1;
            padding: 0;
            margin-top: 10px;
        }}

        canvas.marks {{
            max-width: 100%!important;
            height: auto!important;
        }}

        .stSlider {{
            padding: 0px 13% 0px 13%;
            font-size: 14px;
        }}

        [data-testid] {{
            font-size: 14px;
        }}

        .stForm {{
            padding: 50px;
        }}

        div.stButton > button:first-child {{
        }}

        p.ad_text {{
            font-size: 24px;
            display: inline-flex;
        }}
    </style>
    <div class="header">
        <div id="title">
            <p id='ht'>Human Trafficking</p>
            <h1>Suspicious Meta-Cluster #{meta_index}</h1>
        </div>
        {text}
    </div>
    '''.format(meta_index=st.session_state.index+1, text=stats_str), unsafe_allow_html=True)
    # filter:blur(2px);


# Generic utils
@st.cache
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

    subdf['location'] = [prettify_location(
        *tup) for tup in subdf[['city_id', 'country_id']].values]

    subdf[date_col] = pd.to_datetime(
        subdf[date_col], infer_datetime_format=True)
    subdf[date_col] = subdf[date_col].dt.normalize()

    subdf = subdf.rename(
        columns={date_col: 'days', 'LSH label': 'micro-clusters'}
    ).drop(
        columns=['site_id', 'city_id', 'state_id',
                 'country_id', 'category_id', 'date_posted']
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
    copy_df['location'] = [prettify_location(
        *tup) for tup in df[['city_id', 'country_id']].values]

    days = pd.to_datetime(
        copy_df[date_col], infer_datetime_format=True).dt.normalize()
    copy_df['days'] = days.dt.tz_localize('UTC', ambiguous=True)

    copy_df = gen_locations(copy_df)

    copy_df.to_csv(clean_filename, index=False)

    return copy_df


@st.cache  # (show_spinner=False)
def extract_field(series):
    ''' extract values from Pandas Series, where some entries represent multiple values
        :param series:  Pandas Series to get values from
        :return:        Numpy 1D array of all values (with repetitions)
        '''
    series = series.dropna()
    if not len(series):
        return series

    return np.concatenate(series.apply(lambda val: str(val).split(';')).values)


@st.cache
def pretty_s(s):
    ''' prettify a string for display
        :param s:  string to prettify
        :return     string with spaces and plural '''
    if s.endswith('id'):
        s = s[:-3]
    if s == 'social':
        s = 'social acct'
    return '{}s'.format(s.replace('_', ' '))


@st.cache
def filename_stub(filename):
    ''' strip path and extension from filename
        :param filename:    string of filename
        :return:            stub of filename '''
    return os.path.basename(filename).split('.')[0]


@st.cache(show_spinner=False)
def basic_stats(df, cluster_label='micro-clusters'):
    ''' get basic meta-cluster level stats, not based on time
        :param df:      Pandas DataFrame representing ads from one meta-cluster
        :param cols:    columns of DataFrame containing relevant metadata
        :return:        DataFrame with metadata counts '''

    cols = ('phone', 'email', 'social', 'image_id')
    metadata = {pretty_s(col): [len(extract_field(df[col])), len(
        set(extract_field(df[col])))] for col in cols}
    metadata['ads'] = [len(df), '--']
    metadata['micro-clusters'] = [len(df[cluster_label].unique()), '--']
    metadata['locations'] = [len(df.location.unique()), '--']

    return metadata


@st.cache
def top_n(df, groupby, sortby, n=10):
    ''' get the top n groups from a DataFrame
        :param df:      Pandas DataFrame for one meta-cluster
        :param groupby: column from DataFrame to create groups before aggregation
        :param sortby:  column from DataFrame to aggregate & sort by
        :param n:       number of groups to return
        :return         DataFrame containing data from top n groups'''

    def to_subscript(num): return ''.join(
        [SUBSCRIPT_DICT[s] for s in str(num)])

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
@st.cache
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


@st.cache(allow_output_mutation=True)
def gen_locations(df):
    ''' generate latitude and longitude coordinates given city_id
        :param df:  Pandas DataFrame with column "city_id" to get coordinates from
        :return:    DataFrame with latitude and longitude data '''
    cities_df = read_csv('~/grad_projects/data/aht_data/metadata/cities.csv',
                         keep_cols=['id', 'xcoord', 'ycoord'],
                         rename_cols={'xcoord': 'lat', 'ycoord': 'lon'})

    return pd.merge(df, cities_df, left_on='city_id', right_on='id', sort=False)


@st.cache
def prettify_location(city, country):
    ''' make pretty location string based on city, country
        :param city:    city_id as specified by Marinus
        :param country: country_id as specified by Marinus
        :return:        string of format {state}, {country} '''
    cities_df = read_csv('~/grad_projects/data/aht_data/metadata/cities.csv')
    countries_df = read_csv(
        '~/grad_projects/data/aht_data/metadata/countries.csv')

    # should only happen if nan
    if country not in countries_df.id or city not in cities_df.id:
        return ''
    country_str = countries_df[countries_df.id == country].code.values[0]
    city_str = cities_df[cities_df.id == city].name.values[0]
    return ', '.join([city_str, country_str])


@st.cache
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
@st.cache
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


@st.cache(show_spinner=False)
def cluster_feature_extract(df, cluster_label='micro-clusters', date_col='days', loc_col='location'):
    ''' extract important time-based features for a particular cluster
        :param df:          Pandas DataFrame representing one meta-cluster
        :param date_col:    column from DataFrame representing time data
        :param loc_col:     column from DataFrame representing location (city id)
        :return tuple of DataFrames (1) by cluster, (2) for entire meta-cluster, (3) for metadata'''
    def total(series):
        return len(extract_field(series))

    agg_dict = {name: total for name in FEATURE_RENAMING.keys()}
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
        columns=FEATURE_RENAMING
    )

    # problem: cluster_feature_extract df doesn't have micro-clusters in it, which we are using for topn. Do we need that as arg for topn? Maybe...?


@st.cache(show_spinner=False)
def feature_extract(df, cluster_label='micro-clusters', date_col='days', loc_col='location'):
    micro_cluster_features = cluster_feature_extract(
        df, cluster_label, date_col, loc_col)
    header_stats = basic_stats(df)

    return header_stats, micro_cluster_features


# Graph related utils
# @st.cache#(show_spinner=False)
def construct_metaclusters(filename, df, cols, cluster_label='LSH label'):
    ''' construct metadata graph from dataframe already split into clusters
    :param df:              pandas dataframe containing ad info
    :param cols:            subset of @df.columns to link clusters by
    :param cluster_label:   column from @df.columns containing cluster label
    :return                 nx graph, where each connected component is a meta-cluster '''

    pkl_filename = 'pkl_files/{}.pkl'.format(filename)
    if os.path.exists(pkl_filename):
        return pkl.load(open(pkl_filename, 'rb'))

    metadata_dict = defaultdict(list)
    metadata_graph = nx.Graph()

    for cluster_id, cluster_df in df.groupby(cluster_label):
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


# @st.cache(hash_funcs={types.GeneratorType: id})#, show_spinner=False)
def gen_ccs(graph):
    ''' return generator for connected components, sorted by size
        :param graph:   nx Graph
        :return         generator of connected components '''

    # components = sorted(nx.connected_components(graph), reverse=True, key=len)
    components = nx.connected_components(graph)
    for component in components:
        print('# clusters', len(component))
        yield component


# Text annotation utils
def get_all_template_text(base_dir, labels):
    ''' check a directory for all possible templates and annotate them
        :param directory:   directory to check for subdirs containing templates
        :return:            string to write with annotate_text function '''
    to_write = []
    for label_name, label in labels.items():
        directory = base_dir + str(label)

        is_first = True
        for _, folder in enumerate(os.listdir(directory)):
            result_loc = '{}/{}/text.pkl'.format(directory, folder)
            if is_first:
                is_first = False
            else:
                to_write.append('<br>')

            pickled = pkl.load(open(result_loc, 'rb'))
            to_write += get_template_text(*pickled,
                                          label_name, len(labels) == 10)
            break

    return to_write


def get_template_text(i, template, ads, label_name, templates_only):
    ''' annotate a particular template with relevant ads as calculated from InfoShield
        :param template:    list of tokens in template
        :param ads:         list of tuples of form (type_index, token)
        :param i:           template number
        :return:            string to write with annotated_text for this particular template '''
    index_to_type = {
        -1: ('slot', '#faa'),
        0:  ('const', '#fffae6'),
        1:  ('sub', '#8ef'),
        2:  ('del', '#aaa'),
        3:  ('ins', '#afa'),
    }

    to_write = [
        '<p class=ad_text><b>{}:</b> {}</p>'.format(label_name, template), '<br>']

    if not templates_only:
        for ad_index, ad in enumerate(ads):
            if ad_index >= 20:
                break
            to_write.append('<p><b> Ad #{}:</b>'.format(ad_index+1))
            prev_type = None
            for color_i, token in ad:
                if token == ' ':
                    continue

                curr_type, color = index_to_type[color_i]

                if curr_type == 'const':
                    if prev_type == 'const':
                        prev_token = to_write[-1]
                        to_write[-1] = '{} {}'.format(prev_token, token)
                    else:
                        to_write.append(token.replace(' ', ''))
                    prev_type = curr_type
                    continue

                if curr_type == prev_type:
                    prev_token = to_write[-1][0]
                    to_write[-1] = ('{} {}'.format(prev_token,
                                    token), curr_type, color)
                    continue

                prev_type = curr_type

                to_write.append((token, curr_type, color))

            to_write.append('</p>')

    # now create annotation objects. Couldn't before since we need to access prev_type, etc
    annotated = []
    for tup in to_write:
        # if tup == '<br>':
        #    #annotated.append(annotation(tup, background_color='#f9f9f9'))
        #    annotated.append('<br style="background-color: #f9f9f9">')
        # if type(tup) == str:
        #    annotated.append(annotation(tup, background_color='#f9f9f9', font_size='{}px'.format(BIG_FONT_SIZE)))
        #    continue
        if type(tup) == str:
            annotated.append(
                "<span style='font-size: 24px; padding: 5px'>{}</span>".format(tup))
            continue

        token, curr_type, color = tup
        annotated.append(annotation(
            token, curr_type, background_color=color, font_size='{}px'.format(BIG_FONT_SIZE)))

    return annotated


def write_labels(filename):
    label_filename = '{}-meta-labels.csv'.format(os.path.splitext(filename)[0])
    if os.path.exists(label_filename):
        label_df = read_csv(label_filename)
    else:
        label_df = pd.DataFrame(
            columns=['meta_cluster_label', 'cluster_labels'] + list(st.session_state.labels.keys()))

    # add new row
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
