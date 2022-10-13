import altair as alt
import pandas as pd

from vega_datasets import data

import utils.backend
import utils.format
from annotated_text import annotated_text, annotation

import streamlit as st


alt.data_transformers.enable('csv')

PLOT_HASH_FUNCS = {
    alt.vegalite.v4.api.Selection: lambda x: x.name
}

def templates(directory, df, labels, is_infoshield):
    ''' draw annotated text, handling whether infoshield data exists or not
        :param directory:   directory to look for InfoShield templates in
        :df:                dataframe of text to write
        :labels:            names of cluster ids that are relevant to meta-cluster
        :is_infoshield:     boolean representing whether infoshield data exists for cluster
        :return:            altair annotated text '''

    if is_infoshield:
        to_write = utils.format.get_all_template_text(directory, labels)
    else:
        args = {
            'background_color': '#f9f9f9',
            'font_size': '20px'
        }
        to_write = [annotation(f'{t}:<br>{b}<br>', **args) for t,b in df[['title', 'body']].values]

    annotated_text(*to_write,
                   scrolling=True,
                   height=900,
                   )


@st.cache(hash_funcs=PLOT_HASH_FUNCS, allow_output_mutation=True, suppress_st_warning=True)
def map(subdf, top_map, micro_cluster_selector, date_range):
    ''' generate map with ad location data
        :param subdf:  Pandas DataFrame with latitude, longitude, and count data
        :param top_map: TODO
        :param micro_cluster_selector: TODO
        :param date_range: TODO
        :return:    altair map with ad counts displayed '''

    if len(micro_cluster_selector):
        top_map = {k: v for k, v in top_map.items(
        ) if v in micro_cluster_selector}
    subdf = subdf[subdf['micro-clusters'].isin(top_map.keys())]
    df = subdf[['ad_id', 'days', 'lat', 'lon', 'location']].copy()
    df['micro-clusters'] = subdf['micro-clusters'].apply(
        lambda val: top_map[val])

    date_range = pd.date_range(*date_range)
    df = df[((df.lat != 1) | (df.lon != 1)) & (df.days.isin(date_range))]

    countries = alt.topo_feature(data.world_110m.url, 'countries')

    base = alt.Chart(countries).mark_geoshape(
        fill='#eeeeee',
        stroke='#DDDDDD'
    ).properties(
        height=650,
        width=1000)

    agg_df = utils.backend.aggregate_locations(df)
    center, scale = utils.backend.get_center_scale(agg_df.lat, agg_df.lon)
    domain = [agg_df['count'].min(), agg_df['count'].max()]

    scatter = alt.Chart(agg_df).transform_aggregate(
        groupby=['location'],
        count='sum(count)',
        lat='mean(lat)',
        lon='mean(lon)'
    ).mark_circle(
        color=utils.format.STAT_TO_COLOR['location'],
        fillOpacity=.5,
    ).encode(
        size=alt.Size('count:Q', scale=alt.Scale(
            domain=domain), legend=None),
        longitude='lon:Q',
        latitude='lat:Q',
        tooltip=['location', 'count']
    )

    return (base + scatter).project(
        'equirectangular',
        center=center,
        scale=scale
    )


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def strip_plot(df, micro_cluster_selector, y, facet, tooltip, colorscheme='blues'):
    ''' create strip plot with heatmap
        :param df:      Pandas DataFrame to display
        :param y:       column of DataFrame to use for bubble size
        :param facet:   column of DataFrame to create facet with
        :param tooltip: list of DataFrame columns to include in tooltip
        :return:        altair strip plot '''

    thickness = 1500 / (max(df.days.dt.day) - min(df.days.dt.day) + 1) / 10
    sort_ = sorted(list(df[facet.split(':')[0]].unique()))

    max_val = max(df[y.split(':')[0]])

    def gen_color():
        blue = alt.Color(
            y+':Q', scale=alt.Scale(scheme=colorscheme, domain=[0, max_val]))
        grey = alt.value('lightgray')
        pred = alt.FieldOneOfPredicate(
            'micro-clusters', micro_cluster_selector)

        if len(micro_cluster_selector):
            return alt.condition(pred, blue, grey)

        return blue

    return alt.Chart(df).mark_tick(thickness=thickness, lineHeight=150).encode(
        x=alt.X('days:T',
                axis=alt.Axis(grid=False, tickCount=5,
                              format=utils.format.DATE),
                title=''),
        y=alt.Y(facet,
                axis=alt.Axis(grid=False, domain=False, title='', labelPadding=5,
                              tickWidth=0, labelFontSize=utils.format.BIG_FONT_SIZE),
                sort=sort_,
                ),
        color=gen_color(),
        tooltip=tooltip
    ).properties(
        width=1000,
        height=600
    ).configure_view(
        stroke=None
    ).configure_axis(
        labelFontSize=utils.format.SMALL_FONT_SIZE,
        titleFontSize=utils.format.BIG_FONT_SIZE,
    ).configure_legend(
        gradientLength=400,
        gradientThickness=5,
        labelFontSize=utils.format.SMALL_FONT_SIZE,
        titleFontSize=utils.format.BIG_FONT_SIZE,
        orient='top',
        title=None,
        tickCount=2
    )



@st.cache(
    hash_funcs={alt.vegalite.v4.api.Selection: lambda x: x.name},
    allow_output_mutation=True,
    suppress_st_warning=True
)
def stream_chart(df, micro_cluster_selector):
    def gen_cutoff_str(cutoff_day, op):
        yr = 'year(datum.days)'
        mon = 'month(datum.days)'
        day = 'day(datum.days)'

        return '({yr} {op} {cutoff_yr}) | ({yr} == {cutoff_yr}) & ({mon} {op} {cutoff_mon}) | ({yr} == {cutoff_yr}) & ({mon} == {cutoff_mon}) & ({day} {op} {cutoff_day})'.format(
            yr=yr, mon=mon, day=day, cutoff_yr=cutoff_day.year, cutoff_mon=cutoff_day.month, cutoff_day=cutoff_day.day, op=op
        )

    if len(micro_cluster_selector):
        df = df[df['micro-clusters'].isin(micro_cluster_selector)]

    # handle missing days in df
    days = pd.to_datetime(pd.date_range(
        min(df.days), max(df.days)).date)

    missing_days = set(days) - set(df.days)
    cutoff_day = days[int(len(days) * 5 / 8)]

    impute_df = []
    for day in missing_days:
        for micro_cluster in df['micro-clusters'].unique():
            row = {k: 0 for k in df.columns if k not in (
                'micro-clusters', 'days')}
            row['micro-clusters'] = micro_cluster
            row['days'] = day
            impute_df.append(row)

    bot_df = pd.concat([df, pd.DataFrame({'days': list(days)})])
    top_df = pd.concat([df, pd.DataFrame(impute_df)])

    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection_single(
        nearest=True,
        on='mouseover',
        fields=['days'],
        empty='none'
    )

    domain = list(utils.format.STAT_TO_COLOR.keys())
    range_ = list(utils.format.STAT_TO_COLOR.values())

    # The basic line
    line = alt.Chart(top_df).transform_fold(
        ['ads', 'images', 'phones', 'location',
            'micro-clusters', 'social accts', 'emails'],
        as_=['variable', 'value']
    ).transform_aggregate(
        groupby=['days', 'variable'],
        total='sum(value)'
    ).mark_line(interpolate='step-before').encode(
        x=alt.X('days:T',
                axis=alt.Axis(grid=False, labels=False, title='')),
        y=alt.Y('total:Q',
                impute=alt.ImputeParams(
                    value=0, keyvals=[min(df.days), max(df.days)]),
                axis=alt.Axis(grid=False, tickCount=5), title=''),
        color=alt.Color('variable:N', legend=None,
                        scale=alt.Scale(domain=domain, range=range_)),
        opacity=alt.value(0.5)
    ).transform_filter(
        alt.datum.variable != 'micro-clusters'
    )

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(top_df).transform_fold(
        ['ads', 'images', 'phones', 'location', 'micro-clusters'],
        as_=['variable', 'value']
    ).transform_aggregate(
        groupby=['days', 'variable'],
        total='sum(value)'
    ).mark_point().encode(
        x='days:T',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    left_text = line.mark_text(
        align='left',
        dx=5, dy=-5,
    ).encode(
        text=alt.condition(nearest, 'label:N', alt.value(' ')),
        size=alt.value(20),
        opacity=alt.value(1)
    ).transform_calculate(
        label='datum.total + " " + datum.variable'
    ).transform_filter(
        (alt.datum.total > 0)
    ).transform_filter(
        gen_cutoff_str(cutoff_day, '<')
    )

    right_text = line.mark_text(
        align='right',
        dx=5, dy=-5,
    ).encode(
        text=alt.condition(nearest, 'label:N', alt.value(' ')),
        size=alt.value(20),
        opacity=alt.value(1)
    ).transform_calculate(
        label='datum.total + " " + datum.variable'
    ).transform_filter(
        (alt.datum.total > 0)
    ).transform_filter(
        gen_cutoff_str(cutoff_day, '>')
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(top_df).mark_rule(color='gray').encode(
        x='days:T',
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    c1 = alt.layer(
        line, rules, selectors, points, left_text, right_text
    ).properties(
        width=1000,
        height=500
    )

    # The basic line
    ad_line = alt.Chart(bot_df).transform_aggregate(
        groupby=['days'],
        total='distinct(micro-clusters)'
    ).transform_calculate(
        total='datum.total-1'
    ).mark_line(interpolate='step-before').encode(
        x=alt.X('days:T',
                axis=alt.Axis(grid=False, tickCount=5,
                              format=utils.format.DATE),
                title=''),
        y=alt.Y('total:Q',
                impute=alt.ImputeParams(
                    value=0, keyvals=[min(df.days), max(df.days)]),
                axis=alt.Axis(grid=False, tickCount=1), title=''),
        color=alt.value(utils.format.STAT_TO_COLOR['micro-clusters']),
        opacity=alt.value(0.5)
    )

    # Draw points on the line, and highlight based on selection
    ad_points = ad_line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    left_text = ad_line.mark_text(
        align='left',
        dx=5, dy=-5,
    ).encode(
        text=alt.condition(nearest, 'label:N', alt.value(' ')),
        size=alt.value(20),
        opacity=alt.value(1)
    ).transform_calculate(
        label='datum.total + " micro-clusters"'
    ).transform_filter(
        alt.datum.total > 0
    ).transform_filter(
        gen_cutoff_str(cutoff_day, '<')
    )

    right_text = ad_line.mark_text(
        align='right',
        dx=5, dy=-5,
    ).encode(
        text=alt.condition(nearest, 'label:N', alt.value(' ')),
        size=alt.value(20),
        opacity=alt.value(1)
    ).transform_calculate(
        label='datum.total + " micro-clusters"'
    ).transform_filter(
        alt.datum.total > 0
    ).transform_filter(
        gen_cutoff_str(cutoff_day, '>')
    )

    # Put the five layers into a chart and bind the data
    c2 = alt.layer(
        ad_line, rules, selectors, ad_points, left_text, right_text
    ).properties(
        width=1000,
        height=75
    )

    return alt.vconcat(c1, c2).configure_view(
        stroke=None
    ).configure_axis(
        labelFontSize=utils.format.SMALL_FONT_SIZE,
        titleFontSize=utils.format.BIG_FONT_SIZE,
    ).configure_legend(
        gradientLength=400,
        labelFontSize=utils.format.SMALL_FONT_SIZE,
        titleFontSize=utils.format.BIG_FONT_SIZE,
        orient='top',
        title=None
    )
