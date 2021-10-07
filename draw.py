import altair as alt
import pandas as pd
from datetime import date

from vega_datasets import data

import utils
from annotated_text import annotated_text, annotation

import streamlit as st
import types

alt.data_transformers.enable('csv')


@st.cache(show_spinner=False, suppress_st_warning=True)
def top_row(c1, c2, c3):
    df = pd.DataFrame({'day_posted': [0]})
    dummy_chart = alt.Chart(df).mark_geoshape().encode(
        x='day_posted:T',
        y='day_posted:T',
        opacity=alt.value(0)
    ).properties(
        width=75
    )

    return alt.hconcat(c1, dummy_chart, c2, dummy_chart, c3, center=True).configure_view(
        stroke=None
    ).configure_axis(
        labelFontSize=utils.SMALL_FONT_SIZE,
        titleFontSize=utils.BIG_FONT_SIZE,
    ).configure_legend(
        gradientLength=400,
        labelFontSize=utils.SMALL_FONT_SIZE,
        titleFontSize=utils.BIG_FONT_SIZE,
        orient='top',
        title=None
    )


def templates(directory, df, labels, is_infoshield):
    ''' draw annotated text
        :param directory:   directory to look for InfoShield templates in
        :return:            altair annotated text '''

    if is_infoshield:
        to_write = utils.get_all_template_text(
            directory, labels)
    else:
        to_write = ['{}:<br>{}'.format(*tup)
                    for tup in df[['title', 'body']].values]
        to_write = [annotation(
            text + '<br>', background_color='#f9f9f9', font_size='20px') for text in to_write]

    annotated_text(*to_write,
                   scrolling=True,
                   height=900,
                   )


@st.cache(hash_funcs={alt.vegalite.v4.api.Selection: lambda x: x.name}, allow_output_mutation=True, suppress_st_warning=True)
def map(subdf, top_map, micro_cluster_selector, date_range):
    ''' generate map with ad location data
        :param df:  Pandas DataFrame with latitude, longitude, and count data
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

    agg_df = utils.aggregate_locations(df)
    center, scale = utils.get_center_scale(agg_df.lat, agg_df.lon)
    domain = [agg_df['count'].min(), agg_df['count'].max()]

    scatter = alt.Chart(agg_df).transform_aggregate(
        groupby=['location'],
        count='sum(count)',
        lat='mean(lat)',
        lon='mean(lon)'
    ).mark_circle(
        color=utils.LOCATION_COLOR,
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


def bubble_chart(df, y, facet, tooltip):
    ''' create bubble chart
        :param df:      Pandas DataFrame to display
        :param y:       column of DataFrame to use for bubble size
        :param facet:   column of DataFrame to create facet with
        :param tooltip: list of DataFrame columns to include in tooltip
        :return:        altair bubble chart '''
    return alt.Chart(df).mark_circle().encode(
        x=alt.X('days', axis=alt.Axis(grid=True)),
        y=alt.Y(y, axis=alt.Axis(grid=False, labels=False), title=None),
        color=alt.value('#17becf'),
        row=alt.Row(facet, title=None, header=alt.Header(labelAngle=-45)),
        tooltip=tooltip,
        size=alt.Size(y, scale=alt.Scale(range=[100, 500]))
    ).properties(
        width=450,
        height=400 / len(df)
    ).configure_facet(
        spacing=5
    ).configure_view(
        stroke=None
    )


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def strip_plot(df, micro_cluster_selector, y, facet, tooltip, sort=None, show_labels=True, colorscheme='blues'):
    ''' create strip plot with heatmap
        :param df:      Pandas DataFrame to display
        :param y:       column of DataFrame to use for bubble size
        :param facet:   column of DataFrame to create facet with
        :param tooltip: list of DataFrame columns to include in tooltip
        :return:        altair strip plot '''

    thickness = 1500 / (max(df.days.dt.day) - min(df.days.dt.day) + 1) / 10
    sort_ = sorted(list(df[facet.split(':')[0]].unique()))

    max_val = max(df[y.split(':')[0]])
    print(max_val)

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
                              format=utils.DATE_FORMAT),
                title=''),
        y=alt.Y(facet,
                axis=alt.Axis(grid=False, domain=False, title='', labelPadding=5,
                              tickWidth=0, labelFontSize=utils.BIG_FONT_SIZE),
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
        labelFontSize=utils.SMALL_FONT_SIZE,
        titleFontSize=utils.BIG_FONT_SIZE,
    ).configure_legend(
        gradientLength=400,
        gradientThickness=5,
        labelFontSize=utils.SMALL_FONT_SIZE,
        titleFontSize=utils.BIG_FONT_SIZE,
        orient='top',
        title=None,
        tickCount=2
    )


def bar_chart(data, column):
    ''' create bar charts for displaying categorical data
        :param data:    data from which to display
        :param column:  column on which to create histogram
        :return:        altair bar plot '''

    return alt.Chart(data).mark_area().encode(
        x=alt.X(column, sort='-y', axis=alt.Axis(labels=False, grid=False)),
        y=alt.Y('count():Q', axis=alt.Axis(grid=False),
                title='Clusters (ordered by size)'),
        tooltip=[alt.Tooltip('count()', title='Number of ads in cluster')]
    ).properties(
        width=600,
        height=400
    )


def timeline(data, date_col='day_posted:T'):
    ''' create timeline for # ads each day
        :param data:    data from which to display
        :return:        altair timeline '''

    date_s = date_col.split(':')[0]

    return alt.Chart(data).transform_aggregate(
        num_ads='count()',
        groupby=[date_s]
    ).transform_filter(
        alt.datum.num_ads > 1
    ).mark_point().encode(
        x=alt.X(date_col, title='Day', axis=alt.Axis(grid=False)),
        y=alt.Y('num_ads:Q', title='Number of ads', axis=alt.Axis(grid=False)),
        # tooltip=[alt.Tooltip(date_col, title='Day'), alt.Tooltip('num_ads:Q', title='Number of ads')]
    ).properties(
        width=700,
        height=400
    )


def location_timeline(data, date_col='day_posted:T'):
    ''' create timeline for # unique locations each day
        :param data:    data from which to display
        :return:        altair timeline '''

    date_s = date_col.split(':')[0]

    return alt.Chart(data).transform_aggregate(
        num_locs='distinct(city_id)',
        groupby=[date_s]
    ).mark_point().encode(
        x=alt.X(date_col, title='Day', axis=alt.Axis(grid=False)),
        y=alt.Y('num_locs:Q', title='Number of locations',
                axis=alt.Axis(grid=False)),
        tooltip=[alt.Tooltip(date_col, title='Day'), alt.Tooltip(
            'num_locs:Q', title='Number of locations')]
    ).properties(
        width=700,
        height=400
    )


def contact_bar_chart(data, col):
    ''' create bar chart for metadata information
        :param data:    data from which to display
        :param col:     column name for metadata
        :return:        altair bar chart '''

    return alt.Chart(data).mark_bar().encode(
        x=alt.X('size:Q'),
        y=alt.Y(col, sort='-x'),
        color=alt.value('#40bcc9')
    ).properties(
        width=500,
        height=400
    )


@st.cache(hash_funcs={alt.vegalite.v4.api.Selection: lambda x: x.name}, allow_output_mutation=True, suppress_st_warning=True)
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

    domain = list(utils.STAT_TO_COLOR.keys())
    range_ = list(utils.STAT_TO_COLOR.values())

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
                              format=utils.DATE_FORMAT),
                title=''),
        y=alt.Y('total:Q',
                impute=alt.ImputeParams(
                    value=0, keyvals=[min(df.days), max(df.days)]),
                axis=alt.Axis(grid=False, tickCount=1), title=''),
        color=alt.value(utils.CLUSTER_COLOR),
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
        labelFontSize=utils.SMALL_FONT_SIZE,
        titleFontSize=utils.BIG_FONT_SIZE,
    ).configure_legend(
        gradientLength=400,
        labelFontSize=utils.SMALL_FONT_SIZE,
        titleFontSize=utils.BIG_FONT_SIZE,
        orient='top',
        title=None
    )
