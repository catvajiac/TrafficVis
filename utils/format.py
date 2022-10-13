import streamlit as st
import pandas as pd
import os
import pickle as pkl
from annotated_text import annotation


BIG_FONT_SIZE = 32
SMALL_FONT_SIZE = 28

# these are slightly different colors than the normal colors for these stats
# so that they pop on a dark background
STAT_TO_HEADER_COLOR = {
    'phones': '#d6d81b',
    'emails': '#ff090a',
    'ads': '#1d9bff',
    'micro-clusters': '#ff7f0eff',
    'locations': '#2abc29',
    'images': '#2cd6eb',
    'social accts': '#ac57ff'
}

STAT_TO_COLOR = {
    'phones': '#bcbd22ff',
    'emails': '#d62728ff',
    'ads': '#1f77b4ff',
    'micro-clusters': '#ff7f0eff',
    'images': '#17becfff',
    'location': '#2ca02cff',
    'social accts': '#9467bdff'
}



FEATURE_RENAMING = {
    'ad_id': 'ads',
    'phone': 'phones',
    'image_id': 'images',
    'email': 'emails',
    'social': 'social accts',
}

DATE =  "%e %b %y"

SUBSCRIPT_DICT = {
    '0': '₀',
    '1': '₁',
    '2': '₂',
    '3': '₃',
    '4': '₄',
    '5': '₅',
    '6': '₆',
    '7': '₇',
    '8': '₈',
    '9': '₉'
}

def write_border(stats):
    ''' writes the top border for the interface
        :param stats: dictionary of name: (count, unique_count) for each stat
    '''
    stats_html = '\n'.join(header_stats(name, *tup) for name, tup in stats.items())
    
    # read css file
    with open('config/main.css') as f:
        css = ''.join(f.readlines())

    # create custom banner for top of interface
    header_html = f'''
        <div class="header">
            <div id="title">
                <p id='ht'>Human Trafficking</p>
                <h1>Suspicious Meta-Cluster #{st.session_state.index+1}</h1>
            </div>
            {stats_html}
        </div>'''

    st.write(f'<style>{css}</style>', unsafe_allow_html=True)
    st.write(header_html, unsafe_allow_html=True)


def header_stats(name, count, unique):
    ''' given a stat for the header, create the html templating that displays them
        :param name:    name of stat to display
        :param count:   number of occurances of stat
        :param unique:  number of unique occurances of stat

    '''

    if name in FEATURE_RENAMING:
        name = FEATURE_RENAMING[name]

    return f'''
        <div class="stat", style="color: {STAT_TO_HEADER_COLOR[name]};">
            <p class="stat_name">{name.upper()}</p> 
            <p class='stat_number'>{count}</p>
            {f'<p class="stat_unique">{unique}</p>' if unique != '--' else ''}
        </div>
    '''


@st.cache(suppress_st_warning=True)
def location(city, country):
    ''' make pretty location string based on city, country
        :param city:    city_id as specified by Marinus
        :param country: country_id as specified by Marinus
        :return:        string of format {state}, {country} '''
    cities_df = pd.read_csv('./data/metadata/cities.csv')
    countries_df = pd.read_csv('./data/metadata/countries.csv')

    # should only happen if nan
    if country not in countries_df.id or city not in cities_df.id:
        return ''

    country_str = countries_df[countries_df.id == country].code.values[0]
    city_str = cities_df[cities_df.id == city].name.values[0]
    return ', '.join([city_str, country_str])

def stat(s):
    ''' prettify a stat string for display
        :param s:  string to prettify
        :return     string with spaces and plural '''
    if s.endswith('id'):
        s = s[:-3]
    if s == 'social':
        s = 'social acct'
    return '{}s'.format(s.replace('_', ' '))

# Text annotation utils
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_all_template_text(base_dir, labels):
    ''' check a directory for all possible templates and annotate them
        :param base_dir:    directory to check for subdirs containing templates
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
        :param i:           template number
        :param template:    list of tokens in template
        :param ads:         list of tuples of form (type_index, token)
        :label_name:        TODO
        :templates_only:    TODO
        :return:            string to write with annotated_text for this particular template '''
    index_to_color = {
        -1: ('slot', '#faa'),
        0:  ('const', '#fffae6'),
        1:  ('sub', '#8ef'),
        2:  ('del', '#aaa'),
        3:  ('ins', '#afa'),
    }

    to_write = [f'<p class=ad_text><b>{label_name}:</b> {template}</p><br>']

    if not templates_only:
        for ad_index, ad in enumerate(ads):
            if ad_index >= 20:
                break
            to_write.append(f'<p><b> Ad #{ad_index+1}:</b>')
            prev_type = None
            for color_i, token in ad:
                if token == ' ':
                    continue

                curr_type, color = index_to_color[color_i]

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
        if type(tup) == str:
            annotated.append(f"<span style='font-size: {BIG_FONT_SIZE}px; padding: 5px'>{tup}</span>")
            continue

        token, curr_type, color = tup
        annotated.append(annotation(
            token, curr_type, background_color=color, font_size=f'{BIG_FONT_SIZE}px'))

    return annotated