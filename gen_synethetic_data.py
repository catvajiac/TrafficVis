import datetime as dt
import pandas as pd
import random as rd

from essential_generators import DocumentGenerator



def get_city_ids():
    df = pd.read_csv('data/metadata/cities.csv')
    return df[df.country_id == 2].id.values

def gen_cluster(start_id, num_ads, label_num, city_ids, metadata_dict):
    # you need: ad_id title body phone city_id country_id date_posted

    def get_entry(col):
        return ';'.join([str(rd.choice(metadata_dict[col])) for _ in range(rd.randrange(1, 5))])

    return pd.DataFrame({
        'ad_id': range(start_id, start_id+num_ads),
        'title':        [gen.sentence() for _ in range(num_ads)],
        'body':         [gen.paragraph() for _ in range(num_ads)],
        'phone':        get_entry('phones'),
        'email':        get_entry('emails'),
        'social':       get_entry('social'),
        'image_id':     get_entry('images'),
        'city_id':      [rd.choice(city_ids) for _ in range(num_ads)],
        'country_id':   [2 for _ in range(num_ads)],
        'day_posted':   [dt.date(2021, 1, 1) + dt.timedelta(days=rd.randrange(0, 30)) for _ in range(num_ads)],
        'LSH label':    [label_num for _ in range(num_ads)]
    })


dfs = []
num_ads = 0
cluster_label = 0
city_ids = get_city_ids()

for metacluster in range(2):
    gen = DocumentGenerator()
    metadata_dict = {
        'phones': [gen.phone() for _ in range(rd.randrange(1, 20))],
        'images':  [rd.randrange(10000, 99999) for _ in range(rd.randrange(1, 20))],
        'social':  [gen.word() for _ in range(rd.randrange(1, 20))],
        'emails':  [gen.email() for _ in range(rd.randrange(1, 20))]
    }
    for _ in range(rd.randrange(10, 15)):
        cluster_size = rd.randrange(20, 100)
        dfs.append(gen_cluster(num_ads, cluster_size, cluster_label, city_ids, metadata_dict))
        cluster_label += 1
        num_ads += cluster_size

pd.concat(dfs).to_csv('data/synthetic_data.csv', index=False)
print('Generated as data/synthetic_data.csv')