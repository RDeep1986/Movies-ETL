#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import numpy as np


# In[2]:


file_dir ="./Resources"


# In[3]:


with open(f'{file_dir}/wikipedia.movies.json', mode='r') as file:
    wiki_movies_raw = json.load(file)


# In[4]:


len(wiki_movies_raw)


# In[5]:


# First 5 records
wiki_movies_raw[:5]


# In[6]:


# To see the Last 5 records
wiki_movies_raw[-5:]


# In[7]:


#It’s always a good idea to check records in the middle as well
#Choose a number somewhere around the halfway mark and look at a handful of records after that index.
# Some records in the middle
wiki_movies_raw[3600:3605]


# In[8]:


kaggle_metadata = pd.read_csv(f'{file_dir}/movies_metadata.csv')
ratings = pd.read_csv(f'{file_dir}/ratings.csv')


# In[9]:


#Skill Drill

kaggle_metadata.sample(n=5)


# In[10]:


wiki_movies_df = pd.DataFrame(wiki_movies_raw)


# In[11]:


wiki_movies_df.head()


# In[12]:


wiki_movies_df.columns.tolist()


# In[13]:


wiki_movies=[movie for movie in wiki_movies_raw 
             if ('Director' in movie or 'Directed by' in movie) 
             and 'imdb_link' in movie]
len(wiki_movies)


# In[14]:


wiki_movies_df = pd.DataFrame(wiki_movies)
wiki_movies_df


# In[15]:


wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie]
len(wiki_movies)


# In[16]:


wiki_movies_df = pd.DataFrame(wiki_movies)
wiki_movies_df


# In[17]:


wiki_movies_df[wiki_movies_df['Arabic'].notnull()]


# In[18]:


sorted(wiki_movies_df.columns.tolist())


# In[19]:


# Step 1
def clean_movies(movie):
    movie = dict(movie)
    alt_title = {}
    return movie


# In[20]:


#Step 2
def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune–Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        return movie


# In[21]:


def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune–Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            return movie


# In[22]:


def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune–Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)

        return movie


# In[23]:


def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune–Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles

    return movie


# In[24]:


clean_movies = [clean_movie(movie) for movie in wiki_movies]


# In[25]:


wiki_movies_df = pd.DataFrame(clean_movies)
sorted(wiki_movies_df.columns.tolist())


# In[26]:


def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune–Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles
    
    def change_column_name(old_name, new_name):
        if old_name in movie:
            movie[new_name] = movie.pop(old_name)
    change_column_name('Adaptation by', 'Writer(s)')
    change_column_name('Country of origin', 'Country')
    change_column_name('Directed by', 'Director')
    change_column_name('Distributed by', 'Distributor')
    change_column_name('Edited by', 'Editor(s)')
    change_column_name('Length', 'Running time')
    change_column_name('Original release', 'Release date')
    change_column_name('Music by', 'Composer(s)')
    change_column_name('Produced by', 'Producer(s)')
    change_column_name('Producer', 'Producer(s)')
    change_column_name('Productioncompanies ', 'Production company(s)')
    change_column_name('Productioncompany ', 'Production company(s)')
    change_column_name('Released', 'Release Date')
    change_column_name('Release Date', 'Release date')
    change_column_name('Screen story by', 'Writer(s)')
    change_column_name('Screenplay by', 'Writer(s)')
    change_column_name('Story by', 'Writer(s)')
    change_column_name('Theme music composer', 'Composer(s)')
    change_column_name('Written by', 'Writer(s)')
    
    return movie


# In[27]:


clean_movies = [clean_movie(movie) for movie in wiki_movies]
wiki_movies_df = pd.DataFrame(clean_movies)
wiki_movies_df


# In[28]:


wiki_movies_df.columns.tolist()


# In[29]:


wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')


# In[30]:


wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
print(len(wiki_movies_df))
wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
print(len(wiki_movies_df))
wiki_movies_df.head()


# In[31]:


[[column,wiki_movies_df[column].isnull().sum()] for column in wiki_movies_df.columns]


# In[32]:


[column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]


# In[33]:


wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]


# In[34]:


wiki_movies_df.dtypes


# In[35]:


box_office = wiki_movies_df['Box office'].dropna()


# In[36]:


def is_not_a_string(x):
    return type(x) != str


# In[37]:


box_office[box_office.map(is_not_a_string)]


# In[38]:


box_office[box_office.map(lambda x: type(x) != str)]


# In[39]:


some_list = ['One','Two','Three']
'Mississippi'.join(some_list)


# In[40]:


box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[41]:


box_office


# In[42]:


form_one = r'\$\d+\.?\d*\s*[mb]illion'


# In[43]:


import re
box_office.str.contains(form_one, flags=re.IGNORECASE).sum()


# In[44]:


form_two = r'\$\d{1,3}(?:,\d{3})+'
box_office.str.contains(form_two, flags=re.IGNORECASE).sum()


# In[45]:


matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)


# In[46]:


box_office[~matches_form_one & ~matches_form_two]


# In[47]:


form_one = r'\$\s*\d+\.?\d*\s*[mb]illion'
form_two = r'\$\s*\d{1,3}(?:,\d{3})+'


# In[48]:


form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+'


# In[49]:


form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'


# In[50]:


box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)


# In[51]:


form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'


# In[52]:


box_office.str.extract(f'({form_one}|{form_two})')


# In[53]:


#def parse_dollars(s):
    # if s is not a string, return NaN

    # if input is of the form $###.# million

        # remove dollar sign and " million"

        # convert to float and multiply by a million

        # return value

    # if input is of the form $###.# billion

        # remove dollar sign and " billion"

        # convert to float and multiply by a billion

        # return value

    # if input is of the form $###,###,###

        # remove dollar sign and commas

        # convert to float

        # return value

    # otherwise, return NaN


# In[65]:


def parse_dollars(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan

    # if input is of the form $###.# million
    if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.# billion
    elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " billion"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a billion
        value = float(s) * 10**9

        # return value
        return value

    # if input is of the form $###,###,###
    elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

        # remove dollar sign and commas
        s = re.sub('\$|,','', s)

        # convert to float
        value = float(s)

        # return value
        return value

    # otherwise, return NaN
    else:
        return np.nan


# In[66]:


wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)


# In[67]:


wiki_movies_df.drop('Box office', axis=1, inplace=True)


# In[68]:


budget = wiki_movies_df['Budget'].dropna()


# In[69]:


budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)


# In[70]:


budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)


# In[71]:


matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
budget[~matches_form_one & ~matches_form_two]


# In[72]:


budget = budget.str.replace(r'\[\d+\]\s*', '')
budget[~matches_form_one & ~matches_form_two]


# In[73]:


wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)


# In[74]:


wiki_movies_df.drop('Budget', axis=1, inplace=True)


# In[75]:


release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[76]:


date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
date_form_two = r'\d{4}.[01]\d.[123]\d'
date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
date_form_four = r'\d{4}'


# In[77]:


release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)


# In[78]:


wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)


# In[79]:


wiki_movies_df


# In[80]:


running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[81]:


running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE).sum()


# In[82]:


running_time[running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE) != True]


# In[83]:


running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE).sum()


# In[89]:


running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE) != True]


# In[90]:


running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')


# In[91]:


running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)


# In[92]:


wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)


# In[97]:


wiki_movies_df.drop('Running time', axis=1, inplace=True)


# In[98]:


kaggle_metadata.dtypes


# In[99]:


kaggle_metadata['adult'].value_counts()


# In[100]:


kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]


# In[101]:


kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')


# In[102]:


kaggle_metadata['video'].value_counts()


# In[103]:


kaggle_metadata['video'] == 'True'


# In[104]:


kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'


# In[105]:


kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')


# In[106]:


kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])


# In[107]:


ratings.info(null_counts=True)


# In[108]:


pd.to_datetime(ratings['timestamp'], unit='s')


# In[109]:


ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')


# In[110]:


ratings['rating'].plot(kind='hist')
ratings['rating'].describe()


# In[111]:


movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])


# In[112]:


movies_df[['title_wiki','title_kaggle']]


# In[113]:


movies_df[movies_df['title_wiki'] != movies_df['title_kaggle']][['title_wiki','title_kaggle']]


# In[114]:


movies_df[(movies_df['title_kaggle'] == '') | (movies_df['title_kaggle'].isnull())]


# In[115]:


movies_df.fillna(0).plot(x='running_time', y='runtime', kind='scatter')


# In[116]:


movies_df.fillna(0).plot(x='budget_wiki',y='budget_kaggle', kind='scatter')


# In[117]:


movies_df.fillna(0).plot(x='box_office', y='revenue', kind='scatter')


# In[118]:


movies_df.fillna(0)[movies_df['box_office'] < 10**9].plot(x='box_office', y='revenue', kind='scatter')


# In[119]:


movies_df[['release_date_wiki','release_date_kaggle']].plot(x='release_date_wiki', y='release_date_kaggle', style='.')


# In[120]:


movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')]


# In[122]:


movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index


# In[123]:


movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)


# In[124]:


movies_df[movies_df['release_date_wiki'].isnull()]


# In[125]:


movies_df['Language'].value_counts()


# In[127]:


movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)


# In[128]:


movies_df['original_language'].value_counts(dropna=False)


# In[129]:


movies_df[['Production company(s)','production_companies']]


# In[139]:


movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)


# In[140]:


def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
        , axis=1)
    df.drop(columns=wiki_column, inplace=True)


# In[132]:


fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
movies_df


# In[133]:


for col in movies_df.columns:
    lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
    value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
    num_values = len(value_counts)
    if num_values == 1:
        print(col)


# In[134]:


movies_df['video'].value_counts(dropna=False)


# In[135]:


movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                       'genres','original_language','overview','spoken_languages','Country',
                       'production_companies','production_countries','Distributor',
                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                      ]]


# In[138]:


movies_df.rename({'id':'kaggle_id',
                  'title_kaggle':'title',
                  'url':'wikipedia_url',
                  'budget_kaggle':'budget',
                  'release_date_kaggle':'release_date',
                  'Country':'country',
                  'Distributor':'distributor',
                  'Producer(s)':'producers',
                  'Director':'director',
                  'Starring':'starring',
                  'Cinematography':'cinematography',
                  'Editor(s)':'editors',
                  'Writer(s)':'writers',
                  'Composer(s)':'composers',
                  'Based on':'based_on'
                 }, axis='columns', inplace=True)


# In[141]:


rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()


# In[142]:


rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1) 


# In[143]:


rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1)                 .pivot(index='movieId',columns='rating', values='count')


# In[144]:


rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]


# In[145]:


movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')


# In[147]:


movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)


# In[148]:


#Load


# In[219]:


from sqlalchemy import create_engine
from config import db_password


# In[231]:


db_string = f"postgres://postgres:{db_password}@localhost:5432/movie_data"


# In[232]:


engine = create_engine(db_string)


# In[233]:


movies_df.to_sql(name='movies', con=engine)


# In[235]:


for data in pd.read_csv(f'{file_dir}/ratings.csv', chunksize=1000000):
    data.to_sql(name='ratings', con=engine, if_exists='append')


# In[234]:


import time
rows_imported = 0
# get the start_time from time.time()
for data in pd.read_csv(f'{file_dir}/ratings.csv', chunksize=1000000):

    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
    data.to_sql(name='ratings', con=engine, if_exists='append')
    rows_imported += len(data)

    # add elapsed time to final print out
    print(f'Done.')


# In[213]:


rows_imported = 0
for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):

    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
    data.to_sql(name='ratings', con=engine, if_exists='append')
    rows_imported += len(data)

    print(f'Done.')


# In[214]:


rows_imported = 0
# get the start_time from time.time()
for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):

    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
    data.to_sql(name='ratings', con=engine, if_exists='append')
    rows_imported += len(data)

    # add elapsed time to final print out
    print(f'Done.')


# In[ ]:




