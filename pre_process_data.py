import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm
from math import ceil
import re
from gensim.models import TfidfModel, nmf
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaModel
from multiprocessing import Pool
from functools import partial

tqdm.pandas()
CHUNK_SIZE = 500000


def check():
    if not output_dir.is_dir():
        os.makedirs(output_dir)
    assert (answers_dir.is_dir())
    assert (dataset_dir.is_dir())
    assert (main_answers_file.is_file())
    assert (output_dir.is_dir())


def count_file_lines(file):
    with open(file) as f:
        for count, _ in enumerate(f):
            pass
    return count


def collect_vocabulary(csv_name, chunk_size=100000):
    line_count = count_file_lines(dataset_dir / f'{csv_name}.csv')

    df_iter = pd.read_csv(dataset_dir / f'{csv_name}.csv', usecols=['date', 'user', 'content'], chunksize=chunk_size)

    result_set = set()

    for df in tqdm(df_iter, total=(ceil(line_count / chunk_size))):
        df['content'] = df['content'].str.lower().str.split()
        result_set = result_set.union(*map(set, df['content']))

    return result_set


def chunk_iterator(filename, chunk_size=10000):
    for chunk in tqdm(pd.read_csv(filename, chunksize=chunk_size)):
        for document in chunk['content'].str.lower().str.split().values:
            yield document


def bow_chunk_iterator(filenames, dictionary, chunk_size=10000):
    for filename in filenames:
        print(filename)
        for chunk in tqdm(pd.read_csv(filename, chunksize=chunk_size)):
            for document in chunk['content'].str.lower().str.split().values:
                yield dictionary.doc2bow(document)


def tfidf_chunk_iterator(filenames, dictionary, tfidf, chunk_size=10000):
    for filename in filenames:
        print(filename)
        for chunk in tqdm(pd.read_csv(filename, chunksize=chunk_size)):
            for document in chunk['content'].str.lower().str.split().values:
                yield tfidf[dictionary.doc2bow(document)]


def parallelize(data, func, num_of_processes=8):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)


def parallelize_on_rows(data, func, num_of_processes=8):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)


def process_content_parallelized(filename, dictionary, model, chunk_size=10000, postfix='_lda_content'):
    out_file = output_dir / (filename.stem + postfix + '.csv')
    assert (not out_file.is_file())
    for chunk in tqdm(pd.read_csv(filename, usecols=['id', 'content'], chunksize=chunk_size)):
        chunk['content'] = parallelize_on_rows(
            chunk['content'].str.lower().str.split()
            , lambda doc: model[dictionary.doc2bow(doc)])

        chunk.to_csv(out_file, mode='a')


def process_content(filename, chunk_size=10000, postfix='_lda_content'):
    dictionary = Dictionary.load((output_dir / 'content_dictionary.pkl').as_posix())
    model = LdaModel.load((output_dir / 'lda_model.pkl').as_posix())

    out_file = output_dir / (filename.stem + postfix + '.csv')
    # Path(out_file).touch()
    # print(out_file.is_file())

    # assert(not out_file.is_file())
    for chunk in tqdm(pd.read_csv(filename, usecols=['id', 'content'], chunksize=chunk_size)):
        chunk['content'] = chunk['content'] \
            .str.lower() \
            .str.split() \
            .apply(lambda doc: model[dictionary.doc2bow(doc)])

        chunk.to_csv(out_file, mode='a')


def pre_process_logon(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df.date, format='%m/%d/%Y %H:%M:%S')
    df['day'] = df['date'].dt.floor('D')
    most_common_pc = df \
        .groupby(['user', 'day', 'pc']) \
        .size() \
        .to_frame('count') \
        .reset_index() \
        .sort_values('count', ascending=False) \
        .drop_duplicates(subset=['user', 'day']) \
        .drop(columns=['count']) \
        .sort_values(['user', 'day']) \
        .groupby('user') \
        .pc \
        .agg(pd.Series.mode) \
        .rename('most_common_pc')
    df = df.merge(most_common_pc.to_frame(), left_on='user', right_on='user')
    df['is_usual_pc'] = df['most_common_pc'] == df['pc']

    is_work_time = (8 <= df.date.dt.hour) & (df.date.dt.hour < 17)
    df['is_work_time'] = is_work_time

    df['subtype'] = df['activity']
    df[['id', 'date', 'user', 'is_usual_pc', 'is_work_time', 'subtype']].to_csv(output_dir / 'logon_preprocessed.csv')
    return most_common_pc.to_frame()


def pre_process_device(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df.date, format='%m/%d/%Y %H:%M:%S')
    df = df.merge(most_common_pc, left_on='user', right_on='user', )
    df['is_usual_pc'] = df['most_common_pc'] == df['pc']

    is_work_time = (8 <= df.date.dt.hour) & (df.date.dt.hour < 17)
    df['is_work_time'] = is_work_time

    df['subtype'] = df['activity']
    df[['id', 'date', 'user', 'is_usual_pc', 'is_work_time', 'subtype']].to_csv(
        output_dir / f'device_preprocessed.csv')


def pre_process_file(path):
    df = pd.read_csv(path, usecols=['id', 'date', 'user', 'pc', 'filename'])
    df['date'] = pd.to_datetime(df.date, format='%m/%d/%Y %H:%M:%S')

    df = df.merge(most_common_pc, left_on='user', right_on='user', )
    df['is_usual_pc'] = df['most_common_pc'] == df['pc']

    is_work_time = (8 <= df.date.dt.hour) & (df.date.dt.hour < 17)
    df['is_work_time'] = is_work_time

    file_extensions = df.filename.str[-4:]
    df['subtype'] = file_extensions
    df[['id', 'date', 'user', 'is_usual_pc', 'is_work_time', 'subtype']].to_csv(
        output_dir / f'file_preprocessed.csv')


def pre_process_email(path):
    df = pd.read_csv(path, usecols=['id', 'date', 'user', 'pc', 'to', 'cc', 'bcc', 'from'])
    df = df.fillna('')
    from_domain = df['from'].str.extract('^.+@(.+$)')[0]
    is_external_from = from_domain == 'dtaa.com'
    to_concated = df[['to', 'cc', 'bcc']].progress_apply(lambda x: ';'.join([x.to, x.cc, x.bcc]), axis=1)
    is_external_to = to_concated.progress_apply(
        lambda x: any([re.match('^.+@(.+$)', e).group(1) != 'dtaa.com' for e in x.split(';') if e != '']))
    is_external = is_external_to | is_external_to
    df['date'] = pd.to_datetime(df.date, format='%m/%d/%Y %H:%M:%S')

    df = df.merge(most_common_pc, left_on='user', right_on='user', )
    df['is_usual_pc'] = df['most_common_pc'] == df['pc']

    is_work_time = (8 <= df.date.dt.hour) & (df.date.dt.hour < 17)
    df['is_work_time'] = is_work_time

    df['subtype'] = is_external
    df[['id', 'date', 'user', 'is_usual_pc', 'is_work_time', 'subtype']].to_csv(
        output_dir / f'email_preprocessed.csv')


def pre_process_http(path):
    job_hunting_websites = [
        'careerbuilder.com',
        'craiglist.org',
        'indeed.com',
        'job-hunt.org',
        'jobhuntersbible.com',
        'linkedin.com',
        'monster.com',
        'simplyhired.com',
    ]

    hacktivist_websites = [
        'actualkeylogger.com',
        'best-spy-soft.com',
        'dailykeylogger.com',
        'keylogpc.com',
        'refog.com',
        'relytec.com',
        'softactivity.com',
        'spectorsoft.com',
        'webwatchernow.com',
        'wellresearchedreviews.com',
        'wikileaks.org'
    ]

    filesharing_websites = [
        '4shared.com'
        'dropbox.com',
        'fileserve.com',
        'filefreak.com',
        'filestube.com',
        'megaupload.com',
        'thepiratebay.org'
    ]
    with open(path) as f:
        for count, l in tqdm(enumerate(f)):
            pass
    df_iter = pd.read_csv(dataset_dir / 'http.csv', chunksize=CHUNK_SIZE, usecols=['date', 'user', 'pc', 'url'])
    (output_dir / 'http_preprocessed.csv').unlink()
    first_it = True
    mode = 'w'

    for http_df in tqdm(df_iter, total=ceil(count / CHUNK_SIZE)):
        http_df['date'] = pd.to_datetime(http_df.date, format='%m/%d/%Y %H:%M:%S')

        site_names = http_df['url'].apply(lambda s: re.match('^https?://(www)?([0-9\-\w\.]+)?.+$', s).group(2))
        http_df['site_name'] = site_names

        http_df['subtype'] = 0
        http_df.loc[site_names.isin(job_hunting_websites), 'subtype'] = 1
        http_df.loc[site_names.isin(hacktivist_websites), 'subtype'] = 2
        http_df.loc[site_names.isin(filesharing_websites), 'subtype'] = 3

        http_df = http_df.merge(most_common_pc, left_on='user', right_on='user', )
        http_df['is_usual_pc'] = http_df['most_common_pc'] == http_df['pc']

        is_work_time = (8 <= http_df.date.dt.hour) & (http_df.date.dt.hour < 17)
        http_df['is_work_time'] = is_work_time

        http_df.to_csv(output_dir / 'http_preprocessed.csv', header=first_it, index=False,
                       mode=mode, columns=['id', 'date', 'user', 'is_usual_pc', 'is_work_time', 'subtype', 'site_name'])
        first_it = False
        mode = 'a'


def merge_all_content():
    df_dict = Dictionary(chunk_iterator(dataset_dir / 'email.csv'))
    df_dict.add_documents(chunk_iterator(dataset_dir / 'file.csv'))
    df_dict.add_documents(chunk_iterator(dataset_dir / 'http.csv'))

    df_dict.save((output_dir / 'content_dictionary.pkl').as_posix())


def make_tfidf_model():
    df_dict = Dictionary.load((output_dir / 'content_dictionary.pkl').as_posix())
    tfidf = TfidfModel(
        bow_chunk_iterator([
            dataset_dir / 'email.csv',
            dataset_dir / 'file.csv',
            dataset_dir / 'http.csv'
        ], df_dict))

    tfidf.save((output_dir / 'tfidf_model.pkl').as_posix())


def make_nmf_model():
    df_dict = Dictionary.load((output_dir / 'content_dictionary.pkl').as_posix())
    tfidf = TfidfModel.load((output_dir / 'tfidf_model.pkl').as_posix())
    nmf_model = nmf.Nmf(
        tfidf_chunk_iterator([
            dataset_dir / 'email.csv',
            dataset_dir / 'file.csv',
            dataset_dir / 'http.csv'
        ], df_dict, tfidf,
        ),
        num_topics=100
    )

    nmf_model.save((output_dir / 'nmf_model.pkl').as_posix())


def make_lda_model():
    df_dict = Dictionary.load((output_dir / 'content_dictionary.pkl').as_posix())
    tfidf = TfidfModel.load((output_dir / 'tfidf_model.pkl').as_posix())
    lda_model = LdaModel(
        tfidf_chunk_iterator([
            dataset_dir / 'email.csv',
            dataset_dir / 'file.csv',
            dataset_dir / 'http.csv'
        ], df_dict, tfidf,
        ),
        num_topics=100
    )
    lda_model.save((output_dir / 'lda_model.pkl').as_posix())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", sys.argv[0], "[answer directory] [dataset directory]")
        print("Warning: may cost over 30 hours to run all steps for the dataset over 20G")
        print("hint: comment out the pre_runned process")
    print("Start to process dataset ver", sys.argv[2].split('/')[-1])
    answers_dir = Path(sys.argv[1])
    dataset_dir = Path(sys.argv[2])
    main_answers_file = answers_dir / "insiders.csv"
    output_dir = Path('./_output/')

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    most_common_pc = pre_process_logon(dataset_dir / 'logon.csv')
    print("logon processed")
   
    pre_process_device(dataset_dir / 'device.csv')
    print("device processed")
   
    pre_process_file(dataset_dir / 'file.csv')
    print("file processed")
   
    pre_process_email(dataset_dir / 'email.csv')
    print("email processed")
   
    pre_process_http(dataset_dir / 'http.csv')
    print("http processed")

    merge_all_content()
    print("all content file merged and saved")
    make_tfidf_model()
    print("tfidf model saved")
    make_nmf_model()
    print("nmf model saved")
    make_lda_model()
    print("lda model saved")

    process_content(dataset_dir / 'email.csv')
    print("email content processed")
    process_content(dataset_dir / 'file.csv')
    print("file content processed")
    process_content(dataset_dir / 'http.csv')
    print("http content processed")

