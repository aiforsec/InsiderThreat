import numpy as np
import pandas as pd
import os
from sklearn.externals import joblib
from tqdm import tqdm
import sys

LDAP_dir = "./r5.2/LDAP"
Processed_dir = "./pre_processed"
result_dir = "./processed_log"

ACTION_DIR = {"Logon": "1",
              "Logoff": "2",
              "Connect": "3",
              "Disconnect": "4",
              "http": "5",
              "file": "6",
              "email": "7"}


def get_email_by_id(df, id):
    return df[(df['user_id'] == id)].email.to_string().split()[1]


def get_all_users():
    LDAP_files = []
    for _, _, file in os.walk(LDAP_dir):
        LDAP_files.append(file)
    LDAP_files = np.array(LDAP_files)
    LDAP_files = np.squeeze(LDAP_files)

    user_file = pd.read_csv(os.path.join(LDAP_dir, LDAP_files[0]))

    for file in LDAP_files:
        temp = pd.read_csv(os.path.join(LDAP_dir, file))
        temp = temp[~temp.user_id.isin(user_file.user_id)]
        user_file = user_file.append(temp)

    user_file = user_file.drop(
        columns=["employee_name", "role", "business_unit", "functional_unit", "department", "team", "supervisor"])
    return user_file


def get_all_files():
    file_list = []
    for _, _, file in os.walk(Processed_dir + '/' + user_id):
        file_list.append(file)
    return file_list


def get_all_file_df():
    temp_file_df = []
    for file in file_list[0]:
        if file == 'logon_file.csv':
            temp_file_df.append(
                pd.read_csv(Processed_dir + '/' + user_id + '/' + file).rename(columns={'activity': 'action'}).drop(
                    columns='user'))
            # print(temp_file)
        elif file == 'http_file.csv':
            temp_file_df.append(
                pd.read_csv(Processed_dir + '/' + user_id + '/' + file).rename(columns={'activity': 'action'}).drop(
                    columns='user'))
        elif file == 'file_file.csv':
            temp_file_df.append(
                pd.read_csv(Processed_dir + '/' + user_id + '/' + file).rename(columns={'activity': 'action'}).drop(
                    columns='user'))
        elif file == 'device_file.csv':
            temp_file_df.append(
                pd.read_csv(Processed_dir + '/' + user_id + '/' + file).rename(columns={'activity': 'action'}).drop(
                    columns='user'))
        else:
            print("ERROR")
    return temp_file_df


def merge_all_df(email_file):
    for df in temp_file_df:
        email_file = email_file.append(df)
    return email_file


def key_to_EventId(df, dict_filename):
    df_log_trans = df.copy()
    log_key_sequence = df_log_trans['log key']
    # log_key_sequence = list(log_key_sequence)
    # get the unique list
    items = set(log_key_sequence)
    # define the total number of log keys
    K = None
    K = len(items)
    print("the length of unique log_key_sequence is:", K)
    key_name_dict = {}

    for i, item in enumerate(items):
        # items is a set
        # columns are the lines of log key sequence
        for j in range(len(log_key_sequence)):
            if log_key_sequence[j] == item:
                name = 'E' + str(i)
                # we do not replace the string using Exx in the function
                key_name_dict[name] = item.strip('\n')

    joblib.dump(key_name_dict, dict_filename)

    return log_key_sequence, key_name_dict, K


# save the dict within sequence
def transform_key_k(log_key_sequence, dict):
    print("the length of sequence is {} and the length of dict is {}".format \
              (len(set(log_key_sequence)), len(set(dict.values()))))
    # while set(log_key_sequence) == set(dict.values()):
    for key, value in dict.items():
        for x in log_key_sequence:
            # transform the set type to list type
            log_key_sequence = list(log_key_sequence)
            if value == x:
                log_key_sequence[log_key_sequence.index(x)] = str(key)
            else:
                continue
    return log_key_sequence


if __name__ == "__main__":
    user_file = get_all_users()

    user_id = input("user id\n")
    user_email = get_email_by_id(user_file, user_id)

    test_email_file = Processed_dir + '/' + user_email + "/email_file.csv"
    email_file = pd.read_csv(test_email_file)
    email_file = email_file.drop(columns=["from"])
    email_file['action'] = "email"
    # print(email_file)

    file_list = get_all_files()
    temp_file_df = get_all_file_df()

    temp_final_df = merge_all_df(email_file).sort_values(by=['date'])
    prev_action = "Logoff"
    with open(user_id + "_LogKey.txt", "w") as out_file:
        for _, row in tqdm(temp_final_df.iterrows()):
            if row.action == "Logon" and prev_action != "Logoff":
                out_file.writelines('\n')
            out_file.writelines(ACTION_DIR[row.action] + ' ')
            if row.action == "Logoff":
                out_file.writelines('\n')
            prev_action = row.action
    # temp_final_df.to_csv(user_id+"_LogKey.csv")
