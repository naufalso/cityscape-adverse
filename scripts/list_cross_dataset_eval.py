import os
import pandas as pd
import json
from datetime import datetime

def get_latest_date_folder(path):
    dates = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    dates.sort(key=lambda date: datetime.strptime(date, '%Y%m%d_%H%M%S'), reverse=True)
    return dates[0] if dates else None

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def collect_weather_data(base_dir):
    data_list = []
    for evaluation_type in os.listdir(base_dir): #deeplabv3plus_all_all
        evaluation_type_path = os.path.join(base_dir, evaluation_type)
        # print(evaluation_type_path)
        model_name, train_dataset, eval_dataset = evaluation_type.split('_')
        if os.path.isdir(evaluation_type_path):
            for log_data in os.listdir(evaluation_type_path):
                log_data_path = os.path.join(evaluation_type_path, log_data)
                # print(">", log_data_path)
                if os.path.isdir(log_data_path):
                    latest_date_folder = get_latest_date_folder(evaluation_type_path)
                    # print(latest_date_folder)
                    if latest_date_folder:
                        json_file_path = os.path.join(log_data_path, f"{latest_date_folder}.json")
                        if os.path.isfile(json_file_path):
                            json_data = read_json_file(json_file_path)
                            json_data['model_name'] = model_name
                            json_data['train_dataset'] = train_dataset
                            json_data['eval_dataset'] = eval_dataset
                            data_list.append(json_data)
    
    df = pd.DataFrame(data_list)
    return df

# Usage example:
base_directory = 'work_dirs/filtered_cross_evaluation_eval/deeplabv3plus'
output_csv_path = 'work_dirs/filtered_cross_evaluation_eval_deeplabv3plus.csv'

weather_df = collect_weather_data(base_directory)
weather_df.to_csv(output_csv_path, index=False)
print(weather_df)
