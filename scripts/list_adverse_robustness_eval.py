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
    for weather_type in os.listdir(base_dir):
        weather_type_path = os.path.join(base_dir, weather_type)
        if os.path.isdir(weather_type_path):
            for model_name in os.listdir(weather_type_path):
                model_name_path = os.path.join(weather_type_path, model_name)
                if os.path.isdir(model_name_path):
                    latest_date_folder = get_latest_date_folder(model_name_path)
                    if latest_date_folder:
                        json_file_path = os.path.join(model_name_path, latest_date_folder, f"{latest_date_folder}.json")
                        if os.path.isfile(json_file_path):
                            json_data = read_json_file(json_file_path)
                            json_data['model_name'] = model_name.split('_')[0]
                            json_data['weather_type'] = weather_type
                            json_data['model_config'] = model_name
                            data_list.append(json_data)
    
    df = pd.DataFrame(data_list)
    return df

# Usage example:
base_directory = 'work_dirs/adverse_robustness_eval'
output_csv_path = 'work_dirs/adverse_robustness_eval.csv'

weather_df = collect_weather_data(base_directory)
weather_df.to_csv(output_csv_path, index=False)
print(weather_df)
