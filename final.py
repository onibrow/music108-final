from helper_code import *

file_name = 'www'
list_file_names_www = generate_wav_files(file_name)

run_experiments(3, list_file_names_www, file_name)
