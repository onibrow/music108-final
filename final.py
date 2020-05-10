from helper_code import *

file_name = 'kai'
list_file_names = generate_wav_files(file_name)

run_experiments(40, list_file_names, file_name)
