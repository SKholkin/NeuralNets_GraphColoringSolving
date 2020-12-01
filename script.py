import os
base_dir = os.path.normpath(os.path.join('C:', 'Users', 'holck', 'PycharmProjects', 'geometric', 'datasets', 'ColorData'))
main_txt_dir = os.path.normpath(os.path.join(base_dir, 'processed', 'layout.txt'))
main_txt_dir = 'datasets/ColorData/raw/layout.txt'

raw_dir = 'C:\\Users\\holck\\PycharmProjects\\geometric\\datasets\\ColorData\\raw'

with open(main_txt_dir, 'w+') as main_txt:
    main_txt.truncate()
    for file_name in os.listdir(raw_dir):
        main_txt.write(file_name + '\n')