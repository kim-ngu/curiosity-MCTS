import csv

def write_to_csv(file_name, data):
    with open(file_name, 'a', encoding='UTF8', newline='') as file:
        writer = csv.writer(file, delimiter=';', dialect = 'excel')
        writer.writerow(data)