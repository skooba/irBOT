from tabula import wrapper
import os
import csv
import re

def preprocess_pdfs():
    pdf_folder = 'data/pdf'
    csv_folder = 'data/csv'

    base_command = 'java -jar tabula-1.0.2-jar-with-dependencies.jar -n -p all -a 100,0,730,612 -f TSV -o {} {}'

    for filename in os.listdir(pdf_folder):
        pdf_path = os.path.join(pdf_folder, filename)
        csv_path = os.path.join(csv_folder, filename.replace('.pdf', '.csv'))
        command = base_command.format(csv_path, pdf_path)
        os.system(command)
        read_file = []
        with open (csv_path, 'r') as infile:
            reader = csv.reader(infile)
            count = 0
            start_contents = []
            first_section = []
            dictionary = {}
            for row in reader:
                count += 1
                if 'TABLE OF CONTENTS' in row[0].upper():
                    start_contents = count+1
                if count == start_contents:
                    first_section = re.sub('[^a-zA-Z]+','', row[0])
                if str(first_section).upper() == re.sub('[^a-zA-Z]+','', row[0]).upper():
                    end_contents = count - 1
                read_file.append('\n'.join(row))
        for section in read_file[start_contents-1:end_contents-2]:
            text = re.sub('[\W]+','', section).rstrip('0123456789').upper()
            count = end_contents-1
            for line in read_file[end_contents:]:
                count += 1
                if text == re.sub('[\W]','', line).upper():
                    dictionary[re.sub(r'\t|[0-9]\s*','',read_file[count])] = count
        section_starts = list(dictionary.values())
        count = 0
        for section in list(dictionary.keys()):
            if count < len(section_starts)-2:
                update = {section:re.sub(r'[^A-Z0-9 ]','',' '.join(read_file[section_starts[count]+1:section_starts[count+1]]).upper())}
            else:
                update = {section:re.sub(r'[^A-Z0-9 ]','',' '.join(read_file[section_starts[count]:]).upper())}
            dictionary.update(update)
            count += 1
        print(dictionary)




preprocess_pdfs()
