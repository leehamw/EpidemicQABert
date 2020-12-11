import csv
import json
from os.path import join











def main():
    f=open('data/final/test.csv','w',encoding='utf-8')
    csv_writer = csv.writer(f, delimiter='\t')

    csv_writer.writerow(["id", "docid", "answer"])

    for index in range(1643):
        with open(join('data/final/context_after_generation','{}.json'.format(index+1))) as v:
            js_data = json.load(v)
            print('loading: {}'.format(index + 1))
        id, docid, answer=(js_data['id'], js_data['docid'],js_data['answer'])

        csv_writer.writerow([id, docid, answer])




if __name__ == '__main__':
    main()
