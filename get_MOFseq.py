import csv
import selfies as sf
with open(r"dataset\\MOF_output_test.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    merged_list = []
    for row in reader:
        e_1=list(sf.split_selfies(row['linker_1']))
        e_2=list(sf.split_selfies(row['linker_2']))
        ml=['<T>',row['topo'],row['cat'],'<N>',row['node'],'<E_1>']
        ml=ml+e_1
        ml.append('<E_2>')
        ml=ml+e_2
        merged_list.append(ml)

with open('dataset\\MOFseq_test.txt', 'w') as file:
    for sublist in merged_list:
        for item in sublist:
            file.write(str(item) + ' ')
        file.write('\n')