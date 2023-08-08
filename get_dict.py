import json
with open('dataset\\MOFseq_output.txt', 'r') as f:
    MOF_list= [line.strip() for line in f.readlines()]
MOF_seq = []
for s in MOF_list:
    word_list = s.split()
    MOF_seq.append(word_list)
# print(MOF_seq)

word_to_id = {}
id_to_word = {}
current_id = 1

for lst in MOF_seq:
    for word in lst:
        if word not in word_to_id:
            word_to_id[word] = current_id
            id_to_word[current_id] = word
            current_id += 1
# print(word_to_id)
# print(id_to_word)
my_dict={'symbol2idx':word_to_id,'idx2symbol':id_to_word}
json_str = json.dumps(my_dict)
with open ('dataset\\my_dict_output.json','w') as f:
    f.write(json_str)