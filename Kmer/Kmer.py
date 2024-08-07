# 2 combination
def twoTupleDic():
    AA_list_sort = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M','N','P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'O']
    AA_dict = {}
    numm = 1
    for i in AA_list_sort:
        for j in AA_list_sort:
            AA_dict[i+j] = numm
            numm += 1
    return AA_dict


# 4 combination
def twoTupleDic():
    AA_list_sort = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M','N','P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'O']
    AA_dict = {}
    numm = 1
    for i in AA_list_sort:
        for j in AA_list_sort:
            for k in AA_list_sort:
                for l in AA_list_sort:
                    for m in AA_list_sort:
                        AA_dict[i+j+k+l+m] = numm
                        numm += 1
    return AA_dict

twoTupleDic()

#save dictionary in csv format
import csv
def twoTupleDic():
    AA_list_sort = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M','N','P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'O']
    AA_dict = {}
    numm = 1
    for i in AA_list_sort:
        for j in AA_list_sort:
            for k in AA_list_sort:
                for l in AA_list_sort:
                    for m in AA_list_sort:
                        for n in AA_list_sort:
                            AA_dict[i+j+k+l+m+n] = numm
                            numm += 1
    return AA_dict
    

def save_dict_to_csv(dictionary, csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Name', 'Value'])  # Write header
        for key, value in dictionary.items():
            writer.writerow([key, value])

# Usage:
my_dict = twoTupleDic()
csv_file_path = "six_tuple_dictionary.csv"
save_dict_to_csv(my_dict, csv_file_path)
print(f"Dictionary saved to {csv_file_path}")

df.to_csv('six_tuple_dictionary.csv', index=False)
