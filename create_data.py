import cv2
import os
import csv

#Tạo thư mục lưu 5 người có ảnh nhiều nhất
def data_test():
    data = ['1048', '1394', '1871', '1892', '5459']
    for j in data:
        path_data = 'Data/' + j
        path_test = 'Data_test/' + j
        for i in os.listdir(path_data):
            path = path_data + '/' + i
            img = cv2.imread(path)
            if os.path.exists(path_test) == False:
                os.mkdir(path_test)
            cv2.imwrite(os.path.join(path_test, j + '.' + i), img)

#Ghi tên những người cần dự đoán vào file csv
def write_csv():
    data = []
    for i in os.listdir("lfw_funneled"):
        data.append(i)

    with open('data_name_test.csv', 'w+') as f:
        write = csv.writer(f)
        write.writerow({""})
        for j in data:
            write.writerow({j})

#Đọc file csv
def read_csv(file_name):
    data_name = []
    with open(file_name, 'r') as f:
        read_name = csv.reader(f)
        for row in read_name:
            for e in row:
                data_name.append(e)
    return data_name