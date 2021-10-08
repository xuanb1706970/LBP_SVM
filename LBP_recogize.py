import cv2
import time
import numpy
import os
from create_data import read_csv
import matplotlib.pyplot as plt

recognizer = cv2.face.LBPHFaceRecognizer_create()

#Xu Ly Anh Bang Haar
def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Cascasdes = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = Cascasdes.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
         return None, None

    (x,y,w,h) = faces[0]
    cv2.imshow('faces', gray[y:y+h, x:x+w])
    cv2.waitKey(1)
    return gray[y:y+w, x:x+h], faces[0]

#Luu anh tu haar vao mang de train
def array_faces():
    faces_array = []
    labels_array = []
    for i in os.listdir('Data'):
        path_floder = 'Data/' + i
        labels = int(i)
        for j in os.listdir(path_floder):
            path_image = path_floder + "/" + j
            img = cv2.imread(path_image)
            faces, ret = detect_faces(img)

            if faces is not None:
                faces_array.append(faces)
                labels_array.append(labels)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces_array, labels_array

#Train mô hình
def train():
    start = time.time()

    faces, labels = array_faces()

    recognizer.train(faces, numpy.array(labels))

    recognizer.save('training_lbp.yml')

    print('Thời gian train', time.time() - start, 'seconds.')


#Dự đoán hình ảnh và số ảnh nhận dạng, không nhận dạng vào mảng hai mảng
def predict_image():
    count_data = []
    error_data = []
    count = 0
    error = 0
    data = read_csv('data_name_test.csv')
    recognizer.read('training_lbp.yml')
    for i in os.listdir('Data_test'):
        path_fl = 'Data_test/' + i
        for j in os.listdir(path_fl):
            path_img = path_fl + '/' + j
            ID = int(os.path.split(path_img)[-1].split('.')[0])
            img = cv2.imread(path_img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            Cascasdes = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            faces = Cascasdes.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0 or len(faces) >= 2:
                error += 1
            else:
                (x, y, h, w) = faces[0]
                labels, cof = recognizer.predict(gray[y:y+h, x:x+w])
                label_t = data[labels]
                print(ID)
                print(labels)
                if ID == labels:
                    count += 1
                    print(count)
                    print(label_t)
        count_data.append(count)
        error_data.append(error)
        error = 0
        count = 0
    return count_data, error_data

#count, error = predict_image()
#print(num)
#print(count)
#print(error)

#Chuyển từ số nguyên sang tỉ lệ phần trăm
def sum_count():
    sum_cout = []
    sum_error = []
    sum_fl = [236,121,530,109,144]
    count, error = predict_image()
    for i in range(0, 5):
        t1 = (float(count[i])/float(sum_fl[i])*100)
        t2 = (float(error[i])/float(sum_fl[i])*100)
        sum_cout.append(t1)
        sum_error.append(t2)
    return sum_cout, sum_error

#Lấy tỉ lệ và xây dựng biểu đồ bar
def chart_face():
    sum_correct, sum_error = sum_count()
    print(sum_correct)
    print(sum_error)
    data_name = ['Colin_Powell', 'Donald_Rumsfeld', 'George_W_Bush', 'Gerhard_Schroeder', 'Tony_Blair']
    ind = numpy.arange(len(sum_correct))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width / 2, sum_correct, width,
                    label='Tỉ lệ chính xác')
    rects2 = ax.bar(ind + width / 2, sum_error, width,
                    label='Tỉ lệ sai')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Phần trăm')
    ax.set_title('Biểu đồ tỷ lệ phần trăm nhận diện khuôn mặt')
    ax.set_xticks(ind)
    ax.set_xticklabels(data_name, fontsize=8)
    ax.legend()

    rects = ax.patches

    mega_array = sum_correct + sum_error
    # Make some labels.
    labels = ["%.2f" % i for i in mega_array]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, label,
                ha='center', va='bottom')

    fig.tight_layout()
    plt.show()
    fig.savefig('plt_update.png', dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    train()
    chart_face()