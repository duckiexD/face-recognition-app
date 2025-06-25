import os
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import face_recognition
from PIL import Image, ImageTk
import numpy as np

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.root.geometry("1000x700")
        
        # Инициализация переменных
        self.video_capture = None
        self.current_frame = None
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_this_frame = True
        self.recognition_enabled = True
        self.face_detection_enabled = True
        self.running = False
        self.frame_counter = 0
        
        # Создаем необходимые папки
        if not os.path.exists("known_faces"):
            os.makedirs("known_faces")
        
        # Загрузка сохраненных лиц
        self.load_known_faces()
        
        # Проверка целостности данных
        if not self.verify_data_integrity():
            messagebox.showwarning("Предупреждение", "Обнаружены проблемы с данными. Рекомендуется переобучить модель.")
        
        # Интерфейс
        self.create_widgets()
        
        # Запуск камеры
        self.start_camera()

    def create_widgets(self):
        # Видео-панель
        self.video_frame = tk.LabelFrame(self.root, text="Видеопоток")
        self.video_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        
        # Панель управления
        control_frame = tk.LabelFrame(self.root, text="Управление")
        control_frame.pack(pady=10, padx=10, fill=tk.X)
        
        # Кнопки
        btn_add_face = tk.Button(control_frame, text="Добавить лицо", command=self.add_face)
        btn_add_face.pack(side=tk.LEFT, padx=5, pady=5)
        
        btn_train = tk.Button(control_frame, text="Обучить модель", command=self.train_model)
        btn_train.pack(side=tk.LEFT, padx=5, pady=5)
        
        btn_verify = tk.Button(control_frame, text="Проверить данные", command=self.verify_data)
        btn_verify.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.recognition_var = tk.BooleanVar(value=True)
        chk_recognition = tk.Checkbutton(control_frame, text="Распознавание", 
                                       variable=self.recognition_var, 
                                       command=self.toggle_recognition)
        chk_recognition.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.detection_var = tk.BooleanVar(value=True)
        chk_detection = tk.Checkbutton(control_frame, text="Детекция", 
                                      variable=self.detection_var, 
                                      command=self.toggle_detection)
        chk_detection.pack(side=tk.LEFT, padx=5, pady=5)
        
        btn_quit = tk.Button(control_frame, text="Выход", command=self.quit_app)
        btn_quit.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Информация
        info_frame = tk.LabelFrame(self.root, text="Статус")
        info_frame.pack(pady=10, padx=10, fill=tk.X)
        
        self.info_label = tk.Label(info_frame, text=f"Известных лиц: {len(self.known_face_names)}", justify=tk.LEFT)
        self.info_label.pack(pady=5, padx=5, fill=tk.X)

    def start_camera(self):
        """Инициализация камеры с обработкой ошибок"""
        try:
            self.video_capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            if not self.video_capture.isOpened():
                messagebox.showerror("Ошибка", "Не удалось открыть камеру!")
                return
            
            self.running = True
            self.update_video()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка инициализации камеры: {str(e)}")

    def update_video(self):
        """Основной цикл обновления видео"""
        if not self.running:
            return
            
        try:
            ret, frame = self.video_capture.read()
            if not ret or frame is None:
                self.root.after(10, self.update_video)
                return
                
            # Обрабатываем каждый 3-й кадр для оптимизации
            self.frame_counter += 1
            if self.frame_counter % 3 == 0:
                self.process_frame(frame)
            
            # Обновляем GUI
            self.update_gui()
            self.root.after(10, self.update_video)
            
        except Exception as e:
            print(f"Ошибка в цикле видео: {str(e)}")
            self.root.after(10, self.update_video)

    def process_frame(self, frame):
        """Обработка и распознавание кадра"""
        try:
            # Создаем копию кадра для обработки
            frame_copy = frame.copy()
            
            # Уменьшаем размер для обработки
            small_frame = cv2.resize(frame_copy, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Детекция лиц
            if self.face_detection_enabled:
                self.face_locations = face_recognition.face_locations(
                    rgb_small_frame,
                    model="hog"  # Используем hog для CPU, cnn для GPU
                )
                
                # Распознавание
                if self.recognition_enabled and self.known_face_encodings and self.face_locations:
                    self.face_encodings = face_recognition.face_encodings(
                        rgb_small_frame, 
                        self.face_locations,
                        num_jitters=1
                    )
                    self.face_names = []
                    
                    for face_encoding in self.face_encodings:
                        # Сравнение с известными лицами
                        face_distances = face_recognition.face_distance(
                            self.known_face_encodings, 
                            face_encoding
                        )
                        best_match_index = np.argmin(face_distances)
                        
                        # Устанавливаем порог распознавания
                        if face_distances[best_match_index] < 0.6:  # Можно настроить (0.5-0.6 оптимально)
                            name = self.known_face_names[best_match_index]
                        else:
                            name = "Неизвестно"
                        
                        self.face_names.append(name)
                else:
                    self.face_names = ["Лицо"] * len(self.face_locations)
                
                # Отрисовка результатов
                for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                    # Масштабируем координаты обратно
                    top *= 4; right *= 4; bottom *= 4; left *= 4
                    
                    # Рисуем прямоугольник и подпись
                    color = (0, 255, 0) if name != "Неизвестно" else (0, 0, 255)
                    cv2.rectangle(frame_copy, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(frame_copy, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame_copy, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
            
            # Сохраняем обработанный кадр
            self.current_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"Ошибка обработки кадра: {str(e)}")
            self.current_frame = None

    def update_gui(self):
        """Обновление изображения в интерфейсе"""
        try:
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                img = Image.fromarray(self.current_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
        except Exception as e:
            print(f"Ошибка обновления GUI: {str(e)}")

    def add_face(self):
        """Добавление нового лица с улучшенной обработкой"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение с лицом",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            name = tk.simpledialog.askstring("Ввод", "Введите имя человека:")
            if name:
                try:
                    # Загружаем изображение и конвертируем в RGB
                    image = face_recognition.load_image_file(file_path)
                    
                    # Находим все лица на изображении
                    face_locations = face_recognition.face_locations(image)
                    
                    if not face_locations:
                        messagebox.showerror("Ошибка", "На изображении не найдено лиц!")
                        return
                    
                    # Получаем кодировки для всех найденных лиц
                    face_encodings = face_recognition.face_encodings(image, face_locations, num_jitters=3)
                    
                    if face_encodings:
                        # Добавляем только первое найденное лицо (можно изменить для добавления всех)
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(name)
                        
                        # Сохраняем изображение для будущего использования
                        if not os.path.exists("known_faces"):
                            os.makedirs("known_faces")
                        cv2.imwrite(f"known_faces/{name}.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                        
                        # Сохраняем данные
                        self.save_known_faces()
                        
                        self.info_label.config(text=f"Известных лиц: {len(self.known_face_names)}")
                        messagebox.showinfo("Успех", f"Лицо '{name}' успешно добавлено!")
                    else:
                        messagebox.showerror("Ошибка", "Не удалось получить кодировку лица!")
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Ошибка добавления лица: {str(e)}")

    def train_model(self):
        """Переобучение модели на всех лицах из папки known_faces"""
        try:
            if not os.path.exists("known_faces"):
                os.makedirs("known_faces")
                messagebox.showinfo("Информация", "Папка known_faces создана")
                return
                
            self.known_face_encodings = []
            self.known_face_names = []
            
            for file_name in os.listdir("known_faces"):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    name = os.path.splitext(file_name)[0]
                    image_path = os.path.join("known_faces", file_name)
                    
                    try:
                        image = face_recognition.load_image_file(image_path)
                        face_encodings = face_recognition.face_encodings(image, num_jitters=2)
                        
                        if face_encodings:
                            self.known_face_encodings.append(face_encodings[0])
                            self.known_face_names.append(name)
                            print(f"Добавлено лицо: {name}")
                        else:
                            print(f"Не удалось обработать: {file_name}")
                    except Exception as e:
                        print(f"Ошибка обработки {file_name}: {str(e)}")
                        continue
            
            self.save_known_faces()
            self.info_label.config(text=f"Известных лиц: {len(self.known_face_names)}")
            messagebox.showinfo("Успех", f"Модель обучена на {len(self.known_face_names)} лицах!")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обучения модели: {str(e)}")

    def verify_data(self):
        """Проверка целостности данных"""
        if self.verify_data_integrity():
            messagebox.showinfo("Проверка", "Данные в порядке!")
        else:
            messagebox.showwarning("Проверка", "Обнаружены проблемы с данными!")

    def verify_data_integrity(self):
        """Проверяет соответствие между кодировками и именами"""
        if len(self.known_face_encodings) != len(self.known_face_names):
            print("Ошибка: количество кодировок не соответствует количеству имен!")
            return False
        
        for i, encoding in enumerate(self.known_face_encodings):
            if len(encoding) != 128:
                print(f"Ошибка: некорректная кодировка для {self.known_face_names[i]}")
                return False
        
        return True

    def save_known_faces(self):
        """Сохранение кодировок в файл"""
        try:
            with open("face_encodings.pkl", "wb") as f:
                pickle.dump({
                    "encodings": self.known_face_encodings,
                    "names": self.known_face_names
                }, f)
        except Exception as e:
            print(f"Ошибка сохранения данных: {str(e)}")

    def load_known_faces(self):
        """Загрузка кодировок из файла"""
        try:
            if os.path.exists("face_encodings.pkl"):
                with open("face_encodings.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data["encodings"]
                    self.known_face_names = data["names"]
        except Exception as e:
            print(f"Ошибка загрузки данных: {str(e)}")
            self.known_face_encodings = []
            self.known_face_names = []

    def toggle_recognition(self):
        self.recognition_enabled = self.recognition_var.get()

    def toggle_detection(self):
        self.face_detection_enabled = self.detection_var.get()

    def quit_app(self):
        """Корректное завершение работы"""
        self.running = False
        if self.video_capture:
            self.video_capture.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
