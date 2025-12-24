import cv2
import os
import pickle
import numpy as np
from datetime import datetime
import pyttsx3
import threading

class StableFaceRecognition:
    def __init__(self, dataset_path='dataset'):
        self.dataset_path = dataset_path
        self.known_faces = []  # Список словарей с данными о лицах
        self.encodings_file = 'stable_encodings.pkl'
        
        # Инициализация голосового движка
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
            self.voice_enabled = True
            print("Голосовой движок инициализирован")
        except Exception as e:
            print(f"Ошибка инициализации голоса: {e}")
            self.voice_enabled = False
        
        # Инициализация детектора лиц OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Журнал доступа
        self.access_log_file = 'access_log.csv'
        self.init_access_log()
    
    def init_access_log(self):
        """Инициализация файла журнала доступа"""
        if not os.path.exists(self.access_log_file):
            with open(self.access_log_file, 'w', encoding='utf-8') as f:
                f.write('Дата,Время,Имя,Статус,Уверенность\n')
    
    def log_access(self, name, status, confidence):
        """Записывает событие в журнал доступа"""
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        
        with open(self.access_log_file, 'a', encoding='utf-8') as f:
            f.write(f'{date_str},{time_str},{name},{status},{confidence:.1f}%\n')
    
    def speak_sync(self, text):
        """Озвучивает текст синхронно (ждет завершения)"""
        if not self.voice_enabled:
            print(f"Озвучка отключена: {text}")
            return
            
        try:
            print(f"Озвучиваю: {text}")
            
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            
            import platform
            if platform.system() == 'Darwin':  # macOS
                import subprocess
                try:
                    subprocess.run(['say', '-r', '180', text], check=True)
                    print("Озвучка завершена (системный TTS)")
                    return
                except Exception as e:
                    print(f"Ошибка системного TTS: {e}, пробую pyttsx3")
            
            voices = engine.getProperty('voices')
            if voices:
                engine.setProperty('voice', voices[0].id)
                print(f"Использую голос: {voices[0].name}")
            
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            
            print("Озвучка завершена")
        except Exception as e:
            print(f"Ошибка озвучки: {e}")
            try:
                print("Пробую английскую версию...")
                english_text = text.replace("Здравствуйте", "Hello").replace("Ошибка", "Error").replace("Доступ", "Access")
                if platform.system() == 'Darwin':
                    import subprocess
                    subprocess.run(['say', english_text], check=True)
                else:
                    engine = pyttsx3.init()
                    engine.say(english_text)
                    engine.runAndWait()
                    engine.stop()
                print("Английская озвучка завершена")
            except Exception as e2:
                print(f"Ошибка английской озвучки: {e2}")
    
    def speak(self, text):
        """Озвучивает текст в отдельном потоке"""
        if not self.voice_enabled:
            print(f"Озвучка отключена: {text}")
            return
            
        def speak_thread():
            try:
                print(f"Озвучиваю: {text}")
                
                # Создаем новый экземпляр движка для каждой озвучки
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)
                
                # На macOS пробуем использовать системный TTS через say
                import platform
                if platform.system() == 'Darwin':  # macOS
                    import subprocess
                    try:
                        # Используем системную команду say для macOS
                        subprocess.run(['say', '-r', '180', text], check=True)
                        print("Озвучка завершена (системный TTS)")
                        return
                    except Exception as e:
                        print(f"Ошибка системного TTS: {e}, пробую pyttsx3")
                
                # Если системный TTS не сработал или не macOS, используем pyttsx3
                voices = engine.getProperty('voices')
                if voices:
                    # Выбираем первый доступный голос
                    engine.setProperty('voice', voices[0].id)
                    print(f"Использую голос: {voices[0].name}")
                
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                
                print("Озвучка завершена")
            except Exception as e:
                print(f"Ошибка озвучки: {e}")
        
        thread = threading.Thread(target=speak_thread)
        thread.daemon = True
        thread.start()
        
        # Небольшая задержка для синхронизации
        import time
        time.sleep(0.1)
    
    def extract_face_features(self, face_roi):
        """Извлекает признаки лица для сравнения"""
        # Нормализуем размер
        face_roi = cv2.resize(face_roi, (100, 100))
        
        # Применяем гауссово размытие для сглаживания
        face_roi = cv2.GaussianBlur(face_roi, (5, 5), 0)
        
        # Вычисляем гистограмму
        hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
        
        # Нормализуем гистограмму
        cv2.normalize(hist, hist)
        
        return hist.flatten()
    
    def load_faces_from_dataset(self):
        """Загружает лица из датасета"""
        print("Обновление: Загрузка лиц из датасета...")
        
        if not os.path.exists(self.dataset_path):
            print(f"Ошибка: Папка {self.dataset_path} не найдена!")
            return False
        
        self.known_faces = []
        
        for person_name in os.listdir(self.dataset_path):
            person_path = os.path.join(self.dataset_path, person_name)
            
            if not os.path.isdir(person_path):
                continue
                
            print(f"Обработка: Обрабатываю фотографии {person_name}...")
            person_faces = []
            
            for image_file in os.listdir(person_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_path, image_file)
                    
                    try:
                        img = cv2.imread(image_path)
                        if img is None:
                            print(f"  Ошибка: {image_file} - не удалось загрузить")
                            continue
                        
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
                        
                        if len(faces) > 0:
                            # Берем самое большое лицо
                            x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
                            face_roi = gray[y:y+h, x:x+w]
                            
                            # Извлекаем признаки
                            features = self.extract_face_features(face_roi)
                            
                            person_faces.append({
                                'features': features,
                                'image_file': image_file
                            })
                            
                            print(f"  Успешно: {image_file}")
                        else:
                            print(f"  Ошибка: {image_file} - лицо не найдено")
                            
                    except Exception as e:
                        print(f"  ⚠️ {image_file} - ошибка: {e}")
            
            if person_faces:
                self.known_faces.append({
                    'name': person_name,
                    'faces': person_faces
                })
                print(f"Пользователь: {person_name}: {len(person_faces)} лиц добавлено")
        
        total_faces = sum(len(person['faces']) for person in self.known_faces)
        print(f"Готово: Загружено {total_faces} лиц для {len(self.known_faces)} человек")
        return len(self.known_faces) > 0
    
    def save_encodings(self):
        """Сохраняет данные о лицах в файл"""
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(self.known_faces, f)
        print(f"Сохранение: Данные сохранены в {self.encodings_file}")
    
    def load_encodings(self):
        """Загружает данные о лицах из файла"""
        if os.path.exists(self.encodings_file):
            with open(self.encodings_file, 'rb') as f:
                self.known_faces = pickle.load(f)
            
            total_faces = sum(len(person['faces']) for person in self.known_faces)
            print(f"Загрузка: Загружено {total_faces} лиц для {len(self.known_faces)} человек")
            return True
        return False
    
    def recognize_face(self, face_roi):
        """Распознает лицо по извлеченным признакам"""
        if not self.known_faces:
            return "Неизвестный", 0
        
        # Извлекаем признаки тестового лица
        test_features = self.extract_face_features(face_roi)
        
        best_match = "Неизвестный"
        best_confidence = 0
        
        # Сравниваем с каждым известным человеком
        for person in self.known_faces:
            person_name = person['name']
            person_confidences = []
            
            # Сравниваем со всеми лицами этого человека
            for face_data in person['faces']:
                known_features = face_data['features']
                
                # Вычисляем корреляцию между гистограммами
                correlation = cv2.compareHist(test_features.astype(np.float32), 
                                            known_features.astype(np.float32), 
                                            cv2.HISTCMP_CORREL)
                person_confidences.append(correlation)
            
            # Берем максимальную уверенность для этого человека
            if person_confidences:
                max_confidence = max(person_confidences)
                
                if max_confidence > best_confidence and max_confidence > 0.4:  # Порог
                    best_confidence = max_confidence
                    best_match = person_name
        
        return best_match, best_confidence * 100
    
    def start_camera_recognition(self):
        """Запускает распознавание лиц с камеры"""
        print("Камера: Запуск камеры для распознавания лиц...")
        print("Нажмите 'q' для выхода")
        
        self.speak("Система распознавания лиц запущена")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Ошибка: Не удалось открыть камеру!")
            self.speak("Ошибка камеры")
            return
        
        frame_count = 0
        last_recognition = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Обрабатываем каждый 10-й кадр для стабильности
            frame_count += 1
            if frame_count % 10 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
                
                for (x, y, w, h) in faces:
                    # Извлекаем область лица
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Распознаем лицо
                    name, confidence = self.recognize_face(face_roi)
                    
                    # Рисуем рамку
                    color = (0, 255, 0) if name != "Неизвестный" and confidence > 50 else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Добавляем текст
                    text = f"{name} ({confidence:.1f}%)"
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Озвучиваем только при новом распознавании
                    current_time = datetime.now()
                    if name not in last_recognition or (current_time - last_recognition[name]).seconds > 3:
                        if name != "Неизвестный" and confidence > 50:
                            # Текстовое сообщение
                            print(f"Успешно: Добро пожаловать, {name}! Доступ разрешён.")
                            print(f"   Уровень уверенности: {confidence:.1f}%")
                            
                            # Озвучивание
                            self.speak(f"Здравствуйте, {name}!")
                            
                            # Логирование
                            self.log_access(name, "вход разрешён", confidence)
                        else:
                            # Текстовое сообщение
                            print(f"Ошибка: Ошибка: лицо не распознано. Доступ запрещён.")
                            if confidence > 0:
                                print(f"   Уровень уверенности: {confidence:.1f}%")
                            
                            # Озвучивание
                            self.speak("Ошибка: лицо не распознано. Доступ запрещён.")
                            
                            # Логирование
                            self.log_access("Неизвестный", "доступ запрещён", confidence)
                        
                        last_recognition[name] = current_time
            
            # Добавляем инструкции на экран
            cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Stable Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.speak("Система распознавания отключена")
        print("Камера: Камера отключена")
    
    def recognize_from_image(self, image_path):
        """Распознает лицо на изображении"""
        if not os.path.exists(image_path):
            print(f"Ошибка: Файл {image_path} не найден!")
            return None
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Ошибка: Не удалось загрузить изображение {image_path}")
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        if len(faces) == 0:
            print("Ошибка: Ошибка: лицо не распознано. Доступ запрещён.")
            print("   Причина: лица не найдены на изображении")
            self.speak_sync("Ошибка: лицо не распознано. Доступ запрещён.")
            return None
        
        results = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            name, confidence = self.recognize_face(face_roi)
            
            results.append({
                'name': name,
                'confidence': confidence,
                'location': (x, y, w, h)
            })
            
            if name != "Неизвестный" and confidence > 40:
                # Текстовое сообщение
                print(f"Успешно: Добро пожаловать, {name}! Доступ разрешён.")
                print(f"   Уровень уверенности: {confidence:.1f}%")
                
                # Озвучивание (пробуем и русский и английский)
                try:
                    self.speak_sync(f"Здравствуйте, {name}!")
                except:
                    self.speak_sync(f"Hello, {name}!")
                
                # Логирование
                self.log_access(name, "распознан на фото", confidence)
            else:
                # Текстовое сообщение
                print(f"Ошибка: Ошибка: лицо не распознано. Доступ запрещён.")
                if confidence > 0:
                    print(f"   Уровень уверенности: {confidence:.1f}%")
                
                # Озвучивание (пробуем и русский и английский)
                try:
                    self.speak_sync("Ошибка: лицо не распознано. Доступ запрещён.")
                except:
                    self.speak_sync("Error: face not recognized. Access denied.")
                
                # Логирование
                self.log_access("Неизвестный", "доступ запрещён на фото", confidence)
        
        return results

def main():
    """Главная функция"""
    print("=== СТАБИЛЬНАЯ СИСТЕМА РАСПОЗНАВАНИЯ ЛИЦ ===")
    
    fr = StableFaceRecognition()
    
    # Пытаемся загрузить сохраненные данные
    if not fr.load_encodings():
        print("Данные: Данные не найдены. Создаем новые...")
        if fr.load_faces_from_dataset():
            fr.save_encodings()
        else:
            print("Ошибка: Не удалось загрузить датасет!")
            print("Подсказка: Добавьте фотографии в папку dataset/")
            return
    
    if fr.voice_enabled:
        fr.speak("Добро пожаловать в систему распознавания лиц")
    
    while True:
        print("\nМеню: Выберите действие:")
        print("1. Обработка: Распознать лицо на фотографии")
        print("2. Камера: Запустить распознавание с камеры")
        print("3. Обновление: Обновить датасет")
        print("4. Журнал: Показать журнал доступа")
        print("5. Тест озвучки")
        print("6. Ошибка: Выход")
        
        choice = input("Ваш выбор (1-6): ").strip()
        
        if choice == '1':
            image_path = input("Путь к изображению: ").strip()
            fr.recognize_from_image(image_path)
            
        elif choice == '2':
            fr.start_camera_recognition()
            
        elif choice == '3':
            print("Обновление: Обновление датасета...")
            if fr.load_faces_from_dataset():
                fr.save_encodings()
                if fr.voice_enabled:
                    fr.speak("Датасет обновлен")
                print("Успешно: Датасет обновлен!")
            else:
                print("Ошибка: Ошибка обновления датасета")
                
        elif choice == '4':
            if os.path.exists(fr.access_log_file):
                print("\nЖурнал: Журнал доступа:")
                with open(fr.access_log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines[-10:]:  # Показываем последние 10 записей
                        print(line.strip())
            else:
                print("Журнал: Журнал доступа пуст")
                
        elif choice == '5':
            if fr.voice_enabled:
                fr.speak_sync("Тест озвучки работает отлично!")
                print("Тест озвучки выполнен")
            else:
                print("Ошибка: Озвучка недоступна")
                
        elif choice == '6':
            if fr.voice_enabled:
                fr.speak("До свидания!")
            print("Прощание: До свидания!")
            break
            
        else:
            print("Ошибка: Неверный выбор!")

if __name__ == "__main__":
    main()