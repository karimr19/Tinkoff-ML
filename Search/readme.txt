1. (sudo) pip install flask
2. (sudo) python server.py
3. Зайти на http://localhost, ввести какой-нибудь запрос и убедиться, что всё работает
4. Найти и скачать какой-нибудь датасет (от нескольких сотен мегабайт до нескольких гигабайт) любых текстовых документов — например, песен
5. Построить какой-нибудь базовый индекс -- тут для начала не надо ничего сложного
6. Сделать что-нибудь чуть более сложное для самого ранжирования; стемминг, лемматизация, tf-idf, эмбеддинг word2vec-ом
7. (Опционально) разметить руками несколько десятков примеров и научиться мерять качество

В utils находятся функции обработки текста.
Программа может долго запускаться, для ускорения можно в файле search в функции build_index изменить значение
параметра limit в функции load_vectors.

Ссылка на ноутбук с подготовкой данных и анализом ранжирования: 
https://colab.research.google.com/drive/196cfJN_qJokWkfQa8E0RAl5eiGC2wtYY?usp=sharing