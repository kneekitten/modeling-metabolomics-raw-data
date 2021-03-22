# modeling-metabolomics-raw-data

## Использование

Создание эксперимента на основе xlsx файла:
    
    py main.py -i name.xlsx -o name_experiment

Создание одного файла на основе xlsx файла:

    py main.py -i name.xlsx -o filename.cdf

Либо просто запустить без указания имен (будет читаться стандартный xlsx файл):

    py main.py

## Зависимости
    
Написано с использованием:

    python 3.9.2
    numpy 1.20.1
    scipy 1.6.1
    netCDF4 1.5.6
    openpyxl 3.0.7
    
### установка необходимых модулей
1) Запустить командную строку от имени администратора
2) Выполнить:

       py -m pip install -r requirements.txt
