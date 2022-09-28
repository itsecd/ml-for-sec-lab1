# Машинное обучение для задач информационной безопасности. 
Лаба 1. 
Пчелкина Юлия, группа 6232-010402D

Порядок выполнения
1. Базовый контест (делают все хоть как-то)
1.1. Обучить модель без каких-либо дополнительных условий, которая должна наилучшим образом отработать на тестовой выборке преподавателя с точки зрения F-меры
2. Вторичные контесты
2.1. Максимально логичным образом обучить качественную модель, использующую для работы только 10 признаков из всего исходного множества.
2.2. Обучить модель, обеспечивающую вероятность пропуска бота на уровне не выше 0.03, и имеющую насколько возможно низкую вероятность ложного обнаружения.



## Схема сдачи

1. Сделать форк данного репозитория
2. Выполнить задание согласно выбранному варианту
3. Сделать pull request в данный репозиторий
4. Получить результат в рамках code review с замечаниями по коду и с численным результатом работы кода на тестовой выборке.
5. При необходимости или при желании повысить качество работы программы повторять пп. 3-4. Когда желание что-то менять иссякнет, отправить pull request с чётким указанием на то, что это финальный вариант (может не отличаться от предыдущего, если все замечания уже были исправлены).
7. Получить approve после финального pull request'а
8. Во время онлайн-занятия защитить работу, ответить на вопросы преподавателя

## Более подробные рекомендации по работе с кодом

1. Форк *необходимо* сделать сразу. Для преподавателя это сигнализирует о том, что студент приступил к работе.
2. В описании репозитория нужно указать свои ФИО, номер группы и состав выполненных заданий (может меняться по ходу дела).
3. Желательно почаще делать коммиты. В идеале - как только решена некоторая промежуточная задача.
4. Коммиты *должны* иметь вменяемые описания.
5. Рекомендуется, чтобы ваш репозиторий содержал файлы [.gitignore](https://docs.github.com/en/get-started/getting-started-with-git/ignoring-files) (для них имеется набор [шаблонов](https://github.com/github/gitignore)) и [requirements.txt](https://www.jetbrains.com/help/pycharm/managing-dependencies.html#create-requirements)
