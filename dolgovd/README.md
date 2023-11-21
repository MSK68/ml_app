# Приложение анализа эмоциональной окрашенности текста

Приложение определяет эмоциональную окраску текста.
Работает на основе предобученной модели SamLowe/roberta-base-go-emotions.
Модель обучена на наборе данных go_emotions.
Определяет вероятность отношения к 28-ми меткам эмоциональной окраски:
- admiration (восхищение)
- amusement (удовольствие)
- anger	(злость)
- annoyance (раздражение)
- approval (одобрение)
- caring (забота)
- confusion (смятение)
- curiosity (любопытство)
- desire (желание)
- disappointment (разочарование)
- disapproval (неодобрение)
- disgust (отвращение)
- embarrassment (позор)
- excitement (волнение)
- fear (страх)
- gratitude (благодарность)
- grief (скорбь)
- joy (радость)
- love (любовь)
- nervousness (нервозность)
- optimism (оптимизм)
- pride (гордость)
- realization (осознание)
- relief (облегчение)
- remorse (раскаяние)
- sadness (печаль)
- surprise (удивление)
- neutral (нейтральность)

Запуск приложение осуществляется командой streamlit run <путь_к_main.py>
