import numpy as np
import pandas as pd

data = {
    'Смартфон': ['iPhone 15 Pro', 'iPhone 16 Pro', 'iPhone 17 Pro'],
    'Цена': [70, 85, 120],
    'Автономность': [20, 25, 26],
    'Производительность': [9.0, 9.5, 9.8],
    'Вес': [187, 168, 154]
}

df = pd.DataFrame(data)
print("Исходные данные")
print(df.to_string(index=False))

weights = {
    'Цена': 0.4,        
    'Автономность': 0.3,     
    'Производительность': 0.2,  
    'Вес': 0.10                  
}

def normalize_data(df, weights):
    normalized = df.copy()

    for criterion, weight in weights.items():
        values = df[criterion].values
        min_val = np.min(values)
        max_val = np.max(values)

        if criterion in ['Цена', 'Вес']: 
            normalized[criterion] = (max_val - values) / (max_val - min_val)
        else:  
            normalized[criterion] = (values - min_val) / (max_val - min_val)

    return normalized

def calculate_weighted_sum(normalized_df, weights):
    scores = []

    for i in range(len(normalized_df)):
        score = 0
        for criterion, weight in weights.items():
            score += normalized_df[criterion].iloc[i] * weight
        scores.append(score)

    return scores

print("\nНормализация")
normalized_df = normalize_data(df, weights)
print(normalized_df.round(3).to_string(index=False))

scores = calculate_weighted_sum(normalized_df, weights)

result_df = df.copy()
result_df['Интегральная оценка'] = np.round(scores, 3)
result_df['Ранг'] = result_df['Интегральная оценка'].rank(ascending=False).astype(int)

print("Результат")
result_sorted = result_df.sort_values('Интегральная оценка', ascending=False)
print(result_sorted.to_string(index=False))

best_index = np.argmax(scores)
best_phone = df['Смартфон'].iloc[best_index]
best_score = scores[best_index]

print(f"Рекомендация: {best_phone}")
print(f"Интегральная оценка: {best_score:.3f}")
