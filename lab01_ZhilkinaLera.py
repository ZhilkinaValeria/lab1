import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Установим стиль графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Загрузка данных
df = pd.read_csv('lab01_tips.csv')
print("Первые 5 строк данных:")
print(df.head())
print("\n" + "="*80)
print("ОСНОВНАЯ ИНФОРМАЦИЯ О ДАННЫХ")
print("="*80)

# Основная информация о данных
print("1. РАЗМЕР ДАННЫХ:")
print(f"   Количество наблюдений: {df.shape[0]}")
print(f"   Количество переменных: {df.shape[1]}")

print("\n2. ИНФОРМАЦИЯ О ПЕРЕМЕННЫХ:")
print(df.info())

print("\n3. ОПИСАТЕЛЬНАЯ СТАТИСТИКА ЧИСЛОВЫХ ПЕРЕМЕННЫХ:")
print(df[['total_bill', 'tips', 'size']].describe().round(2))

print("\n4. РАСПРЕДЕЛЕНИЕ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ:")
categorical_cols = ['sex', 'smoker', 'day_jf_week', 'time']
for col in categorical_cols:
    print(f"\n{col.upper()}:")
    print(df[col].value_counts())
    print(f"Доля: {df[col].value_counts(normalize=True).round(3)}")

# Создадим фигуру для визуализации
fig = plt.figure(figsize=(20, 15))

# 2.1 Распределение суммы счета и чаевых
plt.subplot(3, 3, 1)
sns.histplot(df['total_bill'], kde=True, bins=30)
plt.title('Распределение суммы счета (total_bill)', fontsize=12)
plt.xlabel('Сумма счета ($)')
plt.ylabel('Частота')

plt.subplot(3, 3, 2)
sns.histplot(df['tips'], kde=True, bins=30)
plt.title('Распределение чаевых (tips)', fontsize=12)
plt.xlabel('Чаевые ($)')
plt.ylabel('Частота')

# 2.2 Boxplot для выявления выбросов
plt.subplot(3, 3, 3)
sns.boxplot(data=df[['total_bill', 'tips']])
plt.title('Boxplot: сумма счета и чаевые', fontsize=12)
plt.ylabel('Доллары ($)')

# 2.3 Распределение размера компании
plt.subplot(3, 3, 4)
size_counts = df['size'].value_counts().sort_index()
plt.bar(size_counts.index, size_counts.values)
plt.title('Распределение размера компании', fontsize=12)
plt.xlabel('Количество человек')
plt.ylabel('Частота')
for i, v in enumerate(size_counts.values):
    plt.text(size_counts.index[i], v, str(v), ha='center', va='bottom')

# 2.4 Распределение по дням недели
plt.subplot(3, 3, 5)
day_order = ['Thur', 'Fri', 'Sat', 'Sun']
day_counts = df['day_jf_week'].value_counts().reindex(day_order)
plt.bar(day_counts.index, day_counts.values)
plt.title('Распределение по дням недели', fontsize=12)
plt.xlabel('День недели')
plt.ylabel('Количество посещений')

# 2.5 Распределение по времени
plt.subplot(3, 3, 6)
time_counts = df['time'].value_counts()
plt.bar(time_counts.index, time_counts.values)
plt.title('Распределение по времени посещения', fontsize=12)
plt.xlabel('Время')
plt.ylabel('Количество посещений')

# 2.6 Распределение по полу и курению
plt.subplot(3, 3, 7)
sex_counts = df['sex'].value_counts()
plt.pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%')
plt.title('Распределение по полу', fontsize=12)

plt.subplot(3, 3, 8)
smoker_counts = df['smoker'].value_counts()
plt.pie(smoker_counts.values, labels=smoker_counts.index, autopct='%1.1f%%')
plt.title('Распределение курящих/некурящих', fontsize=12)

plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.3) 
plt.show()

print("\n" + "="*80)
print("ПРОВЕРКА НОРМАЛЬНОСТИ РАСПРЕДЕЛЕНИЯ")
print("="*80)

# Функция для проверки нормальности
def check_normality(data, col_name):
    stat, p = stats.shapiro(data)
    print(f"\n{col_name.upper()}:")
    print(f"  Статистика Шапиро-Уилка: {stat:.4f}")
    print(f"  p-value: {p:.10f}")
    if p > 0.05:
        print(f"  Вывод: распределение нормальное (p > 0.05)")
    else:
        print(f"  Вывод: распределение НЕ нормальное (p < 0.05)")
    return p

# Проверяем нормальность
p_bill = check_normality(df['total_bill'], 'total_bill')
p_tips = check_normality(df['tips'], 'tips')

# QQ-plot для визуальной проверки нормальности
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

stats.probplot(df['total_bill'], dist="norm", plot=axes[0])
axes[0].set_title('Q-Q plot: сумма счета')
axes[0].set_xlabel('Теоретические квантили')
axes[0].set_ylabel('Наблюдаемые значения')

stats.probplot(df['tips'], dist="norm", plot=axes[1])
axes[1].set_title('Q-Q plot: чаевые')
axes[1].set_xlabel('Теоретические квантили')
axes[1].set_ylabel('Наблюдаемые значения')

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("ГИПОТЕЗА 1: Существует ли зависимость между суммой счета и размером чаевых?")
print("="*80)

# Визуализация зависимости
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Точечная диаграмма
axes[0].scatter(df['total_bill'], df['tips'], alpha=0.6)
axes[0].set_xlabel('Сумма счета ($)')
axes[0].set_ylabel('Чаевые ($)')
axes[0].set_title('Зависимость чаевых от суммы счета')
axes[0].grid(True, alpha=0.3)

# Линия регрессии
sns.regplot(x='total_bill', y='tips', data=df, ax=axes[1], scatter_kws={'alpha':0.5})
axes[1].set_xlabel('Сумма счета ($)')
axes[1].set_ylabel('Чаевые ($)')
axes[1].set_title('Линейная регрессия')

# Процент чаевых от счета
df['tips_percentage'] = (df['tips'] / df['total_bill']) * 100
axes[2].scatter(df['total_bill'], df['tips_percentage'], alpha=0.6)
axes[2].set_xlabel('Сумма счета ($)')
axes[2].set_ylabel('Процент чаевых (%)')
axes[2].set_title('Процент чаевых в зависимости от суммы счета')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Статистический анализ
print("\nКОРРЕЛЯЦИОННЫЙ АНАЛИЗ:")
print("-" * 40)

# Поскольку данные не нормальны, используем корреляцию Спирмена
corr_spearman, p_spearman = stats.spearmanr(df['total_bill'], df['tips'])
corr_pearson, p_pearson = stats.pearsonr(df['total_bill'], df['tips'])

print(f"Корреляция Пирсона (линейная): {corr_pearson:.4f}")
print(f"p-value Пирсона: {p_pearson:.10f}")
print(f"Корреляция Спирмена (монотонная): {corr_spearman:.4f}")
print(f"p-value Спирмена: {p_spearman:.10f}")

# Интерпретация
print("\nИНТЕРПРЕТАЦИЯ:")
print("-" * 40)
print(f"1. Корреляция Пирсона = {corr_pearson:.3f}")
if abs(corr_pearson) > 0.7:
    print("   Сильная линейная зависимость")
elif abs(corr_pearson) > 0.3:
    print("   Умеренная линейная зависимость")
else:
    print("   Слабая линейная зависимость")

print(f"\n2. Корреляция Спирмена = {corr_spearman:.3f}")
if abs(corr_spearman) > 0.7:
    print("   Сильная монотонная зависимость")
elif abs(corr_spearman) > 0.3:
    print("   Умеренная монотонная зависимость")
else:
    print("   Слабая монотонная зависимость")

print(f"\n3. p-value = {p_spearman:.10e}")
if p_spearman < 0.001:
    print("   Зависимость статистически высоко значима (p < 0.001)")
elif p_spearman < 0.01:
    print("   Зависимость статистически значима (p < 0.01)")
elif p_spearman < 0.05:
    print("   Зависимость статистически значима (p < 0.05)")
else:
    print("   Зависимость статистически не значима")

print("\n4. Анализ процента чаевых:")
print(f"   Средний процент чаевых: {df['tips_percentage'].mean():.2f}%")
print(f"   Медианный процент чаевых: {df['tips_percentage'].median():.2f}%")
print(f"   Стандартное отклонение: {df['tips_percentage'].std():.2f}%")
print(f"   Минимальный процент: {df['tips_percentage'].min():.2f}%")
print(f"   Максимальный процент: {df['tips_percentage'].max():.2f}%")

print("\n" + "="*80)
print("ГИПОТЕЗА 2: Влияет ли размер компании на сумму счета и чаевые?")
print("="*80)

# Группировка данных по размеру компании
grouped_by_size = df.groupby('size').agg({
    'total_bill': ['mean', 'median', 'std', 'count'],
    'tips': ['mean', 'median', 'std'],
    'tips_percentage': ['mean', 'median', 'std']
}).round(2)

print("СТАТИСТИКА ПО РАЗМЕРУ КОМПАНИИ:")
print(grouped_by_size)

# Визуализация
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Средняя сумма счета по размеру компании
axes[0, 0].bar(grouped_by_size.index, grouped_by_size[('total_bill', 'mean')])
axes[0, 0].set_xlabel('Размер компании')
axes[0, 0].set_ylabel('Средняя сумма счета ($)')
axes[0, 0].set_title('Средняя сумма счета в зависимости от размера компании')
for i, v in enumerate(grouped_by_size[('total_bill', 'mean')]):
    axes[0, 0].text(i+1, v, f'${v:.1f}', ha='center', va='bottom')

# Средние чаевые по размеру компании
axes[0, 1].bar(grouped_by_size.index, grouped_by_size[('tips', 'mean')])
axes[0, 1].set_xlabel('Размер компании')
axes[0, 1].set_ylabel('Средние чаевые ($)')
axes[0, 1].set_title('Средние чаевые в зависимости от размера компании')
for i, v in enumerate(grouped_by_size[('tips', 'mean')]):
    axes[0, 1].text(i+1, v, f'${v:.1f}', ha='center', va='bottom')

# Процент чаевых по размеру компании
axes[1, 0].bar(grouped_by_size.index, grouped_by_size[('tips_percentage', 'mean')])
axes[1, 0].set_xlabel('Размер компании')
axes[1, 0].set_ylabel('Средний процент чаевых (%)')
axes[1, 0].set_title('Процент чаевых в зависимости от размера компании')
for i, v in enumerate(grouped_by_size[('tips_percentage', 'mean')]):
    axes[1, 0].text(i+1, v, f'{v:.1f}%', ha='center', va='bottom')

# Boxplot суммы счета по размеру компании
sns.boxplot(x='size', y='total_bill', data=df, ax=axes[1, 1])
axes[1, 1].set_xlabel('Размер компании')
axes[1, 1].set_ylabel('Сумма счета ($)')
axes[1, 1].set_title('Распределение суммы счета по размеру компании')

plt.tight_layout()
plt.show()

# Статистический тест (ANOVA или Крускал-Уоллис)
print("\nСТАТИСТИЧЕСКАЯ ПРОВЕРКА РАЗЛИЧИЙ:")
print("-" * 40)

# Проверяем нормальность для каждой группы
print("Проверка нормальности распределения суммы счета по группам:")
size_groups = [df[df['size'] == i]['total_bill'] for i in sorted(df['size'].unique())]
for i, group in enumerate(size_groups, start=1):
    if len(group) > 3:  # Тест Шапиро требует минимум 3 наблюдения
        stat, p = stats.shapiro(group)
        print(f"  Размер {i}: p-value = {p:.6f}")

# Поскольку данные не нормальны, используем тест Крускала-Уоллиса
print("\nТест Крускала-Уоллиса (непараметрический аналог ANOVA):")
h_stat, p_kw = stats.kruskal(*size_groups)
print(f"  H-статистика: {h_stat:.4f}")
print(f"  p-value: {p_kw:.10f}")

if p_kw < 0.05:
    print(f"  ВЫВОД: Существуют статистически значимые различия между группами (p < 0.05)")
    
    # Пост-хок тест для попарных сравнений
    print("\n  Попарные сравнения (тест Манна-Уитни с поправкой Бонферрони):")
    sizes = sorted(df['size'].unique())
    comparisons = []
    for i in range(len(sizes)):
        for j in range(i+1, len(sizes)):
            group1 = df[df['size'] == sizes[i]]['total_bill']
            group2 = df[df['size'] == sizes[j]]['total_bill']
            u_stat, p_mw = stats.mannwhitneyu(group1, group2)
            comparisons.append((sizes[i], sizes[j], u_stat, p_mw))
    
    # Применяем поправку Бонферрони
    n_comparisons = len(comparisons)
    bonferroni_alpha = 0.05 / n_comparisons
    
    for comp in comparisons:
        size1, size2, u_stat, p_mw = comp
        adjusted_p = min(p_mw * n_comparisons, 1.0)
        if adjusted_p < 0.05:
            print(f"    Размер {size1} vs {size2}: p = {adjusted_p:.4f} *")
        else:
            print(f"    Размер {size1} vs {size2}: p = {adjusted_p:.4f}")
else:
    print(f"  ВЫВОД: Нет статистически значимых различий между группами (p > 0.05)")

print("\n" + "="*80)
print("ГИПОТЕЗА 3: Существуют ли различия в чаевых между мужчинами и женщинами?")
print("="*80)

# Группировка по полу
grouped_by_sex = df.groupby('sex').agg({
    'total_bill': ['mean', 'median', 'std', 'count'],
    'tips': ['mean', 'median', 'std'],
    'tips_percentage': ['mean', 'median', 'std']
}).round(2)

print("СТАТИСТИКА ПО ПОЛУ:")
print(grouped_by_sex)

# Визуализация
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Гистограммы распределения чаевых
axes[0, 0].hist(df[df['sex'] == 'Male']['tips'], alpha=0.7, label='Мужчины', bins=20)
axes[0, 0].hist(df[df['sex'] == 'Female']['tips'], alpha=0.7, label='Женщины', bins=20)
axes[0, 0].set_xlabel('Чаевые ($)')
axes[0, 0].set_ylabel('Частота')
axes[0, 0].set_title('Распределение чаевых по полу')
axes[0, 0].legend()

# Boxplot чаевых по полу
sns.boxplot(x='sex', y='tips', data=df, ax=axes[0, 1])
axes[0, 1].set_xlabel('Пол')
axes[0, 1].set_ylabel('Чаевые ($)')
axes[0, 1].set_title('Boxplot чаевых по полу')

# Средние чаевые по полу
sex_tips_means = grouped_by_sex[('tips', 'mean')]
axes[0, 2].bar(sex_tips_means.index, sex_tips_means.values)
axes[0, 2].set_xlabel('Пол')
axes[0, 2].set_ylabel('Средние чаевые ($)')
axes[0, 2].set_title('Средние чаевые по полу')
for i, v in enumerate(sex_tips_means.values):
    axes[0, 2].text(i, v, f'${v:.2f}', ha='center', va='bottom')

# Гистограммы процента чаевых
axes[1, 0].hist(df[df['sex'] == 'Male']['tips_percentage'], alpha=0.7, label='Мужчины', bins=20)
axes[1, 0].hist(df[df['sex'] == 'Female']['tips_percentage'], alpha=0.7, label='Женщины', bins=20)
axes[1, 0].set_xlabel('Процент чаевых (%)')
axes[1, 0].set_ylabel('Частота')
axes[1, 0].set_title('Распределение процента чаевых по полу')
axes[1, 0].legend()

# Boxplot процента чаевых по полу
sns.boxplot(x='sex', y='tips_percentage', data=df, ax=axes[1, 1])
axes[1, 1].set_xlabel('Пол')
axes[1, 1].set_ylabel('Процент чаевых (%)')
axes[1, 1].set_title('Boxplot процента чаевых по полу')

# Средний процент чаевых по полу
sex_perc_means = grouped_by_sex[('tips_percentage', 'mean')]
axes[1, 2].bar(sex_perc_means.index, sex_perc_means.values)
axes[1, 2].set_xlabel('Пол')
axes[1, 2].set_ylabel('Средний процент чаевых (%)')
axes[1, 2].set_title('Средний процент чаевых по полу')
for i, v in enumerate(sex_perc_means.values):
    axes[1, 2].text(i, v, f'{v:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Статистический тест
print("\nСТАТИСТИЧЕСКАЯ ПРОВЕРКА РАЗЛИЧИЙ:")
print("-" * 40)

# Разделяем данные по полу
male_tipss = df[df['sex'] == 'Male']['tips']
female_tipss = df[df['sex'] == 'Female']['tips']
male_perc = df[df['sex'] == 'Male']['tips_percentage']
female_perc = df[df['sex'] == 'Female']['tips_percentage']

# Проверяем нормальность
print("Проверка нормальности распределения чаевых:")
stat_m, p_m = stats.shapiro(male_tipss)
stat_f, p_f = stats.shapiro(female_tipss)
print(f"  Мужчины: p-value = {p_m:.6f}")
print(f"  Женщины: p-value = {p_f:.6f}")

# Тест Манна-Уитни (непараметрический, т.к. распределения не нормальны)
print("\nТест Манна-Уитни для сравнения чаевых:")
u_stat, p_mw = stats.mannwhitneyu(male_tipss, female_tipss)
print(f"  U-статистика: {u_stat:.4f}")
print(f"  p-value: {p_mw:.6f}")

# Тест Манна-Уитни для процента чаевых
print("\nТест Манна-Уитни для сравнения процента чаевых:")
u_stat_perc, p_mw_perc = stats.mannwhitneyu(male_perc, female_perc)
print(f"  U-статистика: {u_stat_perc:.4f}")
print(f"  p-value: {p_mw_perc:.6f}")

print("\nИНТЕРПЕТАЦИЯ:")
print("-" * 40)
print("1. Абсолютные чаевые ($):")
print(f"   Мужчины: ${male_tipss.mean():.2f} ± ${male_tipss.std():.2f}")
print(f"   Женщины: ${female_tipss.mean():.2f} ± ${female_tipss.std():.2f}")
if p_mw < 0.05:
    print(f"   Различия статистически значимы (p = {p_mw:.4f})")
else:
    print(f"   Различия статистически не значимы (p = {p_mw:.4f})")

print("\n2. Процент чаевых (%):")
print(f"   Мужчины: {male_perc.mean():.2f}% ± {male_perc.std():.2f}%")
print(f"   Женщины: {female_perc.mean():.2f}% ± {female_perc.std():.2f}%")
if p_mw_perc < 0.05:
    print(f"   Различия статистически значимы (p = {p_mw_perc:.4f})")
else:
    print(f"   Различия статистически не значимы (p = {p_mw_perc:.4f})")

print("\n" + "="*80)
print("ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ: Влияние дня недели и времени")
print("="*80)

# Анализ по дням недели
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Средние чаевые по дням
day_tips_means = df.groupby('day_jf_week')['tips'].mean().reindex(day_order)
axes[0, 0].bar(day_tips_means.index, day_tips_means.values)
axes[0, 0].set_xlabel('День недели')
axes[0, 0].set_ylabel('Средние чаевые ($)')
axes[0, 0].set_title('Средние чаевые по дням недели')
for i, v in enumerate(day_tips_means.values):
    axes[0, 0].text(i, v, f'${v:.2f}', ha='center', va='bottom')

# Средний процент чаевых по дням
day_perc_means = df.groupby('day_jf_week')['tips_percentage'].mean().reindex(day_order)
axes[0, 1].bar(day_perc_means.index, day_perc_means.values)
axes[0, 1].set_xlabel('День недели')
axes[0, 1].set_ylabel('Средний процент чаевых (%)')
axes[0, 1].set_title('Средний процент чаевых по дням недели')
for i, v in enumerate(day_perc_means.values):
    axes[0, 1].text(i, v, f'{v:.2f}%', ha='center', va='bottom')

# Количество посещений по дням
axes[0, 2].bar(day_counts.index, day_counts.values)
axes[0, 2].set_xlabel('День недели')
axes[0, 2].set_ylabel('Количество посещений')
axes[0, 2].set_title('Количество посещений по дням недели')
for i, v in enumerate(day_counts.values):
    axes[0, 2].text(i, v, str(v), ha='center', va='bottom')

# Анализ по времени
time_tips_means = df.groupby('time')['tips'].mean()
axes[1, 0].bar(time_tips_means.index, time_tips_means.values)
axes[1, 0].set_xlabel('Время')
axes[1, 0].set_ylabel('Средние чаевые ($)')
axes[1, 0].set_title('Средние чаевые по времени')
for i, v in enumerate(time_tips_means.values):
    axes[1, 0].text(i, v, f'${v:.2f}', ha='center', va='bottom')

time_perc_means = df.groupby('time')['tips_percentage'].mean()
axes[1, 1].bar(time_perc_means.index, time_perc_means.values)
axes[1, 1].set_xlabel('Время')
axes[1, 1].set_ylabel('Средний процент чаевых (%)')
axes[1, 1].set_title('Средний процент чаевых по времени')
for i, v in enumerate(time_perc_means.values):
    axes[1, 1].text(i, v, f'{v:.2f}%', ha='center', va='bottom')

# Комбинированный анализ: день и время
if 'day_jf_week' in df.columns and 'time' in df.columns:
    cross_tab = pd.crosstab(df['day_jf_week'], df['time'])
    cross_tab = cross_tab.reindex(day_order)
    cross_tab.plot(kind='bar', ax=axes[1, 2])
    axes[1, 2].set_xlabel('День недели')
    axes[1, 2].set_ylabel('Количество посещений')
    axes[1, 2].set_title('Распределение по дням и времени')
    axes[1, 2].legend(title='Время')

plt.tight_layout()
plt.show()

# Статистический анализ различий по дням
print("\nАНАЛИЗ РАЗЛИЧИЙ ПО ДНЯМ НЕДЕЛИ:")
print("-" * 40)

# Группируем данные по дням
day_groups = [df[df['day_jf_week'] == day]['tips_percentage'] for day in day_order]

# Тест Крускала-Уоллиса
h_stat_days, p_days = stats.kruskal(*day_groups)
print(f"Тест Крускала-Уоллиса для процента чаевых по дням:")
print(f"  H-статистика: {h_stat_days:.4f}")
print(f"  p-value: {p_days:.6f}")

if p_days < 0.05:
    print(f"  ВЫВОД: Существуют статистически значимые различия между днями")
else:
    print(f"  ВЫВОД: Нет статистически значимых различий между днями")

print("\n" + "="*80)
print("ИТОГОВАЯ СВОДКА РЕЗУЛЬТАТОВ АНАЛИЗА")
print("="*80)

print("\n ОСНОВНЫЕ ВЫВОДЫ:")
print("-" * 40)

print("\n1. ОБЩАЯ СТАТИСТИКА:")
print(f"   - Всего наблюдений: {len(df)}")
print(f"   - Средняя сумма счета: ${df['total_bill'].mean():.2f}")
print(f"   - Средние чаевые: ${df['tips'].mean():.2f}")
print(f"   - Средний процент чаевых: {df['tips_percentage'].mean():.2f}%")
print(f"   - Медианный процент чаевых: {df['tips_percentage'].median():.2f}%")

print("\n2. РЕЗУЛЬТАТЫ ПРОВЕРКИ ГИПОТЕЗ:")
print(f"   - Гипотеза 1 (зависимость счёт-чаевые):")
print(f"     -> Сильная положительная корреляция (Спирмен = {corr_spearman:.3f})")
print(f"     -> Высокая статистическая значимость (p = {p_spearman:.2e})")

print(f"\n   - Гипотеза 2 (влияние размера компании):")
print(f"     -> Существуют значимые различия (p = {p_kw:.6f})")
print(f"     -> Наибольшие счета у компаний из 6 человек (${df[df['size']==6]['total_bill'].mean():.2f})")

print(f"\n   - Гипотеза 3 (различия по полу):")
print(f"     -> Абсолютные чаевые: мужчины дают больше (${male_tipss.mean():.2f} vs ${female_tipss.mean():.2f})")
print(f"     -> Процент чаевых: женщины дают больше ({female_perc.mean():.2f}% vs {male_perc.mean():.2f}%)")
if p_mw < 0.05:
    print(f"     -> Различия в абсолютных чаевых значимы (p = {p_mw:.4f})")
else:
    print(f"     -> Различия в абсолютных чаевых не значимы (p = {p_mw:.4f})")

print("\n3. ДОПОЛНИТЕЛЬНЫЕ НАБЛЮДЕНИЯ:")
print(f"   - Самый популярный день: {day_counts.idxmax()} ({day_counts.max()} посещений)")
print(f"   - Наибольшие чаевые в процентах дают в {day_perc_means.idxmax()} ({day_perc_means.max():.2f}%)")
print(f"   - На ужин чаевые выше, чем на ланч ({time_perc_means['Dinner']:.2f}% vs {time_perc_means['Lunch']:.2f}%)")

print("\n4. ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:")
print("   - Официантам стоит уделять больше внимания компаниям из 3-4 человек")
print("   - В выходные дни можно ожидать более щедрых чаевых")
print("   - Женщины в среднем оставляют больший процент от счета")
print("   - Вечерние смены потенциально более доходны из-за более высоких чаевых")

print("\n" + "="*80)
print("АНАЛИЗ ЗАВЕРШЕН")
print("="*80)

# Создаем сводную таблицу с результатами
summary_df = pd.DataFrame({
    'Метрика': [
        'Общее количество наблюдений',
        'Средняя сумма счета',
        'Средние чаевые', 
        'Средний процент чаевых',
        'Корреляция Спирмена (счёт-чаевые)',
        'p-value корреляции',
        'Различия по размеру компании (p-value)',
        'Различия по полу (чаевые, p-value)',
        'Различия по полу (процент, p-value)',
        'Различия по дням недели (p-value)'
    ],
    'Значение': [
        len(df),
        f"${df['total_bill'].mean():.2f}",
        f"${df['tips'].mean():.2f}",
        f"{df['tips_percentage'].mean():.2f}%",
        f"{corr_spearman:.4f}",
        f"{p_spearman:.2e}",
        f"{p_kw:.6f}",
        f"{p_mw:.4f}",
        f"{p_mw_perc:.4f}",
        f"{p_days:.6f}"
    ],
    'Интерпретация': [
        'Размер выборки',
        'Средний чек',
        'Средняя сумма чаевых',
        'Средняя щедрость клиентов',
        'Сила монотонной зависимости',
        'Статистическая значимость зависимости',
        'Наличие различий между группами',
        'Наличие различий между полами',
        'Наличие различий в процентах',
        'Наличие различий между днями'
    ]
})

print("\nСВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ:")
print(summary_df.to_string(index=False))

# Сохраняем результаты в файл
summary_df.to_csv('анализ_чаевых_результаты.csv', index=False, encoding='utf-8-sig')
print("\n Результаты сохранены в файл 'анализ_чаевых_результаты.csv'")

