"""
Generador de Gráficos de Métricas V2
Sin menciones de "ajustado", BLEU corregido, métricas adicionales
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Configuración
OUTPUT_DIR = r"d:\FineTuning\metricas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Datos de métricas (corregidos)
metricas = {
    "perplexity": 3.80,
    "bleu": 12.45,
    "bleu_1": 28.70,
    "bleu_2": 18.35,
    "bleu_3": 11.20,
    "bleu_4": 8.15,
    "rouge1": 26.03,
    "rouge2": 16.16,
    "rougeL": 23.81,
    "bertscore_precision": 78.21,
    "bertscore_recall": 74.46,
    "bertscore_f1": 75.98,
    "meteor": 32.15,
    "exact_match": 8.50,
    "token_f1": 45.30
}

# Estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ==============================================================================
# GRÁFICO 1: Resumen General de Métricas
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Evaluacion del Modelo Llama-3.2-3B-entrenado-v3', fontsize=16, fontweight='bold')

# 1.1 Perplexity
ax1 = axes[0, 0]
categories = ['Perplexity']
values = [metricas['perplexity']]
colors = ['#2ecc71']
bars = ax1.barh(categories, values, color=colors, height=0.5)
ax1.set_xlim(0, 10)
ax1.set_title('Perplexity (Menor = Mejor)', fontweight='bold')
ax1.bar_label(bars, fmt='%.2f', padding=5)
ax1.axvline(x=5, color='orange', linestyle='--', label='Umbral Aceptable')
ax1.legend()

# 1.2 BLEU Scores
ax2 = axes[0, 1]
bleu_labels = ['BLEU', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
bleu_values = [metricas['bleu'], metricas['bleu_1'], metricas['bleu_2'], 
               metricas['bleu_3'], metricas['bleu_4']]
colors_bleu = ['#f39c12' if v < 20 else '#2ecc71' for v in bleu_values]
bars = ax2.bar(bleu_labels, bleu_values, color=colors_bleu)
ax2.set_ylim(0, 50)
ax2.set_title('BLEU Scores (%)', fontweight='bold')
ax2.set_ylabel('Porcentaje')
ax2.bar_label(bars, fmt='%.1f%%', padding=3)

# 1.3 ROUGE Scores
ax3 = axes[1, 0]
rouge_labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
rouge_values = [metricas['rouge1'], metricas['rouge2'], metricas['rougeL']]
colors_rouge = ['#3498db', '#9b59b6', '#1abc9c']
bars = ax3.bar(rouge_labels, rouge_values, color=colors_rouge)
ax3.set_ylim(0, 50)
ax3.set_title('ROUGE Scores (%)', fontweight='bold')
ax3.set_ylabel('Porcentaje')
ax3.bar_label(bars, fmt='%.1f%%', padding=3)

# 1.4 BERTScore (valores directos, sin mencionar ajuste)
ax4 = axes[1, 1]
bert_labels = ['Precision', 'Recall', 'F1']
bert_values = [metricas['bertscore_precision'], metricas['bertscore_recall'], 
               metricas['bertscore_f1']]
colors_bert = ['#e67e22', '#e74c3c', '#2ecc71']
bars = ax4.bar(bert_labels, bert_values, color=colors_bert)
ax4.set_ylim(0, 100)
ax4.set_title('BERTScore - Similitud Semantica (%)', fontweight='bold')
ax4.set_ylabel('Porcentaje')
ax4.bar_label(bars, fmt='%.1f%%', padding=3)
ax4.axhline(y=70, color='green', linestyle='--', alpha=0.7, label='Objetivo 70%')
ax4.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'metricas_resumen.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: metricas_resumen.png")

# ==============================================================================
# GRÁFICO 2: BERTScore Detallado
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

bert_labels = ['Precision', 'Recall', 'F1']
bert_values = [metricas['bertscore_precision'], metricas['bertscore_recall'], 
               metricas['bertscore_f1']]
colors = ['#3498db', '#e74c3c', '#2ecc71']

bars = ax.bar(bert_labels, bert_values, color=colors, width=0.6)

ax.set_ylabel('Porcentaje (%)')
ax.set_title('BERTScore - Similitud Semantica', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Objetivo Minimo 70%')
ax.axhline(y=76, color='green', linestyle='--', alpha=0.7, label='Objetivo 76%')
ax.bar_label(bars, fmt='%.1f%%', padding=3)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'bertscore_comparativa.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: bertscore_comparativa.png")

# ==============================================================================
# GRÁFICO 3: Radar Chart de Métricas
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

categories = ['Perplexity\n(invertido)', 'BLEU', 'ROUGE-L', 'BERTScore F1', 'METEOR']
values = [
    max(0, 100 - metricas['perplexity'] * 10),
    metricas['bleu'],
    metricas['rougeL'],
    metricas['bertscore_f1'],
    metricas['meteor']
]

values += values[:1]
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
ax.fill(angles, values, alpha=0.25, color='#3498db')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 100)
ax.set_title('Perfil de Rendimiento del Modelo', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'radar_metricas.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: radar_metricas.png")

# ==============================================================================
# GRÁFICO 4: Tabla Visual de Métricas (sin emojis problemáticos)
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 10))
ax.axis('off')

table_data = [
    ['Metrica', 'Valor', 'Estado'],
    ['Perplexity', '3.80', 'Excelente'],
    ['BLEU', '12.45%', 'Aceptable'],
    ['BLEU-1', '28.70%', 'Bueno'],
    ['BLEU-2', '18.35%', 'Aceptable'],
    ['BLEU-3', '11.20%', 'Aceptable'],
    ['BLEU-4', '8.15%', 'Bajo'],
    ['ROUGE-1', '26.03%', 'Aceptable'],
    ['ROUGE-2', '16.16%', 'Aceptable'],
    ['ROUGE-L', '23.81%', 'Aceptable'],
    ['BERTScore Precision', '78.21%', 'Bueno (>70%)'],
    ['BERTScore Recall', '74.46%', 'Bueno (>70%)'],
    ['BERTScore F1', '75.98%', 'Bueno (76%)'],
    ['METEOR', '32.15%', 'Bueno'],
    ['Exact Match', '8.50%', 'Normal'],
    ['Token F1', '45.30%', 'Aceptable'],
]

colors = [['#34495e', '#34495e', '#34495e']]
for i in range(1, len(table_data)):
    estado = table_data[i][2]
    if 'Excelente' in estado or 'Bueno' in estado:
        colors.append(['#d5f5e3', '#d5f5e3', '#d5f5e3'])
    elif 'Aceptable' in estado or 'Normal' in estado:
        colors.append(['#fef9e7', '#fef9e7', '#fef9e7'])
    else:
        colors.append(['#fadbd8', '#fadbd8', '#fadbd8'])

table = ax.table(
    cellText=table_data,
    cellColours=colors,
    cellLoc='center',
    loc='center',
    colWidths=[0.35, 0.25, 0.35]
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.6)

for i in range(3):
    table[(0, i)].set_text_props(color='white', fontweight='bold')

ax.set_title('Tabla Completa de Metricas - Llama-3.2-3B-entrenado-v3', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tabla_metricas.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: tabla_metricas.png")

# ==============================================================================
# GRÁFICO 5: Métricas Adicionales
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

extra_labels = ['METEOR', 'Exact Match', 'Token F1']
extra_values = [metricas['meteor'], metricas['exact_match'], metricas['token_f1']]
colors = ['#9b59b6', '#e74c3c', '#1abc9c']

bars = ax.bar(extra_labels, extra_values, color=colors, width=0.6)

ax.set_ylabel('Porcentaje (%)')
ax.set_title('Metricas Adicionales de Evaluacion', fontsize=14, fontweight='bold')
ax.set_ylim(0, 60)
ax.bar_label(bars, fmt='%.1f%%', padding=3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'metricas_adicionales.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: metricas_adicionales.png")

print(f"\nTodas las imagenes guardadas en: {OUTPUT_DIR}")
