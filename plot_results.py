import pickle
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import model_from_json
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# 1. Load model
print("Loading model...")
with open('mosquito_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_from_json(model_data['architecture'])
model.set_weights(model_data['weights'])

# 2. Load cached test data
print("Loading test data...")
cache = np.load('dataset_cache.npz')
X_test = cache['X_test'] / 255.0
Y_test = cache['Y_test']

# 3. Model predictions
print("Running predictions...")
preds = model.predict(X_test, verbose=0)
y_pred = np.argmax(preds, axis=1)
y_prob_albopictus = preds[:, 1]
y_prob_aegypti = preds[:, 0]

class_names = ['Aedes aegypti', 'Aedes albopictus']
cm = confusion_matrix(Y_test, y_pred)
fpr, tpr, _ = roc_curve(Y_test, y_prob_albopictus)
roc_auc = auc(fpr, tpr)

# 4. Create multi-panel dashboard graph
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Mosquito Species CNN Model - Evaluation & Performance Dashboard', fontsize=18, fontweight='bold', y=0.98)

# Panel 1: Confusion Matrix
ax1 = axes[0, 0]
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
annot = np.empty_like(cm).astype(str)
for i in range(2):
    for j in range(2):
        annot[i, j] = f"{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)"

sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=True, ax=ax1,
            xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14, "weight": "bold"})
ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel('Predicted Species', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Species', fontsize=12, fontweight='bold')

# Panel 2: Metrics Bar Chart (Precision, Recall, F1)
ax2 = axes[0, 1]
metrics_data = {
    'Aedes aegypti': [0.9607, 0.8833, 0.9204],
    'Aedes albopictus': [0.8920, 0.9639, 0.9266]
}
x = np.arange(3)
width = 0.35

rects1 = ax2.bar(x - width/2, metrics_data['Aedes aegypti'], width, label='Aedes aegypti', color='#1f77b4', alpha=0.9)
rects2 = ax2.bar(x + width/2, metrics_data['Aedes albopictus'], width, label='Aedes albopictus', color='#ff7f0e', alpha=0.9)

ax2.set_title('Classification Metrics by Species', fontsize=14, fontweight='bold', pad=12)
ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['Precision', 'Recall', 'F1-Score'], fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1.15)
ax2.legend(loc='lower right', frameon=True, fontsize=11)

# Annotate bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax2.annotate(f'{height*100:.1f}%',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# Panel 3: ROC Curve
ax3 = axes[1, 0]
ax3.plot(fpr, tpr, color='#2ca02c', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.4f})')
ax3.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Baseline (AUC = 0.5000)')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
ax3.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
ax3.set_title(f'ROC Curve (AUC = {roc_auc:.4f})', fontsize=14, fontweight='bold', pad=12)
ax3.legend(loc="lower right", frameon=True, fontsize=11)

# Panel 4: Prediction Confidence Distribution
ax4 = axes[1, 1]
# Probability of actual class
correct_conf = []
incorrect_conf = []
for i in range(len(Y_test)):
    prob = preds[i, y_pred[i]]
    if y_pred[i] == Y_test[i]:
        correct_conf.append(prob)
    else:
        incorrect_conf.append(prob)

bins = np.linspace(0.5, 1.0, 20)
ax4.hist(correct_conf, bins=bins, color='#2ca02c', alpha=0.75, label=f'Correct Predictions (N={len(correct_conf)})')
ax4.hist(incorrect_conf, bins=bins, color='#d62728', alpha=0.85, label=f'Misclassified (N={len(incorrect_conf)})')
ax4.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold', pad=12)
ax4.set_xlabel('Predicted Probability (Confidence)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left', frameon=True, fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.95])

output_filename = 'model_results_graph.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Graph saved locally to {output_filename}")

# Copy to artifacts directory
artifact_dir = r"C:\Users\LAPTOP\.gemini\antigravity-ide\brain\ef638c74-dc0f-4a93-bad2-e2e7aa71a77b"
if os.path.exists(artifact_dir):
    artifact_path = os.path.join(artifact_dir, 'model_results_graph.png')
    shutil.copyfile(output_filename, artifact_path)
    print(f"Copied to artifact dir: {artifact_path}")
