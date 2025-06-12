import matplotlib.pyplot as plt

metrics = ['F1-score', 'Precision', 'Recall', 'Training time', 'Inference speed']
before = [0.04, 0.5, 0.5, 360, 50]
after = [0.937, 0.933, 0.94, 9, 420]

x = range(len(metrics))
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar([i - width/2 for i in x], before, width, label='Avant Optimisation')
bars2 = ax.bar([i + width/2 for i in x], after, width, label='Après Optimisation')

ax.set_ylabel('Valeurs')
ax.set_title('Comparaison des métriques avant et après optimisation')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

plt.show()
