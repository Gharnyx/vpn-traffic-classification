import matplotlib.pyplot as plt

# Pseudo-labeled dataset distribution
labels = ['Non-VPN (62%)', 'VPN (38%)']
sizes = [62, 38]
colors = ['#3498DB', '#E74C3C']

plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, 
        startangle=90, explode=(0.05, 0))
plt.title('Label Distribution in Pseudo-Labeled Dataset', fontsize=14, pad=20)
plt.axis('equal')

plt.savefig('label_distribution.png', dpi=300, bbox_inches='tight')
plt.show()