import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

# Line chart=========================
years = range(2000, 2012)
apples = [0.895, 0.91, 0.919, 0.926, 0.929, 0.931, 0.934, 0.936, 0.937, 0.9375, 0.9372, 0.939]
oranges = [0.962, 0.941, 0.930, 0.923, 0.918, 0.908, 0.907, 0.904, 0.901, 0.898, 0.9, 0.896, ]
plt.plot(years, apples, 'b-x',
         linewidth=4, markersize=12,
         markeredgewidth=4, markeredgecolor='navy')
plt.plot(years, oranges, 'r--o',
         linewidth=4, markersize=12)
plt.title('Crop yield in Hoenn Region')
plt.legend(['Apples', 'Oranges'])
plt.xlabel('Year')
plt.ylabel('Yield (tons) ')
plt.show()

# Scatter plot ===============================
data = sns.load_dataset('iris')
data.sample(5)
sns.scatterplot(data.sepal_length,
                data.sepal_width,
                hue=data.species,
                s=100)
plt.title('Flowers')
plt.show()

# Histogram and frequency distribution
plt.title("Distribution of sepal width")
sns.distplot(data.sepal_width, kde=False)

plt.title("Distribution of Sepal Width")
sns.distplot(data.sepal_width)

# Contour plot================================
plt.title("Flowers")
sns.kdeplot(data.sepal_length, data.sepal_width, shade=True, shade_lowest=False)

# Box plot====================================
tips = sns.load_dataset("tips")
# Chart title
plt.title("Daily Total Bill")
# Draw a nested boxplot to show bills by day and time
sns.boxplot(tips.day, tips.total_bill, hue=tips.smoker)
