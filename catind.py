import os
import numpy, pandas, sklearn.cluster, seaborn, matplotlib
from words import words
words = numpy.array(words)
datadir = os.path.expanduser('~/data/verbsim')
subjects = os.listdir(datadir)
nsubjects = len(subjects)
datatems = [
    'Meadows_Semanticverbclustering800_{}_task4_table.csv',
    'Meadows_Semanticverbclustering800_{}_5_table.csv',
    'Meadows__5aabed2c4eae356b48e40a1f_{}_5_table.csv',
]
nwords = 825

# The number of categories shared by each pair of words, by subject
sharedCats = numpy.zeros([nsubjects, nwords, nwords])
for s, subject in enumerate(subjects):
    for datatem in datatems:
        datafpath = os.path.join(datadir, subject, datatem.format(subject))
        if os.path.isfile(datafpath):
            break
    data = pandas.read_csv(datafpath, index_col=0)
    # Make sure the number of unique words in the data corresponds
    assert nwords == numpy.unique(data.values[data.notnull()]).shape[0]
    for column in data.columns:
        # 1 for words in this category, 0 for others
        cat = data[column].notnull().values
        sharedCats[s, cat, :] = sharedCats[s, cat, :] + cat

# If a pair shares multiple categories (due to copies), weigh the copies 10%
sharedCatsDueToCopies = numpy.zeros([nsubjects, nwords, nwords])
sharedCatsDueToCopies[sharedCats>1] = sharedCats[sharedCats>1]-1
sharedCats = sharedCats.clip(max=1) + (0.1 * sharedCatsDueToCopies)
similarities = sharedCats.sum(axis=0) / sharedCats.sum(axis=0).max()
numpy.fill_diagonal(similarities, 1)
dissimilarities = 1 - similarities


model = sklearn.cluster.AgglomerativeClustering(
    n_clusters=20,
    affinity='precomputed',
    linkage='average'
)
labels = model.fit_predict(dissimilarities)
_, counts = numpy.unique(labels, return_counts=True)

# reorder the RDM
newOrder = labels.argsort()
ixgrid = numpy.ix_(newOrder,newOrder)
orderedWords = words[newOrder]
dissimDf = pandas.DataFrame(
    dissimilarities[ixgrid],
    index=orderedWords,
    columns=orderedWords,
)
dims = (15, 15)
fig, ax = matplotlib.pyplot.subplots(figsize=dims)
ax = seaborn.heatmap(ax=ax, data=dissimDf)
fig.savefig('rdm.png')



"""
In [93]: words.index('kiss')
Out[93]: 423

In [94]: words.index('hug')
Out[94]: 379

In [102]: similarities[423, 379]
Out[102]: .55

"""