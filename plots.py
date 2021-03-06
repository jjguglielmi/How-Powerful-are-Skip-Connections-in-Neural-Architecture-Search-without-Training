
!pip install matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
!git clone https://github.com/jjguglielmi/Neural-Architecture-Search-without-Training.git


def func(string):
  count = ''
  for i in range(0,3):
    if(i!=0): count = count + ' '
    tot = string.split('+')[i].count('skip')
    count = count + str(tot)
  return count

header=['Dataset', 'Arch', 'LogDet', 'TestAccuracy', 'Time']
# if you are running on Google Colab, left the variable 'csvs' like that; instead if you are not, remove '/content/'
csvs = ['/content/Neural-Architecture-Search-without-Training/pre-trained-architectures/out_C10_15k.csv', 
        '/content/Neural-Architecture-Search-without-Training/pre-trained-architectures/out_cifar100_15k.csv', 
        '/content/Neural-Architecture-Search-without-Training/pre-trained-architectures/out_IN_15k.csv']
datasets = ['Cifar10', 'Cifar100', 'ImageNet16-120']
verbose = False
for csv, ds in zip(csvs, datasets):
  df= pd.read_csv(csv, header=None, names=header)
  countdf = df['Arch'].str.count('skip')
  df['count_skip'] = countdf
  df=df.astype({'count_skip': int, 'TestAccuracy': float})

  sns.set_style("white")
  x0 = df.loc[df['count_skip']==0,  ['TestAccuracy']]
  xS= df.loc[df['count_skip']>0, ['TestAccuracy']]
  x0['Skip/notSkip'] = 'notSkip'
  xS['Skip/notSkip'] = 'Skip'
  xTot= pd.concat([x0, xS])
  plt.figure(figsize=(8,5), dpi= 80)
  sns.histplot(xTot, x='TestAccuracy', hue='Skip/notSkip', element='bars')
  
  #plt.title(f'Skip/noSkip for {ds}')
  plt.savefig(f'histSkip_noSkip{ds}.png')

  skip_df = df[df['Arch'].str.contains('skip')]
  skip_count = skip_df['Arch'].str.count('skip')
  skip_df['count_skip'] = skip_count
  acc_count= skip_df[['TestAccuracy', 'count_skip']].copy()
  acc_no_outliers = acc_count.loc[acc_count['TestAccuracy']>20, ['TestAccuracy', 'count_skip']]
  acc_no_outliers.groupby("count_skip").agg(["mean", "std"]).to_csv(f'comparisonMeanSTDSkips{ds}.csv')

  plt.figure(figsize=(8,5), dpi= 80)
  sns.histplot(acc_count, x='TestAccuracy', hue='count_skip', element='poly')
  #plt.title(f'Skip distribution for {ds}')
  plt.savefig(f'histSkip{ds}.png')
  df_skips= acc_count.loc[(acc_count['count_skip']<=3) & (acc_count['TestAccuracy']>20), ['TestAccuracy', 'count_skip']]

  df_skips
  plt.figure(figsize=(8,5), dpi= 80)
  #plt.title(f'Distribution of architectures based on the number of skips ({ds})')
  sns.histplot(df_skips, x='TestAccuracy', hue='count_skip', element='poly', palette = 'copper_r')
  plt.savefig(f'histHighAcc{ds}.png')
  df_skips.groupby("count_skip").agg(["mean", "std"]).to_csv(f'comparisonMeanSTDSkips{ds}.csv') ## table mean/std of arch with 1/2/3 skips without outliers (acc>20)
  tot= skip_df.loc[(skip_df['count_skip']<=3) & (acc_count['TestAccuracy']>20), ['Arch', 'TestAccuracy', 'count_skip']]
  totPosSkip = tot.copy()
  totPosSkip['pos_countSkip'] = totPosSkip['Arch'].apply(lambda x: str(func(x)))
  new = totPosSkip.filter(['TestAccuracy', 'count_skip', 'pos_countSkip'])
  sorted = new.sort_values('TestAccuracy', ascending=False)
  ##sorted.head(5).to_latex()
  sorted.head(10).to_csv(f'first10SortedArchwithSkipPos-{ds}.csv')

