# Apriori Alogorithm on sample transaction dataset

# Importing the libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Sample transaction dataset
dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
              ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
                ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
                    ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
                        ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

# Encoding the dataset
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)

# Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

# Association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Displaying the results

print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)